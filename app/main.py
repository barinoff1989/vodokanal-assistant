"""Точка входа: сборка приложения с реальными зависимостями.

Запуск:

    python -m uvicorn app.main:app --reload

Стенд открывается на http://localhost:8000, API — там же по путям `/v1/...`.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.backend.orchestrator import Orchestrator
from app.config import get_settings
from app.gateway.llm_gateway import LlmGateway
from app.gateway.quota import QuotaManager
from app.kb.search import KnowledgeBase, SentenceTransformerEmbedder
from app.outages.answer import OutageResponder
from app.outages.store import OutageStore
from app.regulated import RegulatedResponder

logger = logging.getLogger(__name__)


def _build_quota() -> QuotaManager | None:
    """Собрать учёт лимитов, если клиент хранилища установлен.

    Подключение к Redis создаётся лениво и не проверяется здесь: по ADR-008
    недоступность хранилища — это ответ `503` на конкретный запрос, а не отказ
    сервиса подняться. Иначе перезапуск Redis требовал бы перезапуска и шлюза.
    """
    settings = get_settings()
    try:
        import redis
        from redis.backoff import NoBackoff
        from redis.retry import Retry

        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=settings.redis_timeout_seconds,
            socket_timeout=settings.redis_timeout_seconds,
            # Повторы выключены намеренно, и это главное из трёх изменений:
            # именно они, а не таймаут, давали основную задержку (26,2 с без
            # таймаута, 8,7 с с таймаутом и повторами, 0,5 с без них).
            #
            # По ADR-008 недоступное хранилище — это `503` + `Retry-After`, то
            # есть повторяет **клиент**, по названному ему времени. Повтор
            # внутри запроса удлиняет ожидание абонента ровно во столько раз,
            # сколько попыток, и делает это невидимым: снаружи виден один
            # долгий запрос, а не четыре быстрых отказа.
            retry=Retry(NoBackoff(), 0),
        )
        quota = QuotaManager(
            client,
            subscriber_limit=settings.quota_subscriber_per_minute,
            session_limit=settings.quota_session_per_minute,
            service_limit=settings.quota_service_per_minute,
            store_retry_after=settings.quota_retry_after_seconds,
        )
    except ImportError:  # pragma: no cover — клиент есть в зависимостях шага 4
        logger.warning("клиент Redis не установлен: лимиты не проверяются")
        quota = None

    return quota


def _build_knowledge_base() -> KnowledgeBase | None:
    """Проиндексировать корпус, если он на месте.

    Отсутствие корпуса — не отказ подняться: поиск выключается, модель отвечает
    без опоры на регламенты. На прототипе это допустимо и заметно по журналу; на
    MVP так работать нельзя, и там пустая база знаний должна ронять запуск.
    """
    settings = get_settings()
    path = Path(settings.kb_corpus_path)
    if not path.exists():
        logger.warning("корпус базы знаний не найден (%s): поиск выключен", path)
        return None

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        local_files_only=settings.embedding_local_files_only,
    )
    return KnowledgeBase.from_file(
        path, embedder, part_max_chars=settings.kb_part_max_chars
    )


def _build_outages() -> OutageResponder | None:
    """Прочитать график отключений, если файл на месте.

    Отметка актуальности берётся от времени чтения, а не от дат внутри файла:
    даты в графике говорят о воде, а не о свежести графика (ADR-013).
    """
    settings = get_settings()
    path = Path(settings.outage_schedule_path)
    if not path.exists():
        logger.warning("график отключений не найден (%s): ответы по нему выключены", path)
        return None

    store = OutageStore.from_file(path, year=datetime.now().year, loaded_at=datetime.now())
    return OutageResponder(store)


def _build_regulated() -> RegulatedResponder:
    """Реестр регламентных ответов (ADR-012).

    Поднимается с `strict=False`: формулировка о качестве воды написана нами и
    владельцем не утверждена (пункт 37 TODO). Дыра видна по предупреждению в
    журнале — та же дисциплина, что у пустой базы знаний. **На MVP строгий режим
    обязателен:** неутверждённый текст там должен ронять запуск, а не тихо
    уходить абоненту как регламентный.
    """
    settings = get_settings()
    return RegulatedResponder(phone=settings.contact_center_phone, strict=False)


@contextmanager
def _stage(name: str) -> Iterator[None]:
    """Сообщить, чем сервис занят, и сколько это заняло.

    **Подъём сервиса занимает около минуты, и это не сбой,** а цена решения
    раздела 50.3: ленивые загрузки вынесены из первого запроса абонента в старт.
    Замер частей в отдельных процессах:

    ===================================================== ======
      обезличиватель: presidio и русская модель spaCy       24,9 с
      база знаний: модель эмбеддингов и индексация          54,0 с
      **`app.main` целиком**                                45,3 с
    ===================================================== ======

    Сумма частей больше целого: `torch` и `transformers` нужны обеим и грузятся
    **один раз на двоих**. Отсюда же следует, что убрать одну из двух моделей
    сэкономит куда меньше, чем стоит она сама.

    До этих сообщений сервис молчал всю минуту, и отличить «грузится» от
    «завис» было нечем — в том числе на демонстрации.
    """
    logger.info("подъём: %s…", name)
    started = time.perf_counter()
    yield
    logger.info("подъём: %s — готово за %.1f с", name, time.perf_counter() - started)


def create() -> object:
    """Собрать приложение. Вынесено функцией ради проверок."""
    from app.api import create_app
    from app.metrics import prometheus as metrics

    # Метрики отдаются отдельным портом, а не адресом основного приложения:
    # правило 4.4 разрешает ровно два адреса, и расширять контракт ради
    # служебной надобности не следует. Их читает система сбора, абоненту они
    # не показываются.
    try:
        metrics.serve(port=int(os.getenv("METRICS_PORT", "9100")))
    except OSError as exc:  # порт занят — сервис всё равно должен подняться
        logger.warning("отдача метрик не запущена: %s", exc)

    started = time.perf_counter()
    settings = get_settings()
    quota = _build_quota()

    # Порядок опроса значим: ответчики возвращают None на чужой теме, но
    # реестр дешевле поиска по графику, а тем у него меньше.
    with _stage("реестр ответов и график отключений"):
        responders = tuple(
            r for r in (_build_regulated(), _build_outages()) if r is not None
        )

    with _stage("слой защиты: обезличиватель и охранители"):
        gateway = LlmGateway(quota=quota, settings=settings)

    with _stage("база знаний: модель эмбеддингов и индексация"):
        knowledge_base = _build_knowledge_base()

    orchestrator = Orchestrator(
        gateway,
        knowledge_base=knowledge_base,
        direct=responders,
        quota=quota,
        settings=settings,
    )
    logger.info("подъём завершён за %.1f с", time.perf_counter() - started)
    return create_app(orchestrator)


app = create()
