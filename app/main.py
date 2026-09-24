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

from app.adapters.inquiry_service import InquiryServiceAdapter
from app.agents.inquiry_type_fallback import InquiryTypeFallbackClassifier
from app.agents.topic_fallback import TopicFallbackClassifier
from app.agents.triage import Triage
from app.backend.orchestrator import Orchestrator
from app.backend.registration import Registrar
from app.backend.sessions import SessionStore
from app.billing.answer import AccountResponder
from app.billing.source import BillingSource
from app.config import get_settings
from app.documents.artifact import ArtifactStore
from app.documents.responder import TemplateResponder
from app.gateway.llm_gateway import LlmGateway
from app.gateway.quota import QuotaManager
from app.gateway.usage import UsageStore
from app.kb.build import build_knowledge_base
from app.outages.answer import OutageResponder
from app.outages.store import OutageStore
from app.quality.judge import AnswerJudge
from app.quality.live import LiveJudge
from app.quality.store import AssessmentStore
from app.regulated import RegulatedResponder
from app.tariffs.answer import TariffResponder
from app.tariffs.store import TariffStore

logger = logging.getLogger(__name__)


def _build_quota() -> tuple[QuotaManager | None, object | None]:
    """Собрать учёт лимитов, если клиент хранилища установлен.

    Подключение к Redis создаётся лениво и не проверяется здесь: по ADR-400
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
            # По ADR-400 недоступное хранилище — это `503` + `Retry-After`, то
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
        return quota, client
    except ImportError:  # pragma: no cover — клиент есть в зависимостях
        logger.warning("клиент Redis не установлен: лимиты не проверяются")
        return None, None


def _build_outages() -> OutageResponder | None:
    """Прочитать график отключений, если файл на месте.

    Отметка актуальности берётся от времени чтения, а не от дат внутри файла:
    даты в графике говорят о воде, а не о свежести графика (ADR-100).
    """
    settings = get_settings()
    path = Path(settings.outage_schedule_path)
    if not path.exists():
        logger.warning("график отключений не найден (%s): ответы по нему выключены", path)
        return None

    store = OutageStore.from_file(path, year=datetime.now().year, loaded_at=datetime.now())
    return OutageResponder(store)


def _build_regulated() -> RegulatedResponder:
    """Реестр регламентных ответов (ADR-100).

    Поднимается с `strict=False`: формулировка о качестве воды написана нами и
    владельцем не утверждена. Дыра видна по предупреждению в
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
    — ленивые загрузки вынесены из первого запроса абонента в старт.
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


def _build_tariffs() -> TariffResponder | None:
    """Прочитать таблицу тарифов, если файл на месте.

    Отметка актуальности берётся из самого файла (`fetched_on`), а не от времени
    чтения: тариф меняется решением регулятора, и абоненту важно, когда прочитана
    **страница**, а не когда поднялся сервис.
    """
    settings = get_settings()
    path = Path(settings.tariff_table_path)
    if not path.exists():
        logger.warning("таблица тарифов не найдена (%s): ответы по ней выключены", path)
        return None

    return TariffResponder(TariffStore.from_file(path))


def _build_billing_source() -> BillingSource | None:
    """Пример данных Биллинга.

    Из него берутся ФИО и лицевой счёт для слотов регистрации, а также факты для
    ответов о лицевом счёте (`AccountResponder`): задолженность, начисления,
    сроки поверки, показания. Один экземпляр на оба пути — файлы читаются при
    старте один раз.
    """
    settings = get_settings()
    path = Path(settings.billing_data_path)
    if not path.exists():
        logger.warning(
            "пример данных Биллинга не найден (%s): слоты и ответы о счёте выключены", path
        )
        return None
    from app.billing.source import CsvBillingSource

    return CsvBillingSource(path)


def _build_usage() -> UsageStore:
    """Хранилище событий использования (база `telemetry` в Postgres).

    Выключается само, если нет Postgres или драйвера: телеметрия важна, но её
    отсутствие не должно ронять сервис. Prometheus при этом работает — пути
    независимы.
    """
    settings = get_settings()
    try:
        import psycopg

        store = UsageStore(lambda: psycopg.connect(settings.telemetry_dsn))
        store.ensure_schema()
        return store
    except Exception as exc:  # noqa: BLE001 — драйвера может не быть, базы тоже
        logger.warning("запись событий использования выключена: %s", exc)
        return UsageStore(None)


def _build_live_judge() -> LiveJudge | None:
    """Судья на живом пути — async-часть Guardrails (C4_L3_LLM).

    Выключается, если `quality_sample_rate = 0` или нет Postgres/драйвера:
    оценка качества не должна ронять сервис, и ночной прогон от неё не зависит.
    """
    settings = get_settings()
    if settings.quality_sample_rate <= 0.0:
        return None
    try:
        import psycopg

        store = AssessmentStore(lambda: psycopg.connect(settings.telemetry_dsn))
        store.ensure_schema()
        return LiveJudge(
            judge=AnswerJudge(settings=settings),
            store=store,
            sample_rate=settings.quality_sample_rate,
            answering_alias=settings.llm_provider,
        )
    except Exception as exc:  # noqa: BLE001 — драйвера может не быть, базы тоже
        logger.warning("судья на живом пути выключен: %s", exc)
        return None


def _build_documents() -> ArtifactStore:
    """Хранилище готовых бланков — дубль S3/MinIO на прототипе.

    Каталог создаётся при первой записи, не здесь: пустой каталог до первого
    бланка незачем. Отказ файловой системы гасится в самом `put` — сборка
    приложения из-за него не падает.
    """
    settings = get_settings()
    return ArtifactStore(
        root=Path(settings.document_store_path),
        ttl_seconds=settings.document_link_ttl_seconds,
    )


def _build_registrar(
    quota_client: object | None, billing: BillingSource | None
) -> tuple[SessionStore, Registrar | None]:
    """Собрать состояние диалога и регистрацию обращений.

    Хранилище сессий и Redis лимитов — **один и тот же экземпляр**: это одна
    служба, и второе подключение к ней означало бы второй таймаут и второй отказ
    там, где отказ один.

    Регистрация выключается сама, если нет Postgres: обращение некуда записать,
    и предлагать его абоненту значило бы обещать несделанное. Ответы на вопросы
    при этом работают — пути независимы.
    """
    settings = get_settings()
    sessions = SessionStore(quota_client, ttl_seconds=settings.session_ttl_seconds)

    try:
        import psycopg

        from app.inquiries.store import InquiryStore

        store = InquiryStore(lambda: psycopg.connect(settings.inquiries_dsn))
        store.ensure_schema()
    except Exception as exc:  # noqa: BLE001 — драйвера может не быть, базы тоже
        logger.warning("регистрация обращений выключена: %s", exc)
        return sessions, None

    return sessions, Registrar(
        sessions,
        adapter=InquiryServiceAdapter(client=store),
        billing=billing,
    )


def create() -> object:
    """Собрать приложение. Вынесено функцией ради проверок."""
    from app.api import create_app
    from app.metrics import prometheus as metrics

    # Метрики отдаются отдельным портом, а не адресом основного приложения:
    # контракт разрешает ровно два адреса, и расширять контракт ради
    # служебной надобности не следует. Их читает система сбора, абоненту они
    # не показываются.
    try:
        metrics.serve(port=int(os.getenv("METRICS_PORT", "9100")))
    except OSError as exc:  # порт занят — сервис всё равно должен подняться
        logger.warning("отдача метрик не запущена: %s", exc)

    started = time.perf_counter()
    settings = get_settings()
    quota, redis_client = _build_quota()
    billing = _build_billing_source()
    usage = _build_usage()
    live_judge = _build_live_judge()
    documents = _build_documents()
    sessions, registrar = _build_registrar(redis_client, billing)

    # Порядок опроса значим: ответчики возвращают None на чужой теме, но
    # реестр дешевле поиска по графику, а тем у него меньше.
    with _stage("прямые ответчики: реестр, тарифы, отключения, лицевой счёт, бланки"):
        account = AccountResponder(billing) if billing is not None else None
        templates = TemplateResponder(
            billing=billing, settings=settings, sessions=sessions, artifacts=documents
        )
        responders = tuple(
            r
            for r in (
                _build_regulated(),
                _build_tariffs(),
                _build_outages(),
                account,
                templates,
            )
            if r is not None
        )

    with _stage("слой защиты: обезличиватель и охранители"):
        gateway = LlmGateway(
            quota=quota, settings=settings, usage=usage, live_judge=live_judge
        )

    with _stage("база знаний: модель эмбеддингов и индексация"):
        # Qdrant — Backend только читает уже наполненную коллекцию
        # (`scripts/reindex_kb.py` наполняет её отдельным прогоном), не считает
        # эмбеддинги всего корпуса заново на каждом рестарте. `memory` не
        # переживает рестарт сама по себе, поэтому там reindex остаётся
        # обязательным — `build_knowledge_base` это учитывает сам.
        knowledge_base = build_knowledge_base(
            settings, reindex=settings.vector_store != "qdrant"
        )

    # Запасные классификаторы (тема и тип обращения) переиспользуют уже
    # поднятую модель эмбеддингов базы знаний — вторая копия весов не
    # грузится. Базы знаний нет (пустой корпус) — оба
    # классификатора выключены, как и сам поиск.
    triage = Triage(
        model_fallback=(
            TopicFallbackClassifier(knowledge_base.embedder)
            if knowledge_base is not None
            else None
        ),
        type_fallback=(
            InquiryTypeFallbackClassifier(knowledge_base.embedder)
            if knowledge_base is not None
            else None
        ),
    )
    orchestrator = Orchestrator(
        gateway,
        knowledge_base=knowledge_base,
        direct=responders,
        quota=quota,
        settings=settings,
        sessions=sessions,
        registrar=registrar,
        billing=billing,
        triage=triage,
    )
    logger.info("подъём завершён за %.1f с", time.perf_counter() - started)
    return create_app(orchestrator, documents=documents)


app = create()
