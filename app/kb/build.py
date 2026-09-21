"""Сборка индекса базы знаний из всех её источников.

Вынесено из `app/main.py` отдельным модулем **не ради красоты**: в `main`
выражение `app = create()` выполняется на уровне модуля, и любой импорт оттуда
поднимает всё приложение целиком — метрики, шлюз, модели. Скрипту замера это
обошлось бы в минуту на пустом месте, а сам замер стал бы мерить не индекс, а
подъём сервиса (журнал, раздел 72.2).

Второй довод важнее первого: пока сборка жила в `main`, скрипт замера собирал
индекс **по-своему** — только из корпуса FAQ. То есть мерил не ту базу знаний,
с которой работает сервис. Один источник истины на сборку убирает этот класс
расхождения целиком.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.gateway.pii_filter import FILTERED_ENTITIES, PiiSanitizer
from app.kb.documents import load_directory
from app.kb.search import (
    KbItem,
    KnowledgeBase,
    SentenceTransformerEmbedder,
    faq_items,
)

__all__ = ["PersonalDataInCorpusError", "build_knowledge_base", "collect_items"]

logger = logging.getLogger(__name__)



class PersonalDataInCorpusError(RuntimeError):
    """В документе базы знаний нашлись персональные данные (ADR-015).

    **Ошибка, а не предупреждение.** Предупреждение в журнале — способ не
    заметить: одна такая дыра в проекте уже заведена осознанно (пустая база
    знаний, раздел 62.5), и заводить вторую нельзя. Документ с персональными
    данными в базе знаний — это 152-ФЗ, а не вопрос качества ответа.

    Находка здесь имеет последствие: документ не берётся в корпус, и человек об
    этом узнаёт. На пути абонента последствия нет — там остаётся только испортить
    текст, что прежняя редакция и делала.
    """

    def __init__(self, chunk_id: str, entities: tuple[str, ...]) -> None:
        super().__init__(
            f"в документе {chunk_id!r} найдены персональные данные: "
            f"{', '.join(entities)}. Документ в базу знаний не берётся (ADR-015). "
            "Уберите данные из источника и пересоберите корпус."
        )
        self.chunk_id = chunk_id
        self.entities = entities


def check_no_personal_data(items: list[KbItem]) -> None:
    """Проверить корпус на персональные данные — один раз, при приёме.

    Проверяется заголовок вместе с телом: разрезание на части происходит позже, и
    данные могли бы оказаться на стыке.

    **Отказ вызывают только сильные свидетельства**, и это не послабление, а
    условие работоспособности. Первая редакция проверки принимала любую находку —
    и отказалась брать наш собственный корпус, найдя «персональные данные» в
    словах «Перерасчёт» и «Документы». То есть ложные срабатывания, ради
    избавления от которых ADR-015 и написан, вернулись бы на другом конце:
    вместо порчи текста — отказ собрать базу знаний.

    Что считается сильным свидетельством: СНИЛС, ИНН, лицевой счёт и телефон —
    у них контрольные суммы и слова-подсказки; почта — жёсткая форма, на которой
    разбор языка ложных находок не давал (раздел 56.10). Настоящий лицевой счёт
    в документе они находят, а слово «Перерасчёт» — нет.

    Что отбрасывается: `PERSON` и `LOCATION` от разбора языка — те самые
    :data:`FILTERED_ENTITIES`, для которых обезличиватель и так держит отдельный
    отсев по форме и словарю.

    **Чего проверка не поймает:** ФИО в тексте документа. Это принятая цена, и
    она названа прямо: имя без номера, без ИНН и без телефона неотличимо от
    фамилии автора регламента, а таких в документах владельца будет много.
    Блокировать по ним значило бы не собрать корпус вовсе.

    Обойти проверку можно только правкой кода: она стоит внутри `collect_items`,
    а другого пути в корпус нет. Это и есть условие, при котором ADR-015 остаётся
    верным — риск назван в самом решении.
    """
    sanitizer = PiiSanitizer()
    for item in items:
        spans = [
            span
            for span in sanitizer.find(item.title + "\n" + item.body)
            if span.entity_type not in FILTERED_ENTITIES
        ]
        if spans:
            raise PersonalDataInCorpusError(
                item.chunk_id, tuple(sorted({span.entity_type for span in spans}))
            )


def collect_items(settings: Settings | None = None) -> list[KbItem]:
    """Записи корпуса из всех источников: FAQ и документы Word.

    Корпус FAQ обязателен, документы — нет: каталога может не быть, и тогда
    индексируется одно FAQ. Пустой корпус — не отказ подняться, а выключенный
    поиск; на MVP так работать нельзя (пункт 54 TODO).
    """
    settings = settings or get_settings()
    corpus = Path(settings.kb_corpus_path)
    items = faq_items(corpus) if corpus.exists() else []

    # Документы Word — источники 2, 5 и 8 каталога. На прототипе они
    # синтетические, и признак этого доходит до фрагмента, а оттуда до ответа.
    sections = load_directory(Path(settings.kb_documents_path))
    items.extend(
        KbItem(
            chunk_id=section.chunk_id,
            title=section.heading,
            body=section.body,
            source_title=section.source_title,
            source_url=None,
            synthetic=section.synthetic,
            inquiry_type=section.inquiry_type,
        )
        for section in sections
    )

    # Проверка стоит здесь, а не в вызывающем коде: `collect_items` — единственный
    # путь в корпус, и другого места, где её нельзя было бы забыть, нет.
    check_no_personal_data(items)
    return items


def build_knowledge_base(
    settings: Settings | None = None, *, reindex: bool = True
) -> KnowledgeBase | None:
    """Проиндексировать корпус — либо открыть уже готовый индекс в Qdrant.

    Отсутствие корпуса (или пустая коллекция Qdrant) — не отказ подняться: поиск
    выключается, модель отвечает без опоры на регламенты. На прототипе это
    допустимо и заметно по журналу; на MVP так работать нельзя, и там пустая
    база знаний должна ронять запуск.

    :param reindex: пересобрать корпус и загрузить его в хранилище заново.
        По умолчанию — да, и для `memory` это единственный осмысленный режим:
        хранилище живёт только в этом процессе, открыть чужой индекс негде.
        Для `qdrant` можно поставить `False` — тогда Backend лишь открывает
        уже наполненную коллекцию (`QdrantVectorStore.attach`) вместо того,
        чтобы на каждом рестарте заново считать эмбеддинги всего корпуса
        (минуты на CPU без ускорителя). Пересборку данных в этом режиме
        делает отдельный прогон — `python scripts/reindex_kb.py` — так
        целевая архитектура и разносит это на ETL Worker и читающий Backend
        (раздел 5.5 контекста), просто без своего планировщика задач: на
        объёме прототипа он не нужен, запускается вручную, когда меняется
        корпус.
    """
    settings = settings or get_settings()

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        local_files_only=settings.embedding_local_files_only,
    )

    reranker = None
    if settings.reranker_enabled:
        # Ленивый импорт: та же причина, что у qdrant-client ниже —
        # пакет не должен требоваться, пока опция не выбрана явно.
        from app.kb.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            settings.reranker_model, local_files_only=settings.embedding_local_files_only
        )
        # Индексация прогревает Embedder сама (encode() вызывается прямо
        # сейчас, при сборке корпуса) — у реранкера такого триггера нет, и
        # без явного прогрева загрузка спряталась бы в первый запрос абонента
        # (та же ошибка, что уже находили четыре раза, см. docstring reranker.py).
        reranker.warm_up()
        logger.info("переранжирование: включено (%s)", settings.reranker_model)

    if settings.vector_store == "qdrant" and not reindex:
        from app.kb.qdrant_store import QdrantVectorStore

        attached_store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection)
        count = attached_store.attach()
        if count == 0:
            logger.warning(
                "коллекция Qdrant %r пуста или не создана (%s): поиск выключен, "
                "пока не выполнен `python scripts/reindex_kb.py`",
                settings.qdrant_collection,
                settings.qdrant_url,
            )
            return None
        logger.info(
            "база знаний: открыта коллекция Qdrant %r, %d фрагментов",
            settings.qdrant_collection,
            count,
        )
        return KnowledgeBase(attached_store, embedder, reranker)

    items = collect_items(settings)
    if not items:
        logger.warning(
            "корпус базы знаний пуст (%s, %s): поиск выключен",
            settings.kb_corpus_path,
            settings.kb_documents_path,
        )
        return None

    synthetic = sum(item.synthetic for item in items)
    logger.info(
        "база знаний: %d фрагментов, синтетических %d", len(items), synthetic
    )

    store = None
    if settings.vector_store == "qdrant":
        # Ленивый импорт: qdrant-client не должен требоваться, пока хранилище
        # не выбрано явно (тот же приём, что sentence_transformers в Embedder).
        from app.kb.qdrant_store import QdrantVectorStore

        # blue-green: пишем в новую версию коллекции, алиас переключается
        # только после полной загрузки — поиск по нему не видит недособранного.
        store = QdrantVectorStore(
            settings.qdrant_url, settings.qdrant_collection, blue_green=True
        )
        logger.info(
            "векторное хранилище: Qdrant (%s, алиас %s)",
            settings.qdrant_url,
            settings.qdrant_collection,
        )

    knowledge_base = KnowledgeBase.from_items(
        items,
        embedder,
        part_max_chars=settings.kb_part_max_chars,
        store=store,
        reranker=reranker,
    )
    if store is not None:
        published = store.publish()
        logger.info("алиас %s -> %s", settings.qdrant_collection, published)
    return knowledge_base
