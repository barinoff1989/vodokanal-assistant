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
        )
        for section in sections
    )

    # Проверка стоит здесь, а не в вызывающем коде: `collect_items` — единственный
    # путь в корпус, и другого места, где её нельзя было бы забыть, нет.
    check_no_personal_data(items)
    return items


def build_knowledge_base(settings: Settings | None = None) -> KnowledgeBase | None:
    """Проиндексировать корпус, если он на месте.

    Отсутствие корпуса — не отказ подняться: поиск выключается, модель отвечает
    без опоры на регламенты. На прототипе это допустимо и заметно по журналу; на
    MVP так работать нельзя, и там пустая база знаний должна ронять запуск.
    """
    settings = settings or get_settings()
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

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        local_files_only=settings.embedding_local_files_only,
    )
    return KnowledgeBase.from_items(
        items, embedder, part_max_chars=settings.kb_part_max_chars
    )
