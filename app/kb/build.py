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
from app.kb.documents import load_directory
from app.kb.search import (
    KbItem,
    KnowledgeBase,
    SentenceTransformerEmbedder,
    faq_items,
)

__all__ = ["build_knowledge_base", "collect_items"]

logger = logging.getLogger(__name__)


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
