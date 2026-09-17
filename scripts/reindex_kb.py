"""Переиндексировать базу знаний в Qdrant — прототипный ETL Worker.

Целевая архитектура разносит эту работу на отдельный контейнер (ETL Worker,
запускаемый Celery Beat по расписанию — контекст, раздел про K8s CronJob vs
Celery Beat) и держит Backend только читающей стороной. На прототипе своей
очереди задач нет и не нужна: пайплайн одношаговый (парсинг → чанки →
эмбеддинг → upsert), запускается вручную, когда меняется корпус
(`kb/faq_voronezh.json` или синтетические документы `scripts/generate_kb_docs.py`).

ПОЧЕМУ ОТДЕЛЬНЫМ ПРОГОНОМ, А НЕ ПРИ КАЖДОМ СТАРТЕ BACKEND. До этого скрипта
`build_knowledge_base()` пересчитывала эмбеддинги всего корпуса на каждом
рестарте Backend, даже когда хранилище — Qdrant, то есть данные и так
переживают рестарт (замерено: около 95 секунд на 51 записи без ускорителя,
раздел «Собрана цепочка на Qdrant», журнал). С этим скриптом Backend при
`VECTOR_STORE=qdrant` только открывает уже наполненную коллекцию
(`QdrantVectorStore.attach`, `build_knowledge_base(reindex=False)`) —
переиндексация происходит здесь, по явному запуску.

ДЛЯ `VECTOR_STORE=memory` ЭТОТ СКРИПТ БЕСПОЛЕЗЕН. Хранилище живёт только в
памяти процесса Backend — переиндексировать его отдельно от Backend негде,
пересборка при каждом старте и есть штатный путь (ADR-006, «Отступление
прототипа»).

ЗАПУСК

    python scripts/reindex_kb.py
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.kb.build import build_knowledge_base

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()

    if settings.vector_store != "qdrant":
        print(
            f"VECTOR_STORE={settings.vector_store!r}, а не 'qdrant' — переиндексация "
            "отдельным прогоном имеет смысл только для Qdrant. Для 'memory' Backend "
            "строит корпус сам при каждом старте, здесь делать нечего."
        )
        return 1

    knowledge_base = build_knowledge_base(settings, reindex=True)
    if knowledge_base is None:
        print("корпус пуст — переиндексация не выполнена, коллекция не создана")
        return 1

    print(
        f"переиндексация завершена: {len(knowledge_base)} фрагментов "
        f"в коллекции {settings.qdrant_collection!r} ({settings.qdrant_url})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
