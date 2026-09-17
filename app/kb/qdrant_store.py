"""Qdrant как хранилище векторов базы знаний (ADR-006, минимальная реализация).

Реализует протокол `VectorStore` (`app/kb/search.py`) — `KnowledgeBase` не знает
о существовании Qdrant, просто получает другую реализацию `search_parts`.

ЧТО ЗДЕСЬ ЕСТЬ. Один фрагмент — несколько точек Qdrant (заголовок/вопрос плюс
части ответа, как и в памяти), payload несёт всё нужное для восстановления
`ContextChunk`, коллекция создаётся лениво при первой загрузке (размер вектора
берётся из самих данных, а не захардкожен). Поиск — точный: лимит выборки равен
числу точек в коллекции, поэтому на объёме прототипа это не приближение
HNSW-графа, а полный перебор через Qdrant — то же самое, что раньше делал
Python-цикл, просто по сети.

ЧЕГО ЗДЕСЬ НЕТ — ОСОЗНАННО, ЭТО НЕ ВЕСЬ ADR-006. Payload-фильтр по
`inquiry_type` не создаётся (нечем фильтровать — таксономия документов не
размечена); blue-green переключение alias при реиндексе не реализовано
(переиндексация здесь — `upsert` тех же ID, оверрайт на месте); проверка
несовпадения версии модели эмбеддингов с алертом (ADR-006, п. 7 подтверждения)
не написана. Каждое — отдельная задача дорожной карты (ADR-021), не забытая
часть этой.

`qdrant-client` импортируется лениво (как `sentence_transformers` в
`SentenceTransformerEmbedder`) — модуль не должен требовать пакет, если Qdrant
не выбран (`settings.vector_store != "qdrant"`).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from app.kb.search import _Entry
from app.models import ContextChunk

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

__all__ = ["QdrantVectorStore"]

# Фиксированный namespace для uuid5: id точки детерминирован по (chunk_id, часть),
# а не случаен — повторный upsert того же фрагмента переписывает те же точки,
# а не плодит дубли. Значение само по себе не имеет смысла, важна стабильность.
_NAMESPACE = uuid.UUID("6f6e5f8e-6b1a-4e2b-9f2a-2f7c2a9d6b40")


def _point_id(chunk_id: str, part_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{chunk_id}:{part_index}"))


def _chunk_from_payload(payload: dict[str, Any]) -> ContextChunk:
    # relevance_score здесь — заглушка: тот же приём, что у `_Entry.chunk` в
    # памяти, реальное значение подставляет `KnowledgeBase.search()`.
    return ContextChunk(
        chunk_id=payload["chunk_id"],
        text=payload["text"],
        source_title=payload["source_title"],
        source_url=payload.get("source_url"),
        synthetic=payload.get("synthetic", False),
        relevance_score=0.0,
    )


class QdrantVectorStore:
    """Векторное хранилище на Qdrant вместо памяти процесса.

    :param url: адрес Qdrant (`http://localhost:6333`) либо `:memory:` —
        встроенный режим `qdrant-client` без сервера, для тестов.
    :param collection: имя коллекции. Отдельное имя на окружение убережёт от
        путаницы, если несколько процессов используют одну и ту же Qdrant.

    Два способа наполнить `_point_count`/`_chunk_ids`, которыми живут
    `search_parts`/`__len__`: `upsert` (пишущая сторона, `scripts/reindex_kb.py`)
    и `attach` (читающая сторона, `build_knowledge_base(reindex=False)`,
    боевой путь Backend при `VECTOR_STORE=qdrant`) — см. докстринг `attach`."""

    def __init__(self, url: str, collection: str) -> None:
        self._url = url
        self._collection = collection
        self._client: QdrantClient | None = None
        self._point_count = 0
        self._chunk_ids: set[str] = set()

    def _ensure_client(self) -> QdrantClient:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = (
                QdrantClient(location=":memory:")
                if self._url == ":memory:"
                else QdrantClient(url=self._url)
            )
        return self._client

    def upsert(self, entries: Sequence[_Entry]) -> None:
        from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

        client = self._ensure_client()
        points = [
            PointStruct(
                id=_point_id(entry.chunk.chunk_id, part_index),
                vector=list(vector),
                payload={
                    "chunk_id": entry.chunk.chunk_id,
                    "text": entry.chunk.text,
                    "source_title": entry.chunk.source_title,
                    "source_url": entry.chunk.source_url,
                    "synthetic": entry.chunk.synthetic,
                },
            )
            for entry in entries
            for part_index, vector in enumerate(entry.vectors)
        ]
        if not points:
            return

        if not client.collection_exists(self._collection):
            # Не из `points[0].vector`: у PointStruct он типизирован широким
            # объединением (в т.ч. без длины), хотя сюда всегда кладём list.
            vector_size = len(entries[0].vectors[0])
            client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                # m/ef — параметры ADR-006 (раздел 8.2), не выдумка модуля.
                hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            )

        client.upsert(self._collection, points=points)
        self._point_count += len(points)
        self._chunk_ids.update(entry.chunk.chunk_id for entry in entries)

    def attach(self) -> int:
        """Открыть уже наполненную коллекцию для поиска, не переиндексируя.

        `upsert` — единственное место, где раньше заполнялись `_point_count` и
        `_chunk_ids`; для процесса, который сам ничего не писал (Backend при
        `build_knowledge_base(reindex=False)` — переиндексация ушла в отдельный
        прогон, `scripts/reindex_kb.py`), они остались бы нулевыми, и
        `search_parts`/`KnowledgeBase.search` решили бы, что хранилище пусто,
        хотя в Qdrant уже есть данные. `attach` читает реальное состояние
        коллекции вместо того, чтобы полагаться на историю вызовов `upsert`
        в этом процессе.

        Возвращает число фрагментов (не точек) — то же, что покажет `len()`
        после вызова. Коллекции ещё нет — `0`, это не ошибка: `reindex_kb.py`
        просто не запускали ни разу.
        """
        client = self._ensure_client()
        if not client.collection_exists(self._collection):
            return 0

        chunk_ids: set[str] = set()
        offset = None
        while True:
            points, offset = client.scroll(
                self._collection, with_payload=["chunk_id"], with_vectors=False,
                limit=256, offset=offset,
            )
            chunk_ids.update(
                point.payload["chunk_id"] for point in points if point.payload
            )
            if offset is None:
                break

        info = client.get_collection(self._collection)
        self._point_count = info.points_count or 0
        self._chunk_ids = chunk_ids
        return len(self._chunk_ids)

    def search_parts(self, vector: Sequence[float]) -> list[tuple[ContextChunk, float]]:
        if self._point_count == 0:
            return []
        client = self._ensure_client()
        # Лимит — все точки коллекции: на объёме прототипа это точный перебор,
        # не приближение (см. докстринг модуля).
        result = client.query_points(
            self._collection, query=list(vector), limit=self._point_count
        )

        best: dict[str, tuple[ContextChunk, float]] = {}
        for point in result.points:
            chunk = _chunk_from_payload(point.payload or {})
            prev = best.get(chunk.chunk_id)
            if prev is None or point.score > prev[1]:
                best[chunk.chunk_id] = (chunk, point.score)
        return list(best.values())

    def __len__(self) -> int:
        return len(self._chunk_ids)
