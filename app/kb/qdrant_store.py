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

BLUE-GREEN (`blue_green=True`, `publish()`): переиндексация пишет в новую
версию коллекции, читатели ищут по алиасу и не видят недособранного корпуса;
алиас переключается одной атомарной операцией. Без флага `upsert` пишет
прямо в коллекцию (оверрайт на месте, как раньше).

ФИЛЬТР ПО `inquiry_type` — внутри запроса (`query_filter`), не над выдачей:
тип совпал или у фрагмента типа нет (общий регламент). Payload-индексы
созданы по `inquiry_type` и `source_title` — по остальным полям ADR-006
(`department`, `effective_date`) данных в корпусе пока нет.

ЧЕГО ЗДЕСЬ НЕТ — ОСОЗНАННО, ЭТО НЕ ВЕСЬ ADR-006. Проверка несовпадения
версии модели эмбеддингов с алертом (ADR-006, п. 7 подтверждения) не написана —
отдельная задача дорожной карты (ADR-021), не забытая часть этой.

`qdrant-client` импортируется лениво (как `sentence_transformers` в
`SentenceTransformerEmbedder`) — модуль не должен требовать пакет, если Qdrant
не выбран (`settings.vector_store != "qdrant"`).
"""

from __future__ import annotations

import time
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
    :param collection: имя коллекции — либо алиас, если включён `blue_green`.
        Отдельное имя на окружение убережёт от путаницы, если несколько
        процессов используют одну и ту же Qdrant.
    :param blue_green: переиндексация без простоя поиска (ADR-006, раздел
        «Реиндексация»). `upsert` пишет в НОВУЮ физическую коллекцию
        `<collection>__v<время>`, а читатели всё это время ищут по алиасу
        `<collection>`, который указывает на старую; `publish()` переключает
        алиас одним атомарным вызовом. Выключено — `upsert` пишет прямо в
        `collection`, как было (оверрайт на месте, поиск в это время видит
        смесь старых и новых данных).

    Два способа наполнить `_point_count`/`_chunk_ids`, которыми живут
    `search_parts`/`__len__`: `upsert` (пишущая сторона, `scripts/reindex_kb.py`)
    и `attach` (читающая сторона, `build_knowledge_base(reindex=False)`,
    боевой путь Backend при `VECTOR_STORE=qdrant`) — см. докстринг `attach`."""

    def __init__(self, url: str, collection: str, *, blue_green: bool = False) -> None:
        self._url = url
        self._collection = collection
        self._blue_green = blue_green
        self._target: str | None = None
        self._client: QdrantClient | None = None
        self._point_count = 0
        self._chunk_ids: set[str] = set()

    def _write_target(self) -> str:
        """Куда пишет `upsert`: сама коллекция либо новая версия для blue-green."""
        if not self._blue_green:
            return self._collection
        if self._target is None:
            self._target = f"{self._collection}__v{time.strftime('%Y%m%dT%H%M%S')}"
        return self._target

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
        from qdrant_client.models import (
            Distance,
            HnswConfigDiff,
            PayloadSchemaType,
            PointStruct,
            VectorParams,
        )

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
                    # Общий фрагмент (без типа) не пишется в payload вовсе:
                    # `IsEmptyCondition` в фильтре ловит и отсутствующее поле.
                    **({"inquiry_type": entry.inquiry_type} if entry.inquiry_type else {}),
                },
            )
            for entry in entries
            for part_index, vector in enumerate(entry.vectors)
        ]
        if not points:
            return

        target = self._write_target()
        if not client.collection_exists(target):
            # Не из `points[0].vector`: у PointStruct он типизирован широким
            # объединением (в т.ч. без длины), хотя сюда всегда кладём list.
            vector_size = len(entries[0].vectors[0])
            client.create_collection(
                target,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                # m/ef — параметры ADR-006 (раздел 8.2), не выдумка модуля.
                hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            )
            # Индексы payload (ADR-006, раздел 8.2): без них фильтр — проход
            # по всем точкам. Только поля, по которым есть данные: `department`
            # и `effective_date` корпус пока не несёт, и индекс по пустому полю
            # ничего не ускоряет.
            # Встроенный режим (`:memory:`) индексов payload не поддерживает —
            # там это было бы предупреждение без эффекта.
            if self._url != ":memory:":
                for field_name in ("inquiry_type", "source_title"):
                    client.create_payload_index(
                        target,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )

        client.upsert(target, points=points)
        self._point_count += len(points)
        self._chunk_ids.update(entry.chunk.chunk_id for entry in entries)

    def publish(self, *, keep_previous: int = 1) -> str | None:
        """Переключить алиас на только что записанную коллекцию (blue-green).

        Одна атомарная операция `update_collection_aliases`: удаление старой
        привязки и создание новой идут вместе, читатель не увидит момента, когда
        алиаса нет. Возвращает имя опубликованной коллекции; `None` — нечего
        публиковать (`blue_green` выключен либо `upsert` ничего не записал).

        Старые версии не удаляются сразу: `keep_previous` последних остаются
        для отката — вернуть алиас на прежнюю коллекцию можно одной командой,
        пока она цела. Остальные удаляются, иначе каждая переиндексация
        оставляла бы полную копию корпуса.

        Коллекция, чьё имя совпадает с алиасом (прежняя схема — `upsert` прямо в
        `collection`), мешает создать алиас с тем же именем и удаляется один
        раз при переходе. Это единственный момент с коротким провалом поиска;
        дальше переключения атомарны.
        """
        from qdrant_client.models import (
            AliasOperations,
            CreateAlias,
            CreateAliasOperation,
            DeleteAlias,
            DeleteAliasOperation,
        )

        if not self._blue_green or self._target is None:
            return None
        client = self._ensure_client()

        bound = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
        operations: list[AliasOperations] = []
        if self._collection in bound:
            operations.append(
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=self._collection))
            )
        elif client.collection_exists(self._collection):
            client.delete_collection(self._collection)  # прежняя схема, см. докстринг
        operations.append(
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=self._target, alias_name=self._collection
                )
            )
        )
        client.update_collection_aliases(change_aliases_operations=operations)

        prefix = f"{self._collection}__v"
        versions = sorted(
            c.name for c in client.get_collections().collections
            if c.name.startswith(prefix) and c.name != self._target
        )
        for stale in versions[: max(0, len(versions) - keep_previous)]:
            client.delete_collection(stale)
        return self._target

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

    def search_parts(
        self, vector: Sequence[float], inquiry_type: str | None = None
    ) -> list[tuple[ContextChunk, float]]:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchValue,
            PayloadField,
        )

        if self._point_count == 0:
            return []
        client = self._ensure_client()
        # Фильтр — внутри запроса, а не над готовой выдачей (ADR-006): тип
        # совпал ИЛИ у фрагмента типа нет (общий регламент подходит под любой).
        query_filter = (
            Filter(
                should=[
                    FieldCondition(key="inquiry_type", match=MatchValue(value=inquiry_type)),
                    IsEmptyCondition(is_empty=PayloadField(key="inquiry_type")),
                ]
            )
            if inquiry_type is not None
            else None
        )
        # Лимит — все точки коллекции: на объёме прототипа это точный перебор,
        # не приближение (см. докстринг модуля).
        result = client.query_points(
            self._collection,
            query=list(vector),
            query_filter=query_filter,
            limit=self._point_count,
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
