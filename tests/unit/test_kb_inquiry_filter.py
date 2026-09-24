"""Фильтр по типу обращения внутри поиска (ADR-200).

Одни и те же сценарии на обоих хранилищах: смена `VECTOR_STORE` не должна
менять, какие фрагменты находятся. Qdrant — встроенный режим, без сервера.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from app.kb.qdrant_store import QdrantVectorStore
from app.kb.search import (
    InMemoryVectorStore,
    KnowledgeBase,
    VectorStore,
    _Entry,
)
from app.models import ContextChunk


class _Embedder:
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]


def _entry(chunk_id: str, vector: list[float], inquiry_type: str | None) -> _Entry:
    return _Entry(
        chunk=ContextChunk(
            chunk_id=chunk_id, text=chunk_id, source_title="s", relevance_score=0.0
        ),
        vectors=(vector,),
        inquiry_type=inquiry_type,
    )


STORES: dict[str, Callable[[], VectorStore]] = {
    "memory": InMemoryVectorStore,
    "qdrant": lambda: QdrantVectorStore(":memory:", "filter-test"),
}


@pytest.fixture(params=sorted(STORES))
def kb(request: pytest.FixtureRequest) -> KnowledgeBase:
    store = STORES[request.param]()
    store.upsert(
        [
            # Чужой тип ближе всех к запросу — без фильтра он бы вытеснил остальных.
            _entry("sealing-1", [1.0, 0.0], "meter_sealing"),
            _entry("verify-1", [0.9, 0.1], "meter_verification"),
            _entry("general-1", [0.7, 0.7], None),
        ]
    )
    return KnowledgeBase(store, _Embedder())


def _ids(kb: KnowledgeBase, inquiry_type: str | None, top_k: int | None = None) -> list[str]:
    found = kb.search(
        "запрос", top_n=10, threshold=0.0, top_k=top_k, inquiry_type=inquiry_type
    )
    return [c.chunk_id for c in found]


def test_без_фильтра_находится_всё(kb: KnowledgeBase):
    assert set(_ids(kb, None)) == {"sealing-1", "verify-1", "general-1"}


def test_чужой_тип_исключается(kb: KnowledgeBase):
    assert "sealing-1" not in _ids(kb, "meter_verification")


def test_свой_тип_и_общие_проходят(kb: KnowledgeBase):
    assert set(_ids(kb, "meter_verification")) == {"verify-1", "general-1"}


def test_тип_без_документов_оставляет_только_общие(kb: KnowledgeBase):
    assert _ids(kb, "refund") == ["general-1"]


def test_фильтр_действует_до_отбора_top_k(kb: KnowledgeBase):
    """Смысл правила: без фильтра `top_k=1` взял бы `sealing-1` и после отсева
    не осталось бы ничего; с фильтром в поиске нужный фрагмент не теряется."""
    assert _ids(kb, "meter_verification", top_k=1) == ["verify-1"]

