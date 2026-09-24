"""Проверки Qdrant-хранилища векторов (app/kb/qdrant_store.py).

Qdrant здесь — встроенный режим `qdrant-client` (`location=":memory:"`), не
внешний сервис: тесты воспроизводимы без `docker compose up qdrant`. Каждый
`QdrantVectorStore` поднимает свой изолированный клиент, поэтому коллекции с
одинаковым именем в разных тестах друг другу не мешают.

Сценарии дублируют `test_kb_search.py` (тот же порядок, порог, top_n) —
цель именно в этом: доказать, что смена хранилища не меняет поведение
`KnowledgeBase.search()` ни на бит, только реализацию внутри.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from app.kb.qdrant_store import QdrantVectorStore
from app.kb.search import PASSAGE_PREFIX, QUERY_PREFIX, KnowledgeBase, _Entry
from app.models import ContextChunk


class FakeEmbedder:
    """Вектор по точному тексту (с префиксом) — тот же приём, что в test_kb_search.py."""

    def __init__(self, table: dict[str, Sequence[float]] | None = None) -> None:
        self._table = table or {}

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = []
        for text in texts:
            body = text.removeprefix(QUERY_PREFIX).removeprefix(PASSAGE_PREFIX)
            key = next((k for k in self._table if body.startswith(k)), None)
            vectors.append(self._table[key] if key else [0.0, 1.0])
        return vectors


def _corpus_file(tmp_path: Path, items: list[dict[str, str]]) -> Path:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def _item(chunk_id: str, question: str) -> dict[str, str]:
    return {
        "chunk_id": chunk_id,
        "question": question,
        "answer": f"ответ на {question}",
        "source_title": "FAQ",
        "source_url": "https://example.test/faq/",
    }


def _store(name: str) -> QdrantVectorStore:
    return QdrantVectorStore(":memory:", name)


# --- поведение через KnowledgeBase — то же, что у InMemoryVectorStore ---------- #


def test_фрагменты_идут_от_самого_близкого(tmp_path: Path):
    table = {
        "ближний": [1.0, 0.0],
        "средний": [0.7, 0.7],
        "дальний": [0.0, 1.0],
        "запрос": [1.0, 0.0],
    }
    path = _corpus_file(
        tmp_path,
        [_item("c1", "дальний"), _item("c2", "ближний"), _item("c3", "средний")],
    )
    kb = KnowledgeBase.from_file(path, FakeEmbedder(table), store=_store("order"))
    found = kb.search("запрос", top_n=3, threshold=0.0)

    assert [c.chunk_id for c in found] == ["c2", "c3", "c1"]
    assert found[0].relevance_score >= found[1].relevance_score


def test_порог_отсекает_далёкое(tmp_path: Path):
    table = {"ближний": [1.0, 0.0], "дальний": [0.0, 1.0], "запрос": [1.0, 0.0]}
    path = _corpus_file(tmp_path, [_item("c1", "ближний"), _item("c2", "дальний")])
    kb = KnowledgeBase.from_file(path, FakeEmbedder(table), store=_store("threshold"))

    assert [c.chunk_id for c in kb.search("запрос", top_n=3, threshold=0.5)] == ["c1"]


def test_пустой_результат_это_законный_ответ(tmp_path: Path):
    table = {"дальний": [0.0, 1.0], "запрос": [1.0, 0.0]}
    path = _corpus_file(tmp_path, [_item("c1", "дальний")])
    kb = KnowledgeBase.from_file(path, FakeEmbedder(table), store=_store("empty-result"))

    assert kb.search("запрос", top_n=3, threshold=0.5) == []


def test_в_промпт_уходит_не_больше_запрошенного(tmp_path: Path):
    path = _corpus_file(tmp_path, [_item(f"c{n}", "текст") for n in range(10)])
    kb = KnowledgeBase.from_file(path, FakeEmbedder(), store=_store("top-n"))

    assert len(kb.search("запрос", top_n=3, threshold=0.0)) == 3


# --- поведение самого хранилища ------------------------------------------------- #


def test_близость_берётся_по_лучшей_части_а_не_по_средней():
    """Тот же принцип, что у `_Entry.similarity` в памяти — здесь через Qdrant:
    несколько точек одного фрагмента, близость фрагмента — максимум среди них."""
    store = _store("best-part")
    entry = _Entry(
        chunk=ContextChunk(
            chunk_id="c", text="t", source_title="s", source_url="u", relevance_score=0.0
        ),
        vectors=([1.0, 0.0], [0.0, 1.0]),
    )
    store.upsert([entry])

    (chunk, score) = store.search_parts([1.0, 0.0])[0]
    assert chunk.chunk_id == "c"
    assert score == 1.0


def test_len_считает_фрагменты_а_не_точки():
    """У фрагмента несколько точек (заголовок + части ответа) — `len` не должен
    путать число точек с числом фрагментов, иначе `top_k`-семантика поедет."""
    store = _store("len")
    entry = _Entry(
        chunk=ContextChunk(
            chunk_id="c", text="t", source_title="s", source_url="u", relevance_score=0.0
        ),
        vectors=([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]),
    )
    store.upsert([entry])

    assert len(store) == 1


def test_пустое_хранилище_ничего_не_находит():
    store = _store("empty-store")
    assert store.search_parts([1.0, 0.0]) == []
    assert len(store) == 0


# --- attach: открыть коллекцию, не переиндексируя (build_knowledge_base) ------- #


def test_attach_на_несуществующую_коллекцию_возвращает_ноль():
    """`reindex_kb.py` ни разу не запускали — это не ошибка, а пустая база."""
    store = _store("attach-missing")
    assert store.attach() == 0
    assert len(store) == 0
    assert store.search_parts([1.0, 0.0]) == []


def test_attach_видит_данные_чужого_upsert():
    """Тот же сценарий, что Backend/`build_knowledge_base(reindex=False)`:
    `attach` вызывается в процессе, который сам ничего не писал — данные в
    Qdrant появились другим прогоном (`scripts/reindex_kb.py`). `:memory:`
    Qdrant изолирован по клиенту, а не по процессу, поэтому второй стор
    получает клиент первого напрямую — так же, как если бы это был общий
    Qdrant по сети (`http://localhost:6333`), только без реального сервера.
    """
    writer = _store("attach-shared")
    writer.upsert(
        [
            _Entry(
                chunk=ContextChunk(
                    chunk_id="c1", text="t1", source_title="s", source_url="u",
                    relevance_score=0.0,
                ),
                vectors=([1.0, 0.0],),
            ),
            _Entry(
                chunk=ContextChunk(
                    chunk_id="c2", text="t2", source_title="s", source_url="u",
                    relevance_score=0.0,
                ),
                vectors=([0.0, 1.0], [0.9, 0.1]),
            ),
        ]
    )

    reader = _store("attach-shared")
    reader._client = writer._client  # тот же приём, что использовал бы общий сервер

    assert reader.attach() == 2
    assert len(reader) == 2
    found = reader.search_parts([1.0, 0.0])
    assert {chunk.chunk_id for chunk, _ in found} == {"c1", "c2"}


# --- blue-green: переиндексация без простоя поиска ------------------------------ #


def _entry(chunk_id: str, vector: list[float]) -> _Entry:
    return _Entry(
        chunk=ContextChunk(
            chunk_id=chunk_id, text=chunk_id, source_title="s", source_url="u",
            relevance_score=0.0,
        ),
        vectors=(vector,),
    )


def _bg_store(client_from: QdrantVectorStore | None = None) -> QdrantVectorStore:
    store = QdrantVectorStore(":memory:", "kb", blue_green=True)
    if client_from is not None:
        store._client = client_from._client
    return store


def test_blue_green_читатель_видит_старое_пока_не_publish():
    first = _bg_store()
    first.upsert([_entry("old", [1.0, 0.0])])
    first.publish()

    # Второй прогон пишет в новую версию — читатель по алиасу его пока не видит.
    second = _bg_store(first)
    second._target = "kb__v29990101T000000"
    second.upsert([_entry("new", [1.0, 0.0])])

    reader = _store_on("kb", first)
    assert reader.attach() == 1
    assert {c.chunk_id for c, _ in reader.search_parts([1.0, 0.0])} == {"old"}

    second.publish()
    reader2 = _store_on("kb", first)
    assert reader2.attach() == 1
    assert {c.chunk_id for c, _ in reader2.search_parts([1.0, 0.0])} == {"new"}


def _store_on(name: str, client_from: QdrantVectorStore) -> QdrantVectorStore:
    store = _store(name)
    store._client = client_from._client
    return store


def test_publish_без_upsert_ничего_не_делает():
    assert _bg_store().publish() is None


def test_publish_без_blue_green_ничего_не_делает():
    store = _store("plain")
    store.upsert([_entry("a", [1.0, 0.0])])
    assert store.publish() is None


def test_publish_оставляет_только_заданное_число_прежних_версий():
    store = _bg_store()
    store._ensure_client()  # общий клиент для всех прогонов ниже
    for stamp in ("20260101T000001", "20260101T000002", "20260101T000003", "20260101T000004"):
        run = _bg_store(store)
        run._target = f"kb__v{stamp}"
        run.upsert([_entry(stamp, [1.0, 0.0])])
        run.publish(keep_previous=1)

    client = store._ensure_client()
    names = sorted(c.name for c in client.get_collections().collections)
    assert names == ["kb__v20260101T000003", "kb__v20260101T000004"]


def test_publish_заменяет_прежнюю_коллекцию_с_тем_же_именем_что_алиас():
    """Переход со старой схемы: физическая коллекция `kb` мешает алиасу `kb`."""
    legacy = _store("kb")
    legacy.upsert([_entry("legacy", [1.0, 0.0])])

    fresh = _bg_store(legacy)
    fresh.upsert([_entry("fresh", [1.0, 0.0])])
    assert fresh.publish() is not None

    reader = _store_on("kb", legacy)
    assert reader.attach() == 1
    assert {c.chunk_id for c, _ in reader.search_parts([1.0, 0.0])} == {"fresh"}


# --- протокол клиента и лимит выдачи (замер 24 сентября 2026, ADR-200) ---------- #


def test_клиент_создаётся_с_grpc_и_портом(monkeypatch):
    """`prefer_grpc` и порт доходят до `QdrantClient`: по REST поиск стоил ≈48 мс, по gRPC ≈6 мс."""
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeClient)
    QdrantVectorStore(
        "http://qdrant:6333", "kb", prefer_grpc=True, grpc_port=6335
    )._ensure_client()

    assert captured == {"url": "http://qdrant:6333", "prefer_grpc": True, "grpc_port": 6335}


def _many(store: QdrantVectorStore, count: int) -> None:
    """`count` фрагментов по одной точке, все по разным направлениям — порядок известен."""
    store.upsert(
        [
            _Entry(
                chunk=ContextChunk(
                    chunk_id=f"c{i}", text="t", source_title="s", source_url="u", relevance_score=0.0
                ),
                vectors=([1.0, i / count],),
            )
            for i in range(count)
        ]
    )


def test_на_малом_объёме_выдача_точная_даже_с_top_k():
    """До EXACT_SCAN_MAX_POINTS точек лимит — все точки, `top_k` ничего не режет."""
    store = _store("exact")
    _many(store, 100)

    assert len(store.search_parts([1.0, 0.0], top_k=3)) == 100


def test_на_большом_объёме_лимит_ограничен_top_k():
    """Выше границы запрашивается `top_k * PARTS_HEADROOM` точек, а не вся коллекция:
    возврат всех 100 000 точек стоил 1,3–3,4 с."""
    from app.kb.qdrant_store import EXACT_SCAN_MAX_POINTS, PARTS_HEADROOM

    store = _store("bounded")
    _many(store, EXACT_SCAN_MAX_POINTS + 100)

    found = store.search_parts([1.0, 0.0], top_k=3)
    assert len(found) == 3 * PARTS_HEADROOM
    # верх выдачи точный: самый близкий фрагмент — c0 (вектор [1, 0])
    assert max(found, key=lambda pair: pair[1])[0].chunk_id == "c0"
    # без `top_k` — по-прежнему все
    assert len(store.search_parts([1.0, 0.0])) == EXACT_SCAN_MAX_POINTS + 100
