"""Проверки поиска по базе знаний.

Модель эмбеддингов подменяется: проверяется **сам поиск** — порядок, порог,
отсечение по числу фрагментов, — а не качество эмбеддингов. Качество моделью не
проверишь ни быстро, ни воспроизводимо, и его место в замере на эталонном
наборе (пункт 14 сведённого TODO), а не в модульном тесте.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.kb.search import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    KnowledgeBase,
    cosine,
)

CORPUS = Path(__file__).resolve().parents[2] / "kb" / "faq_voronezh.json"


class FakeEmbedder:
    """Вектор по первой букве текста — этого хватает, чтобы задать порядок.

    Запоминает, что ему передали: префиксы `query:` и `passage:` — свойство
    модели e5, и их пропажа тихо испортила бы близость, ничего не сломав.
    """

    def __init__(self, table: dict[str, Sequence[float]] | None = None) -> None:
        self.seen: list[str] = []
        self._table = table or {}

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.seen.extend(texts)
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


# --- косинусная близость ------------------------------------------------------ #


def test_совпадающие_векторы_дают_единицу():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_перпендикулярные_дают_ноль():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_оценка_не_выходит_за_границы_контракта():
    """`relevance_score` объявлен на отрезке от нуля до единицы.

    Погрешность вычислений даёт на совпадающих векторах единицу с хвостом, а
    противоположные направления — отрицательную близость; и то, и другое не
    прошло бы проверку схемы.
    """
    assert 0.0 <= cosine([1.0, 1.0], [1.0, 1.0]) <= 1.0
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0


# --- поиск --------------------------------------------------------------------- #


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
    kb = KnowledgeBase.from_file(path, FakeEmbedder(table))
    found = kb.search("запрос", top_n=3, threshold=0.0)

    assert [c.chunk_id for c in found] == ["c2", "c3", "c1"]
    assert found[0].relevance_score >= found[1].relevance_score


def test_порог_отсекает_далёкое(tmp_path: Path):
    """Ниже порога контекст считается не найденным (раздел 8.2).

    Замер на настоящем корпусе: посторонние вопросы набирают 0,74–0,79, по делу
    0,81–0,92. Без порога борщ и погода уходили бы в промпт наравне с
    регламентом.
    """
    table = {"ближний": [1.0, 0.0], "дальний": [0.0, 1.0], "запрос": [1.0, 0.0]}
    path = _corpus_file(tmp_path, [_item("c1", "ближний"), _item("c2", "дальний")])
    kb = KnowledgeBase.from_file(path, FakeEmbedder(table))

    assert [c.chunk_id for c in kb.search("запрос", top_n=3, threshold=0.5)] == ["c1"]


def test_пустой_результат_это_законный_ответ(tmp_path: Path):
    """Лучше сказать «не знаю», чем дать модели чужой фрагмент.

    Найденное ниже порога — не сбой поиска, а сигнал включить запасной путь без
    модели: с чужим контекстом она ответит уверенно и неверно.
    """
    table = {"дальний": [0.0, 1.0], "запрос": [1.0, 0.0]}
    path = _corpus_file(tmp_path, [_item("c1", "дальний")])
    kb = KnowledgeBase.from_file(path, FakeEmbedder(table))

    assert kb.search("запрос", top_n=3, threshold=0.5) == []


def test_в_промпт_уходит_не_больше_запрошенного(tmp_path: Path):
    """`top_n = 3` (ADR-006). Лишние фрагменты — лишние токены и лишний шум."""
    path = _corpus_file(tmp_path, [_item(f"c{n}", "текст") for n in range(10)])
    kb = KnowledgeBase.from_file(path, FakeEmbedder())

    assert len(kb.search("запрос", top_n=3, threshold=0.0)) == 3


def test_пустой_запрос_ничего_не_ищет(tmp_path: Path):
    path = _corpus_file(tmp_path, [_item("c1", "текст")])
    kb = KnowledgeBase.from_file(path, FakeEmbedder())

    assert kb.search("   ", top_n=3, threshold=0.0) == []


# --- префиксы модели ----------------------------------------------------------- #


def test_фрагменты_и_запрос_идут_с_разными_префиксами(tmp_path: Path):
    """Свойство модели e5, а не украшение.

    Без префиксов близость считается по другому распределению, и порог,
    подобранный с ними, перестаёт что-либо означать. Пропажа префикса ничего не
    ломает — она тихо портит выдачу, поэтому сторожится тестом.
    """
    path = _corpus_file(tmp_path, [_item("c1", "текст")])
    embedder = FakeEmbedder()
    kb = KnowledgeBase.from_file(path, embedder)
    kb.search("вопрос абонента", top_n=1, threshold=0.0)

    assert any(t.startswith(PASSAGE_PREFIX) for t in embedder.seen)
    assert any(t.startswith(QUERY_PREFIX) for t in embedder.seen)


# --- настоящий корпус ---------------------------------------------------------- #


def test_настоящий_корпус_индексируется(tmp_path: Path):
    """Форма корпуса и форма индекса обязаны совпадать.

    Проверяется на настоящем файле: расхождение полей проявилось бы не ошибкой,
    а пустым поиском.
    """
    kb = KnowledgeBase.from_file(CORPUS, FakeEmbedder())
    assert len(kb) >= 15


def test_источник_доходит_до_фрагмента():
    """`SourceRef` показывается абоненту: без адреса ответ негде проверить."""
    kb = KnowledgeBase.from_file(CORPUS, FakeEmbedder())
    found = kb.search("вопрос", top_n=1, threshold=0.0)

    assert found
    assert found[0].source_url
    assert found[0].source_title
    assert found[0].chunk_id


# --- веса берутся из кэша, а не из сети (пункт 65) ----------------------------- #


def test_по_умолчанию_модель_берётся_только_из_кэша():
    """На MVP сервис живёт в закрытой сети, где `huggingface.co` недоступен.

    Обращение к нему при старте обернулось бы там ожиданием сетевых таймаутов.
    На прототипе оно же стоило 14 секунд подъёма из 46."""
    from app.kb.search import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("любая/модель")
    assert embedder._local_files_only is True


def test_настройка_доходит_до_загрузчика_модели():
    """Значение из настроек не должно потеряться по дороге: без этой проверки
    оно осталось бы объявленным и неподключённым — тот же класс ошибки, что с
    прогревом (журнал, раздел 69)."""
    from app.config import get_settings
    from app.main import _build_knowledge_base

    knowledge_base = _build_knowledge_base()
    if knowledge_base is None:
        pytest.skip("корпус базы знаний не найден")

    embedder = knowledge_base._embedder
    assert embedder._local_files_only is get_settings().embedding_local_files_only


@pytest.mark.slow
def test_пустой_кэш_объясняет_себя_а_не_выглядит_отказом_сети():
    """Отказ из-за отсутствия весов в кэше неотличим от сетевой ошибки.

    Без подсказки следующий человек будет искать причину не там — поэтому
    сообщение называет и команду скачивания, и способ разрешить сеть."""
    from app.kb.search import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("несуществующая/модель-для-проверки")
    with pytest.raises(RuntimeError) as caught:
        embedder.warm_up()

    message = str(caught.value)
    assert "локальном кэше" in message
    assert "make install-search" in message


# --- вопрос и ответ индексируются порознь (пункт 62) --------------------------- #


def _knowledge_base():
    """Настоящий корпус на настоящей модели — эти проверки про устройство индекса.

    Подменённая модель здесь не годится: проверяется, что вопрос и ответ попали
    в индекс порознь, а это свойство сборки, а не поиска.
    """
    from app.main import _build_knowledge_base

    knowledge_base = _build_knowledge_base()
    if knowledge_base is None:
        pytest.skip("корпус базы знаний не найден")
    return knowledge_base


def test_у_фрагмента_столько_векторов_сколько_частей():
    """Вопрос и ответ — два вида одного фрагмента, а не один склеенный.

    Склейка топила короткий вопрос в длинном ответе: у `faq-11` вопрос 24 знака
    против 611, и дословный вопрос корпуса набирал 0,844 — ниже порога, при том
    что ответ на него лежит в базе знаний буквально."""
    knowledge_base = _knowledge_base()
    assert all(len(entry.vectors) == 2 for entry in knowledge_base._entries)


def test_близость_берётся_по_лучшей_части_а_не_по_средней():
    """Среднее вернуло бы то самое разбавление, ради устранения которого части
    и разведены."""
    from app.kb.search import _Entry
    from app.models import ContextChunk

    entry = _Entry(
        chunk=ContextChunk(
            chunk_id="c", text="t", source_title="s", source_url="u", relevance_score=0.0
        ),
        vectors=([1.0, 0.0], [0.0, 1.0]),
    )
    assert entry.similarity([1.0, 0.0]) == pytest.approx(1.0)


def test_в_промпт_уходит_пара_целиком_а_не_найденная_часть():
    """Единица поиска перестала совпадать с единицей контекста.

    Ищется по частям, но модель обязана получить ответ вместе с вопросом, к
    которому он относится, — иначе фрагмент теряет смысл."""
    from app.config import get_settings

    knowledge_base = _knowledge_base()
    found = knowledge_base.search("Почему начисляются пени?", top_n=1, threshold=0.0)
    corpus = json.loads(Path(get_settings().kb_corpus_path).read_text(encoding="utf-8"))
    pair = next(c for c in corpus if c["chunk_id"] == found[0].chunk_id)
    assert pair["question"] in found[0].text
    assert pair["answer"] in found[0].text
