"""Проверки запасного классификатора темы: k-NN по эмбеддингам примеров.

Эмбеддер подменён на детерминированный (вектор = позиция темы в простом
базисе) — сторожим саму логику (порог, разрыв, сортировку), не качество
модели эмбеддингов. Отдельный `live`-тест проверяет числа
на настоящей `SentenceTransformerEmbedder` — он тяжёлый и не идёт по умолчанию.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.agents.topic_fallback import TopicFallbackClassifier, _Example
from app.kb.search import PASSAGE_PREFIX, QUERY_PREFIX
from app.taxonomy import Topic

# Три оси в простом базисе — примеры и запросы получают векторы вручную,
# чтобы косинус между ними был предсказуем и проверяем в уме.
_OUTAGE_NEAR = (1.0, 0.0, 0.0)
_OUTAGE_FAR = (0.6, 0.8, 0.0)  # тот же топик, но дальше — для проверки разрыва
_TARIFF_NEAR = (0.0, 1.0, 0.0)
_UNRELATED = (0.0, 0.0, 1.0)


class FakeEmbedder:
    """Отдаёт заранее заданный вектор по точному тексту (с префиксом)."""

    def __init__(self, table: dict[str, Sequence[float]]) -> None:
        self._table = table

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._table[text] for text in texts]


_EXAMPLES = (
    _Example("пример отключения", Topic.OUTAGE),
    _Example("другой пример отключения", Topic.OUTAGE),
    _Example("пример тарифа", Topic.TARIFF),
)


def _classifier(query_vector: Sequence[float], **overrides: object) -> TopicFallbackClassifier:
    table = {
        PASSAGE_PREFIX + "пример отключения": _OUTAGE_NEAR,
        PASSAGE_PREFIX + "другой пример отключения": _OUTAGE_FAR,
        PASSAGE_PREFIX + "пример тарифа": _TARIFF_NEAR,
        QUERY_PREFIX + "вопрос": query_vector,
    }
    defaults: dict[str, object] = {"examples": _EXAMPLES}
    return TopicFallbackClassifier(FakeEmbedder(table), **{**defaults, **overrides})


async def test_ближайший_пример_даёт_тему():
    clf = _classifier(_OUTAGE_NEAR, threshold=0.5, margin=0.0)
    assert await clf.classify("вопрос") is Topic.OUTAGE


async def test_ниже_порога_это_none():
    clf = _classifier(_UNRELATED, threshold=0.5, margin=0.0)
    assert await clf.classify("вопрос") is None


async def test_маленький_разрыв_до_другой_темы_это_none():
    # Вектор ровно между outage и tariff — разрыва между лучшей темой и
    # второй (другой) почти нет, несмотря на то что порог формально взят.
    clf = _classifier((0.72, 0.7, 0.0), threshold=0.5, margin=0.5)
    assert await clf.classify("вопрос") is None


async def test_разрыв_считается_только_до_другой_темы():
    # Второй ближайший пример — тоже outage; разрыв меряется до первого
    # ПОСЛЕ него примера с другой темой, а не до второго по счёту вообще.
    clf = _classifier(_OUTAGE_NEAR, threshold=0.5, margin=0.3)
    assert await clf.classify("вопрос") is Topic.OUTAGE


async def test_пустой_запрос_это_none():
    clf = _classifier(_OUTAGE_NEAR, threshold=0.0, margin=0.0)
    assert await clf.classify("   ") is None


async def test_без_примеров_это_none():
    clf = TopicFallbackClassifier(FakeEmbedder({}), examples=())
    assert await clf.classify("что угодно") is None


async def test_отказ_эмбеддера_не_пробрасывается():
    class BrokenEmbedder:
        def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            raise RuntimeError("модель не загружена")

    clf = TopicFallbackClassifier(BrokenEmbedder(), examples=_EXAMPLES)
    assert await clf.classify("вопрос") is None


async def test_примеры_проэмбечены_один_раз():
    calls = []

    class CountingEmbedder:
        def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            calls.append(list(texts))
            if len(texts) == 1:
                return [_OUTAGE_NEAR]
            return [_OUTAGE_NEAR, _OUTAGE_FAR, _TARIFF_NEAR]

    clf = TopicFallbackClassifier(CountingEmbedder(), examples=_EXAMPLES, threshold=0.0, margin=0.0)
    await clf.classify("первый")
    await clf.classify("второй")
    # Один вызов на примеры (кэшируется) плюс по одному на каждый запрос.
    example_calls = [c for c in calls if len(c) == len(_EXAMPLES)]
    assert len(example_calls) == 1


# --- живой эмбеддер: проверка чисел (тяжёлый, только с --live) --- #


@pytest.mark.live
async def test_реальные_случаи_сессии_живой_эмбеддер():
    """Два целевых случая — «отключение» и разбора «подключения» — обязаны
    находить свою тему на реальном пороге; явная бессмыслица — нет."""
    from app.agents.topic_fallback import EXAMPLES
    from app.kb.search import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    embedder.warm_up()
    clf = TopicFallbackClassifier(embedder, examples=EXAMPLES)

    assert await clf.classify("По моему адресу до скольки отключение?") is Topic.OUTAGE
    assert (
        await clf.classify("нужно оформить новое подключение к сети водоснабжения")
        is Topic.TEMPLATE
    )
    assert await clf.classify("Спасибо за помощь") is None
    assert await clf.classify("здравствуйте, подскажите пожалуйста") is None

    # «подключить» и «включат» — общий корень, эмбеддер
    # путал новое подключение с восстановлением после отключения (margin
    # 0.0301 — почти прошло прежний разрыв 0.03). Живая проверка регрессии.
    assert await clf.classify("Как подключить воду в дом?") is Topic.TEMPLATE
    assert await clf.classify("У меня нет воды с самого утра") is Topic.OUTAGE
