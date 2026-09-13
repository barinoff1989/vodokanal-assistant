"""Проверки запасного классификатора типа обращения: k-NN по эмбеддингам примеров.

Зеркало `test_topic_fallback.py` — тот же приём, другая ось (`InquiryType`,
семь регистрируемых типов вместо шести тем). Эмбеддер подменён на
детерминированный: сторожим логику (порог, разрыв, сортировку, кэш примеров),
не качество модели. `live`-тест проверяет реальный эмбеддер на нескольких
случаях из `EXAMPLES`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.agents.inquiry_type_fallback import InquiryTypeFallbackClassifier, _Example
from app.kb.search import PASSAGE_PREFIX, QUERY_PREFIX
from app.taxonomy import InquiryType

_VERIFICATION_NEAR = (1.0, 0.0, 0.0)
_VERIFICATION_FAR = (0.6, 0.8, 0.0)  # тот же тип, но дальше — для проверки разрыва
_SEALING_NEAR = (0.0, 1.0, 0.0)
_UNRELATED = (0.0, 0.0, 1.0)


class FakeEmbedder:
    """Отдаёт заранее заданный вектор по точному тексту (с префиксом)."""

    def __init__(self, table: dict[str, Sequence[float]]) -> None:
        self._table = table

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._table[text] for text in texts]


_EXAMPLES = (
    _Example("пример поверки", InquiryType.METER_VERIFICATION, real=False),
    _Example("другой пример поверки", InquiryType.METER_VERIFICATION, real=False),
    _Example("пример опломбировки", InquiryType.METER_SEALING, real=False),
)


def _classifier(
    query_vector: Sequence[float], **overrides: object
) -> InquiryTypeFallbackClassifier:
    table = {
        PASSAGE_PREFIX + "пример поверки": _VERIFICATION_NEAR,
        PASSAGE_PREFIX + "другой пример поверки": _VERIFICATION_FAR,
        PASSAGE_PREFIX + "пример опломбировки": _SEALING_NEAR,
        QUERY_PREFIX + "вопрос": query_vector,
    }
    defaults: dict[str, object] = {"examples": _EXAMPLES}
    return InquiryTypeFallbackClassifier(FakeEmbedder(table), **{**defaults, **overrides})


async def test_ближайший_пример_даёт_тип():
    clf = _classifier(_VERIFICATION_NEAR, threshold=0.5, margin=0.0)
    assert await clf.classify("вопрос") is InquiryType.METER_VERIFICATION


async def test_ниже_порога_это_none():
    clf = _classifier(_UNRELATED, threshold=0.5, margin=0.0)
    assert await clf.classify("вопрос") is None


async def test_маленький_разрыв_до_другого_типа_это_none():
    clf = _classifier((0.72, 0.7, 0.0), threshold=0.5, margin=0.5)
    assert await clf.classify("вопрос") is None


async def test_разрыв_считается_только_до_другого_типа():
    # Второй ближайший пример — тоже поверка; разрыв меряется до первого
    # ПОСЛЕ него примера с другим типом, а не до второго по счёту вообще.
    clf = _classifier(_VERIFICATION_NEAR, threshold=0.5, margin=0.3)
    assert await clf.classify("вопрос") is InquiryType.METER_VERIFICATION


async def test_пустой_запрос_это_none():
    clf = _classifier(_VERIFICATION_NEAR, threshold=0.0, margin=0.0)
    assert await clf.classify("   ") is None


async def test_без_примеров_это_none():
    clf = InquiryTypeFallbackClassifier(FakeEmbedder({}), examples=())
    assert await clf.classify("что угодно") is None


async def test_отказ_эмбеддера_не_пробрасывается():
    class BrokenEmbedder:
        def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            raise RuntimeError("модель не загружена")

    clf = InquiryTypeFallbackClassifier(BrokenEmbedder(), examples=_EXAMPLES)
    assert await clf.classify("вопрос") is None


async def test_примеры_проэмбечены_один_раз():
    calls = []

    class CountingEmbedder:
        def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            calls.append(list(texts))
            if len(texts) == 1:
                return [_VERIFICATION_NEAR]
            return [_VERIFICATION_NEAR, _VERIFICATION_FAR, _SEALING_NEAR]

    clf = InquiryTypeFallbackClassifier(
        CountingEmbedder(), examples=_EXAMPLES, threshold=0.0, margin=0.0
    )
    await clf.classify("первый")
    await clf.classify("второй")
    example_calls = [c for c in calls if len(c) == len(_EXAMPLES)]
    assert len(example_calls) == 1


def test_примеры_ограничены_семью_регистрируемыми_типами():
    from app.agents.inquiry_type_fallback import EXAMPLES
    from app.backend.registration import REGISTRABLE

    assert {example.inquiry_type for example in EXAMPLES} <= REGISTRABLE


# --- живой эмбеддер: тяжёлый, только с --live ---------------------------- #


@pytest.mark.live
async def test_реальные_формулировки_живой_эмбеддер():
    from app.agents.inquiry_type_fallback import EXAMPLES
    from app.kb.search import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    embedder.warm_up()
    clf = InquiryTypeFallbackClassifier(embedder, examples=EXAMPLES)

    assert await clf.classify("Хочу заказать поверку счётчика воды") is (
        InquiryType.METER_VERIFICATION
    )
    assert await clf.classify("Нужно опломбировать новый счётчик") is (
        InquiryType.METER_SEALING
    )
    assert await clf.classify("Спасибо за помощь") is None
