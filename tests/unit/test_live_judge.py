"""Проверки судьи на живом пути (async-часть Guardrails).

Судья и хранилище подменены: проверяется, что оценка пишется строкой, что
пропуски (конфликт / недоступность) тоже пишутся строкой, и что выборка
управляется `sample_rate`.
"""

from __future__ import annotations

from app.models import Channel, GenerateRequest, GenerationParameters, RequestMetadata
from app.quality.judge import JudgeConflict, JudgeUnavailable, QualityScores
from app.quality.live import LiveJudge
from app.quality.store import QualityAssessment
from app.taxonomy import Topic


class FakeJudge:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    async def score(self, **_: object) -> QualityScores:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeStore:
    def __init__(self) -> None:
        self.rows: list[QualityAssessment] = []

    def record(self, assessment: QualityAssessment) -> None:
        self.rows.append(assessment)


def _request() -> GenerateRequest:
    return GenerateRequest(
        system="помощник",
        query="Как заказать поверку?",
        parameters=GenerationParameters(stream=True),
        metadata=RequestMetadata(
            subscriber_id="2100202213", session_id="s-1",
            channel=Channel.LK_WEB, topic=Topic.GENERAL,
        ),
    )


def _live(judge: object, store: FakeStore, *, rate: float = 1.0) -> LiveJudge:
    return LiveJudge(
        judge=judge,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        sample_rate=rate,
        answering_alias="yandexgpt",
    )


async def test_оценка_пишется_строкой_scored():
    store = FakeStore()
    judge = FakeJudge(QualityScores(
        faithfulness=0.9, answer_relevancy=0.6, judge_model="qwen", reasoning="норм"
    ))
    await _live(judge, store).assess(_request(), "ответ", "trace-1", context=["ctx"])

    assert len(store.rows) == 1
    row = store.rows[0]
    assert row.outcome == "scored"
    assert row.faithfulness == 0.9
    assert row.answer_relevancy == 0.6
    assert row.provisional is True
    assert row.trace_id == "trace-1"
    assert row.topic == "general"


async def test_конфликт_судьи_пишется_строкой():
    store = FakeStore()
    await _live(FakeJudge(JudgeConflict("совпали")), store).assess(
        _request(), "ответ", "t", context=["ctx"]
    )
    assert store.rows[0].outcome == "conflict"
    assert store.rows[0].faithfulness is None


async def test_недоступность_судьи_пишется_строкой():
    store = FakeStore()
    await _live(FakeJudge(JudgeUnavailable("молчит")), store).assess(
        _request(), "ответ", "t", context=["ctx"]
    )
    assert store.rows[0].outcome == "unavailable"


async def test_любая_ошибка_судьи_не_пробрасывается():
    store = FakeStore()
    await _live(FakeJudge(RuntimeError("бум")), store).assess(
        _request(), "ответ", "t", context=["ctx"]
    )
    assert store.rows == []  # строки нет, но и исключения нет


def test_выборка_ноль_никогда_не_сэмплирует():
    assert _live(None, FakeStore(), rate=0.0).would_sample() is False


def test_выборка_единица_всегда_сэмплирует():
    assert _live(None, FakeStore(), rate=1.0).would_sample() is True
