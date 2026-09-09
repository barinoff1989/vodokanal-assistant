"""Проверки судьи качества ответа (LLM-judge, код-шаг 10).

Модель-судья подменяется функцией `complete`: тесты не поднимают ни Ollama, ни
LiteLLM. Проверяется разбор вердикта, правило «судья ≠ отвечающая модель» и
устойчивость к тому, что локальная модель обернёт JSON в пояснение.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.quality import AnswerJudge, JudgeConflict, JudgeUnavailable, QualityScores


def _response(content: str, *, model: str = "ollama/qwen2.5:7b-instruct") -> Any:
    """Ответ в форме LiteLLM: raw.choices[0].message.content + raw.model."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model=model)


def _judge(complete: Any, *, alias: str = "local-test") -> AnswerJudge:
    return AnswerJudge(settings=Settings(), model_alias=alias, complete=complete)


async def _clean_verdict(**_: Any) -> Any:
    return _response(
        '{"faithfulness": 0.9, "answer_relevancy": 0.75, "reasoning": "по контексту"}'
    )


async def test_вердикт_разбирается_в_оценки():
    judge = _judge(_clean_verdict)
    scores = await judge.score(
        question="Как часто поверять счётчик?",
        answer="Раз в шесть лет.",
        context=["Поверка счётчика — раз в шесть лет."],
    )
    assert isinstance(scores, QualityScores)
    assert scores.faithfulness == 0.9
    assert scores.answer_relevancy == 0.75
    assert scores.judge_model == "ollama/qwen2.5:7b-instruct"
    assert scores.provisional is False


async def test_json_в_обёртке_из_текста_всё_равно_разбирается():
    async def wrapped(**_: Any) -> Any:
        return _response(
            "Вот моя оценка:\n"
            '{"faithfulness": 1.0, "answer_relevancy": 0.8}\n'
            "Надеюсь, помог."
        )

    scores = await _judge(wrapped).score(question="в", answer="о", context=["к"])
    assert scores.faithfulness == 1.0
    assert scores.answer_relevancy == 0.8


async def test_проценты_приводятся_к_доле():
    async def percents(**_: Any) -> Any:
        return _response('{"faithfulness": "80%", "answer_relevancy": 95}')

    scores = await _judge(percents).score(question="в", answer="о", context=["к"])
    assert scores.faithfulness == pytest.approx(0.8)
    assert scores.answer_relevancy == pytest.approx(0.95)


async def test_судья_совпал_с_отвечающей_моделью_поднимает_конфликт():
    """Failover генерации: оценка не выполняется (ADR-007)."""

    async def must_not_be_called(**_: Any) -> Any:  # pragma: no cover
        raise AssertionError("модель не должна вызываться при конфликте")

    judge = _judge(must_not_be_called, alias="local-test")
    with pytest.raises(JudgeConflict):
        await judge.score(
            question="в",
            answer="о",
            context=["к"],
            answering_alias="local-test",
        )


async def test_разные_псевдонимы_конфликта_не_дают():
    judge = _judge(_clean_verdict, alias="local-test")
    assert judge.conflicts_with("yandexgpt") is False
    scores = await judge.score(
        question="в", answer="о", context=["к"], answering_alias="yandexgpt"
    )
    assert scores.faithfulness == 0.9


async def test_модель_не_ответила_это_judge_unavailable():
    async def boom(**_: Any) -> Any:
        raise TimeoutError("Ollama не отвечает")

    with pytest.raises(JudgeUnavailable):
        await _judge(boom).score(question="в", answer="о", context=["к"])


async def test_ответ_без_json_это_judge_unavailable():
    async def prose(**_: Any) -> Any:
        return _response("Ответ хороший, обоснованный.")

    with pytest.raises(JudgeUnavailable):
        await _judge(prose).score(question="в", answer="о", context=["к"])


async def test_нет_обеих_оценок_это_judge_unavailable():
    async def half(**_: Any) -> Any:
        return _response('{"faithfulness": 0.9}')

    with pytest.raises(JudgeUnavailable):
        await _judge(half).score(question="в", answer="о", context=["к"])


async def test_оценки_за_пределами_диапазона_зажимаются():
    async def wild(**_: Any) -> Any:
        return _response('{"faithfulness": -0.5, "answer_relevancy": 1.4}')

    scores = await _judge(wild).score(question="в", answer="о", context=["к"])
    assert scores.faithfulness == 0.0
    # 1.4 > 1 трактуется как «140» → делится на 100 → 0.014
    assert 0.0 <= scores.answer_relevancy <= 1.0


def test_псевдоним_по_умолчанию_из_настроек():
    judge = AnswerJudge(settings=Settings())
    assert judge.model_alias == Settings().judge_provider


def test_passes_проверяет_оба_порога():
    high = QualityScores(faithfulness=0.9, answer_relevancy=0.85, judge_model="x")
    low_rel = QualityScores(faithfulness=0.9, answer_relevancy=0.7, judge_model="x")
    assert high.passes(faithfulness_min=0.85, relevancy_min=0.8) is True
    assert low_rel.passes(faithfulness_min=0.85, relevancy_min=0.8) is False
