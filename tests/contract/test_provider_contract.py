"""Контрактный тест провайдерского уровня.

Критерий: один и тот же набор проверок проходит и на
локальной модели, и на внешней, без изменений в коде. Здесь это выражено
буквально — тесты параметризованы по backend'у, а тело у них общее.

Внешний провайдер на прототипе подменяется заглушкой: обращаться к платному
API ради контрактной проверки не нужно, а форма ответа у LiteLLM одинакова
для всех провайдеров — именно на этом и держится гибридная стратегия ADR-200.
"""

from __future__ import annotations

import pytest

from app.gateway.local_test_backend import (
    LocalTestBackend,
    ProviderChunk,
    ProviderResult,
)

MESSAGES = [{"role": "user", "content": "Когда нужна поверка счётчика?"}]
ANSWER = "Поверка счётчика проводится раз в шесть лет."


def _response(model: str) -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": ANSWER}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 9},
    }


def _stream(model: str):
    async def factory(**_):
        async def gen():
            for piece in ["Поверка ", "счётчика ", "проводится ", "раз в шесть лет."]:
                yield {"choices": [{"delta": {"content": piece}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return gen()

    return factory


def _backend(model: str, streaming: bool = False) -> LocalTestBackend:
    """Один и тот же класс обслуживает оба провайдера — различие только в конфиге."""
    if streaming:
        return LocalTestBackend(model=model, completion_fn=_stream(model))

    async def completion(**_):
        return _response(model)

    return LocalTestBackend(model=model, completion_fn=completion)


# Локальная тестовая модель и управляемый API на территории РФ.
PROVIDERS = ["qwen2.5:7b-instruct-q4_K_M", "GigaChat-Pro"]


@pytest.mark.parametrize("model", PROVIDERS)
async def test_форма_результата_одинакова_у_всех_провайдеров(model):
    result = await _backend(model).complete(MESSAGES)

    assert isinstance(result, ProviderResult)
    assert result.text == ANSWER
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 30
    assert result.usage.completion_tokens == 9


@pytest.mark.parametrize("model", PROVIDERS)
async def test_поток_одинаков_у_всех_провайдеров(model):
    chunks = [c async for c in _backend(model, streaming=True).stream(MESSAGES)]

    assert all(isinstance(c, ProviderChunk) for c in chunks)
    assert [c.delta for c in chunks if not c.done] == [
        "Поверка ",
        "счётчика ",
        "проводится ",
        "раз в шесть лет.",
    ]
    assert chunks[-1].done is True
    assert chunks[-1].result is not None
    assert chunks[-1].result.text == ANSWER


@pytest.mark.parametrize("model", PROVIDERS)
async def test_вызывающий_код_не_ветвится_по_провайдеру(model):
    """Смена провайдера не требует ни одной строки кода на стороне вызывающего."""
    backend = _backend(model)
    result = await backend.complete(MESSAGES)

    # Ни имя провайдера, ни его особенности в результат не протекают —
    # наружу видна только общая форма.
    assert set(result.__slots__) == {"text", "model", "finish_reason", "usage"}
