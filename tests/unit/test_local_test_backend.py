"""Модульные тесты локального backend'а.

Ни один тест здесь не требует ни запущенной Ollama, ни установленного LiteLLM:
обе зависимости внедряются. Живая проверка на реальной модели — отдельный
тест с меткой `live`, он пропускается в обычном прогоне.
"""

from __future__ import annotations

import pytest

from app.gateway.local_test_backend import (
    LocalBackendForbidden,
    LocalBackendUnavailable,
    LocalModelMissing,
    LocalTestBackend,
    ProviderResult,
)

MESSAGES = [
    {"role": "system", "content": "Ты помощник абонента водоканала."},
    {"role": "user", "content": "Когда нужна поверка счётчика?"},
]


# --------------------------------------------------------------------------- #
# Заглушки
# --------------------------------------------------------------------------- #


def fake_response(text: str = "Раз в шесть лет.") -> dict:
    """Ответ в форме LiteLLM (словарём — так же читается кодом, как объект)."""
    return {
        "model": "ollama_chat/qwen2.5:7b-instruct-q4_K_M",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
    }


def make_backend(**kwargs) -> LocalTestBackend:
    async def default_completion(**_):
        return fake_response()

    kwargs.setdefault("completion_fn", default_completion)
    return LocalTestBackend(**kwargs)


# --------------------------------------------------------------------------- #
# Защита боевого контура
# --------------------------------------------------------------------------- #


def test_запрещена_в_production():
    """Тестовая модель не должна попасть в боевой контур даже по ошибке конфига."""
    backend = make_backend(app_env="production")
    with pytest.raises(LocalBackendForbidden):
        backend.ensure_allowed()


@pytest.mark.parametrize("env", ["local", "dev"])
def test_разрешена_вне_production(env):
    make_backend(app_env=env).ensure_allowed()


async def test_генерация_в_production_отклоняется_до_вызова_модели():
    """Запрет срабатывает раньше, чем произойдёт обращение к модели."""
    called = False

    async def spy(**_):
        nonlocal called
        called = True
        return fake_response()

    backend = make_backend(app_env="production", completion_fn=spy)
    with pytest.raises(LocalBackendForbidden):
        await backend.complete(MESSAGES)
    assert called is False


# --------------------------------------------------------------------------- #
# Преполётная проверка: два случая должны различаться
# --------------------------------------------------------------------------- #


async def test_преполёт_проходит_когда_модель_загружена():
    async def tags(_url):
        return {"models": [{"name": "qwen2.5:7b-instruct-q4_K_M"}]}

    await make_backend(http_get=tags).preflight()


async def test_преполёт_сообщает_что_ollama_не_запущена():
    async def tags(_url):
        raise ConnectionError("connection refused")

    backend = make_backend(http_get=tags)
    with pytest.raises(LocalBackendUnavailable) as exc:
        await backend.preflight()
    assert "ollama serve" in str(exc.value)


async def test_преполёт_сообщает_что_модель_не_загружена():
    """Отдельная ошибка со строкой команды — иначе случай неотличим от сбоя сети."""

    async def tags(_url):
        return {"models": [{"name": "llama3:latest"}]}

    backend = make_backend(http_get=tags)
    with pytest.raises(LocalModelMissing) as exc:
        await backend.preflight()
    assert "ollama pull" in str(exc.value)
    assert "llama3:latest" in str(exc.value)


async def test_преполёт_учитывает_тег_latest():
    """Модель без тега в настройках и с тегом latest в Ollama — это одно и то же."""

    async def tags(_url):
        return {"models": [{"name": "qwen2.5:latest"}]}

    await make_backend(model="qwen2.5", http_get=tags).preflight()


# --------------------------------------------------------------------------- #
# Провайдерский контракт
# --------------------------------------------------------------------------- #


async def test_ответ_приводится_к_провайдерскому_контракту():
    result = await make_backend().complete(MESSAGES)

    assert isinstance(result, ProviderResult)
    assert result.text == "Раз в шесть лет."
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 42
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 49


async def test_имя_модели_для_litellm_собирается_с_префиксом():
    backend = make_backend(model="qwen2.5:7b-instruct-q4_K_M")
    assert backend.litellm_model == "ollama_chat/qwen2.5:7b-instruct-q4_K_M"


async def test_пустой_ответ_не_ломает_разбор():
    """Провайдер вправе вернуть пустой ответ — это не повод падать исключением."""

    async def empty(**_):
        return {"model": "m", "choices": [{"message": {}, "finish_reason": "length"}]}

    result = await make_backend(completion_fn=empty).complete(MESSAGES)
    assert result.text == ""
    assert result.finish_reason == "length"
    assert result.usage.total_tokens == 0


async def test_параметры_доходят_до_вызова():
    captured: dict = {}

    async def spy(**kwargs):
        captured.update(kwargs)
        return fake_response()

    backend = make_backend(completion_fn=spy, base_url="http://localhost:11434")
    await backend.complete(MESSAGES, temperature=0.3, max_tokens=256)

    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 256
    assert captured["stream"] is False
    assert captured["api_base"] == "http://localhost:11434"
    assert captured["messages"] == MESSAGES


# --------------------------------------------------------------------------- #
# Поток — правило потоковой выдачи
# --------------------------------------------------------------------------- #


async def test_поток_отдаёт_фрагменты_по_мере_генерации():
    """Фрагменты должны приходить по одному, а не одним куском в конце."""

    async def streaming(**_):
        async def gen():
            for piece in ["Раз ", "в шесть ", "лет."]:
                yield {"choices": [{"delta": {"content": piece}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return gen()

    chunks = [c async for c in make_backend(completion_fn=streaming).stream(MESSAGES)]

    deltas = [c.delta for c in chunks if not c.done]
    assert deltas == ["Раз ", "в шесть ", "лет."]

    final = chunks[-1]
    assert final.done is True
    assert final.result is not None
    assert final.result.text == "Раз в шесть лет."
    assert final.result.finish_reason == "stop"


async def test_поток_всегда_завершается_финальным_фрагментом():
    """Даже пустой поток обязан закрыться — обрыв соединения недопустим."""

    async def empty_stream(**_):
        async def gen():
            if False:  # pragma: no cover — намеренно пустой генератор
                yield {}

        return gen()

    chunks = [c async for c in make_backend(completion_fn=empty_stream).stream(MESSAGES)]
    assert len(chunks) == 1
    assert chunks[0].done is True
    assert chunks[0].result is not None
    assert chunks[0].result.text == ""


async def test_поток_в_production_отклоняется():
    backend = make_backend(app_env="production")
    with pytest.raises(LocalBackendForbidden):
        [c async for c in backend.stream(MESSAGES)]
