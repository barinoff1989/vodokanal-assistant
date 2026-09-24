"""Живая проверка на реальной модели.

Единственный тест, которому нужны запущенная Ollama, загруженная
модель и установленный LiteLLM. В обычном прогоне пропускается — запускать
явно:

    pytest -m live

Он не дублирует контрактные тесты, а проверяет то, что заглушкой проверить
нельзя: что модель действительно поднимается, отвечает по-русски и отдаёт
ответ потоком, а не одним куском в конце.
"""

from __future__ import annotations

import os
import time

import pytest

from app.gateway.local_test_backend import LocalTestBackend

pytestmark = pytest.mark.live

MESSAGES = [
    {
        "role": "system",
        "content": "Ты помощник абонента водоканала. Отвечай кратко и по-русски.",
    },
    {"role": "user", "content": "Как часто нужно поверять счётчик воды?"},
]


def _backend() -> LocalTestBackend:
    return LocalTestBackend(
        model=os.getenv("LOCAL_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


@pytest.fixture(autouse=True)
def _require_ollama() -> None:
    """Пропустить живые тесты, если Ollama не поднята или модель не загружена.

    Без этой проверки `pytest -m live` на машине без Ollama давал бы падение
    вместо пропуска — красный прогон по причине, не связанной с кодом.
    """
    import httpx

    backend = _backend()
    try:
        response = httpx.get(f"{backend.base_url}/api/tags", timeout=2.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — причина недоступности здесь не важна
        pytest.skip(f"Ollama недоступна на {backend.base_url}: {exc}. См. `make model-pull`.")

    loaded = {model.get("name", "") for model in response.json().get("models", [])}
    family = backend.model.split(":")[0]
    if not any(name.startswith(family) for name in loaded):
        pytest.skip(f"Модель {backend.model} не загружена. Выполните `make model-pull`.")


async def test_модель_поднята_и_загружена():
    """Преполёт на живой Ollama: сервис отвечает, нужная модель на месте."""
    await _backend().preflight()


async def test_модель_отвечает_по_русски():
    result = await _backend().complete(MESSAGES, max_tokens=200)

    assert result.text.strip(), "модель вернула пустой ответ"
    # Кириллица в ответе — минимальная проверка того, что модель пригодна
    # для русскоязычных промптов проекта.
    assert any("а" <= c.lower() <= "я" for c in result.text), (
        f"в ответе нет кириллицы, модель непригодна для проекта: {result.text[:200]!r}"
    )


async def test_ответ_приходит_потоком_а_не_одним_куском():
    """Правило потоковой выдачи: локальный backend обязан уметь поток так же, как внешний.

    Замеряется время до первого фрагмента. Если оно почти совпадает с временем
    полного ответа — значит, ответ собрался целиком и лишь потом отдан,
    а это нарушение потоковой отдачи.
    """
    started = time.perf_counter()
    first_delta_at: float | None = None
    pieces = 0

    async for chunk in _backend().stream(MESSAGES, max_tokens=200):
        if chunk.delta and first_delta_at is None:
            first_delta_at = time.perf_counter() - started
        if chunk.delta:
            pieces += 1

    total = time.perf_counter() - started

    assert pieces > 1, "поток пришёл одним фрагментом — это не поток"
    assert first_delta_at is not None
    assert first_delta_at < total * 0.8, (
        f"первый фрагмент пришёл на {first_delta_at:.2f} с из {total:.2f} с — "
        "похоже, ответ собирается целиком перед отдачей"
    )
