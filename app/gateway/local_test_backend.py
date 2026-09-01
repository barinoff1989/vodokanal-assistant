"""Локальная тестовая модель как ещё один backend шлюза.

Назначение (шаг 0.5 плана разработки): дать возможность гонять промпты,
проверки Security Layer и нагрузочные профили без обращения к платному
внешнему API. Модель **не отвечает абонентам** — на MVP генерацию выполняет
управляемый API на территории РФ (ADR-002).

Почему обёртка тонкая. LiteLLM поддерживает Ollama как провайдера штатно
(`ollama_chat/<model>`), поэтому само переключение обеспечивается конфигурацией
`litellm_config.yaml`, а не кодом. Этому модулю остаётся то, чего конфигурация
не делает:

1. **Преполётная проверка** — сказать понятным текстом, что Ollama не запущена
   или модель не загружена. Без неё обе ситуации выглядят как невнятная сетевая
   ошибка, а смысл шага в быстрых офлайн-проверках.
2. **Единая форма результата** — вызывающий код не должен ветвиться в
   зависимости от того, локальная модель отвечает или внешняя.
3. **Запрет на использование в production** — локальная модель тестовая, и
   попасть в боевой контур она не должна даже по ошибке конфигурации.

Контракт. Модуль реализует **провайдерский** контракт, одинаковый для локальной
и внешней модели, а не публичный контракт `/v1/generate` из раздела 7 контекста:
тот принадлежит уровню выше и собирается шлюзом на шаге 4. Формулировка плана
«отвечает по тому же контракту, что и внешняя модель» относится именно к
провайдерскому уровню — иначе обёртка дублировала бы шлюз.

Логирование. Промпты и ответы не логируются: этот же кодовый путь используется
с внешним провайдером, а логирование сырых персональных данных запрещено
правилом 4.2. В журнал уходят только имя модели и статистика токенов.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
DEFAULT_TIMEOUT = 120.0

# Префикс провайдера LiteLLM для чат-эндпоинта Ollama.
_LITELLM_PREFIX = "ollama_chat/"


# --------------------------------------------------------------------------- #
# Ошибки
# --------------------------------------------------------------------------- #


class LocalBackendError(RuntimeError):
    """Базовая ошибка локального backend'а."""


class LocalBackendForbidden(LocalBackendError):
    """Попытка использовать тестовую модель в production."""


class LocalBackendUnavailable(LocalBackendError):
    """Ollama не отвечает по указанному адресу."""


class LocalModelMissing(LocalBackendError):
    """Ollama работает, но нужная модель не загружена."""


# --------------------------------------------------------------------------- #
# Провайдерский контракт — одинаковый для локальной и внешней модели
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Usage:
    """Расход токенов. Нужен для квот и для строки OPEX в отчёте по мощностям."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Результат генерации в форме, не зависящей от провайдера."""

    text: str
    model: str
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True)
class ProviderChunk:
    """Фрагмент потока. `done=True` приходит последним и несёт итоговый результат."""

    delta: str = ""
    done: bool = False
    result: ProviderResult | None = None


Message = dict[str, str]


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


class LocalTestBackend:
    """Тонкая обёртка над локальной моделью, подключённой через LiteLLM.

    Параметры передаются явно, а не читаются из настроек внутри: так модуль
    остаётся проверяемым без окружения и без установленного LiteLLM. Сборка
    из настроек — в :meth:`from_settings`.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        app_env: str = "local",
        completion_fn: Callable[..., Any] | None = None,
        http_get: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.app_env = app_env
        # Обе зависимости внедряются для тестов; по умолчанию берутся лениво,
        # чтобы импорт модуля не требовал ни litellm, ни сети.
        self._completion_fn = completion_fn
        self._http_get = http_get

    # -- сборка ------------------------------------------------------------- #

    @classmethod
    def from_settings(cls) -> LocalTestBackend:
        """Собрать backend из настроек приложения (ленивый импорт конфигурации)."""
        from app.config import get_settings

        s = get_settings()
        return cls(
            model=s.local_model,
            base_url=s.ollama_base_url,
            timeout=s.local_timeout_seconds,
            app_env=s.app_env,
        )

    @property
    def litellm_model(self) -> str:
        """Имя модели в терминах LiteLLM."""
        return f"{_LITELLM_PREFIX}{self.model}"

    # -- защита ------------------------------------------------------------- #

    def ensure_allowed(self) -> None:
        """Запретить тестовую модель в боевом контуре.

        Локальная модель не проходила оценку качества на эталонном наборе и не
        предназначена для ответов абонентам. Ошибка конфигурации не должна
        приводить к тому, что абонент получит ответ от неё.
        """
        if self.app_env == "production":
            raise LocalBackendForbidden(
                "Локальная тестовая модель недопустима в production: "
                "генерацию выполняет управляемый API на территории РФ (ADR-002). "
                "Проверьте LLM_PROVIDER в окружении."
            )

    # -- преполётная проверка ----------------------------------------------- #

    async def preflight(self) -> None:
        """Убедиться, что Ollama запущена и нужная модель загружена.

        Вызывается перед первым обращением. Разделяет два случая, которые иначе
        выглядят одинаково невнятно: сервис не поднят и модель не скачана.
        """
        self.ensure_allowed()
        tags_url = f"{self.base_url}/api/tags"

        try:
            payload = await self._get_json(tags_url)
        except Exception as exc:  # noqa: BLE001 — причина уходит в сообщение
            raise LocalBackendUnavailable(
                f"Ollama не отвечает по адресу {self.base_url}. "
                f"Запустите её командой `ollama serve` и повторите. Причина: {exc}"
            ) from exc

        available = {m.get("name", "") for m in payload.get("models", [])}
        if not self._model_present(available):
            raise LocalModelMissing(
                f"Модель {self.model} не загружена. "
                f"Выполните `ollama pull {self.model}`. "
                f"Доступны: {', '.join(sorted(available)) or '—'}"
            )

    def _model_present(self, available: set[str]) -> bool:
        """Ollama возвращает имена с тегом; сравниваем с учётом тега по умолчанию."""
        if self.model in available:
            return True
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in available

    async def _get_json(self, url: str) -> dict[str, Any]:
        if self._http_get is not None:
            return await self._http_get(url)

        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data

    # -- генерация ---------------------------------------------------------- #

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> ProviderResult:
        """Сгенерировать ответ целиком."""
        self.ensure_allowed()
        raw = await self._acompletion(
            model=self.litellm_model,
            messages=list(messages),
            api_base=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
            stream=False,
        )
        result = self._to_result(raw)
        # В журнал — только модель и токены. Ни промпта, ни ответа (правило 4.2).
        logger.info(
            "local-test complete: model=%s prompt_tokens=%d completion_tokens=%d",
            result.model,
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
        )
        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[ProviderChunk]:
        """Сгенерировать ответ потоком.

        Поток обязателен на всём пути до абонента (правило 4.3), поэтому
        локальный backend обязан уметь то же, что и внешний: отдавать фрагменты
        по мере генерации, а не собирать ответ целиком.
        """
        self.ensure_allowed()
        raw_stream = await self._acompletion(
            model=self.litellm_model,
            messages=list(messages),
            api_base=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
            stream=True,
        )

        collected: list[str] = []
        finish_reason = "stop"

        async for raw_chunk in raw_stream:
            delta = self._extract_delta(raw_chunk)
            reason = self._extract_finish_reason(raw_chunk)
            if reason:
                finish_reason = reason
            if delta:
                collected.append(delta)
                yield ProviderChunk(delta=delta)

        text = "".join(collected)
        yield ProviderChunk(
            done=True,
            result=ProviderResult(
                text=text,
                model=self.litellm_model,
                finish_reason=finish_reason,
                # Ollama не всегда отдаёт статистику в потоке; оценка по факту
                # выдачи честнее, чем ноль, но помечена как приблизительная
                # в docs/model-choice.md.
                usage=Usage(completion_tokens=len(text.split())),
            ),
        )

    # -- работа с ответом LiteLLM ------------------------------------------- #

    async def _acompletion(self, **kwargs: Any) -> Any:
        if self._completion_fn is not None:
            return await self._completion_fn(**kwargs)

        from litellm import acompletion

        return await acompletion(**kwargs)

    @staticmethod
    def _to_result(raw: Any) -> ProviderResult:
        """Привести ответ LiteLLM к провайдерскому контракту."""
        choice = _first_choice(raw)
        message = _get(choice, "message") or {}
        usage_raw = _get(raw, "usage") or {}

        return ProviderResult(
            text=_get(message, "content") or "",
            model=_get(raw, "model") or "",
            finish_reason=_get(choice, "finish_reason") or "stop",
            usage=Usage(
                prompt_tokens=int(_get(usage_raw, "prompt_tokens") or 0),
                completion_tokens=int(_get(usage_raw, "completion_tokens") or 0),
            ),
        )

    @staticmethod
    def _extract_delta(raw_chunk: Any) -> str:
        choice = _first_choice(raw_chunk)
        delta = _get(choice, "delta") or {}
        return str(_get(delta, "content") or "")

    @staticmethod
    def _extract_finish_reason(raw_chunk: Any) -> str | None:
        choice = _first_choice(raw_chunk)
        reason = _get(choice, "finish_reason")
        return str(reason) if reason else None


# --------------------------------------------------------------------------- #
# Вспомогательное: LiteLLM отдаёт объекты, тесты — словари
# --------------------------------------------------------------------------- #


def _get(obj: Any, key: str) -> Any:
    """Достать поле и из словаря, и из объекта — форма ответа зависит от версии."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first_choice(raw: Any) -> Any:
    choices = _get(raw, "choices")
    if not choices:
        return None
    return choices[0]
