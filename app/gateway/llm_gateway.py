"""Шлюз к модели: единственная точка, через которую идут все обращения.

Ради этого компонента в ADR-002 выбран вариант с единым шлюзом: смена провайдера
должна быть правкой конфигурации, а не переписыванием бизнес-логики. Отсюда и
устройство — здесь нет ничего про водоканал, только порядок проверок и перевод
ошибок в формат RFC 7807.

ПОРЯДОК ПРОВЕРОК ЗАФИКСИРОВАН РАЗДЕЛОМ 41.3 И МЕНЯТЬ ЕГО НЕЛЬЗЯ::

    Backend -> Quota Manager -> PII Sanitizer -> LiteLLM -> Guardrails -> Backend

Лимиты идут первыми осознанно: отклонить запрос по превышению дешевле, чем
сначала тратить процессор на обезличивание. Правило 4.2 при этом не нарушается —
персональные данные всё равно скрыты до обращения к провайдеру, а учёт лимитов
ведётся по идентификаторам абонента и сессии и текста запроса не касается.

ПОТОК ОБЯЗАТЕЛЕН (правило 4.3). Основной способ вызова — :meth:`stream`, отдающий
события по мере готовности. Ответ целиком (:meth:`generate`) нужен для служебных
вызовов вроде классификации, где промежуточный вывод некому показывать.

ОШИБКИ. Наружу уходит :class:`ProblemDetail` — одно тело на все случаи, включая
ошибку внутри уже начатого потока (решение 4 в `app/models.py`). Коды:

===== ============================================================
  400  запрос не прошёл проверку схемы
  429  превышен лимит (раздел 35.1)
  502  провайдер ответил ошибкой
  503  хранилище счётчиков недоступно (ADR-008)
  504  провайдер не ответил вовремя
===== ============================================================
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from app.config import Settings, get_settings
from app.gateway.guardrails import StreamGuard, SyncGuardrails
from app.gateway.pii_filter import PiiSanitizer
from app.gateway.quota import QuotaManager, QuotaStoreUnavailableError
from app.models import (
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    MetadataEvent,
    PiiReport,
    ProblemDetail,
    SourceRef,
    TokenEvent,
    Usage,
)

__all__ = [
    "ERROR_BASE",
    "GatewayEvent",
    "LlmGateway",
    "ProviderError",
    "ProviderTimeoutError",
]

ERROR_BASE = "https://vodokanal.example/errors"
"""Основа для поля `type` в теле ошибки. По RFC 7807 это опознаватель типа
проблемы, а не адрес, который обязан открываться."""

GatewayEvent = TokenEvent | MetadataEvent | DoneEvent | ErrorEvent


class ProviderError(RuntimeError):
    """Провайдер ответил ошибкой. Отображается в `502`."""


class ProviderTimeoutError(ProviderError):
    """Провайдер не ответил вовремя. Отображается в `504`."""


class LlmGateway:
    """Шлюз: проверки, вызов модели, охрана ответа.

    Все зависимости внедряются снаружи. Это не про чистоту ради чистоты: без
    подмены провайдера проверки требовали бы платных обращений к внешнему API, а
    без подмены хранилища — поднятого Redis.

    :param quota: учёт лимитов. ``None`` — лимиты не проверяются; допустимо
        только для служебных вызовов внутри процесса, не для пути абонента.
    :param completion: функция обращения к модели. По умолчанию берётся LiteLLM
        лениво, чтобы импорт модуля не тянул тяжёлую зависимость.
    """

    def __init__(
        self,
        *,
        quota: QuotaManager | None = None,
        sanitizer: PiiSanitizer | None = None,
        guardrails: SyncGuardrails | None = None,
        completion: Callable[..., Awaitable[Any]] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._sanitizer = sanitizer if sanitizer is not None else PiiSanitizer()
        self._guardrails = (
            guardrails if guardrails is not None else SyncGuardrails(sanitizer=self._sanitizer)
        )
        self._quota = quota
        self._completion = completion

    # -- вспомогательное ------------------------------------------------------ #

    @staticmethod
    def new_trace_id() -> str:
        """Опознаватель запроса для журнала и для жалобы абонента."""
        return f"trace-{uuid.uuid4().hex[:16]}"

    def problem(
        self, slug: str, title: str, status: int, trace_id: str, detail: str | None = None
    ) -> ProblemDetail:
        return ProblemDetail(
            type=f"{ERROR_BASE}/{slug}",
            title=title,
            status=status,
            detail=detail,
            instance="/v1/generate",
            trace_id=trace_id,
        )

    def _prepare(self, request: GenerateRequest) -> tuple[list[dict[str, str]], PiiReport]:
        """Собрать сообщения для модели, обезличив всё, что уходит наружу.

        Обезличивается не только вопрос абонента, но и найденный контекст:
        фрагменты базы знаний в теории могут содержать пример с настоящими
        данными, и правило 4.2 не делает для них исключения.
        """
        clean_query, report = self._sanitizer.sanitize(request.query)

        pieces: list[str] = []
        for chunk in request.context:
            clean_chunk, chunk_report = self._sanitizer.sanitize(chunk.text)
            pieces.append(f"[{chunk.source_title}]\n{clean_chunk}")
            if chunk_report.pii_detected:
                report = PiiReport(
                    pii_detected=True,
                    entities=sorted(set(report.entities) | set(chunk_report.entities)),
                )

        content = clean_query
        if pieces:
            content = "Контекст:\n" + "\n\n".join(pieces) + "\n\nВопрос: " + clean_query

        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": content},
        ]
        return messages, report

    def _check_quota(self, request: GenerateRequest, trace_id: str) -> ProblemDetail | None:
        """Проверить лимиты. Возвращает тело ошибки либо ``None``, если можно."""
        if self._quota is None:
            return None
        try:
            decision = self._quota.check(
                request.metadata.subscriber_id, request.metadata.session_id
            )
        except QuotaStoreUnavailableError:
            # ADR-008: отказ хранилища — это 503, а не 429. Код 429 означал бы
            # «вы превысили лимит» и возлагал причину на абонента, который ни при
            # чём, а виджет тихо повторил бы запрос вместо сообщения о сбое.
            return self.problem(
                "quota-store-unavailable",
                "Service temporarily unavailable",
                503,
                trace_id,
                "Сервис временно недоступен. Попробуйте повторить запрос позже.",
            )
        if not decision.allowed:
            return self.problem(
                "quota-exceeded", "Too many requests", 429, trace_id, decision.detail
            )
        return None

    @staticmethod
    def _sources(request: GenerateRequest) -> list[SourceRef]:
        """Источники ответа — из того же контекста, что ушёл в модель.

        Раздел 18 требует, чтобы ответ содержал источники; поля принимались на
        входе, но не возвращались (раздел 7.6, пункт 2). Здесь это исправлено.
        """
        return [
            SourceRef(
                chunk_id=chunk.chunk_id,
                source_title=chunk.source_title,
                source_url=chunk.source_url,
                relevance_score=chunk.relevance_score,
            )
            for chunk in request.context
        ]

    async def _call(
        self,
        request: GenerateRequest,
        messages: list[dict[str, str]],
        *,
        stream: bool,
    ) -> Any:
        """Обратиться к модели через LiteLLM либо через подменённую функцию."""
        completion = self._completion
        if completion is None:
            from litellm import acompletion

            completion = acompletion

        return await completion(
            messages=messages,
            max_tokens=request.parameters.max_tokens,
            temperature=request.parameters.temperature,
            stream=stream,
            **self._provider_params(),
        )

    def _provider_params(self) -> dict[str, Any]:
        """Расшифровать псевдоним провайдера в параметры вызова.

        Псевдонимы (`local-test`, `yandexgpt`, `gigachat`) описаны и в
        `litellm_config.yaml`, но эту подстановку выполняет только прокси
        LiteLLM, которого в прототипе нет: мы обращаемся к `acompletion`
        напрямую. Живой вызов это и вскрыл — псевдоним уходил провайдеру как
        имя модели и не распознавался.

        > **[ИЗВЕСТНОЕ ДУБЛИРОВАНИЕ]** Пока прокси не введён, `litellm_config.yaml`
        > остаётся описанием, а рабочий источник — этот метод. Когда прокси
        > появится, всё наоборот: конфигурация станет живой, а метод исчезнет.
        > Владелец значений — `app/config.py`, чтобы расхождение не завелось
        > в третьем месте.
        """
        settings = self._settings
        alias = settings.llm_provider

        if alias == "yandexgpt":
            return {
                "model": settings.yandex_model,
                "api_base": settings.yandex_api_base,
                "api_key": settings.yandex_api_key,
            }
        if alias == "gigachat":
            return {"model": "gigachat/GigaChat-Pro"}
        return {
            "model": f"ollama_chat/{settings.local_model}",
            "api_base": settings.ollama_base_url,
        }

    # -- ответ целиком --------------------------------------------------------- #

    async def generate(self, request: GenerateRequest) -> GenerateResponse | ProblemDetail:
        """Получить ответ целиком. Для служебных вызовов, не для пути абонента."""
        trace_id = self.new_trace_id()

        if (problem := self._check_quota(request, trace_id)) is not None:
            return problem

        messages, report = self._prepare(request)

        try:
            raw = await self._call(request, messages, stream=False)
        except ProviderTimeoutError as exc:
            return self.problem("upstream-timeout", "LLM provider timeout", 504, trace_id, str(exc))
        except Exception as exc:  # noqa: BLE001 — любая ошибка провайдера это 502
            return self.problem(
                "upstream-unavailable", "LLM provider unavailable", 502, trace_id, str(exc)
            )

        answer = _extract_text(raw)
        verdict = self._guardrails.check(answer)
        if not verdict.allowed:
            answer = verdict.text_for_subscriber or ""

        return GenerateResponse(
            answer=answer,
            model=self._settings.llm_provider,
            finish_reason=verdict.finish_reason or FinishReason.STOP,
            usage=_extract_usage(raw),
            sources=self._sources(request),
            pii_report=report,
            routing={"provider": self._settings.llm_provider},
            trace_id=trace_id,
        )

    # -- поток ------------------------------------------------------------------ #

    async def stream(self, request: GenerateRequest) -> AsyncIterator[GatewayEvent]:
        """Отдать ответ потоком — основной путь абонента (правило 4.3).

        События идут в том же порядке, что описан в разделе 7.3: куски текста,
        затем метаданные, затем завершение. Ошибка на любом шаге превращается в
        событие ошибки с тем же телом RFC 7807, что и у обычного ответа.
        """
        trace_id = self.new_trace_id()

        if (problem := self._check_quota(request, trace_id)) is not None:
            yield ErrorEvent(problem=problem)
            return

        messages, report = self._prepare(request)

        try:
            raw_stream = await self._call(request, messages, stream=True)
        except ProviderTimeoutError as exc:
            yield ErrorEvent(
                problem=self.problem(
                    "upstream-timeout", "LLM provider timeout", 504, trace_id, str(exc)
                )
            )
            return
        except Exception as exc:  # noqa: BLE001
            yield ErrorEvent(
                problem=self.problem(
                    "upstream-unavailable", "LLM provider unavailable", 502, trace_id, str(exc)
                )
            )
            return

        guard = StreamGuard(guardrails=self._guardrails)
        blocked = False

        async for piece in raw_stream:
            delta = _extract_delta(piece)
            if not delta:
                continue
            verdict = guard.feed(delta)
            if not verdict.allowed:
                # Охранитель сработал: наружу уходит заглушка вместо остатка
                # ответа. Уже отданные куски отозвать нельзя, поэтому проверка
                # стоит до выдачи, а не после.
                blocked = True
                yield TokenEvent(delta=verdict.text_for_subscriber or "")
                break
            yield TokenEvent(delta=delta)

        yield MetadataEvent(sources=self._sources(request))
        yield DoneEvent(
            finish_reason=FinishReason.GUARDRAIL if blocked else FinishReason.STOP,
            usage=Usage(),
            trace_id=trace_id,
        )
        # Отчёт об обезличивании собран, но никуда не пишется: приёмник появится
        # на шаге 10 вместе с телеметрией. Здесь он остаётся частью ответа
        # целиком (`generate`) — см. `pii_report` там.
        _ = report


# --- разбор ответа провайдера --------------------------------------------------- #
# LiteLLM приводит ответы разных провайдеров к общему виду, но объект остаётся
# чужим. Разбор вынесен в отдельные функции, чтобы подмена в тестах не требовала
# воспроизводить всю его структуру.


def _extract_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return str(raw.choices[0].message.content or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def _extract_delta(piece: Any) -> str:
    if isinstance(piece, str):
        return piece
    try:
        return str(piece.choices[0].delta.content or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def _extract_usage(raw: Any) -> Usage:
    try:
        usage = raw.usage
        return Usage(
            prompt_tokens=int(usage.prompt_tokens or 0),
            completion_tokens=int(usage.completion_tokens or 0),
        )
    except (AttributeError, TypeError, ValueError):
        return Usage()
