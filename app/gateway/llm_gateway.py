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

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.gateway.guardrails import StreamGuard, SyncGuardrails
from app.gateway.pii_filter import PiiSanitizer
from app.gateway.quota import QuotaManager, QuotaStoreUnavailableError
from app.metrics import prometheus as metrics
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
    "DirectAnswer",
    "DirectResponder",
    "GatewayEvent",
    "LlmGateway",
    "ProviderError",
    "ProviderTimeoutError",
]

UPSTREAM_DETAIL = "Сервис ответа временно недоступен. Попробуйте повторить запрос."
"""Что видит абонент при сбое провайдера.

Внутренний текст ошибки наружу не идёт: он содержит адреса, порты и подробности
устройства системы, абоненту непонятен и раскрывает лишнее. Техническая причина
пишется в журнал рядом с опознавателем запроса — по нему жалоба связывается с
записью.
"""

ERROR_BASE = "https://vodokanal.example/errors"
"""Основа для поля `type` в теле ошибки. По RFC 7807 это опознаватель типа
проблемы, а не адрес, который обязан открываться."""

GatewayEvent = TokenEvent | MetadataEvent | DoneEvent | ErrorEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DirectAnswer:
    """Готовый ответ, для которого модель не нужна.

    Два случая обходят генерацию по противоположным причинам: график отключений
    меняется постоянно и требует точности (ADR-013), утверждённая формулировка о
    качестве воды не меняется вовсе (ADR-012). Общее у них то, что обоим нужен
    поиск по ключу, а не по смыслу.
    """

    text: str
    disclaimer: str | None = None


class DirectResponder(Protocol):
    """Тот, кто умеет ответить без модели.

    Шлюз знает только это — не про водоканал, не про адреса и не про
    отключения. Предметная логика живёт в своём модуле и внедряется снаружи,
    иначе утверждение из описания шлюза перестало бы быть правдой при первой же
    теме.
    """

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:
        """``None`` — «не мой случай», запрос идёт обычным путём."""
        ...


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
        direct: DirectResponder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._direct = direct
        self._sanitizer = sanitizer if sanitizer is not None else PiiSanitizer()
        self._guardrails = (
            guardrails if guardrails is not None else SyncGuardrails(sanitizer=self._sanitizer)
        )
        self._quota = quota
        self._completion = completion
        self._router_instance: Any = None

    def warmup(self) -> None:
        """Прогреть тяжёлые зависимости до первого запроса абонента.

        Обе грузятся лениво, и без прогрева эти секунды платит первый абонент.
        Замер на стенде дал 8,0 с до первого куска ответа при цели в 500 мс,
        хотя провайдер отвечал за 0,28 с. Разбор по слагаемым:

        ===================================== ======
          импорт LiteLLM                        7,0 с
          загрузка языковой модели               4,0 с
          первый вызов после прогрева            1,3 с
          последующие вызовы                     0,28 с
        ===================================== ======

        Ленивая загрузка удобна в проверках — они не тянут ни LiteLLM, ни
        языковую модель, — но в сервисе её нужно выполнить заранее.
        """
        self._sanitizer.sanitize("прогрев")
        if self._completion is None:
            # Сборка маршрутизатора, а не вызов модели: тратить токены на прогрев
            # незачем, а семь секунд уходило именно на разбор библиотеки, который
            # здесь и происходит.
            self._router()

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
        sanitize_started = time.perf_counter()
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

        metrics.PII_SANITIZE_SECONDS.observe(time.perf_counter() - sanitize_started)
        metrics.record_pii(report.entities)

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
            metrics.record_quota_rejection(
                decision.scope.value if decision.scope else "unknown"
            )
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
        if self._completion is not None:
            return await self._completion(
                model=self._settings.llm_provider,
                messages=messages,
                max_tokens=request.parameters.max_tokens,
                temperature=request.parameters.temperature,
                stream=stream,
            )

        return await self._router().acompletion(
            model=self._settings.llm_provider,
            messages=messages,
            max_tokens=request.parameters.max_tokens,
            temperature=request.parameters.temperature,
            stream=stream,
        )

    def _router(self) -> Any:
        """Маршрутизатор по конфигурации. Собирается один раз за жизнь шлюза.

        Псевдоним провайдера (`local-test`, `yandexgpt`, `gigachat`) расшифровывает
        сама конфигурация — теперь она читается, а не лежит описанием. Отсюда же
        берутся цепочки запасных провайдеров: требование ADR-002 о переключении
        при сбое исполняется, а не только описано.
        """
        if self._router_instance is None:
            from app.gateway.model_router import build_router

            self._router_instance = build_router(self._settings)
        return self._router_instance

    # -- ответ целиком --------------------------------------------------------- #

    def _direct_answer(self, request: GenerateRequest) -> DirectAnswer | None:
        """Спросить прямого ответчика — до обезличивания и до модели.

        Порядок: лимиты остаются первыми (раздел 41.3), иначе появился бы путь
        в обход них, которым можно давить сервис бесплатно. Обезличивание
        пропускается осознанно: наружу ничего не уходит, защищать нечего.
        Охранители к готовому ответу тоже не применяются — они проверяют то,
        что сочинила модель, а здесь сочинять было некому.
        """
        if self._direct is None:
            return None
        return self._direct.answer(request, now=datetime.now())

    async def generate(self, request: GenerateRequest) -> GenerateResponse | ProblemDetail:
        """Получить ответ целиком. Для служебных вызовов, не для пути абонента."""
        trace_id = self.new_trace_id()

        if (problem := self._check_quota(request, trace_id)) is not None:
            return problem

        if (direct := self._direct_answer(request)) is not None:
            return GenerateResponse(
                answer=direct.text,
                model="direct",
                confidence_score=1.0,
                disclaimer=direct.disclaimer,
                trace_id=trace_id,
                routing={"path": "direct"},
            )

        messages, report = self._prepare(request)

        try:
            raw = await self._call(request, messages, stream=False)
        except ProviderTimeoutError as exc:
            logger.warning("провайдер не ответил вовремя: %s | %s", exc, trace_id)
            return self.problem(
                "upstream-timeout", "LLM provider timeout", 504, trace_id, UPSTREAM_DETAIL
            )
        except Exception as exc:  # noqa: BLE001 — любая ошибка провайдера это 502
            logger.warning("провайдер ответил ошибкой: %s | %s", exc, trace_id)
            return self.problem(
                "upstream-unavailable", "LLM provider unavailable", 502, trace_id,
                UPSTREAM_DETAIL,
            )

        answer = _extract_text(raw)
        check_started = time.perf_counter()
        verdict = self._guardrails.check(answer)
        metrics.GUARDRAILS_CHECK_SECONDS.observe(time.perf_counter() - check_started)
        metrics.record_guardrail(
            blocked=not verdict.allowed,
            reason=verdict.reason.value if verdict.reason else None,
        )
        if not verdict.allowed:
            answer = verdict.text_for_subscriber or ""

        metrics.record_provider_used(
            alias=self._settings.llm_provider, model=str(getattr(raw, "model", ""))
        )
        usage = _extract_usage(raw)
        metrics.record_tokens(
            prompt=usage.prompt_tokens, completion=usage.completion_tokens
        )

        return GenerateResponse(
            answer=answer,
            model=self._settings.llm_provider,
            finish_reason=verdict.finish_reason or FinishReason.STOP,
            usage=usage,
            sources=self._sources(request),
            pii_report=report,
            routing={
                "provider": self._settings.llm_provider,
                "model": str(getattr(raw, "model", "")),
            },
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
        started = time.perf_counter()
        channel = request.metadata.channel.value
        inquiry_type = (
            request.metadata.inquiry_type.value if request.metadata.inquiry_type else None
        )

        if (problem := self._check_quota(request, trace_id)) is not None:
            metrics.record_response(
                channel=channel, inquiry_type=inquiry_type, outcome="error"
            )
            yield ErrorEvent(problem=problem)
            return

        if (direct := self._direct_answer(request)) is not None:
            metrics.ASSISTANT_TTFT_SECONDS.observe(time.perf_counter() - started)
            metrics.record_response(
                channel=channel, inquiry_type=inquiry_type, outcome="success"
            )
            yield TokenEvent(delta=direct.text)
            yield MetadataEvent(confidence_score=1.0, disclaimer=direct.disclaimer)
            yield DoneEvent(
                finish_reason=FinishReason.STOP,
                usage=Usage(),
                routing={"path": "direct"},
                trace_id=trace_id,
            )
            return

        messages, report = self._prepare(request)

        try:
            raw_stream = await self._call(request, messages, stream=True)
        except ProviderTimeoutError as exc:
            logger.warning("провайдер не ответил вовремя: %s | %s", exc, trace_id)
            metrics.record_response(
                channel=channel, inquiry_type=inquiry_type, outcome="error"
            )
            yield ErrorEvent(
                problem=self.problem(
                    "upstream-timeout", "LLM provider timeout", 504, trace_id,
                    UPSTREAM_DETAIL,
                )
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("провайдер ответил ошибкой: %s | %s", exc, trace_id)
            metrics.record_response(
                channel=channel, inquiry_type=inquiry_type, outcome="error"
            )
            yield ErrorEvent(
                problem=self.problem(
                    "upstream-unavailable", "LLM provider unavailable", 502, trace_id,
                    UPSTREAM_DETAIL,
                )
            )
            return

        guard = StreamGuard(guardrails=self._guardrails)
        blocked = False
        first_delta_at: float | None = None
        answering_model = ""

        async for piece in raw_stream:
            if not answering_model:
                answering_model = str(getattr(piece, "model", ""))
            delta = _extract_delta(piece)
            if not delta:
                continue
            if first_delta_at is None:
                first_delta_at = time.perf_counter() - started
            verdict = guard.feed(delta)
            if not verdict.allowed:
                # Охранитель сработал: наружу уходит заглушка вместо остатка
                # ответа. Уже отданные куски отозвать нельзя, поэтому проверка
                # стоит до выдачи, а не после.
                blocked = True
                yield TokenEvent(delta=verdict.text_for_subscriber or "")
                break
            yield TokenEvent(delta=delta)

        if answering_model:
            metrics.record_provider_used(
                alias=self._settings.llm_provider, model=answering_model
            )
        metrics.record_guardrail(
            blocked=blocked,
            reason=guard.verdict.reason.value if guard.verdict.reason else None,
        )
        metrics.record_response(
            channel=channel,
            inquiry_type=inquiry_type,
            outcome="blocked" if blocked else "ok",
            ttft=first_delta_at,
            total=time.perf_counter() - started,
        )

        yield MetadataEvent(sources=self._sources(request))
        yield DoneEvent(
            finish_reason=FinishReason.GUARDRAIL if blocked else FinishReason.STOP,
            usage=Usage(),
            pii_report=report,
            routing={
                "provider": self._settings.llm_provider,
                "model": answering_model,
            },
            trace_id=trace_id,
        )


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
