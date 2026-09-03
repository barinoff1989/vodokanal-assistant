"""Backend: выбор пути ответа, поиск контекста, сборка запроса к шлюзу.

По разделу 5.2 обязанности разделены так: **Backend** отвечает за оркестрацию,
сессии, поиск и сборку промпта, **LLM Gateway** — за абстракцию над моделями,
лимиты и слой защиты. Связи в разделе 5.3 подписаны в ту же сторону: Backend
ищет в Vector DB, Backend передаёт шлюзу промпт **с уже собранным контекстом**.

ЗАЧЕМ ЭТОТ МОДУЛЬ ПОЯВИЛСЯ ИМЕННО СЕЙЧАС. Оркестратора не было, и его работу
подхватывал шлюз: ответчик отключений жил внутри него, а демо-стенд обращался к
шлюзу напрямую. Работало — но каждый следующий предметный путь (поиск,
регламентные ответы, классификация) увеличивал бы это отступление. Дешевле
остановиться на третьем, чем на шестом.

Порядок плана нарушен осознанно: шаг 7 сделан раньше шага 6. Причина — поиск
готов, а класть его в шлюз значит нарушить раздел 5.3 сразу после того, как
разобрались, почему так делать нельзя.

ТРИ ПУТИ ОТВЕТА, И ТОЛЬКО ОДИН ИЗ НИХ ИДЁТ К МОДЕЛИ

===================== ====================================================
 `topic = outage`      Точный поиск по графику отключений (ADR-013)
 `topic = water_quality` Утверждённая формулировка дословно (ADR-012)
 всё остальное         Поиск по базе знаний, затем генерация
===================== ====================================================

Первые два минуют модель по противоположным причинам: график меняется постоянно
и требует точности, регламентная формулировка не меняется вовсе. Общее у них то,
что обоим нужен поиск по ключу, а не генерация по найденному.

ГДЕ ПРОВЕРЯЮТСЯ ЛИМИТЫ. Ровно один раз на запрос, но в разных местах — и это не
недосмотр. Запрос, доходящий до модели, проверяет шлюз (порядок quota-first,
раздел 41.3). Запрос, отвечаемый напрямую, до шлюза не доходит, поэтому лимит
проверяет Backend: ответ без модели ничего не стоит нам, но обработка запроса
стоит, и путь в обход лимитов стал бы способом бесплатно давить сервис.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.agents.advisor import Advisor
from app.agents.triage import Triage
from app.config import Settings, get_settings
from app.gateway.llm_gateway import GatewayEvent, LlmGateway
from app.gateway.quota import QuotaManager, QuotaStoreUnavailableError
from app.kb.search import KnowledgeBase
from app.metrics import prometheus as metrics
from app.models import (
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    MetadataEvent,
    ProblemDetail,
    TokenEvent,
    Usage,
)

__all__ = ["DirectAnswer", "DirectResponder", "Orchestrator"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DirectAnswer:
    """Готовый ответ, для которого модель не нужна."""

    text: str
    disclaimer: str | None = None


class DirectResponder(Protocol):
    """Тот, кто умеет ответить без модели.

    Backend знает только это. Про водоканал, адреса и отключения знает сам
    ответчик, и внедряется он снаружи — иначе предметная логика расползлась бы
    по оркестрации так же, как до этого расползалась по шлюзу.
    """

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:
        """``None`` — «не мой случай», запрос идёт дальше обычным путём."""
        ...


class Orchestrator:
    """Выбирает путь ответа и снабжает шлюз контекстом.

    :param gateway: шлюз к модели. Единственный, кто разговаривает с провайдером.
    :param knowledge_base: поиск по базе знаний. ``None`` — контекст не ищется,
        и модель отвечает по общим знаниям; на прототипе это допустимо, на MVP
        означало бы ответ без опоры на регламенты.
    :param direct: ответчики, отвечающие без модели. Опрашиваются по порядку.
    :param quota: учёт лимитов для путей, которые до шлюза не доходят.
    """

    def __init__(
        self,
        gateway: LlmGateway,
        *,
        knowledge_base: KnowledgeBase | None = None,
        direct: tuple[DirectResponder, ...] = (),
        quota: QuotaManager | None = None,
        triage: Triage | None = None,
        advisor: Advisor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._gateway = gateway
        self._kb = knowledge_base
        self._direct = direct
        self._quota = quota
        self._settings = settings if settings is not None else get_settings()
        self._triage = triage if triage is not None else Triage()
        self._advisor = (
            advisor
            if advisor is not None
            else Advisor(contact_phone=self._settings.contact_center_phone)
        )

    # --- выбор пути ------------------------------------------------------- #

    def _triaged(self, request: GenerateRequest) -> GenerateRequest:
        """Проставить тему и тип обращения, если их не проставили снаружи.

        Классификация идёт **до** выбора пути: тема решает, звать ли модель
        вообще, и узнавать её после вызова было бы поздно.

        Результат кладётся в метаданные запроса, а не остаётся у оркестратора:
        по ним ответчики решают, их ли это случай, и по ним же разбиваются
        метрики. Второе место хранения того же значения развело бы их при первой
        правке.
        """
        result = self._triage.classify(
            request.query,
            topic=request.metadata.topic,
            inquiry_type=request.metadata.inquiry_type,
        )
        if (
            request.metadata.topic is result.topic
            and request.metadata.inquiry_type is result.inquiry_type
        ):
            return request
        metadata = request.metadata.model_copy(
            update={"topic": result.topic, "inquiry_type": result.inquiry_type}
        )
        return request.model_copy(update={"metadata": metadata})

    def _direct_answer(self, request: GenerateRequest) -> DirectAnswer | None:
        now = datetime.now()
        for responder in self._direct:
            if (answer := responder.answer(request, now=now)) is not None:
                return answer
        return None

    def _quota_problem(self, request: GenerateRequest, trace_id: str) -> ProblemDetail | None:
        """Лимит для пути, который до шлюза не дойдёт.

        Повторяет решения шлюза, а не заводит свои: `429` при превышении,
        `503` при недоступности хранилища (ADR-008) — иначе абонент получал бы
        разные ответы на одну и ту же причину в зависимости от пути.
        """
        if self._quota is None:
            return None
        try:
            decision = self._quota.check(
                request.metadata.subscriber_id, request.metadata.session_id
            )
        except QuotaStoreUnavailableError:
            return self._gateway.problem(
                "quota-store-unavailable",
                "Service temporarily unavailable",
                503,
                trace_id,
                "Сервис временно недоступен. Попробуйте повторить запрос позже.",
                retry_after=self._quota.store_retry_after,
            )
        if not decision.allowed:
            metrics.record_quota_rejection(decision.scope.value if decision.scope else "unknown")
            return self._gateway.problem(
                "quota-exceeded",
                "Too many requests",
                429,
                trace_id,
                decision.detail,
                retry_after=decision.retry_after,
            )
        return None

    def _with_context(self, request: GenerateRequest) -> GenerateRequest:
        """Найти контекст и вложить его в запрос.

        Уже переданный контекст не трогается: служебные вызовы и проверки
        передают его сами, и перетирать переданное значило бы делать поведение
        зависимым от того, настроен ли поиск.

        Пустая выдача — законный исход, а не сбой: ниже порога контекст
        считается не найденным (раздел 8.2), и модель отвечает без опоры. Что
        она при этом обязана сказать «не знаю» — забота промпта, не поиска.
        """
        if self._kb is None or request.has_context:
            return request

        # `vector_db_pending_queries` была объявлена под виджет дашборда и не
        # писалась ничем — та же болезнь, что разбирал раздел 54.1. Здесь она
        # наконец получает источник: глубина очереди поиска и есть число
        # запросов, находящихся в поиске прямо сейчас.
        metrics.VECTOR_DB_PENDING.inc()
        try:
            chunks = self._kb.search(
                request.query,
                top_n=self._settings.rerank_top_n,
                threshold=self._settings.score_threshold,
                top_k=self._settings.vector_top_k,
            )
        finally:
            metrics.VECTOR_DB_PENDING.dec()
        if not chunks:
            logger.info("контекст не найден выше порога: %s", self._settings.score_threshold)
        return request.model_copy(update={"context": chunks})

    def _advice(self, request: GenerateRequest) -> DirectAnswer | None:
        """Есть ли на чём отвечать. ``None`` — контекст есть, идём к модели.

        Требование шага 6 плана: при пустом контексте генерация не вызывается.
        Модель без контекста не молчит — она отвечает связно, уверенно и на
        общих сведениях, которых в регламентах водоканала может не быть вовсе.
        Отличить такой ответ от настоящего не сможет ни абонент, ни мы.

        **Правило применяется, только если поиск состоялся.** Пустой контекст —
        свидетельство промаха поиска, а не его отсутствия: когда база знаний не
        настроена, отказывать было бы не на что опереться, и ассистент молчал бы
        на любой вопрос.

        Это осознанная дыра, и она видна: `app/main.py` при отсутствии корпуса
        пишет предупреждение в журнал. На MVP так работать нельзя — там пустая
        база знаний должна ронять запуск, а не понижать качество молча.
        """
        if self._kb is None:
            return None
        advice = self._advisor.advise(request.context)
        if advice.answer is None:
            return None
        return DirectAnswer(text=advice.answer)

    # --- пути ответа ------------------------------------------------------- #

    def _route(self, request: GenerateRequest) -> tuple[GenerateRequest, DirectAnswer | None]:
        """Пройти путь до решения: классификация, ответчики, поиск, консультант.

        Возвращает подготовленный запрос и готовый ответ, если модель не нужна.
        Порядок здесь и есть устройство Backend, поэтому он собран в одном месте,
        а не размазан по двум точкам входа: поток и ответ целиком обязаны
        принимать одинаковые решения.
        """
        request = self._triaged(request)

        if (answer := self._direct_answer(request)) is not None:
            return request, answer

        request = self._with_context(request)
        return request, self._advice(request)

    async def generate(self, request: GenerateRequest) -> GenerateResponse | ProblemDetail:
        """Ответ целиком. Для служебных вызовов, не для пути абонента."""
        request, answer = self._route(request)
        if answer is None:
            return await self._gateway.generate(request)

        trace_id = self._gateway.new_trace_id()
        if (problem := self._quota_problem(request, trace_id)) is not None:
            return problem
        return GenerateResponse(
            answer=answer.text,
            model="direct",
            confidence_score=1.0,
            disclaimer=answer.disclaimer,
            trace_id=trace_id,
            routing={"path": "direct"},
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[GatewayEvent]:
        """Ответ потоком — основной путь абонента (правило 4.3)."""
        started = time.perf_counter()
        request, answer = self._route(request)

        if answer is not None:
            trace_id = self._gateway.new_trace_id()
            if (problem := self._quota_problem(request, trace_id)) is not None:
                yield ErrorEvent(problem=problem)
                return

            channel = request.metadata.channel.value
            inquiry_type = (
                request.metadata.inquiry_type.value if request.metadata.inquiry_type else None
            )
            metrics.ASSISTANT_TTFT_SECONDS.observe(time.perf_counter() - started)
            metrics.record_response(channel=channel, inquiry_type=inquiry_type, outcome="success")

            yield TokenEvent(delta=answer.text)
            yield MetadataEvent(confidence_score=1.0, disclaimer=answer.disclaimer)
            yield DoneEvent(
                finish_reason=FinishReason.STOP,
                usage=Usage(),
                routing={"path": "direct"},
                trace_id=trace_id,
            )
            return

        async for event in self._gateway.stream(request):
            yield event
