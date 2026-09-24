"""Backend: выбор пути ответа, поиск контекста, сборка запроса к шлюзу.

Обязанности разделены так: **Backend** отвечает за оркестрацию,
сессии, поиск и сборку промпта, **LLM Gateway** — за абстракцию над моделями,
лимиты и слой защиты. Связи на диаграмме контейнеров подписаны в ту же сторону: Backend
ищет в Vector DB, Backend передаёт шлюзу промпт **с уже собранным контекстом**.

ЗАЧЕМ ЭТОТ МОДУЛЬ ПОЯВИЛСЯ ИМЕННО СЕЙЧАС. Оркестратора не было, и его работу
подхватывал шлюз: ответчик отключений жил внутри него, а демо-стенд обращался к
шлюзу напрямую. Работало — но каждый следующий предметный путь (поиск,
регламентные ответы, классификация) увеличивал бы это отступление. Дешевле
остановиться на третьем, чем на шестом.

Порядок работ нарушен осознанно. Причина — поиск
готов, а класть его в шлюз значит нарушить границу Backend и шлюза сразу после того, как
разобрались, почему так делать нельзя.

ТРИ ПУТИ ОТВЕТА, И ТОЛЬКО ОДИН ИЗ НИХ ИДЁТ К МОДЕЛИ

===================== ====================================================
 `topic = outage`      Точный поиск по графику отключений (ADR-100)
 `topic = water_quality` Утверждённая формулировка дословно (ADR-100)
 всё остальное         Поиск по базе знаний, затем генерация
===================== ====================================================

Первые два минуют модель по противоположным причинам: график меняется постоянно
и требует точности, регламентная формулировка не меняется вовсе. Общее у них то,
что обоим нужен поиск по ключу, а не генерация по найденному.

ГДЕ ПРОВЕРЯЮТСЯ ЛИМИТЫ. Ровно один раз на запрос, но в разных местах — и это не
недосмотр. Запрос, доходящий до модели, проверяет шлюз (порядок quota-first). Запрос, отвечаемый
напрямую, до шлюза не доходит, поэтому лимит
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
from app.agents.intents import Condition, Part, PartKind, split_intents
from app.agents.triage import Triage
from app.backend.registration import REGISTRABLE, Registrar
from app.backend.sessions import SessionStore, SubscriberMismatch
from app.billing.source import BillingSource
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
    PiiReport,
    ProblemDetail,
    SessionState,
    SuggestedAction,
    TokenEvent,
    Usage,
)
from app.taxonomy import InquiryType

__all__ = ["DirectAnswer", "DirectResponder", "Orchestrator"]

logger = logging.getLogger(__name__)

NO_ACTIONS_RULE = (
    " Ты только консультируешь: не передаёшь показания, не проверяешь долг и лицевой "
    "счёт, не регистрируешь и не оплачиваешь. Никогда не пиши, что ты это сделал или "
    "делаешь. Если просят выполнить операцию, скажи, что не можешь, и объясни, как "
    "сделать это самому."
)
"""Добавляется к системной части запроса, который идёт к модели.

Модель без этого правила отвечала «Передал показания 1234. Проверил наличие
долга — его нет», хотя записи и проверки в этом пути нет. Второй рубеж — правило
`unperformed_action` в охранителях шлюза (`app/gateway/guardrails.py`): промпту
модель может не подчиниться, а выученное правило она обойти не может."""


@dataclass(frozen=True, slots=True)
class DirectAnswer:
    """Готовый ответ, для которого модель не нужна."""

    text: str
    disclaimer: str | None = None
    document_url: str | None = None
    """Ссылка на готовый бланк заявления для печати (тема `template`).

    Ответчик образцов кладёт заполненный бланк в хранилище и возвращает ссылку;
    оркестратор пробрасывает её в `MetadataEvent`/`GenerateResponse`, а виджет
    показывает панелью. `None` — обычный прямой ответ без документа."""


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
    :param billing: источник профиля абонента. Задан — адрес в метаданных
        запроса берётся отсюда по идентификатору абонента, а не из того, что
        прислал клиент (см. :meth:`_with_address`).
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
        sessions: SessionStore | None = None,
        registrar: Registrar | None = None,
        billing: BillingSource | None = None,
    ) -> None:
        self._gateway = gateway
        self._kb = knowledge_base
        self._direct = direct
        self._quota = quota
        self._settings = settings if settings is not None else get_settings()
        self._sessions = sessions
        self._registrar = registrar
        self._billing = billing
        # Кнопки, относящиеся к текущему ответу. Живут на экземпляре, а не
        # в возвращаемом значении, чтобы не менять подпись `_route`, общую
        # для потока и ответа целиком.
        self._pending_actions: tuple[SuggestedAction, ...] = ()
        self._triage = triage if triage is not None else Triage()
        self._advisor = (
            advisor
            if advisor is not None
            else Advisor(contact_phone=self._settings.contact_center_phone)
        )

    def warmup(self) -> None:
        """Прогреть тяжёлые зависимости до первого запроса абонента.

        **Этот метод — не удобство, а восстановление сломанной связи.** Прогрев
        появился позже: ленивые загрузки LiteLLM и языковой модели
        обезличивателя оплачивал первый абонент, и стенд показал 8,0 с до первого
        куска при цели 500 мс. Интерфейс вызывает прогрев у того, кого ему
        внедрили, — `getattr(resolved, "warmup", None)`.

        Пока внедряли шлюз, всё работало. С появлением Backend
        внедрять стали **оркестратор**, у которого метода не было: `getattr`
        молча вернул `None`, и прогрев перестал выполняться вовсе. Замер
        `scripts/measure_first_request.py` показал это прямо — «старт сервиса»
        занимал 0,01 с и с прогревом, и без него.

        Проверка на это была и не помогла: она подставляла собственную заглушку
        с методом `warmup` и доказывала, что интерфейс вызывает его **у того, у
        кого он есть**. Что настоящий объект метод потерял, такая проверка
        увидеть не может — о чём предупреждали давно: «метод мог
        остаться написанным и неподключённым».
        """
        self._gateway.warmup()

    # --- выбор пути ------------------------------------------------------- #

    async def _triaged(self, request: GenerateRequest) -> GenerateRequest:
        """Проставить тему и тип обращения, если их не проставили снаружи.

        Классификация идёт **до** выбора пути: тема решает, звать ли модель
        вообще, и узнавать её после вызова было бы поздно.

        Результат кладётся в метаданные запроса, а не остаётся у оркестратора:
        по ним ответчики решают, их ли это случай, и по ним же разбиваются
        метрики. Второе место хранения того же значения развело бы их при первой
        правке.

        Асинхронный метод — `Triage.classify_async` может звать запасные
        классификаторы (эмбеддинги) для темы и для типа обращения, когда
        правила/ключевые слова ничего не нашли (`app/agents/topic_fallback.py`,
        `app/agents/inquiry_type_fallback.py`).
        """
        result = await self._triage.classify_async(
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

    def _with_address(self, request: GenerateRequest) -> GenerateRequest:
        """Проставить адрес абонента из Биллинга по его идентификатору.

        **Адрес принадлежит Биллингу (файл `ЛСФЛ`, источник 1), а не запросу.**
        Личный кабинет его показывает, но ведёт — Биллинг; на MVP Backend и так
        читает оттуда данные абонента. Поэтому значение, присланное
        клиентом в метаданных, **не используется**: клиент мог бы прислать чужой
        адрес и получить график отключений по нему (та же логика, что у проверки
        `subscriber_id` в `SessionStore`, риск Spoofing).

        `billing` не задан — поведение прежнее (адрес из запроса): проверки
        ответчика отключений подают его сами. Абонента нет в Биллинге — адрес
        сбрасывается: отвечать про место, которого мы не подтвердили, нельзя.
        """
        if self._billing is None:
            return request
        account = self._billing.account(request.metadata.subscriber_id)
        resolved = account.address if account is not None else None
        if resolved == request.metadata.address:
            return request
        metadata = request.metadata.model_copy(update={"address": resolved})
        return request.model_copy(update={"metadata": metadata})

    # --- регистрация обращения (Этап 2) -------------------------------------- #

    def _session(self, request: GenerateRequest) -> SessionState | None:
        """Состояние диалога, если оно вообще ведётся.

        `None` — состояние не хранится: Redis недоступен либо хранилище не
        подключено. Это не ошибка ответа: без состояния работает всё, кроме
        регистрации, а она без состояния и не должна работать.
        """
        if self._sessions is None:
            return None
        try:
            return self._sessions.load(
                request.metadata.session_id, request.metadata.subscriber_id
            )
        except SubscriberMismatch as exc:
            # Чужую сессию не отдаём и новую под тем же ключом не заводим:
            # иначе подмена `session_id` молча создавала бы рабочий диалог.
            logger.warning("сессия запрошена не тем абонентом: %s", exc)
            return None

    def _registration_answer(
        self, request: GenerateRequest
    ) -> tuple[DirectAnswer, tuple[SuggestedAction, ...]] | None:
        """Ответ на нажатие кнопки под черновиком.

        Стоит **до** классификации и поиска: намерение задано признаком, искать
        по слову «Подтверждаю» нечего, а звать модель — тем более.
        """
        if request.metadata.intent is None or self._registrar is None:
            return None

        session = self._session(request)
        if session is None:
            return (
                DirectAnswer(
                    text=(
                        "Черновик обращения не найден — возможно, прошло слишком "
                        "много времени. Опишите, пожалуйста, вопрос заново."
                    )
                ),
                (),
            )

        outcome = self._registrar.apply(request, session)
        self._sessions.save(session)  # type: ignore[union-attr]
        return DirectAnswer(text=outcome.text), outcome.actions

    def _offer_registration(
        self, request: GenerateRequest
    ) -> tuple[str, tuple[SuggestedAction, ...]]:
        """Приписка с черновиком к обычному ответу, если тип это допускает."""
        if self._registrar is None or self._sessions is None:
            return "", ()
        session = self._session(request)
        if session is None:
            return "", ()

        outcome = self._registrar.offer(request, session)
        if outcome is None:
            return "", ()
        self._sessions.save(session)
        return outcome.text, outcome.actions

    def _direct_answer(self, request: GenerateRequest) -> DirectAnswer | None:
        now = datetime.now()
        for responder in self._direct:
            if (answer := responder.answer(request, now=now)) is not None:
                return answer
        return None

    def _quota_problem(self, request: GenerateRequest, trace_id: str) -> ProblemDetail | None:
        """Лимит для пути, который до шлюза не дойдёт.

        Повторяет решения шлюза, а не заводит свои: `429` при превышении,
        `503` при недоступности хранилища (ADR-400) — иначе абонент получал бы
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
        считается не найденным, и модель отвечает без опоры. Что
        она при этом обязана сказать «не знаю» — забота промпта, не поиска.
        """
        if self._kb is None or request.has_context:
            return request

        # `vector_db_pending_queries` была объявлена под виджет дашборда и не
        # писалась ничем — та же болезнь. Здесь она
        # наконец получает источник: глубина очереди поиска и есть число
        # запросов, находящихся в поиске прямо сейчас.
        metrics.VECTOR_DB_PENDING.inc()
        try:
            # Тип обращения ищет вместе с поиском, не после (ADR-200, правило
            # 4.8). `other` и отсутствие типа — «не знаем»: фильтровать по
            # догадке значило бы потерять нужный фрагмент, поэтому без фильтра.
            inquiry_type = request.metadata.inquiry_type
            chunks = self._kb.search(
                request.query,
                top_n=self._settings.rerank_top_n,
                threshold=self._settings.score_threshold,
                top_k=self._settings.vector_top_k,
                inquiry_type=(
                    inquiry_type.value
                    if inquiry_type is not None and inquiry_type is not InquiryType.OTHER
                    else None
                ),
            )
        finally:
            metrics.VECTOR_DB_PENDING.dec()
        if not chunks:
            logger.info("контекст не найден выше порога: %s", self._settings.score_threshold)
        return request.model_copy(update={"context": chunks})

    def _advice(self, request: GenerateRequest) -> DirectAnswer | None:
        """Есть ли на чём отвечать. ``None`` — контекст есть, идём к модели.

        Требование проекта: при пустом контексте генерация не вызывается.
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

    async def _route(
        self, request: GenerateRequest
    ) -> tuple[GenerateRequest, DirectAnswer | None]:
        """Пройти путь до решения: классификация, ответчики, поиск, консультант.

        Возвращает подготовленный запрос и готовый ответ, если модель не нужна.
        Порядок здесь и есть устройство Backend, поэтому он собран в одном месте,
        а не размазан по двум точкам входа: поток и ответ целиком обязаны
        принимать одинаковые решения.
        """
        # Нажатие кнопки под черновиком разбирается до классификации и поиска:
        # намерение задано признаком, искать по слову «Подтверждаю» нечего.
        if (decided := self._registration_answer(request)) is not None:
            self._pending_actions = decided[1]
            return request, decided[0]

        request = await self._triaged(request)
        request = self._with_address(request)
        self._pending_actions = ()

        if (direct := self._direct_answer(request)) is not None:
            return request, direct

        request = self._with_context(request)
        return request, self._advice(request)

    # --- несколько задач в одной реплике ---------------------------------------- #

    def _plan(self, request: GenerateRequest) -> list[Part] | None:
        """Части реплики, если задач несколько (или единственная — операция без выполнения).

        `None` — обычный путь. Не разбираются: нажатие кнопки, запрос с уже
        готовым контекстом (служебные вызовы) и запрос, где тему или тип
        проставили снаружи: пришедшее извне значение считается более осведомлённым.
        """
        meta = request.metadata
        if (
            meta.intent is not None
            or request.has_context
            or meta.topic is not None
            or meta.inquiry_type is not None
        ):
            return None
        parts = split_intents(request.query)
        if len(parts) >= 2 or parts[0].kind is PartKind.UNSUPPORTED:
            return parts
        return None

    def _has_debt(self, request: GenerateRequest) -> bool | None:
        """Есть ли долг по счёту абонента. `None` — счёта нет или Биллинг не подключён."""
        if self._billing is None:
            return None
        account = self._billing.account(request.metadata.subscriber_id)
        return None if account is None else account.has_debt

    def _refusal(self, part: Part) -> str:
        """Текст для операции, которую помощник не выполняет: что не сделано и как сделать."""
        return (
            f"Эту операцию помощник выполнить не может: {part.action}. Сделайте это в "
            f"личном кабинете или по телефону контакт-центра "
            f"{self._settings.contact_center_phone}. Я ничего не передавал, не оплачивал "
            "и не менял."
        )

    def _condition_text(self, part: Part, debt: bool | None) -> str | None:
        """Почему условная часть не выполняется. `None` — условие выполнено."""
        if part.condition is Condition.UNKNOWN or debt is None:
            return (
                f"Условие «если {part.condition_text}» я проверить не могу, поэтому эту "
                "часть не выполняю. Уточните после ответа на предыдущие части и "
                "повторите просьбу."
            )
        if part.condition is Condition.NO_DEBT and debt:
            return (
                f"Условие «если {part.condition_text}» не выполнено: по счёту есть "
                "задолженность. Эту часть не выполняю."
            )
        if part.condition is Condition.HAS_DEBT and not debt:
            return (
                f"Условие «если {part.condition_text}» не выполнено: задолженности по "
                "счёту нет. Эту часть не выполняю."
            )
        return None

    @staticmethod
    def _label(part: Part) -> str:
        text = part.text.strip(" .?!,;")
        if part.kind is PartKind.CONDITIONAL:
            text = f"если {part.condition_text} — {text}"
        return text if len(text) <= 80 else text[:77] + "…"

    async def _stream_parts(
        self, request: GenerateRequest, parts: list[Part], started: float
    ) -> AsyncIterator[GatewayEvent]:
        """Ответить на каждую часть реплики по очереди — один поток, один итог.

        Часть отвечается тем же путём, что и отдельная реплика: правилами и
        ответчиками, затем базой знаний, затем моделью. Что не сделано, говорится
        прямо: операция без выполнения, невыполненное условие, непроверяемое
        условие. Регистрация предлагается по **первой** подходящей части — в
        состоянии диалога один черновик; об остальных абонент предупреждён.
        """
        trace_id = self._gateway.new_trace_id()
        if (problem := self._quota_problem(request, trace_id)) is not None:
            yield ErrorEvent(problem=problem)
            return

        metrics.record_multi_intent(len(parts))
        numbered = len(parts) > 1
        prompt_tokens = completion_tokens = 0
        pii = PiiReport()
        disclaimer: str | None = None
        offer_source: GenerateRequest | None = None
        skipped_offers: list[str] = []
        debt = self._has_debt(request)

        for number, part in enumerate(parts, start=1):
            if numbered:
                head = f"**{number}. {self._label(part)}**\n"
                yield TokenEvent(delta=("\n\n" if number > 1 else "") + head)

            if part.kind is PartKind.UNSUPPORTED:
                yield TokenEvent(delta=self._refusal(part))
                continue
            if part.kind is PartKind.CONDITIONAL and (why := self._condition_text(part, debt)):
                yield TokenEvent(delta=why)
                continue

            sub_meta = request.metadata.model_copy(update={"topic": None, "inquiry_type": None})
            sub = request.model_copy(update={"query": part.text, "metadata": sub_meta})
            sub = await self._triaged(sub)
            sub = self._with_address(sub)
            if (direct := self._direct_answer(sub)) is not None:
                yield TokenEvent(delta=direct.text)
                disclaimer = disclaimer or direct.disclaimer
                continue
            sub = self._with_context(sub)
            if (advice := self._advice(sub)) is not None:
                yield TokenEvent(delta=advice.text)
                continue

            sub = sub.model_copy(update={"system": sub.system + NO_ACTIONS_RULE})
            async for event in self._gateway.stream(sub):
                if isinstance(event, TokenEvent):
                    yield event
                elif isinstance(event, ErrorEvent):
                    yield event
                    return
                elif isinstance(event, MetadataEvent):
                    disclaimer = disclaimer or event.disclaimer
                elif isinstance(event, DoneEvent):
                    prompt_tokens += event.usage.prompt_tokens
                    completion_tokens += event.usage.completion_tokens
                    if event.pii_report.pii_detected:
                        pii = PiiReport(
                            pii_detected=True,
                            entities=sorted({*pii.entities, *event.pii_report.entities}),
                        )
            if sub.metadata.inquiry_type in REGISTRABLE:
                if offer_source is None:
                    offer_source = sub
                else:
                    skipped_offers.append(self._label(part))

        actions: tuple[SuggestedAction, ...] = ()
        if offer_source is not None:
            offer, actions = self._offer_registration(offer_source)
            if offer:
                yield TokenEvent(delta=offer)
                if skipped_offers:
                    yield TokenEvent(
                        delta=(
                            "\n\nОбращение по остальным задачам ("
                            + "; ".join(skipped_offers)
                            + ") оформлю отдельно: напишите их следующим сообщением после "
                            "подтверждения этого."
                        ).replace("\\n", "\n")
                    )

        metrics.ASSISTANT_TTFT_SECONDS.observe(time.perf_counter() - started)
        metrics.record_response(
            channel=request.metadata.channel.value, inquiry_type=None, outcome="success"
        )
        yield MetadataEvent(
            confidence_score=1.0,
            disclaimer=disclaimer,
            suggested_actions=list(actions),
        )
        yield DoneEvent(
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            pii_report=pii,
            routing={"path": "multi", "parts": len(parts)},
            trace_id=trace_id,
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse | ProblemDetail:
        """Ответ целиком. Для служебных вызовов, не для пути абонента.

        Несколько задач в реплике здесь **не разбираются**: разбор идёт только на
        пути абонента (`stream`); служебные вызовы получают прежнее поведение.
        """
        request, answer = await self._route(request)
        if answer is None:
            response = await self._gateway.generate(request)
            if isinstance(response, GenerateResponse):
                offer, actions = self._offer_registration(request)
                if offer:
                    response = response.model_copy(
                        update={
                            "answer": response.answer + offer,
                            "suggested_actions": list(actions),
                        }
                    )
            return response

        trace_id = self._gateway.new_trace_id()
        if (problem := self._quota_problem(request, trace_id)) is not None:
            return problem
        return GenerateResponse(
            answer=answer.text,
            model="direct",
            confidence_score=1.0,
            disclaimer=answer.disclaimer,
            document_url=answer.document_url,
            suggested_actions=list(self._pending_actions),
            trace_id=trace_id,
            routing={"path": "direct"},
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[GatewayEvent]:
        """Ответ потоком — основной путь абонента."""
        started = time.perf_counter()
        if (parts := self._plan(request)) is not None:
            async for event in self._stream_parts(request, parts, started):
                yield event
            return
        request, answer = await self._route(request)

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
            yield MetadataEvent(
                confidence_score=1.0,
                disclaimer=answer.disclaimer,
                document_url=answer.document_url,
                suggested_actions=list(self._pending_actions),
            )
            yield DoneEvent(
                finish_reason=FinishReason.STOP,
                usage=Usage(),
                routing={"path": "direct"},
                trace_id=trace_id,
            )
            return

        offer, actions = self._offer_registration(request)
        request = request.model_copy(update={"system": request.system + NO_ACTIONS_RULE})
        async for event in self._gateway.stream(request):
            if offer and isinstance(event, MetadataEvent):
                # Приписка идёт последним куском текста, а не отдельным полем:
                # абонент читает ответ подряд, и черновик — его продолжение.
                yield TokenEvent(delta=offer)
                yield event.model_copy(update={"suggested_actions": list(actions)})
                continue
            yield event
