"""Выдача бланка заявления абоненту — образцом из реестра, без модели.

Тема `template` (ADR-300) уводит вопрос сюда. Абонент просит бланк, чтобы
заполнить и подать его самостоятельно.

ТРИ РОДА ПОЛЕЙ (см. `templates.py`): из Биллинга по лицевому счёту подставляются
без вопросов; недостающие, которые абонент может назвать словами, ассистент
**спрашивает по одному** и вставляет ответы; паспортные данные и реквизиты
остаются прочерком для заполнения на бумаге.

ПОЧЕМУ ЭТО НЕ РЕГИСТРАЦИЯ ОБРАЩЕНИЯ. Регистрация (Этап 2) — ассистент оформляет
заявку сам после подтверждения абонента. Здесь ассистент ничего не оформляет и
никуда не пишет: собирает поля и отдаёт готовый текст. Подтверждение записи, гейты HITL и
идемпотентность к этому пути не применяются. Состояние сбора живёт в той же
сессии (`SessionState.template_fill`), что и черновик регистрации, но это
отдельное поле — данные бумажного бланка, а не обращения.

ПОЧЕМУ БЕЗ МОДЕЛИ. Текст бланка обязан не меняться от запроса к запросу и
опираться на нормативную базу (тот же довод, что у ADR-100). Вопросы к абоненту
тоже фиксированные: спросить «когда вам удобно?» модель не нужна.

ПРЕДМЕТНАЯ ЛОГИКА ЖИВЁТ ЗДЕСЬ, А НЕ В ОРКЕСТРАЦИИ — как у ответчиков отключений,
тарифов, регламентных ответов и фактов лицевого счёта.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

from app.backend.orchestrator import DirectAnswer
from app.backend.sessions import SessionStore, SubscriberMismatch
from app.billing.source import BillingSource, Meter
from app.config import Settings, get_settings
from app.documents.artifact import ArtifactStore, render_html
from app.documents.templates import (
    TEMPLATES,
    is_template_request,
    match_template,
    render_template,
)
from app.models import GenerateRequest, SessionState, TemplateFill
from app.taxonomy import Topic

__all__ = ["TemplateResponder"]

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "Это ориентировочный образец, не официальная форма. "
    "Актуальный бланк уточните в контакт-центре."
)

_TITLES: dict[str, str] = {tid: t.title for tid, t in TEMPLATES.items()}

# Абонент бросает заполнение. Требуется короткая реплика — иначе «не надо
# снимать счётчик» в ответ на вопрос о способе поверки прервало бы сбор.
_CANCEL: frozenset[str] = frozenset(
    {"отмена", "отменить", "стоп", "хватит", "отбой", "не нужно", "не надо",
     "передумал", "передумала", "отмени"}
)
# Абонент не знает поле — оно останется прочерком.
_SKIP: frozenset[str] = frozenset(
    {"пропустить", "пропуск", "не знаю", "незнаю", "позже", "потом", "дальше", "-", "—"}
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("ё", "е"))


def _meter_line(meter: Meter) -> str:
    parts: list[str] = []
    if meter.verify_by is not None:
        parts.append(f"поверка до {meter.verify_by:%d.%m.%Y}")
    if meter.reading:
        reading = f"показание {meter.reading}"
        if meter.reading_date is not None:
            reading += f" от {meter.reading_date:%d.%m.%Y}"
        parts.append(reading)
    return "; ".join(parts) if parts else "данных нет"


@dataclass(frozen=True, slots=True)
class TemplateResponder:
    """Отдаёт бланк заявления, собирая недостающие поля в диалоге.

    ``None`` — «не мой случай», запрос идёт обычным путём.

    :param billing: источник профиля для подстановки полей. ``None`` — бланк
        выдаётся с пустыми полями.
    :param sessions: где живёт состояние сбора между репликами. ``None`` —
        сбор не ведётся, бланк отдаётся сразу с прочерками.
    :param artifacts: хранилище готовых бланков для печати. ``None`` — бланк
        отдаётся только текстом в ленте, ссылки на файл нет.
    """

    billing: BillingSource | None = None
    settings: Settings | None = None
    sessions: SessionStore | None = None
    artifacts: ArtifactStore | None = None

    # -- вход ------------------------------------------------------------- #

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:
        session = self._load(request)
        lowered = _norm(request.query)
        fill = session.template_fill if session is not None else None

        if fill is not None and lowered in _CANCEL:
            session.template_fill = None  # type: ignore[union-attr]
            self._save(session)
            return DirectAnswer(
                text="Хорошо, бланк не заполняем. Обращайтесь, если понадобится."
            )

        if request.metadata.topic is Topic.TEMPLATE:
            template_id = match_template(lowered)
            if template_id is None:
                return DirectAnswer(text=self._menu())
            return self._start(request, session, template_id, lowered, now)

        # Свободная реплика во время сбора — это ответ на заданный вопрос. Но
        # если абонент явно переключился на другую тему (задолженность, тариф,
        # отключение), пусть ответит профильный ответчик; сбор подождёт.
        if session is not None and fill is not None and fill.pending is not None:
            if request.metadata.topic is Topic.GENERAL:
                return self._receive(request, session, fill, now)
            logger.info("сбор бланка отложен: абонент спросил про %s", request.metadata.topic)

        return None

    # -- сессия ---------------------------------------------------------- #

    def _load(self, request: GenerateRequest) -> SessionState | None:
        if self.sessions is None:
            return None
        try:
            return self.sessions.load(
                request.metadata.session_id, request.metadata.subscriber_id
            )
        except SubscriberMismatch as exc:
            logger.warning("бланк: сессия запрошена не тем абонентом: %s", exc)
            return None

    def _save(self, session: SessionState | None) -> None:
        if self.sessions is not None and session is not None:
            self.sessions.save(session)

    # -- сбор ---------------------------------------------------------- #

    def _start(
        self,
        request: GenerateRequest,
        session: SessionState | None,
        template_id: str,
        lowered: str,
        now: datetime,
    ) -> DirectAnswer:
        template = TEMPLATES[template_id]
        prefix = "" if is_template_request(lowered) else "Похоже, вам нужен бланк заявления. "

        # Нет куда хранить состояние или нечего спрашивать — отдаём сразу.
        if session is None or not template.collect:
            return self._deliver(
                prefix
                + "Ниже — образец: часть полей заполнена по вашему лицевому счёту, "
                "остальные заполните сами.",
                template_id,
                request.metadata.subscriber_id,
                {},
                now,
            )

        session.template_fill = TemplateFill(
            template_id=template_id, pending=template.collect[0].name, answers={}
        )
        self._save(session)
        return DirectAnswer(
            text=(
                prefix
                + f"Помогу с заявлением «{template.title}». "
                f"Задам {len(template.collect)} "
                + _plural(len(template.collect), "вопрос", "вопроса", "вопросов")
                + " — на любой можно ответить «пропустить», а всё бросить — «отмена».\n\n"
                + template.collect[0].question
            )
        )

    def _receive(
        self,
        request: GenerateRequest,
        session: SessionState,
        fill: TemplateFill,
        now: datetime,
    ) -> DirectAnswer:
        template = TEMPLATES[fill.template_id]
        reply = request.query.strip()
        assert fill.pending is not None
        fill.answers[fill.pending] = "" if _norm(reply) in _SKIP else reply

        nxt = next(
            (f.name for f in template.collect if f.name not in fill.answers), None
        )
        if nxt is not None:
            fill.pending = nxt
            self._save(session)
            question = next(f.question for f in template.collect if f.name == nxt)
            return DirectAnswer(text=question)

        fill.pending = None
        session.template_fill = None
        self._save(session)
        return self._deliver(
            "Готово. Заполненный бланк ниже — проверьте и подайте привычным способом.",
            fill.template_id,
            request.metadata.subscriber_id,
            dict(fill.answers),
            now,
        )

    # -- отрисовка ------------------------------------------------------ #

    def _deliver(
        self,
        lead: str,
        template_id: str,
        subscriber_id: str,
        collected: dict[str, str],
        now: datetime,
    ) -> DirectAnswer:
        """Готовый бланк: текст в ленте и, если хранилище задано, ссылка на лист
        для печати.

        Ссылка — дополнение, а не замена: текст абонент получает всегда, и путь
        не ломается, если хранилище недоступно (`ArtifactStore.put` вернёт
        ``None``). Тот же принцип best-effort, что у телеметрии.
        """
        body = self._render(template_id, subscriber_id, collected, now)
        url = None
        if self.artifacts is not None:
            title = _TITLES.get(template_id, "Заявление")
            url = self.artifacts.put(render_html(title, body, disclaimer=_DISCLAIMER))
        if url is not None:
            lead += " Печатную версию откройте по ссылке под ответом."
        return DirectAnswer(text=lead + "\n\n" + body, document_url=url)

    def _render(
        self,
        template_id: str,
        subscriber_id: str,
        collected: dict[str, str],
        now: datetime,
    ) -> str:
        prefill = self._prefill(subscriber_id, now)
        prefill.update(collected)
        rendered = render_template(
            template_id, prefill=prefill, company_phone=self._phone()
        )
        return rendered + self._reference_block(subscriber_id)

    def _phone(self) -> str:
        settings = self.settings if self.settings is not None else get_settings()
        return settings.contact_center_phone

    def _menu(self) -> str:
        lines = ["Подскажите, какой бланк нужен. Есть образцы заявлений:"]
        lines += [f"  - {title}" for title in _TITLES.values()]
        lines.append('\nНапример: «дай бланк заявления на поверку».')
        return "\n".join(lines)

    def _prefill(self, subscriber_id: str, now: datetime) -> dict[str, str | None]:
        values: dict[str, str | None] = {"date": now.strftime("%d.%m.%Y")}
        if self.billing is None:
            return values
        account = self.billing.account(subscriber_id)
        if account is not None:
            values["full_name"] = account.full_name
            values["account_number"] = account.number
            values["address"] = account.address
        # Данные счётчика — только когда прибор ровно один: при двух и более
        # какой из них имеет в виду абонент, из запроса неизвестно. Оба он
        # увидит в справочном блоке под бланком.
        meters = self.billing.meters(subscriber_id)
        if len(meters) == 1:
            meter = meters[0]
            values["meter_kind"] = meter.kind
            values["meter_serial"] = meter.serial
            values["meter_reading"] = meter.reading or None
            if meter.verify_by is not None:
                values["meter_verify_by"] = meter.verify_by.strftime("%d.%m.%Y")
        return values

    def _reference_block(self, subscriber_id: str) -> str:
        """Данные лицевого счёта под бланком — чтобы абонент не искал их сам."""
        if self.billing is None:
            return ""
        account = self.billing.account(subscriber_id)
        if account is None:
            return ""

        lines = [
            "",
            "—" * 60,
            "Данные из вашего лицевого счёта (для полей выше):",
            f"  ФИО: {account.full_name}",
            f"  Лицевой счёт: {account.number}",
            f"  Адрес: {account.address}",
        ]
        for meter in self.billing.meters(subscriber_id):
            lines.append(f"  Счётчик {meter.kind} № {meter.serial}: {_meter_line(meter)}")
        if account.has_debt:
            debt = f"{account.debt:,.2f}".replace(",", " ").replace(".", ",")
            lines.append(
                f"  Задолженность: {debt} ₽ "
                "(для справки укажите вид «о наличии задолженности»)"
            )
        else:
            lines.append(
                "  Задолженности нет (для справки — вид «об отсутствии задолженности»)"
            )
        return "\n".join(lines) + "\n"


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many
