"""Регистрация обращения из диалога: черновик, подтверждение, запись.

ПОЧЕМУ ОТДЕЛЬНЫЙ МОДУЛЬ, А НЕ ВЕТКА В ОРКЕСТРАТОРЕ

Оркестратор выбирает путь ответа. Здесь — путь **записи**, и у него своя
последовательность: предложить, дождаться явного подтверждения, выполнить,
показать номер. Смешение читалось бы как одна ветка среди трёх, а это не ветка
ответа: она растянута на несколько реплик и держит состояние.

ТРИ РЕПЛИКИ, А НЕ ОДНА

    1. «Хочу заказать поверку»   -> ответ по базе знаний + черновик + кнопки
    2. «Подтверждаю»             -> запись, номер обращения
       «Отменить»                -> возврат к правке, запись отказа в аудит

Между репликами состояние живёт в `SessionStore`. Второй реплики может не быть
вовсе — абонент вправе передумать, и черновик просто истечёт по сроку.

ОТКУДА БЕРУТСЯ СЛОТЫ

Не из формы. Обязательных полей три — ФИО, лицевой счёт, основание, — и все три
уже известны: первые два в профиле Биллинга по идентификатору абонента, третье —
сам вопрос, который абонент задал. Спрашивать у человека то, что система про
него знает, — худший вид формы.

Форма слотов (`Slot Form` на C3 виджета) остаётся для случаев, где данных нет:
обращение за третье лицо, уточнение адреса объекта. На прототипе такого нет, и
писать её значило бы завести код без потребителя — тот же довод, по которому не
написаны агенты `form` и `draft_validator` (журнал, раздел 62.7).
"""

from __future__ import annotations

import logging

from app.adapters.base import ConfirmationRequiredError, SubscriberMismatchError
from app.adapters.inquiry_service import Inquiry, InquiryServiceAdapter
from app.backend.sessions import SessionStore
from app.billing.source import BillingSource
from app.fsm import InquiryEvent, advance
from app.models import (
    GenerateRequest,
    Intent,
    SessionState,
    Slots,
    SuggestedAction,
)
from app.taxonomy import InquiryType, display_name

__all__ = ["RegistrationOutcome", "Registrar", "REGISTRABLE"]

logger = logging.getLogger(__name__)

REGISTRABLE: frozenset[InquiryType] = frozenset(
    {
        InquiryType.METER_VERIFICATION,
        InquiryType.METER_SEALING,
        InquiryType.METER_INSTALLATION,
        InquiryType.METER_UNSEALING,
        InquiryType.CERTIFICATE,
        InquiryType.INSPECTION,
        InquiryType.DOCUMENT_SUBMISSION,
    }
)
"""Типы, по которым ассистент предлагает подать обращение.

Перечень **закрытый и короткий намеренно.** Предлагать регистрацию на каждый
вопрос — значит превратить подсказку в шум, а абонента приучить нажимать не
глядя. Здесь только то, что абонент и так заказывает заявкой: поверка,
опломбировка, установка и снятие прибора учёта, справка, обследование, приём
документов.

Чего здесь нет и почему:

* `EMERGENCY` — аварии идут в Диспетчерскую, а не в микросервис обращений
  (ADR-009), и путь у них другой;
* `DEBT`, `PENALTY`, `ACCRUAL_RECALCULATION` — это вопросы «почему столько», а
  не заявки; на них отвечает база знаний;
* `OTHER` — свалка по измерению (журнал, раздел 56.4): предлагать регистрацию
  того, что не распозналось, значит регистрировать неизвестно что.
"""

DRAFT_TEMPLATE = (
    "Обращение: {display}\n"
    "Абонент: {full_name}\n"
    "Лицевой счёт: {account}\n"
    "Основание: {reason}"
)


class RegistrationOutcome:
    """Что регистрация просит сделать оркестратор.

    Не событие и не ответ: оркестратор сам решает, каким путём это отдать —
    потоком или целиком. Здесь только существо.
    """

    __slots__ = ("actions", "text")

    def __init__(self, text: str, actions: tuple[SuggestedAction, ...] = ()) -> None:
        self.text = text
        self.actions = actions


class Registrar:
    """Ведёт обращение от предложения до записи.

    :param sessions: где живёт состояние между репликами.
    :param adapter: адаптер записи (шаг 9). `None` — регистрация выключена.
    :param billing: откуда берутся ФИО и лицевой счёт.
    """

    def __init__(
        self,
        sessions: SessionStore,
        *,
        adapter: InquiryServiceAdapter | None = None,
        billing: BillingSource | None = None,
    ) -> None:
        self._sessions = sessions
        self._adapter = adapter
        self._billing = billing

    @property
    def enabled(self) -> bool:
        return self._adapter is not None

    # -- предложение --------------------------------------------------------- #

    def offer(self, request: GenerateRequest, session: SessionState) -> RegistrationOutcome | None:
        """Предложить подать обращение, если тип это допускает.

        Черновик собирается **до** показа абоненту и сохраняется в состоянии:
        то, что он подтвердит, и то, что уйдёт в запись, обязано быть одним и тем
        же текстом. Собрать черновик заново на второй реплике значило бы
        зарегистрировать не то, что показали.
        """
        inquiry_type = request.metadata.inquiry_type
        if not self.enabled or inquiry_type is None or inquiry_type not in REGISTRABLE:
            return None

        slots = self._slots_from_profile(request)
        if slots is None:
            # Без ФИО и счёта черновик неполон, и `advance` его не пропустит.
            # Молчать здесь правильнее, чем предлагать то, что не соберётся.
            logger.info("профиль абонента не найден: обращение не предлагается")
            return None

        session.inquiry_type = inquiry_type
        session.slots = slots
        session.draft_text = DRAFT_TEMPLATE.format(
            display=display_name(inquiry_type),
            full_name=slots.full_name,
            account=slots.account_number,
            reason=slots.reason,
        )

        advance(session, InquiryEvent.CLASSIFY)
        advance(session, InquiryEvent.FILL)
        advance(session, InquiryEvent.FINALIZE_DRAFT)
        advance(session, InquiryEvent.REQUEST_CONFIRM)

        return RegistrationOutcome(
            text=(
                "\n\nМогу оформить обращение:\n"
                + session.draft_text
                + "\n\nПодтвердите — и я его зарегистрирую."
            ),
            actions=(
                SuggestedAction(
                    action=Intent.CONFIRM.value,
                    label="Подтвердить обращение",
                    inquiry_type=inquiry_type,
                ),
                SuggestedAction(action=Intent.REJECT.value, label="Отменить"),
            ),
        )

    # -- решение абонента ---------------------------------------------------- #

    def apply(self, request: GenerateRequest, session: SessionState) -> RegistrationOutcome:
        """Выполнить то, что абонент выбрал под черновиком.

        Рассогласование — не ошибка абонента: состояние могло истечь, а виджет
        отстать. Поэтому ответ объясняет, а не отказывает кодом.
        """
        intent = request.metadata.intent
        if not session.awaits_confirmation:
            return RegistrationOutcome(
                text=(
                    "Черновик обращения не найден — возможно, прошло слишком много "
                    "времени. Опишите, пожалуйста, вопрос заново."
                )
            )

        if intent is Intent.REJECT:
            advance(session, InquiryEvent.REJECT)
            session.record("subscriber", "draft:rejected")
            return RegistrationOutcome(
                text="Обращение не отправлено. Если нужно — уточните, что исправить."
            )

        return self._submit(request, session)

    def _submit(self, request: GenerateRequest, session: SessionState) -> RegistrationOutcome:
        assert self._adapter is not None  # проверено в `offer`, иначе черновика нет
        inquiry = Inquiry(
            inquiry_type=session.inquiry_type or InquiryType.OTHER,
            subject=display_name(session.inquiry_type or InquiryType.OTHER),
            body=session.draft_text or "",
            slots=session.slots.model_dump(exclude_none=True),
        )
        try:
            result = self._adapter.register(
                session, request.metadata.subscriber_id, inquiry
            )
        except (ConfirmationRequiredError, SubscriberMismatchError) as exc:
            # Гейты адаптера сработали позже нашей проверки состояния — значит
            # состояние и запрос разошлись. Записываем и отвечаем, а не падаем.
            logger.warning("запись отклонена гейтом адаптера: %s", exc)
            session.record("adapter", "write:refused", reason=type(exc).__name__)
            return RegistrationOutcome(
                text=(
                    "Не удалось зарегистрировать обращение. Обратитесь, пожалуйста, "
                    "в контакт-центр."
                )
            )

        advance(session, InquiryEvent.SUBMIT)
        session.submission_result = {
            "operation_id": result.operation_id,
            "system": result.system,
            "repeated": result.repeated,
        }
        advance(session, InquiryEvent.COMPLETE)

        if result.repeated:
            return RegistrationOutcome(
                text=f"Это обращение уже зарегистрировано, его номер {result.operation_id}."
            )
        return RegistrationOutcome(
            text=(
                f"Обращение зарегистрировано, номер {result.operation_id}. "
                "Срок рассмотрения — 15 рабочих дней."
            )
        )

    # -- слоты ---------------------------------------------------------------- #

    def _slots_from_profile(self, request: GenerateRequest) -> Slots | None:
        """Собрать обязательные слоты из профиля и вопроса абонента."""
        if self._billing is None:
            return None
        account = self._billing.account(request.metadata.subscriber_id)
        if account is None:
            return None

        slots = Slots(
            full_name=account.full_name,
            account_number=account.number,
            address=account.address,
            reason=request.query,
        )
        return slots if slots.is_complete else None
