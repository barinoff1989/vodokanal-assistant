"""Конечный автомат обращения: какие переходы допустимы и какие запрещены.

Состояния и события взяты из раздела 10 контекста (`InquiryFSM`), путь записи —
из диаграммы последовательностей Этапа 2. На этом модуле стоит правило 4.7:
операция записи возможна ровно из одного состояния, и это проверяется здесь, а
не в каждом адаптере по отдельности.

БЕЗ БИБЛИОТЕКИ — И ЭТО ОСОЗНАННОЕ ОТСТУПЛЕНИЕ ОТ ПЛАНА.
План разработки предполагал, что шаг 2 останется на библиотеке `transitions`,
которую использовал прототип. Здесь она не подключается по двум причинам:

1. Дефект 11.9 прошлой версии — пробелы в ключах словаря переходов
   (`{"trigger ": "classify "}`). Библиотека настраивается строковыми ключами и
   на таком не ругается: она просто не находит нужный переход. Перечисления
   вместо строк убирают этот класс ошибки целиком, а он в проекте уже случался.
2. Семь состояний и восемь переходов — это таблица на двадцать строк. Внешняя
   зависимость ради неё окупается только если нужны сложные возможности
   библиотеки, а они не нужны.

Отступление **не предрешает** ненаписанный ADR о переходе на LangGraph (пункт 10
сведённого TODO): наоборот, отсутствие привязки к какой-либо библиотеке делает
любой из вариантов дешевле. Решение обратимо — таблица переходов останется той
же, поменяется только исполнитель.
"""

from __future__ import annotations

from enum import StrEnum

from app.models import InquiryState, SessionState

__all__ = [
    "ALLOWED_FROM_AWAITING_CONFIRMATION",
    "TRANSITIONS",
    "IllegalTransitionError",
    "IncompleteDraftError",
    "InquiryEvent",
    "advance",
    "allowed_events",
    "can",
    "is_terminal",
    "next_state",
]


class InquiryEvent(StrEnum):
    """События, двигающие обращение (раздел 10, переходы `InquiryFSM`).

    `reject` в исходном перечне отсутствовал, хотя раздел 6.3 (шаг 7) прямо
    говорит «абонент подтверждает или отклоняет», а раздел 43.3 требует писать
    отказ в аудит с пометкой, что запись не выполнялась. Без этого события отказ
    абонента было бы некуда деть, кроме как оставить обращение висеть в
    ожидании подтверждения.
    """

    CLASSIFY = "classify"
    FILL = "fill"
    FINALIZE_DRAFT = "finalize_draft"
    REQUEST_CONFIRM = "request_confirm"
    SUBMIT = "submit"
    REJECT = "reject"
    COMPLETE = "complete"


class IllegalTransitionError(RuntimeError):
    """Переход не предусмотрен автоматом.

    Отдельный тип, а не `ValueError`: попытка недопустимого перехода — это почти
    всегда ошибка в оркестрации, и её нужно отличать от неверных данных
    пользователя, которые обрабатываются иначе.
    """

    def __init__(self, state: InquiryState, event: InquiryEvent) -> None:
        self.state = state
        self.event = event
        super().__init__(
            f"переход {event.value!r} недопустим из состояния {state.value!r}; "
            f"разрешены: {', '.join(e.value for e in allowed_events(state)) or '—'}"
        )


class IncompleteDraftError(IllegalTransitionError):
    """Черновик нельзя завершить: не хватает обязательных полей.

    Наследуется от недопустимого перехода, потому что по сути это он и есть, но
    отдельный тип позволяет вызывающему коду отличить «оркестрация пошла не
    туда» от «у абонента ещё не собраны данные» — во втором случае нужно не
    падать, а спросить недостающее.
    """

    def __init__(self, state: InquiryState, missing: tuple[str, ...]) -> None:
        self.missing = missing
        RuntimeError.__init__(
            self,
            "черновик неполон, не хватает обязательных полей: " + ", ".join(missing),
        )
        self.state = state
        self.event = InquiryEvent.FINALIZE_DRAFT


# Таблица переходов. Читается как «из состояния по событию в состояние».
# Всё, чего в ней нет, запрещено — перечислять запреты не нужно.
TRANSITIONS: dict[tuple[InquiryState, InquiryEvent], InquiryState] = {
    # Классификация обращения (раздел 6.1, шаг 3).
    (InquiryState.INIT, InquiryEvent.CLASSIFY): InquiryState.CLASSIFIED,
    # Переклассификация допустима: абонент может уточнить суть в следующей
    # реплике, и запрет заставил бы начинать диалог заново.
    (InquiryState.CLASSIFIED, InquiryEvent.CLASSIFY): InquiryState.CLASSIFIED,
    (InquiryState.FILLING, InquiryEvent.CLASSIFY): InquiryState.CLASSIFIED,
    # Сбор недостающих полей. Событие повторяемое: слоты заполняются по одному,
    # в течение нескольких реплик.
    (InquiryState.CLASSIFIED, InquiryEvent.FILL): InquiryState.FILLING,
    (InquiryState.FILLING, InquiryEvent.FILL): InquiryState.FILLING,
    # Черновик готов. Полноту обязательных полей проверяет Draft Validator
    # (раздел 41.2, шаг 17a диаграммы) — здесь она тоже проверяется, см. advance.
    (InquiryState.FILLING, InquiryEvent.FINALIZE_DRAFT): InquiryState.DRAFT_READY,
    # Черновик показан абоненту, ждём его решения. Это и есть привал HITL.
    (InquiryState.DRAFT_READY, InquiryEvent.REQUEST_CONFIRM): InquiryState.AWAITING_CONFIRMATION,
    # Абонент подтвердил — только отсюда возможна запись во внешние системы.
    (InquiryState.AWAITING_CONFIRMATION, InquiryEvent.SUBMIT): InquiryState.SUBMITTED,
    # Абонент отказался — возвращаемся к правке, а не в тупик: раздел 3 требует
    # «возможность отмены или исправления до отправки».
    (InquiryState.AWAITING_CONFIRMATION, InquiryEvent.REJECT): InquiryState.FILLING,
    # Заявка зарегистрирована, абоненту показан результат.
    (InquiryState.SUBMITTED, InquiryEvent.COMPLETE): InquiryState.COMPLETED,
}


ALLOWED_FROM_AWAITING_CONFIRMATION: frozenset[InquiryEvent] = frozenset(
    {InquiryEvent.SUBMIT, InquiryEvent.REJECT}
)
"""Из состояния ожидания есть ровно два выхода: подтвердить или отказаться.

Вынесено отдельным именем, чтобы правило 4.7 можно было проверить тестом, а не
вычитывать из таблицы глазами.
"""

TERMINAL_STATES: frozenset[InquiryState] = frozenset({InquiryState.COMPLETED})


def allowed_events(state: InquiryState) -> tuple[InquiryEvent, ...]:
    """Какие события допустимы из состояния. Порядок — как в таблице."""
    return tuple(event for (src, event) in TRANSITIONS if src is state)


def can(state: InquiryState, event: InquiryEvent) -> bool:
    """Допустим ли переход. Без побочных действий — для проверок и подсказок."""
    return (state, event) in TRANSITIONS


def next_state(state: InquiryState, event: InquiryEvent) -> InquiryState:
    """Куда приведёт переход.

    :raises IllegalTransitionError: перехода нет в таблице.
    """
    try:
        return TRANSITIONS[(state, event)]
    except KeyError:
        raise IllegalTransitionError(state, event) from None


def is_terminal(state: InquiryState) -> bool:
    """Завершено ли обращение: дальше двигаться некуда."""
    return state in TERMINAL_STATES


def advance(
    session: SessionState,
    event: InquiryEvent,
    *,
    actor: str = "orchestrator",
) -> InquiryState:
    """Выполнить переход и записать его в аудит.

    Аудит пишется здесь, а не вызывающим кодом, по той же причине, по которой
    здесь же стоит проверка допустимости: правило 4.7 требует аудита на всех
    путях записи, а требование, которое нужно помнить, рано или поздно забудут.

    Перед завершением черновика дополнительно проверяется полнота обязательных
    полей — это Draft Validator из раздела 41.2. Проверка здесь не заменяет его
    как компонент, а не даёт обойти: без неё черновик с пустым именем дошёл бы
    до показа абоненту.

    :raises IllegalTransitionError: переход не предусмотрен таблицей.
    :raises IncompleteDraftError: черновик неполон (частный случай предыдущего).
    """
    target = next_state(session.current_state, event)

    if event is InquiryEvent.FINALIZE_DRAFT and not session.slots.is_complete:
        raise IncompleteDraftError(session.current_state, session.slots.missing_required())

    previous = session.current_state
    session.current_state = target
    session.record(
        actor,
        f"transition:{event.value}",
        from_state=previous.value,
        to_state=target.value,
    )
    return target
