"""Проверки конечного автомата обращения.

Главное здесь — не то, что разрешённые переходы работают, а то, что
**запрещённые не работают**. Правило 4.7 держится именно на запретах: если
операцию записи удастся выполнить в обход подтверждения абонента, нарушится
несущее требование проекта, и никакая проверка ниже по стеку этого не поймает.
"""

from __future__ import annotations

import itertools

import pytest

from app.fsm import (
    ALLOWED_FROM_AWAITING_CONFIRMATION,
    TRANSITIONS,
    IllegalTransitionError,
    IncompleteDraftError,
    InquiryEvent,
    advance,
    allowed_events,
    can,
    is_terminal,
    next_state,
)
from app.models import InquiryState, SessionState, Slots


def _session(
    state: InquiryState = InquiryState.INIT, *, complete_slots: bool = False
) -> SessionState:
    slots = (
        Slots(full_name="Иванов Иван Петрович", account_number="1234567890", reason="Проверка")
        if complete_slots
        else Slots()
    )
    return SessionState(
        session_id="sess-1", subscriber_id="sub-1", current_state=state, slots=slots
    )


# --- сквозной путь ----------------------------------------------------------- #


def test_полный_путь_обращения_проходится():
    """Тот же порядок шагов, что на диаграмме последовательностей Этапа 2."""
    session = _session(complete_slots=True)
    for event, expected in [
        (InquiryEvent.CLASSIFY, InquiryState.CLASSIFIED),
        (InquiryEvent.FILL, InquiryState.FILLING),
        (InquiryEvent.FINALIZE_DRAFT, InquiryState.DRAFT_READY),
        (InquiryEvent.REQUEST_CONFIRM, InquiryState.AWAITING_CONFIRMATION),
        (InquiryEvent.SUBMIT, InquiryState.SUBMITTED),
        (InquiryEvent.COMPLETE, InquiryState.COMPLETED),
    ]:
        assert advance(session, event) is expected

    assert is_terminal(session.current_state)


def test_каждый_шаг_пути_попал_в_аудит():
    """Правило 4.7 требует аудита на всех путях записи."""
    session = _session(complete_slots=True)
    for event in (
        InquiryEvent.CLASSIFY,
        InquiryEvent.FILL,
        InquiryEvent.FINALIZE_DRAFT,
        InquiryEvent.REQUEST_CONFIRM,
        InquiryEvent.SUBMIT,
    ):
        advance(session, event)

    assert len(session.audit_log) == 5
    assert session.audit_log[-1].action == "transition:submit"
    assert session.audit_log[-1].details == {
        "from_state": "awaiting_confirmation",
        "to_state": "submitted",
    }


# --- правило 4.7: запись только после подтверждения --------------------------- #


@pytest.mark.parametrize(
    "state",
    [s for s in InquiryState if s is not InquiryState.AWAITING_CONFIRMATION],
)
def test_запись_невозможна_ни_из_какого_другого_состояния(state: InquiryState):
    """Несущая проверка модуля: обойти подтверждение абонента нельзя.

    Перебираются все состояния, а не пара характерных: при добавлении нового
    состояния тест обязан заметить, что из него внезапно стала возможна запись.
    """
    assert not can(state, InquiryEvent.SUBMIT)
    with pytest.raises(IllegalTransitionError):
        advance(_session(state), InquiryEvent.SUBMIT)


def test_из_ожидания_подтверждения_ровно_два_выхода():
    """Подтвердить или отказаться — третьего быть не должно."""
    assert set(allowed_events(InquiryState.AWAITING_CONFIRMATION)) == (
        ALLOWED_FROM_AWAITING_CONFIRMATION
    )


def test_отказ_абонента_возвращает_к_правке():
    """Раздел 3: должна быть возможность отмены или исправления до отправки."""
    session = _session(InquiryState.AWAITING_CONFIRMATION)
    assert advance(session, InquiryEvent.REJECT) is InquiryState.FILLING


def test_отказ_абонента_попадает_в_аудит():
    """Раздел 43.3: отказ пишется в аудит с пометкой, что запись не выполнялась."""
    session = _session(InquiryState.AWAITING_CONFIRMATION)
    advance(session, InquiryEvent.REJECT)
    assert session.audit_log[-1].action == "transition:reject"
    assert session.audit_log[-1].details["to_state"] == "filling"


# --- полнота черновика (Draft Validator, раздел 41.2) ------------------------- #


def test_неполный_черновик_не_завершается():
    """Иначе абоненту показали бы черновик с пустым именем."""
    session = _session(InquiryState.FILLING)
    with pytest.raises(IncompleteDraftError) as excinfo:
        advance(session, InquiryEvent.FINALIZE_DRAFT)
    assert set(excinfo.value.missing) == {"full_name", "account_number", "reason"}


def test_состояние_не_меняется_при_неудачном_переходе():
    """Переход либо происходит целиком, либо не происходит вовсе."""
    session = _session(InquiryState.FILLING)
    with pytest.raises(IncompleteDraftError):
        advance(session, InquiryEvent.FINALIZE_DRAFT)
    assert session.current_state is InquiryState.FILLING
    assert session.audit_log == []


def test_неполный_черновик_отличим_от_ошибки_оркестрации():
    """Первое — повод спросить недостающее, второе — поломка. Реакция разная."""
    session = _session(InquiryState.FILLING)
    with pytest.raises(IncompleteDraftError):
        advance(session, InquiryEvent.FINALIZE_DRAFT)
    # Но недопустимый переход из того же состояния — обычная ошибка перехода.
    with pytest.raises(IllegalTransitionError) as excinfo:
        advance(_session(InquiryState.FILLING), InquiryEvent.COMPLETE)
    assert not isinstance(excinfo.value, IncompleteDraftError)


def test_полный_черновик_завершается():
    session = _session(InquiryState.FILLING, complete_slots=True)
    assert advance(session, InquiryEvent.FINALIZE_DRAFT) is InquiryState.DRAFT_READY


# --- запреты в целом ---------------------------------------------------------- #


def test_все_переходы_вне_таблицы_запрещены():
    """Проверяются все пары «состояние × событие», а не выборочные случаи."""
    for state, event in itertools.product(InquiryState, InquiryEvent):
        expected = (state, event) in TRANSITIONS
        assert can(state, event) is expected
        if not expected:
            with pytest.raises(IllegalTransitionError):
                next_state(state, event)


def test_из_завершённого_состояния_выхода_нет():
    assert allowed_events(InquiryState.COMPLETED) == ()
    assert is_terminal(InquiryState.COMPLETED)


def test_незавершённые_состояния_имеют_продолжение():
    """Состояние без выходов, кроме завершённого, — это тупик в диалоге."""
    for state in InquiryState:
        if not is_terminal(state):
            assert allowed_events(state), f"тупиковое состояние: {state}"


def test_текст_ошибки_подсказывает_допустимые_события():
    """Без подсказки отладка оркестрации превращается в чтение таблицы."""
    with pytest.raises(IllegalTransitionError, match="classify"):
        next_state(InquiryState.INIT, InquiryEvent.SUBMIT)


# --- повторяемые события ------------------------------------------------------ #


def test_слоты_можно_заполнять_несколькими_репликами():
    session = _session(InquiryState.FILLING)
    assert advance(session, InquiryEvent.FILL) is InquiryState.FILLING
    assert advance(session, InquiryEvent.FILL) is InquiryState.FILLING


def test_переклассификация_допустима():
    """Абонент может уточнить суть — запрет заставил бы начинать диалог заново."""
    session = _session(InquiryState.FILLING)
    assert advance(session, InquiryEvent.CLASSIFY) is InquiryState.CLASSIFIED


def test_каждое_состояние_достижимо():
    """Недостижимое состояние — признак ошибки в таблице переходов."""
    reachable = {InquiryState.INIT}
    frontier = [InquiryState.INIT]
    while frontier:
        state = frontier.pop()
        for event in allowed_events(state):
            target = next_state(state, event)
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    assert reachable == set(InquiryState)
