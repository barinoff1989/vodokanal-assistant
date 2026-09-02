"""Проверки схем данных.

Проверяется не то, что Pydantic умеет складывать поля, а то, что модели держат
конкретные требования проекта: ограничение длины запроса, нормализация типа
обращения, запрет персональных данных в отчёте и в журнале, обязательность
подтверждения перед записью.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import (
    MAX_QUERY_LENGTH,
    AuditEntry,
    Channel,
    ContextChunk,
    ErrorEvent,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    GenerationParameters,
    InquiryState,
    PiiReport,
    ProblemDetail,
    RequestMetadata,
    SessionState,
    Slots,
    TriageResult,
    Usage,
)
from app.taxonomy import InquiryType, UnknownInquiryTypeError


def _metadata(**overrides) -> RequestMetadata:
    defaults = {"subscriber_id": "sub-123456", "session_id": "sess-12345678"}
    return RequestMetadata(**{**defaults, **overrides})


def _request(**overrides) -> GenerateRequest:
    defaults = {
        "system": "Ты помощник абонента водоканала.",
        "query": "Когда нужно поверять счётчик?",
        "metadata": _metadata(),
    }
    return GenerateRequest(**{**defaults, **overrides})


# --- ограничения контракта (раздел 7.2) ------------------------------------- #


def test_слишком_длинный_запрос_отклоняется():
    with pytest.raises(ValidationError):
        _request(query="а" * (MAX_QUERY_LENGTH + 1))


def test_запрос_предельной_длины_принимается():
    """Граница должна быть включающей, иначе ровно на пределе будет отказ."""
    assert len(_request(query="а" * MAX_QUERY_LENGTH).query) == MAX_QUERY_LENGTH


def test_пустая_системная_часть_отклоняется():
    """Без неё модель отвечает вне роли (раздел 7.2)."""
    with pytest.raises(ValidationError):
        _request(system="")


def test_контекст_может_быть_пустым():
    """Допустимо, но включает запасной путь — это проверяется отдельно."""
    request = _request()
    assert request.context == []
    assert request.has_context is False


def test_непредусмотренное_поле_отклоняется():
    """`extra=forbid`: опечатка в имени поля должна ломать разбор сразу.

    Иначе значение молча потеряется, и это повторит класс ошибки с пробелами
    в ключах словаря из прошлого прототипа (раздел 11.9).
    """
    with pytest.raises(ValidationError):
        _request(unknown_field="значение")


def test_оценка_релевантности_ограничена_единицей():
    with pytest.raises(ValidationError):
        ContextChunk(chunk_id="c1", text="t", source_title="s", relevance_score=1.5)


# --- нормализация типа обращения на границе --------------------------------- #


def test_тип_обращения_нормализуется_при_разборе():
    """Пробелы и регистр снимаются на входе, а не тремя строками ниже."""
    assert _metadata(inquiry_type=" METER_VERIFICATION ").inquiry_type is (
        InquiryType.METER_VERIFICATION
    )


def test_значение_прошлой_версии_переводится():
    assert _metadata(inquiry_type="перерасчёт ").inquiry_type is InquiryType.ACCRUAL_RECALCULATION


def test_тип_обращения_необязателен_до_классификации():
    """На первом обращении абонента типа ещё нет."""
    assert _metadata().inquiry_type is None


def test_неизвестный_тип_в_метаданных_это_ошибка():
    """Строгий разбор на границе: мягкая подмена скрыла бы поломку."""
    with pytest.raises((ValidationError, UnknownInquiryTypeError)):
        _metadata(inquiry_type="выдуманный_тип")


def test_канал_по_умолчанию_веб():
    assert _metadata().channel is Channel.LK_WEB


def test_департамента_в_метаданных_нет():
    """Решение 2 модуля: департаменты не пользователи чат-интерфейса.

    Тест закрепляет отход от раздела 7.1, чтобы поле не вернули по невнимательности.
    """
    assert "department" not in RequestMetadata.model_fields
    with pytest.raises(ValidationError):
        _metadata(department="Абонентский отдел")


# --- бюджет ответа (пункт 31 TODO) ------------------------------------------ #


def test_длина_ответа_по_умолчанию_укладывается_в_бюджет():
    """Замер показал: 3 секунды DoD — это примерно 150–200 токенов."""
    assert GenerationParameters().max_tokens <= 200


def test_поток_включён_по_умолчанию():
    """Правило 4.3: ответ доходит до абонента постепенно."""
    assert GenerationParameters().stream is True


# --- персональные данные ---------------------------------------------------- #


def test_текст_запроса_не_попадает_в_печатное_представление():
    """Правило 4.2 чаще нарушают отладочным выводом, а не отправкой наружу."""
    request = _request(query="Иванов Иван, лицевой счёт 1234567890")
    assert "Иванов" not in repr(request)
    assert "1234567890" not in repr(request)


def test_слоты_не_раскрываются_в_печатном_представлении():
    slots = Slots(full_name="Иванов Иван Петрович", account_number="1234567890")
    assert "Иванов" not in repr(slots)
    assert "1234567890" not in repr(slots)


def test_в_отчёт_об_обезличивании_нельзя_положить_значение():
    """Ловится самый вероятный промах — номер счёта вместо имени сущности."""
    with pytest.raises(ValidationError):
        PiiReport(pii_detected=True, entities=["1234567890"])


def test_отчёт_принимает_имена_сущностей():
    report = PiiReport(pii_detected=True, entities=["SNILS", "PHONE", "ACCOUNT_NUMBER"])
    assert report.entities == ["SNILS", "PHONE", "ACCOUNT_NUMBER"]


def test_ответ_модели_не_раскрывается_в_печатном_представлении():
    response = GenerateResponse(answer="Ваш долг 1234 рубля", model="m", trace_id="t")
    assert "1234" not in repr(response)


# --- слоты и полнота черновика ---------------------------------------------- #


def test_пустые_слоты_неполны():
    assert Slots().missing_required() == ("full_name", "account_number", "reason")
    assert Slots().is_complete is False


def test_полные_слоты_считаются_полными():
    slots = Slots(full_name="Иванов И. И.", account_number="1234567890", reason="Проверка")
    assert slots.missing_required() == ()
    assert slots.is_complete is True


def test_пробельное_значение_не_считается_заполненным():
    """Иначе черновик уйдёт на подтверждение с пустым по сути полем."""
    slots = Slots(full_name="  ", account_number="1234567890", reason="Проверка")
    assert "full_name" in slots.missing_required()


# --- состояние обращения ----------------------------------------------------- #


def test_новое_состояние_начальное():
    state = SessionState(session_id="s", subscriber_id="sub")
    assert state.current_state is InquiryState.INIT
    assert state.awaits_confirmation is False


def test_ожидание_подтверждения_распознаётся():
    """Правило 4.7: запись возможна только из этого состояния."""
    state = SessionState(
        session_id="s",
        subscriber_id="sub",
        current_state=InquiryState.AWAITING_CONFIRMATION,
    )
    assert state.awaits_confirmation is True


def test_запись_аудита_добавляется():
    state = SessionState(session_id="s", subscriber_id="sub")
    entry = state.record("backend", "classified", inquiry_type="debt")
    assert state.audit_log == [entry]
    assert entry.actor == "backend"
    assert entry.details["inquiry_type"] == "debt"


def test_у_записи_аудита_есть_время():
    assert AuditEntry(actor="a", action="b").at is not None


# --- классификация ----------------------------------------------------------- #


def test_результат_классификации_нормализует_тип():
    result = TriageResult(inquiry_type=" DEBT ", confidence=0.9)
    assert result.inquiry_type is InquiryType.DEBT


def test_уверенность_ограничена_единицей():
    with pytest.raises(ValidationError):
        TriageResult(inquiry_type="debt", confidence=1.4)


def test_запасной_режим_помечается_явно():
    """Долю таких ответов нужно видеть в метриках, а не выводить из уверенности."""
    assert TriageResult(inquiry_type="debt", confidence=0.5).fallback_used is False
    assert TriageResult(inquiry_type="debt", confidence=0.5, fallback_used=True).fallback_used


# --- ошибки и события потока -------------------------------------------------- #


def test_ошибка_соответствует_rfc7807():
    problem = ProblemDetail(
        type="https://vodokanal.example/errors/upstream-llm-unavailable",
        title="LLM provider unavailable",
        status=502,
        trace_id="t-1",
    )
    assert set(problem.model_dump(exclude_none=True)) >= {"type", "title", "status"}


def test_недопустимый_код_состояния_отклоняется():
    with pytest.raises(ValidationError):
        ProblemDetail(type="t", title="t", status=99)


def test_у_события_ошибки_в_потоке_есть_схема():
    """Пункт 3 раздела 7.6: раньше схемы не было вовсе."""
    event = ErrorEvent(problem=ProblemDetail(type="t", title="t", status=504))
    assert event.event == "error"
    assert event.problem.status == 504


def test_обрыв_охранителем_отличим_от_нормального_завершения():
    """Иначе в аналитике блокировка выглядела бы как обычный ответ (раздел 35.1)."""
    assert FinishReason.GUARDRAIL != FinishReason.STOP


def test_расход_токенов_суммируется():
    assert Usage(prompt_tokens=100, completion_tokens=50).total_tokens == 150
