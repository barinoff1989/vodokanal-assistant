"""Проверки адаптеров записи.

Шесть проверок, каждая названа так, чтобы связь была видна.
Сквозная мысль: гейты держатся запретами, а не соглашениями. Соглашение
однажды забудут — запрет упадёт.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.adapters.base import (
    ConfirmationRequiredError,
    SubscriberMismatchError,
    WriteAdapter,
)
from app.adapters.billing_adapter import (
    ALLOWED_WRITE_OPERATIONS,
    BillingAdapter,
    MeterReading,
    UnsupportedBillingOperationError,
)
from app.adapters.document_generator import DocumentGenerator
from app.adapters.inquiry_service import (
    DispatchAdapter,
    EmergencyRequest,
    Inquiry,
    InquiryServiceAdapter,
)
from app.models import InquiryState, SessionState
from app.taxonomy import InquiryType


class FakeBilling:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def get_account(self, account_number: str) -> dict[str, Any]:
        return {"account_number": account_number, "balance": -1240.5}

    def submit_meter_reading(self, *args: Any) -> str:
        self.calls.append(args)
        return f"op-{len(self.calls)}"


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def create_inquiry(self, payload: dict[str, Any], key: str) -> str:
        self.calls.append((payload, key))
        return f"inq-{len(self.calls)}"

    def create_incident(self, payload: dict[str, Any], key: str) -> str:
        self.calls.append((payload, key))
        return f"inc-{len(self.calls)}"

    def get_incidents(self, address: str) -> list[dict[str, Any]]:
        return [{"address": address, "status": "в работе"}]


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, key: str, body: bytes, content_type: str) -> None:
        self.objects[key] = body

    def presigned_url(self, key: str, ttl_seconds: int) -> str:
        return f"https://storage.example/{key}?ttl={ttl_seconds}"


def _session(state: InquiryState = InquiryState.AWAITING_CONFIRMATION) -> SessionState:
    return SessionState(session_id="sess-1", subscriber_id="sub-1", current_state=state)


READING = MeterReading(
    account_number="4501230011", meter_serial="М-77", value=1420, period="2026-09"
)


# --- проверка 1: запись только после подтверждения ------------- #


@pytest.mark.parametrize(
    "state", [s for s in InquiryState if s is not InquiryState.AWAITING_CONFIRMATION]
)
def test_запись_невозможна_вне_ожидания_подтверждения(state: InquiryState):
    """Перебираются все состояния: новое не должно открыть обход незаметно."""
    adapter = BillingAdapter(FakeBilling())
    with pytest.raises(ConfirmationRequiredError):
        adapter.submit_meter_reading(_session(state), "sub-1", READING)


def test_прямой_вызов_адаптера_тоже_отклоняется():
    """Гейт стоит в адаптере, а не только в оркестраторе.

    Адаптер можно вызвать напрямую, и тогда единственная проверка на уровне
    оркестратора оказалась бы в стороне.
    """
    billing = FakeBilling()
    with pytest.raises(ConfirmationRequiredError):
        BillingAdapter(billing).submit_meter_reading(
            _session(InquiryState.DRAFT_READY), "sub-1", READING
        )
    assert billing.calls == [], "до внешней системы дойти не должно"


# --- проверка 2: идемпотентность ---------------------------------------------- #


def test_повтор_с_тем_же_ключом_не_создаёт_вторую_операцию():
    billing = FakeBilling()
    adapter = BillingAdapter(billing)
    session = _session()

    first = adapter.submit_meter_reading(session, "sub-1", READING)
    second = adapter.submit_meter_reading(session, "sub-1", READING)

    assert first.operation_id == second.operation_id
    assert second.repeated is True
    assert len(billing.calls) == 1, "второго обращения к системе быть не должно"


def test_ключ_идемпотентности_уходит_во_внешнюю_систему():
    """Наш учёт ключей живёт в памяти и перезапуск не переживает.

    Без передачи ключа самой системе повтор после перезапуска создал бы вторую
    операцию.
    """
    billing = FakeBilling()
    adapter = BillingAdapter(billing)
    result = adapter.submit_meter_reading(_session(), "sub-1", READING)
    assert result.idempotency_key in billing.calls[0]


def test_ключ_не_зависит_от_времени():
    """Иначе повтор того же действия получал бы новый ключ, и защиты нет."""
    adapter = BillingAdapter(FakeBilling())
    session = _session()
    first = adapter.idempotency_key(session, "submit_meter_reading", "4501230011")
    second = adapter.idempotency_key(session, "submit_meter_reading", "4501230011")
    assert first == second


def test_показание_за_другой_период_это_другая_операция():
    billing = FakeBilling()
    adapter = BillingAdapter(billing)
    session = _session()
    adapter.submit_meter_reading(session, "sub-1", READING)
    other = MeterReading("4501230011", "М-77", 1500, "2026-10")
    adapter.submit_meter_reading(session, "sub-1", other)
    assert len(billing.calls) == 2


# --- проверка 3: привязка абонента к сессии (Spoofing и IDOR) ------------------ #


def test_запись_по_чужому_абоненту_отклоняется():
    """Расхождение Р4: проверка была показана на диаграммах, но нигде не реализована."""
    billing = FakeBilling()
    with pytest.raises(SubscriberMismatchError):
        BillingAdapter(billing).submit_meter_reading(_session(), "sub-999", READING)
    assert billing.calls == []


def test_сообщение_об_ошибке_не_сводит_два_идентификатора():
    """Текст ошибки может уйти в журнал; сопоставление чужих идентификаторов —
    уже сведение данных."""
    with pytest.raises(SubscriberMismatchError) as excinfo:
        BillingAdapter(FakeBilling()).submit_meter_reading(_session(), "sub-999", READING)
    assert "sub-999" not in str(excinfo.value)
    assert "sub-1" not in str(excinfo.value)


def test_проверка_абонента_есть_на_каждом_пути_записи():
    """Не в одном адаптере, а во всех трёх — иначе дыра остаётся в пропущенном."""
    session = _session()
    with pytest.raises(SubscriberMismatchError):
        InquiryServiceAdapter(FakeApi()).register(
            session, "sub-999", Inquiry(InquiryType.DEBT, "Долг", "текст")
        )
    with pytest.raises(SubscriberMismatchError):
        DispatchAdapter(FakeApi()).submit(
            session, "sub-999", EmergencyRequest("ул. Речная, 14", "прорыв")
        )


# --- проверка 4: отказ абонента фиксируется ------------------------------------ #


def test_выполненная_запись_попадает_в_аудит():
    session = _session()
    BillingAdapter(FakeBilling()).submit_meter_reading(session, "sub-1", READING)
    assert session.audit_log[-1].action == "write:performed"
    assert session.audit_log[-1].details["system"] == "billing"


def test_повтор_отмечается_в_аудите_отдельно():
    """Иначе по журналу не отличить «выполнили дважды» от «защита сработала»."""
    session = _session()
    adapter = BillingAdapter(FakeBilling())
    adapter.submit_meter_reading(session, "sub-1", READING)
    adapter.submit_meter_reading(session, "sub-1", READING)
    assert [entry.action for entry in session.audit_log] == [
        "write:performed",
        "write:repeated",
    ]


def test_в_аудит_не_попадают_значения_показаний():
    """В журнал идут идентификаторы и названия, не сами данные абонента."""
    session = _session()
    BillingAdapter(FakeBilling()).submit_meter_reading(session, "sub-1", READING)
    written = str(session.audit_log[-1].details)
    assert "4501230011" not in written
    assert "1420" not in written


# --- проверка 5: область записи в Биллинг сужена (ADR-100) --------------------- #


def test_разрешена_ровно_одна_операция_записи():
    assert {"submit_meter_reading"} == ALLOWED_WRITE_OPERATIONS


@pytest.mark.parametrize(
    "operation", ["recalculate", "write_off_debt", "update_subscriber", "charge"]
)
def test_любая_другая_операция_биллинга_отклоняется(operation: str):
    """Разница между «мы этого не написали» и «это запрещено» должна быть видна."""
    with pytest.raises(UnsupportedBillingOperationError, match=operation):
        BillingAdapter(FakeBilling()).write(operation)


def test_чтение_биллинга_гейтов_записи_не_проходит():
    """Показать баланс — обычное действие Этапа 1, подтверждения не требует."""
    account = BillingAdapter(FakeBilling()).fetch_account("4501230011")
    assert account["balance"] == -1240.5


# --- проверка 6: одна внешняя система на путь (ADR-100) ------------------------ #


def test_у_каждого_адаптера_ровно_одна_система():
    """Архитектурная проверка: расхождение Р6 устранено тем, что пар не бывает."""
    systems = {
        BillingAdapter.system,
        InquiryServiceAdapter.system,
        DispatchAdapter.system,
    }
    assert len(systems) == 3
    assert all(isinstance(name, str) and name != "unknown" for name in systems)


def test_авария_не_регистрируется_как_обращение():
    """Решение ADR-100 отменяет прежнее утверждение.

    Если бы авария писалась в обе системы, вернулось бы расхождение Р6.
    """
    api = FakeApi()
    with pytest.raises(ValueError, match="Диспетчерской"):
        InquiryServiceAdapter(api).register(
            _session(), "sub-1", Inquiry(InquiryType.EMERGENCY, "Авария", "прорыв трубы")
        )
    assert api.calls == []


def test_авария_уходит_в_диспетчерскую():
    api = FakeApi()
    result = DispatchAdapter(api).submit(
        _session(), "sub-1", EmergencyRequest("ул. Речная, 14", "прорыв")
    )
    assert result.system == "dispatch"
    assert len(api.calls) == 1


def test_обычное_обращение_уходит_в_микросервис():
    api = FakeApi()
    result = InquiryServiceAdapter(api).register(
        _session(), "sub-1", Inquiry(InquiryType.DEBT, "Задолженность", "прошу сверку")
    )
    assert result.system == "inquiry_service"


def test_чтение_аварий_доступно_без_подтверждения():
    """Путь Этапа 1: статус аварий включается в ответ абоненту."""
    assert DispatchAdapter(FakeApi()).fetch_incidents("ул. Речная, 14")


# --- путь документа ------------------------------------------------ #


def test_документ_доходит_до_ссылки():
    """Путь прослеживается целиком: от формирования до ссылки для абонента."""
    store = FakeStore()
    session = _session()
    document = DocumentGenerator(store).build(
        session, "заявление-перерасчёт", {"ФИО": "Иванов И. И."}
    )
    assert document.url.startswith("https://")
    assert document.object_key in store.objects
    assert session.audit_log[-1].action == "document:ready"


def test_готовый_документ_не_выдаётся_до_подтверждения():
    """До подтверждения абонент видит черновик — это другое состояние документа."""
    with pytest.raises(RuntimeError, match="подтверждения"):
        DocumentGenerator(FakeStore()).build(
            _session(InquiryState.DRAFT_READY), "заявление", {}
        )


def test_ссылка_имеет_ограниченный_срок():
    """Ссылка ведёт на документ с персональными данными и вечной быть не может."""
    document = DocumentGenerator(FakeStore()).build(_session(), "заявление", {})
    assert 0 < document.ttl_seconds <= 7 * 24 * 3600


def test_повторное_формирование_не_плодит_копии():
    """Имя объекта детерминировано содержимым."""
    store = FakeStore()
    values = {"ФИО": "Иванов И. И."}
    first = DocumentGenerator(store).build(_session(), "заявление", values)
    second = DocumentGenerator(store).build(_session(), "заявление", values)
    assert first.object_key == second.object_key
    assert len(store.objects) == 1


# --- основа --------------------------------------------------------------------- #


def test_основа_не_выполняет_запись_сама():
    """Базовый класс обязан быть переопределён — иначе тихо «выполнил» ничего."""
    adapter = WriteAdapter()
    with pytest.raises(NotImplementedError):
        adapter.perform(_session(), "sub-1", key_parts=("x",), payload={})


def test_сессия_попадает_в_запись_обращения():
    """По жалобе абонента нужно поднять диалог, из которого выросло обращение.

    Без сессии запись сирота: известно, кто подал, но не известно, что именно
    ему показывали перед подтверждением — а показанное и подтверждённое обязаны
    совпадать."""
    seen: dict[str, Any] = {}

    class Client:
        def create_inquiry(self, payload: dict[str, Any], key: str) -> str:
            seen.update(payload)
            return "42"

    session = SessionState(session_id="sess-77", subscriber_id="sub-1")
    session.current_state = InquiryState.AWAITING_CONFIRMATION
    adapter = InquiryServiceAdapter(client=Client())

    adapter.register(
        session,
        "sub-1",
        Inquiry(
            inquiry_type=InquiryType.METER_VERIFICATION,
            subject="Заказать поверку",
            body="черновик",
        ),
    )

    assert seen["session_id"] == "sess-77"
