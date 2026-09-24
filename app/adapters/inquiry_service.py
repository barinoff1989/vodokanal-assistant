"""Сервис работы с обращениями: регистрация обращения и аварийной заявки.

Контейнер назывался «Application Router» до версии 4.5 и был переименован под
фактическую топологию: он регистрирует обращения, а не «отправляет заявки
подразделениям» — подрядчики читают обращения из микросервиса сами.

ДВА ПУТИ, И КАЖДЫЙ ПИШЕТ РОВНО В ОДНУ СИСТЕМУ (ADR-100):

* обычное обращение — в микросервис обращений;
* **аварийная заявка — только в Диспетчерскую**, в микросервисе обращений она
  не регистрируется.

Второе отменяет прежнее утверждение. Цена решения названа явно:
аварии не попадут ни в корпус для дообучения, ни в аналитику по типам обращений,
ни в эталонный набор — примеры по типу `emergency` придётся брать из Диспетчерской
или составлять синтетически.

> **[ДУБЛИ, ПОСТРОЕННЫЕ НА ДОПУЩЕНИЯХ]** Состав полей и способ вызова — ожидаемая
> форма, а не подтверждённый факт: описания интерфейсов не получены.
> Протокол микросервиса обращений помечен как
> предположение с самого начала.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.adapters.base import WriteAdapter, WriteResult
from app.models import SessionState
from app.taxonomy import InquiryType

__all__ = ["DispatchAdapter", "EmergencyRequest", "Inquiry", "InquiryServiceAdapter"]


class _ApiLike(Protocol):
    """Минимум, который нужен от клиента чужого API.

    Раньше здесь стоял `__getattr__`: под него подходило что угодно, и проверка
    типов ничего не проверяла. Теперь названы сами методы — на прототипе их
    предоставляет таблица в Postgres (`app/inquiries/store.py`), на MVP
    предоставит обёртка над REST владельца.
    """

    def create_inquiry(self, payload: dict[str, Any], key: str) -> str: ...


class _DispatchLike(Protocol):
    """То же для Диспетчерской: чтение аварий и передача заявки."""

    def get_incidents(self, address: str) -> list[dict[str, Any]]: ...

    def create_incident(self, payload: dict[str, Any], key: str) -> str: ...


@dataclass(frozen=True, slots=True)
class Inquiry:
    """Обращение абонента, регистрируемое в микросервисе обращений."""

    inquiry_type: InquiryType
    subject: str
    body: str
    slots: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmergencyRequest:
    """Аварийная заявка, передаваемая в Диспетчерскую систему."""

    address: str
    description: str
    contact_phone: str | None = None


class InquiryServiceAdapter(WriteAdapter):
    """Регистрация обращения в микросервисе обращений.

    Аварии сюда не попадают: для них есть :class:`DispatchAdapter`. Разделение
    на два адаптера, а не два метода одного, сделано намеренно — так правило
    «одна внешняя система на адаптер» проверяется тестом, а не соблюдается
    на честном слове.
    """

    system = "inquiry_service"

    def __init__(self, client: _ApiLike | None = None) -> None:
        super().__init__()
        self._client = client

    def register(
        self, session: SessionState, subscriber_id: str, inquiry: Inquiry
    ) -> WriteResult:
        """Зарегистрировать обращение после подтверждения абонента."""
        if inquiry.inquiry_type is InquiryType.EMERGENCY:
            raise ValueError(
                "аварийные заявки регистрируются в Диспетчерской системе, "
                "а не в микросервисе обращений (ADR-100)"
            )

        return self.perform(
            session,
            subscriber_id,
            key_parts=("register", inquiry.inquiry_type.value, inquiry.subject),
            payload={
                "inquiry_type": inquiry.inquiry_type.value,
                "subject": inquiry.subject,
                "body": inquiry.body,
                "subscriber_id": subscriber_id,
                # Сессия идёт в запись, а не остаётся в аудите: по жалобе
                # абонента нужно поднять диалог, из которого выросло обращение.
                # Без неё запись сирота — известно, кто подал, но не известно,
                # что именно ему показывали перед подтверждением.
                "session_id": session.session_id,
                "slots": inquiry.slots,
            },
        )

    def _call_system(self, payload: dict[str, Any], key: str) -> WriteResult:
        if self._client is None:
            raise RuntimeError("клиент микросервиса обращений не задан")
        number = self._client.create_inquiry(payload, key)
        return WriteResult(
            operation_id=str(number or uuid.uuid4().hex[:12]),
            system=self.system,
            idempotency_key=key,
            payload=payload,
        )


class DispatchAdapter(WriteAdapter):
    """Передача аварийной заявки в Диспетчерскую систему.

    Единственный путь для аварий (ADR-100). На Этапе 1 та же система только
    читается — статус аварий включается в ответ абоненту; чтение гейтов записи
    не проходит.
    """

    system = "dispatch"

    def __init__(self, client: _DispatchLike | None = None) -> None:
        super().__init__()
        self._client = client

    def fetch_incidents(self, address: str) -> list[dict[str, Any]]:
        """Прочитать аварии по адресу — путь Этапа 1, без гейтов записи."""
        if self._client is None:
            raise RuntimeError("клиент Диспетчерской системы не задан")
        return list(self._client.get_incidents(address) or [])

    def submit(
        self, session: SessionState, subscriber_id: str, request: EmergencyRequest
    ) -> WriteResult:
        """Передать аварийную заявку после подтверждения абонента."""
        return self.perform(
            session,
            subscriber_id,
            key_parts=("emergency", request.address, request.description),
            payload={
                "address": request.address,
                "description": request.description,
                "contact_phone": request.contact_phone,
                "subscriber_id": subscriber_id,
            },
        )

    def _call_system(self, payload: dict[str, Any], key: str) -> WriteResult:
        if self._client is None:
            raise RuntimeError("клиент Диспетчерской системы не задан")
        incident_id = self._client.create_incident(payload, key)
        return WriteResult(
            operation_id=str(incident_id or uuid.uuid4().hex[:12]),
            system=self.system,
            idempotency_key=key,
            payload=payload,
        )
