"""Адаптер Биллинга: чтение данных абонента и передача показаний счётчика.

ОБЛАСТЬ ЗАПИСИ СУЖЕНА ДО ПОКАЗАНИЙ (ADR-009).
Через помощника в Биллинг передаются **только показания счётчика**. Перерасчёт,
списание, изменение данных абонента и любые другие операции — не выполняются.
Попытка вызвать иную операцию отклоняется здесь, а не «не предусмотрена»: разница
между «мы этого не написали» и «это запрещено» становится видна ровно тогда,
когда кто-то попробует дописать.

ТОЛЬКО ЧЕРЕЗ XML-RPC ВЛАДЕЛЬЦА (ADR-003).
Прямого обращения к базе Биллинга нет и быть не может: схема чужой базы не
контракт и меняется без предупреждения. Клиент XML-RPC входит в стандартную
библиотеку, внешняя зависимость не нужна.

> **[ДУБЛЬ, ПОСТРОЕННЫЙ НА ДОПУЩЕНИЯХ]** Имена методов и состав полей ниже —
> ожидаемая форма, а не подтверждённый факт: описания интерфейсов у владельцев
> систем ещё не получены (пункт 20 сведённого TODO, часть A10 запроса данных).
> Сверить до подключения к настоящему Биллингу.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.adapters.base import WriteAdapter, WriteResult
from app.models import SessionState

__all__ = [
    "ALLOWED_WRITE_OPERATIONS",
    "BillingAdapter",
    "MeterReading",
    "UnsupportedBillingOperationError",
]

ALLOWED_WRITE_OPERATIONS: frozenset[str] = frozenset({"submit_meter_reading"})
"""Единственная разрешённая операция записи (ADR-009).

Вынесено отдельным именем, чтобы ограничение можно было проверить тестом, а не
вычитывать из кода глазами.
"""


class UnsupportedBillingOperationError(RuntimeError):
    """Запрошена операция записи, не входящая в разрешённую область.

    Отдельный тип, а не «метод не найден»: сообщение должно объяснять, что
    операция запрещена решением, а не отсутствует по недосмотру.
    """

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"операция {operation!r} не входит в область записи Биллинга (ADR-009); "
            f"разрешено только: {', '.join(sorted(ALLOWED_WRITE_OPERATIONS))}"
        )
        self.operation = operation


@dataclass(frozen=True, slots=True)
class MeterReading:
    """Показание счётчика, передаваемое в Биллинг."""

    account_number: str
    meter_serial: str
    value: int
    period: str
    """Расчётный период в виде ``ГГГГ-ММ``. Входит в ключ идемпотентности:
    показание за один период передаётся один раз, повтор не создаёт второго."""


class _XmlRpcLike(Protocol):
    """Минимум, который нужен от клиента XML-RPC. Ради подмены в проверках."""

    def __getattr__(self, name: str) -> Any: ...


class BillingAdapter(WriteAdapter):
    """Обращение к Биллингу через его внешний XML-RPC API.

    :param client: клиент XML-RPC. Внедряется снаружи: настоящий Биллинг внутри
        закрытой сети, и на прототипе вместо него работает дубль.
    """

    system = "billing"

    def __init__(self, client: _XmlRpcLike | None = None) -> None:
        super().__init__()
        self._client = client

    # -- чтение (Этап 1) ---------------------------------------------------- #

    def fetch_account(self, account_number: str) -> dict[str, Any]:
        """Прочитать данные лицевого счёта.

        Чтение не проходит гейты записи: подтверждение абонента требуется для
        операций записи (правило 4.7), а показать баланс — обычное действие
        Этапа 1.
        """
        if self._client is None:
            raise RuntimeError("клиент Биллинга не задан")
        result = self._client.get_account(account_number)
        return dict(result) if result else {}

    # -- запись (Этап 2, только показания) ---------------------------------- #

    def submit_meter_reading(
        self, session: SessionState, subscriber_id: str, reading: MeterReading
    ) -> WriteResult:
        """Передать показание счётчика.

        Единственная операция записи в Биллинг (ADR-009). Проходит все гейты
        основы: подтверждение абонента, принадлежность сессии, идемпотентность.
        """
        return self.perform(
            session,
            subscriber_id,
            key_parts=(
                "submit_meter_reading",
                reading.account_number,
                reading.meter_serial,
                reading.period,
            ),
            payload={
                "account_number": reading.account_number,
                "meter_serial": reading.meter_serial,
                "value": reading.value,
                "period": reading.period,
            },
        )

    def write(self, operation: str, *args: Any, **kwargs: Any) -> WriteResult:
        """Общая точка записи — отклоняет всё, кроме разрешённого.

        Существует затем, чтобы попытка добавить операцию мимо решения ADR-009
        падала явно, а не проходила незамеченной.
        """
        raise UnsupportedBillingOperationError(operation)

    # -- обращение к системе ------------------------------------------------- #

    def _call_system(self, payload: dict[str, Any], key: str) -> WriteResult:
        if self._client is None:
            raise RuntimeError("клиент Биллинга не задан")

        # Ключ идемпотентности передаётся самой системе: без него повтор,
        # переживший перезапуск нашего процесса, создал бы вторую операцию —
        # наш учёт ключей живёт в памяти и перезапуск не переживает.
        operation_id = self._client.submit_meter_reading(
            payload["account_number"],
            payload["meter_serial"],
            payload["value"],
            payload["period"],
            key,
        )
        return WriteResult(
            operation_id=str(operation_id or uuid.uuid4().hex[:12]),
            system=self.system,
            idempotency_key=key,
            payload=payload,
        )
