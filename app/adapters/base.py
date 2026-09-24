"""Общая основа адаптеров записи: гейты, которые нельзя обойти.

Три требования проекта повторяются на каждом пути записи, и повторять их кодом
в каждом адаптере значило бы однажды забыть:

* **Подтверждение абонента** — запись возможна ровно из состояния
  ожидания подтверждения. Проверка стоит здесь, а не в оркестраторе: адаптер
  можно вызвать и напрямую, и тогда единственная проверка оказалась бы в стороне.
* **Привязка абонента к сессии** — закрывает угрозы Spoofing и IDOR.
  Расхождение Р4 отмечало, что проверка «показана, но не реализована нигде»;
  здесь она реализована.
* **Идемпотентность** — повтор с тем же ключом не создаёт вторую операцию.

ОДНА ВНЕШНЯЯ СИСТЕМА НА ПУТЬ (ADR-100).
Каждый адаптер обращается ровно к одной чужой системе. Это не стилистика: пока
записей было две, требовалось согласование при частичном отказе (расхождение Р6),
непроверяемое на прототипе. Сузив область, проект убрал не проблему, а её
причину. Тест сторожит, что адаптер не приобрёл вторую систему.

ТОЛЬКО ЧЕРЕЗ API ВЛАДЕЛЬЦА (ADR-400).
Ни один адаптер не обращается к базе чужой системы напрямую — только через её
интерфейс. Схема чужой базы не контракт и меняется без предупреждения.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from app.models import InquiryState, SessionState

__all__ = [
    "ConfirmationRequiredError",
    "SubscriberMismatchError",
    "WriteAdapter",
    "WriteResult",
]


class ConfirmationRequiredError(RuntimeError):
    """Запись запрошена вне состояния ожидания подтверждения."""

    def __init__(self, state: InquiryState) -> None:
        super().__init__(
            f"запись невозможна из состояния {state.value!r}: "
            "без подтверждения абонента запись запрещена"
        )
        self.state = state


class SubscriberMismatchError(RuntimeError):
    """Операция запрошена не над тем абонентом, чья сессия аутентифицирована.

    Это и есть угроза IDOR из разбора безопасности: подменив идентификатор в
    запросе, можно было бы выполнить запись по чужому лицевому счёту.
    """

    def __init__(self, session_subscriber: str, requested: str) -> None:
        # В сообщение не попадает ни один из идентификаторов: текст ошибки может
        # уйти в журнал или наружу, а сопоставление двух чужих идентификаторов —
        # уже сведение данных.
        super().__init__("абонент операции не совпадает с абонентом сессии")
        self.session_subscriber = session_subscriber
        self.requested = requested


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Итог операции записи."""

    operation_id: str
    system: str
    idempotency_key: str
    repeated: bool = False
    """Повтор с уже виденным ключом. Вторая операция при этом не создавалась."""

    payload: dict[str, Any] = field(default_factory=dict)


class WriteAdapter:
    """Основа адаптера: гейты и учёт ключей идемпотентности.

    :param system: имя внешней системы. Одно на адаптер — см. ADR-100.
    """

    system: str = "unknown"

    def __init__(self) -> None:
        # Ключи хранятся в памяти процесса: на прототипе этого достаточно, в
        # рабочем контуре учёт переедет в общее хранилище — иначе повтор, попавший
        # на другую реплику, создал бы вторую операцию.
        self._seen: dict[str, WriteResult] = {}

    # -- гейты ------------------------------------------------------------- #

    @staticmethod
    def ensure_confirmed(session: SessionState) -> None:
        """Убедиться, что абонент подтвердил операцию.

        :raises ConfirmationRequiredError: состояние не то.
        """
        if session.current_state is not InquiryState.AWAITING_CONFIRMATION:
            raise ConfirmationRequiredError(session.current_state)

    @staticmethod
    def ensure_same_subscriber(session: SessionState, subscriber_id: str) -> None:
        """Убедиться, что операция запрошена над абонентом этой сессии.

        :raises SubscriberMismatchError: идентификаторы не совпадают.
        """
        if session.subscriber_id != subscriber_id:
            raise SubscriberMismatchError(session.subscriber_id, subscriber_id)

    def idempotency_key(self, session: SessionState, *parts: str) -> str:
        """Собрать ключ операции.

        В ключ входит сессия и существо операции, но **не время**: иначе повтор
        того же действия получал бы новый ключ, и защита не работала бы вовсе.
        """
        material = "|".join([self.system, session.session_id, *parts])
        return hashlib.sha256(material.encode()).hexdigest()[:32]

    # -- выполнение --------------------------------------------------------- #

    def perform(
        self,
        session: SessionState,
        subscriber_id: str,
        *,
        key_parts: tuple[str, ...],
        payload: dict[str, Any],
    ) -> WriteResult:
        """Пройти гейты и выполнить запись, если она ещё не выполнялась.

        Порядок проверок: подтверждение, затем принадлежность абоненту, затем
        идемпотентность. Первые две дешевле и отсекают заведомо недопустимое до
        любых обращений наружу.
        """
        self.ensure_confirmed(session)
        self.ensure_same_subscriber(session, subscriber_id)

        key = self.idempotency_key(session, *key_parts)
        if key in self._seen:
            previous = self._seen[key]
            session.record(
                self.system,
                "write:repeated",
                system=self.system,
                idempotency_key=key,
                operation_id=previous.operation_id,
            )
            return WriteResult(
                operation_id=previous.operation_id,
                system=self.system,
                idempotency_key=key,
                repeated=True,
                payload=previous.payload,
            )

        result = self._call_system(payload, key)
        self._seen[key] = result
        session.record(
            self.system,
            "write:performed",
            system=self.system,
            idempotency_key=key,
            operation_id=result.operation_id,
        )
        return result

    def _call_system(self, payload: dict[str, Any], key: str) -> WriteResult:
        """Обратиться к внешней системе. Переопределяется наследником."""
        raise NotImplementedError
