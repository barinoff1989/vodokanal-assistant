"""Проверки состояния диалога.

Главное, что здесь сторожится, — **две угрозы разом**. Состояние хранит ФИО,
лицевой счёт и черновик, и по нему же решается, можно ли выполнять запись.
Ошибка тут не роняет сервис: она отдаёт чужой черновик или
позволяет подтвердить чужую операцию.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.backend.sessions import KEY_PREFIX, SessionStore, SubscriberMismatch
from app.models import InquiryState, SessionState


class FakeRedis:
    """Дубль хранилища: помнит значения и последний срок жизни."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


class BrokenRedis:
    """Хранилище, которое отказало. Любой вызов — ошибка соединения."""

    def get(self, key: str) -> Any:
        raise ConnectionError("соединение отсутствует")

    def set(self, key: str, value: str, ex: int | None = None) -> Any:
        raise ConnectionError("соединение отсутствует")

    def delete(self, key: str) -> Any:
        raise ConnectionError("соединение отсутствует")


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def store(redis: FakeRedis) -> SessionStore:
    return SessionStore(redis, ttl_seconds=1800)


def test_первая_реплика_заводит_новое_состояние(store: SessionStore):
    session = store.load("sess-1", "sub-1")
    assert session.current_state is InquiryState.INIT
    assert session.draft_text is None


def test_состояние_переживает_реплику(store: SessionStore):
    """Ради этого модуль и написан: между «показали черновик» и «абонент
    подтвердил» проходит отдельный HTTP-запрос."""
    session = store.load("sess-1", "sub-1")
    session.current_state = InquiryState.AWAITING_CONFIRMATION
    session.draft_text = "Заявка на поверку прибора учёта"
    store.save(session)

    restored = store.load("sess-1", "sub-1")
    assert restored.current_state is InquiryState.AWAITING_CONFIRMATION
    assert restored.draft_text == "Заявка на поверку прибора учёта"
    assert restored.awaits_confirmation


def test_чужая_сессия_не_отдаётся(store: SessionStore):
    """Угрозы Spoofing и IDOR: подменив `session_id`, можно было бы прочитать
    чужой черновик с ФИО и лицевым счётом."""
    session = store.load("sess-1", "sub-1")
    session.draft_text = "Черновик первого абонента"
    store.save(session)

    with pytest.raises(SubscriberMismatch):
        store.load("sess-1", "sub-2")


def test_проверка_стоит_до_ответа_а_не_только_в_гейте_адаптера(store: SessionStore):
    """Гейт адаптера защищает путь записи. Но чужое состояние успело бы попасть
    в ответ абоненту раньше, чем дело дошло бы до записи."""
    session = store.load("sess-1", "sub-1")
    session.current_state = InquiryState.AWAITING_CONFIRMATION
    store.save(session)

    with pytest.raises(SubscriberMismatch):
        store.load("sess-1", "чужой")


def test_срок_жизни_проставляется(store: SessionStore, redis: FakeRedis):
    """Срок — единственный механизм удаления, и без него хранилище растёт
    неограниченно."""
    store.save(store.load("sess-1", "sub-1"))
    assert redis.expirations[f"{KEY_PREFIX}sess-1"] == 1800


def test_срок_отсчитывается_от_последней_реплики(store: SessionStore, redis: FakeRedis):
    """Иначе абонент, который думает над черновиком дольше срока, терял бы его
    на полуслове."""
    session = store.load("sess-1", "sub-1")
    store.save(session)
    redis.expirations.clear()
    store.save(session)
    assert redis.expirations[f"{KEY_PREFIX}sess-1"] == 1800


def test_завершённый_диалог_забывается(store: SessionStore, redis: FakeRedis):
    """Срок жизни — верхняя граница, а не замена уборке: держать ПДн дольше,
    чем они нужны, незачем."""
    store.save(store.load("sess-1", "sub-1"))
    store.drop("sess-1")
    assert redis.values == {}


def test_отказ_хранилища_не_роняет_ответ():
    """Не `503`, в отличие от хранилища лимитов (ADR-400).

    Потеря состояния не открывает дыру, а закрывает путь: без состояния
    обращение никогда не окажется в ожидании подтверждения, и запись не
    выполнится. Правило подтверждения записи остаётся выполненным по построению, а вопрос про
    пени по-прежнему получает ответ.
    """
    store = SessionStore(BrokenRedis(), ttl_seconds=1800)
    session = store.load("sess-1", "sub-1")

    assert session.current_state is InquiryState.INIT
    store.save(session)  # не бросает
    store.drop("sess-1")  # тоже


def test_отказ_хранилища_не_даёт_подтвердить_запись():
    """Обратная сторона того же: подтверждать станет нечего, и это правильно."""
    store = SessionStore(BrokenRedis(), ttl_seconds=1800)
    saved = SessionState(session_id="sess-1", subscriber_id="sub-1")
    saved.current_state = InquiryState.AWAITING_CONFIRMATION
    store.save(saved)

    restored = store.load("sess-1", "sub-1")
    assert not restored.awaits_confirmation


def test_без_хранилища_вовсе_сервис_работает():
    """`None` — допустимое состояние: Redis может быть не установлен."""
    store = SessionStore(None, ttl_seconds=1800)
    session = store.load("sess-1", "sub-1")
    store.save(session)
    assert store.load("sess-1", "sub-1").current_state is InquiryState.INIT
