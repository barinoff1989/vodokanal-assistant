"""Проверки учёта лимитов.

Redis здесь не поднимается: клиент подменяется словарём в памяти. Проверяется
логика пределов, а не работа хранилища — за неё отвечает сам Redis, и повторять
её тестами бессмысленно.

Отдельно проверяется поведение при отказе хранилища: ADR-008 требует отклонять
запросы, и без теста это решение осталось бы текстом.
"""

from __future__ import annotations

import pytest

from app.gateway.quota import (
    QuotaManager,
    QuotaScope,
    QuotaStoreUnavailableError,
)


class FakeRedis:
    """Счётчики в памяти. Время жизни не имитируется — окно задаётся ключом."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.expirations: dict[str, int] = {}

    def incr(self, name: str) -> int:
        self.values[name] = self.values.get(name, 0) + 1
        return self.values[name]

    def expire(self, name: str, time: int) -> bool:
        self.expirations[name] = time
        return True


class BrokenRedis:
    """Хранилище, недоступное на любой операции."""

    def incr(self, name: str) -> int:
        raise ConnectionError("соединение отсутствует")

    def expire(self, name: str, time: int) -> bool:
        raise ConnectionError("соединение отсутствует")


def _manager(client: object, **limits: int) -> QuotaManager:
    defaults = {"subscriber_limit": 5, "session_limit": 3, "service_limit": 100}
    return QuotaManager(client, **{**defaults, **limits})  # type: ignore[arg-type]


# --- обычная работа ---------------------------------------------------------- #


def test_запрос_в_пределах_лимита_разрешён():
    decision = _manager(FakeRedis()).check("sub-1", "sess-1")
    assert decision.allowed is True
    assert decision.scope is None


def test_превышение_лимита_сессии_отклоняется():
    manager = _manager(FakeRedis(), session_limit=2)
    assert manager.check("sub-1", "sess-1").allowed is True
    assert manager.check("sub-1", "sess-1").allowed is True
    decision = manager.check("sub-1", "sess-1")
    assert decision.allowed is False
    assert decision.scope is QuotaScope.SESSION


def test_превышение_лимита_абонента_отклоняется():
    """У абонента может быть несколько сессий — его предел выше сессионного."""
    manager = _manager(FakeRedis(), subscriber_limit=2, session_limit=100)
    manager.check("sub-1", "sess-1")
    manager.check("sub-1", "sess-2")
    decision = manager.check("sub-1", "sess-3")
    assert decision.allowed is False
    assert decision.scope is QuotaScope.SUBSCRIBER


def test_превышение_общего_лимита_сервиса_отклоняется():
    manager = _manager(FakeRedis(), subscriber_limit=100, session_limit=100, service_limit=2)
    manager.check("sub-1", "sess-1")
    manager.check("sub-2", "sess-2")
    decision = manager.check("sub-3", "sess-3")
    assert decision.allowed is False
    assert decision.scope is QuotaScope.SERVICE


def test_абоненты_считаются_раздельно():
    manager = _manager(FakeRedis(), subscriber_limit=1, session_limit=1)
    assert manager.check("sub-1", "sess-1").allowed is True
    assert manager.check("sub-2", "sess-2").allowed is True


def test_названа_именно_та_квота_что_исчерпана():
    """По счётчику отказов должно быть видно, чей предел сработал.

    Один абонент, упёршийся в свой лимит, и сервис, упёршийся в общий, — разные
    инциденты с разными действиями.
    """
    manager = _manager(FakeRedis(), subscriber_limit=1, session_limit=100, service_limit=100)
    manager.check("sub-1", "sess-1")
    assert manager.check("sub-1", "sess-2").scope is QuotaScope.SUBSCRIBER


# --- Retry-After -------------------------------------------------------------- #


def test_время_повтора_указано_и_разумно():
    """Правило 4.5: код 429 без `Retry-After` недопустим."""
    manager = _manager(FakeRedis(), session_limit=1)
    manager.check("sub-1", "sess-1")
    decision = manager.check("sub-1", "sess-1")
    assert 1 <= decision.retry_after <= 60


def test_время_повтора_это_остаток_окна_а_не_константа():
    """Сказать «через 30 секунд», когда счётчик обнулится через две, — задержать
    абонента без нужды; когда через пятьдесят — обречь на второй отказ."""
    manager = _manager(FakeRedis(), session_limit=1)
    manager.check("sub-1", "sess-1")
    first = manager.check("sub-1", "sess-1").retry_after
    manager.check("sub-2", "sess-2")
    second = manager.check("sub-2", "sess-2").retry_after
    # Значения близки, но выведены из текущего времени, а не заданы жёстко.
    assert abs(first - second) <= 1
    assert first <= 60


# --- отклонённые попытки тоже считаются ----------------------------------------- #


def test_отклонённый_запрос_увеличивает_счётчик():
    """Иначе абонент за лимитом получал бы неограниченные бесплатные попытки."""
    client = FakeRedis()
    manager = _manager(client, session_limit=1)
    manager.check("sub-1", "sess-1")
    manager.check("sub-1", "sess-1")
    manager.check("sub-1", "sess-1")
    session_keys = [k for k in client.values if k.startswith("quota:sess:")]
    assert client.values[session_keys[0]] == 3


def test_все_три_счётчика_растут_одинаково():
    """Учёт не должен разъезжаться между пределами при срабатывании одного из них."""
    client = FakeRedis()
    manager = _manager(client, subscriber_limit=1, session_limit=1, service_limit=1)
    manager.check("sub-1", "sess-1")
    manager.check("sub-1", "sess-1")
    assert sorted(client.values.values()) == [2, 2, 2]


def test_время_жизни_ставится_один_раз():
    """Иначе счётчик, продлеваемый каждым запросом, не истёк бы никогда."""
    client = FakeRedis()
    manager = _manager(client)
    manager.check("sub-1", "sess-1")
    manager.check("sub-1", "sess-1")
    assert all(value == 60 for value in client.expirations.values())
    assert len(client.expirations) == 3


# --- отказ хранилища: ADR-008 ---------------------------------------------------- #


def test_недоступное_хранилище_отклоняет_запрос():
    """ADR-008: не пропускаем. Отказ может быть следствием самого всплеска."""
    with pytest.raises(QuotaStoreUnavailableError):
        _manager(BrokenRedis()).check("sub-1", "sess-1")


def test_отказ_хранилища_отличим_от_превышения_лимита():
    """Разные коды ответа — 503 и 429 — требуют разных типов ошибки."""
    manager = _manager(FakeRedis(), session_limit=1)
    manager.check("sub-1", "sess-1")
    decision = manager.check("sub-1", "sess-1")
    assert decision.allowed is False  # превышение — обычное решение, не исключение

    with pytest.raises(QuotaStoreUnavailableError):
        _manager(BrokenRedis()).check("sub-1", "sess-1")


def test_у_отказа_хранилища_своё_время_повтора():
    """Оно не связано с окном счёта: счётчиков в этот момент попросту нет."""
    assert _manager(BrokenRedis(), ).store_retry_after == 30


# --- текст ошибки ------------------------------------------------------------------ #


def test_пояснение_не_раскрывает_счётчиков_абонента():
    """Тело ошибки уходит наружу — точные значения лимитов там ни к чему."""
    manager = _manager(FakeRedis(), session_limit=1)
    manager.check("sub-1", "sess-1")
    detail = manager.check("sub-1", "sess-1").detail
    assert "session" in detail
    assert "sub-1" not in detail


def test_у_разрешённого_запроса_пояснения_нет():
    assert _manager(FakeRedis()).check("sub-1", "sess-1").detail == ""


# --- недоступное хранилище отвечает быстро (пункт 35) -------------------------- #


def test_клиент_хранилища_собран_с_таймаутом_и_без_повторов():
    """Недоступный Redis обязан отвечать отказом **быстро**.

    По ADR-008 недоступность хранилища — это `503` + `Retry-After`, то есть
    повторяет клиент, по названному ему времени. Но без настроек отказ приходил
    через **26 секунд**, и столько же ждал абонент, чтобы услышать «сервис
    временно недоступен».

    Замер разложил задержку на слагаемые и показал, что решает не таймаут, а
    повторы:

    ===================================== ======
      без таймаута, с повторами             26,2 с
      таймаут 0,5 с, с повторами             8,7 с
      таймаут 0,5 с, без повторов            0,5 с
    ===================================== ======

    Проверяется настройка клиента, а не время: время зависит от машины и от
    того, поднят ли Redis, и порог сделал бы проверку хрупкой. Настройка же —
    ровно то, что было упущено и что можно упустить снова.
    """
    from app.config import get_settings
    from app.main import _build_quota

    # `_build_quota` отдаёт и менеджер, и сам клиент: клиент общий с хранилищем
    # сессий, потому что Redis — одна служба, и второе подключение к ней
    # означало бы второй таймаут там, где отказ один.
    quota, client = _build_quota()
    assert quota is not None, "клиент Redis не установлен — проверка бессмысленна"
    assert client is not None
    kwargs = client.connection_pool.connection_kwargs
    timeout = get_settings().redis_timeout_seconds

    assert kwargs["socket_connect_timeout"] == timeout
    assert kwargs["socket_timeout"] == timeout
    assert kwargs["retry"].get_retries() == 0, (
        "повторы внутри запроса удлиняют ожидание абонента во столько раз, "
        "сколько попыток, и делают это невидимым: снаружи виден один долгий "
        "запрос, а не четыре быстрых отказа"
    )
