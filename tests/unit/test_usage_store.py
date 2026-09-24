"""Проверки хранилища событий использования (Postgres).

Быстрые проверки на дубле соединения: сторожат состав запроса и то, что отказ
хранилища не пробрасывается наружу. База для этого не нужна. Один живой прогон
против Postgres помечен `live` — он проверяет применение схемы.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from app.gateway.usage import SCHEMA_PATH, UsageEvent, UsageStore

EVENT = UsageEvent(
    trace_id="trace-1",
    subscriber_id="2100202213",
    session_id="sess-1",
    channel="lk_web",
    provider_alias="local-test",
    outcome="ok",
    model="local-test-model",
    inquiry_type="meter_verification",
    topic="general",
    prompt_tokens=120,
    completion_tokens=40,
    ttft_ms=210,
    total_ms=1400,
    pii_detected=True,
    pii_entities=["PHONE_NUMBER"],
)


class FakeCursor:
    def __init__(self, owner: FakeConnection) -> None:
        self._owner = owner

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self._owner.executed.append((sql, params))


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.commits = 0

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


class BrokenConnection:
    def __enter__(self) -> BrokenConnection:
        raise ConnectionError("Postgres недоступен")

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def conn() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def store(conn: FakeConnection) -> UsageStore:
    return UsageStore(lambda: conn)


def test_выключенное_хранилище_молча_ничего_не_делает():
    UsageStore(None).record(EVENT)  # не должно бросить


def test_событие_записывается_одним_insert(store: UsageStore, conn: FakeConnection):
    store.ensure_schema()
    commits_before = conn.commits
    store.record(EVENT)
    inserts = [sql for sql, _ in conn.executed if "INSERT INTO usage_events" in sql]
    assert len(inserts) == 1
    assert conn.commits == commits_before + 1  # схема уже применена — только вставка


def test_в_запрос_идут_все_поля_события(store: UsageStore, conn: FakeConnection):
    store.record(EVENT)
    _, params = next(
        row for row in conn.executed if "INSERT INTO usage_events" in row[0]
    )
    assert params is not None
    assert EVENT.trace_id in params
    assert EVENT.subscriber_id in params
    assert EVENT.model in params
    assert EVENT.prompt_tokens in params
    assert EVENT.ttft_ms in params
    # pii_entities уходит как JSON-строка, не как список
    assert json.dumps(EVENT.pii_entities, ensure_ascii=False) in params


def test_отказ_хранилища_не_пробрасывается(caplog: pytest.LogCaptureFixture):
    """Телеметрия не имеет права уронить ответ абоненту."""
    store = UsageStore(BrokenConnection)
    store.record(EVENT)  # не должно бросить
    assert "не записано" in caplog.text.lower() or caplog.records


@pytest.mark.live
def test_схема_применяется_к_настоящей_базе():
    dsn = os.getenv("TELEMETRY_DSN")
    if not dsn:
        pytest.skip("TELEMETRY_DSN не задан — живой прогон пропущен")
    import psycopg

    store = UsageStore(lambda: psycopg.connect(dsn))
    store.ensure_schema()
    store.record(EVENT)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM usage_events WHERE trace_id = %s", ("trace-1",))
        assert cur.fetchone()[0] >= 1


def test_путь_схемы_существует():
    assert SCHEMA_PATH.exists()
    assert "usage_events" in SCHEMA_PATH.read_text(encoding="utf-8")
