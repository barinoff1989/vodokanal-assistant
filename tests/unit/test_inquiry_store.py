"""Проверки хранилища обращений (прототип).

Быстрые проверки идут на дубле соединения: они сторожат состав запроса и
обработку повтора, и для этого база не нужна. Один живой прогон против
настоящего Postgres помечен `live` — он проверяет то, чего дубль проверить не
может: что схема применяется и что уникальность ключа держит **база**, а не наш
код перед вставкой.

Разделение не формальность. Дубль всегда согласен с тем, кто его писал: именно
поэтому в проекте шесть раз находились дефекты на стыке с внешним миром, которых
тесты не видели.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from app.inquiries.store import SCHEMA_PATH, InquiryStore

PAYLOAD = {
    "inquiry_type": "meter_verification",
    "subscriber_id": "2100707718",
    "session_id": "sess-1",
    "subject": "Заказать поверку",
    "body": "Необходима поверка счётчика горячей воды.",
    "slots": {"full_name": "Михайлова Т. А."},
}


class FakeCursor:
    def __init__(self, owner: FakeConnection) -> None:
        self._owner = owner
        self._result: list[Any] | None = None

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self._owner.executed.append((sql, params))
        if "INSERT" in sql:
            key = params[-1] if params else None
            if key in self._owner.rows:
                self._result = None  # конфликт: строка не создана
            else:
                self._owner.next_id += 1
                self._owner.rows[key] = self._owner.next_id
                self._result = [self._owner.next_id]
        elif "SELECT id" in sql:
            key = params[0] if params else None
            found = self._owner.rows.get(key)
            self._result = [found] if found is not None else None
        else:
            self._result = None

    def fetchone(self) -> list[Any] | None:
        return self._result


class FakeConnection:
    """Дубль соединения: помнит вставленные ключи и записанные запросы."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.rows: dict[Any, int] = {}
        self.next_id = 0
        self.commits = 0

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


@pytest.fixture
def conn() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def store(conn: FakeConnection) -> InquiryStore:
    return InquiryStore(lambda: conn)


def test_обращение_записывается_и_возвращает_номер(store: InquiryStore):
    assert store.create_inquiry(PAYLOAD, "key-1") == "1"


def test_повтор_по_ключу_не_создаёт_вторую_запись(store: InquiryStore, conn: FakeConnection):
    """Абонент нажал «подтверждаю» дважды — обращение должно остаться одно."""
    first = store.create_inquiry(PAYLOAD, "key-1")
    second = store.create_inquiry(PAYLOAD, "key-1")

    assert first == second
    assert len(conn.rows) == 1


def test_разные_ключи_создают_разные_обращения(store: InquiryStore, conn: FakeConnection):
    store.create_inquiry(PAYLOAD, "key-1")
    store.create_inquiry(PAYLOAD, "key-2")
    assert len(conn.rows) == 2


def test_уникальность_проверяет_база_а_не_код(store: InquiryStore, conn: FakeConnection):
    """Между проверкой «есть ли такой ключ» и вставкой помещается второй запрос.

    Поэтому в запросе обязан стоять `ON CONFLICT`, а не предварительный SELECT."""
    store.create_inquiry(PAYLOAD, "key-1")
    insert = next(sql for sql, _ in conn.executed if "INSERT" in sql)
    assert "ON CONFLICT (idempotency_key) DO NOTHING" in insert


def test_слоты_уходят_в_базу_текстом_json(store: InquiryStore, conn: FakeConnection):
    """Состав слотов зависит от типа обращения; колонка на каждый слот
    превратила бы таблицу в разреженную при первом же новом типе."""
    store.create_inquiry(PAYLOAD, "key-1")
    _, params = next((s, p) for s, p in conn.executed if "INSERT" in s)
    assert json.loads(params[5]) == {"full_name": "Михайлова Т. А."}


def test_схема_применяется_один_раз(store: InquiryStore, conn: FakeConnection):
    """Иначе за каждое обращение платили бы лишним обращением к базе."""
    store.create_inquiry(PAYLOAD, "key-1")
    store.create_inquiry(PAYLOAD, "key-2")
    assert sum(1 for sql, _ in conn.executed if "CREATE TABLE" in sql) == 1


def test_схема_лежит_одним_файлом_и_он_на_месте():
    """Postgres применяет её при создании тома, хранилище — на живой базе.

    Два места, где написана схема, разошлись бы при первой правке."""
    assert SCHEMA_PATH.exists()
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS inquiries" in text
    assert "idempotency_key TEXT         NOT NULL UNIQUE" in text


# --- живой прогон -------------------------------------------------------------- #

DSN = os.getenv(
    "INQUIRIES_DSN",
    "postgresql://vodokanal:vodokanal_local@localhost:5432/inquiries_stub",
)


@pytest.mark.live
def test_живой_postgres_держит_уникальность_ключа():
    """Проверяет то, чего дубль проверить не может.

    Дубль повторяет **наше** представление о поведении базы. Здесь уникальность
    обеспечивает сама база: если ограничения в схеме не окажется, дубль об этом
    не узнает, а этот прогон упадёт.
    """
    psycopg = pytest.importorskip("psycopg")
    store = InquiryStore(lambda: psycopg.connect(DSN))

    key = f"test-{os.urandom(6).hex()}"
    first = store.create_inquiry({**PAYLOAD, "subject": "Живой прогон"}, key)
    second = store.create_inquiry({**PAYLOAD, "subject": "Живой прогон"}, key)

    assert first == second

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM inquiries WHERE idempotency_key = %s", (key,))
        assert cur.fetchone()[0] == 1
        cur.execute("DELETE FROM inquiries WHERE idempotency_key = %s", (key,))
        conn.commit()
