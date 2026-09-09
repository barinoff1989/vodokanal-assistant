"""Проверки хранилища отчётов о качестве (роль ClickHouse на прототипе).

Дубль соединения: сторожат состав запроса и то, что отказ хранилища не роняет
прогон. Один живой прогон против Postgres помечен `live`.
"""

from __future__ import annotations

import os

import pytest

from app.quality.store import SCHEMA_PATH, QualityReport, QualityStore

REPORT = QualityReport(
    judge_alias="local-test",
    judge_model="ollama/qwen2.5:7b",
    answering_alias="yandexgpt",
    golden_set_version="abc123def456",
    questions_total=30,
    questions_scored=28,
    faithfulness_avg=0.91,
    answer_relevancy_avg=0.84,
    pass_rate=0.75,
)

SKIPPED = QualityReport(
    judge_alias="local-test",
    answering_alias="local-test",
    questions_total=30,
    skipped=True,
    skip_reason="судья и отвечающая модель совпали (local-test)",
)


class FakeCursor:
    def __init__(self, owner: FakeConnection) -> None:
        self._owner = owner

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> None:
        self._owner.executed.append((sql, params))


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object]] = []
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
def store(conn: FakeConnection) -> QualityStore:
    return QualityStore(lambda: conn)


def test_выключенное_хранилище_молча_ничего_не_делает():
    QualityStore(None).record(REPORT)


def test_отчёт_записывается_одним_insert(store: QualityStore, conn: FakeConnection):
    store.ensure_schema()
    commits_before = conn.commits
    store.record(REPORT)
    inserts = [sql for sql, _ in conn.executed if "INSERT INTO quality_reports" in sql]
    assert len(inserts) == 1
    assert conn.commits == commits_before + 1


def test_в_запрос_идут_средние_и_версия(store: QualityStore, conn: FakeConnection):
    store.record(REPORT)
    _, params = next(
        row for row in conn.executed if "INSERT INTO quality_reports" in row[0]
    )
    assert params is not None
    assert REPORT.judge_alias in params
    assert REPORT.judge_model in params
    assert REPORT.golden_set_version in params
    assert REPORT.faithfulness_avg in params
    assert REPORT.pass_rate in params


def test_пропуск_пишется_строкой(store: QualityStore, conn: FakeConnection):
    """Пропуск из-за failover — это строка со skipped=true, а не отсутствие записи."""
    store.record(SKIPPED)
    _, params = next(
        row for row in conn.executed if "INSERT INTO quality_reports" in row[0]
    )
    assert True in params
    assert SKIPPED.skip_reason in params


def test_отказ_хранилища_не_пробрасывается(caplog: pytest.LogCaptureFixture):
    QualityStore(BrokenConnection).record(REPORT)
    assert "не записан" in caplog.text.lower() or caplog.records


def test_путь_схемы_существует():
    assert SCHEMA_PATH.exists()
    assert "quality_reports" in SCHEMA_PATH.read_text(encoding="utf-8")


@pytest.mark.live
def test_схема_применяется_к_настоящей_базе():
    dsn = os.getenv("TELEMETRY_DSN")
    if not dsn:
        pytest.skip("TELEMETRY_DSN не задан — живой прогон пропущен")
    import psycopg

    store = QualityStore(lambda: psycopg.connect(dsn))
    store.ensure_schema()
    store.record(REPORT)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM quality_reports WHERE golden_set_version = %s",
            (REPORT.golden_set_version,),
        )
        assert cur.fetchone()[0] >= 1
