"""Запись отчётов о качестве — `quality_report` раздела 35.1.

На MVP итог оценки судьёй уходит в ClickHouse рядом с телеметрией использования;
на прототипе ClickHouse не развёрнут, и его роль занимает таблица
`quality_reports` в базе `telemetry` (той же, что `usage_events`).

Зерно — прогон, не ответ: ночной прогон пишет средние по эталонному набору плюс
долю ответов выше стартовых порогов. Сравнивают между собой именно прогоны
(базовая линия и регрессия, раздел 36.3).

Запись best-effort — как `UsageStore`: отчёт качества не должен ронять прогон.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["SCHEMA_PATH", "QualityReport", "QualityStore"]

logger = logging.getLogger(__name__)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "postgres"
    / "init"
    / "04_quality_reports.sql"
)
"""Единственное место, где написана схема — как у `app/gateway/usage.py`."""


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Итог одного прогона судьи по эталонному набору."""

    judge_alias: str
    judge_model: str = ""
    answering_alias: str = ""
    golden_set_version: str = ""
    questions_total: int = 0
    questions_scored: int = 0
    faithfulness_avg: float | None = None
    answer_relevancy_avg: float | None = None
    pass_rate: float | None = None
    skipped: bool = False
    skip_reason: str | None = None


class QualityStore:
    """Запись отчётов о качестве в Postgres (роль ClickHouse на прототипе).

    :param connect: как получить соединение (функция, а не готовое соединение).
        `None` — запись выключена, `record` молча ничего не делает.
    """

    def __init__(self, connect: Any | None) -> None:
        self._connect = connect
        self._schema_applied = False

    @property
    def enabled(self) -> bool:
        return self._connect is not None

    def ensure_schema(self) -> None:
        """Применить схему, если её ещё нет. Идемпотентно (`CREATE ... IF NOT EXISTS`)."""
        if self._connect is None or self._schema_applied:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            conn.commit()
        self._schema_applied = True

    def record(self, report: QualityReport) -> None:
        """Записать отчёт. Отказ хранилища — предупреждение, не исключение."""
        if self._connect is None:
            return
        try:
            self.ensure_schema()
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO quality_reports (
                            golden_set_version, judge_alias, judge_model,
                            answering_alias, questions_total, questions_scored,
                            faithfulness_avg, answer_relevancy_avg, pass_rate,
                            skipped, skip_reason
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            report.golden_set_version,
                            report.judge_alias,
                            report.judge_model,
                            report.answering_alias,
                            report.questions_total,
                            report.questions_scored,
                            report.faithfulness_avg,
                            report.answer_relevancy_avg,
                            report.pass_rate,
                            report.skipped,
                            report.skip_reason,
                        ),
                    )
                conn.commit()
        except Exception as exc:  # noqa: BLE001 — отчёт качества не роняет прогон
            logger.warning("отчёт о качестве не записан (%s)", exc)
