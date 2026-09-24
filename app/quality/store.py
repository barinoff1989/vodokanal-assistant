"""Запись качества — два зерна: прогон (`quality_reports`) и ответ (`quality_assessments`).

Итог оценки судьёй пишется рядом с телеметрией использования: таблицы лежат в
базе `telemetry` в Postgres (той же, что `usage_events`).

* `QualityStore` → `quality_reports`. Зерно — прогон: ночной прогон
  (`scripts/run_quality_eval.py`) пишет средние по эталонному набору плюс долю
  ответов выше стартовых порогов. Сравнивают между собой именно прогоны (базовая
  линия и регрессия).
* `AssessmentStore` → `quality_assessments`. Зерно — ответ: судья оценивает
  конкретный живой ответ после того, как поток дошёл до абонента (async-часть
  Guardrails, C4_L3_LLM). Для разреза качества в реальном времени.

Запись best-effort — как `UsageStore`: качество не должно ронять ни прогон, ни
ответ абоненту.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ASSESSMENT_SCHEMA_PATH",
    "SCHEMA_PATH",
    "AssessmentStore",
    "QualityAssessment",
    "QualityReport",
    "QualityStore",
]

logger = logging.getLogger(__name__)

_INIT = Path(__file__).resolve().parents[2] / "docker" / "postgres" / "init"

SCHEMA_PATH = _INIT / "04_quality_reports.sql"
"""Единственное место, где написана схема — как у `app/gateway/usage.py`."""

ASSESSMENT_SCHEMA_PATH = _INIT / "05_quality_assessments.sql"


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
    """Запись отчётов о качестве в Postgres.

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


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Оценка судьёй одного живого ответа."""

    trace_id: str
    subscriber_id: str
    session_id: str
    outcome: str
    """`scored` | `unavailable` | `conflict`."""

    topic: str | None = None
    inquiry_type: str | None = None
    judge_model: str = ""
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    provisional: bool = True
    detail: str | None = None


class AssessmentStore:
    """Запись пооответных оценок качества в Postgres.

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
        """Применить схему, если её ещё нет. Идемпотентно."""
        if self._connect is None or self._schema_applied:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ASSESSMENT_SCHEMA_PATH.read_text(encoding="utf-8"))
            conn.commit()
        self._schema_applied = True

    def record(self, assessment: QualityAssessment) -> None:
        """Записать оценку. Отказ хранилища — предупреждение, не исключение."""
        if self._connect is None:
            return
        try:
            self.ensure_schema()
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO quality_assessments (
                            trace_id, subscriber_id, session_id, topic,
                            inquiry_type, judge_model, faithfulness,
                            answer_relevancy, provisional, outcome, detail
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            assessment.trace_id,
                            assessment.subscriber_id,
                            assessment.session_id,
                            assessment.topic,
                            assessment.inquiry_type,
                            assessment.judge_model,
                            assessment.faithfulness,
                            assessment.answer_relevancy,
                            assessment.provisional,
                            assessment.outcome,
                            assessment.detail,
                        ),
                    )
                conn.commit()
        except Exception as exc:  # noqa: BLE001 — оценка не роняет ответ абоненту
            logger.warning("оценка качества ответа не записана (%s)", exc)
