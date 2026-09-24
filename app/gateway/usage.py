"""События использования LLM Gateway — «Billing Callback» диаграммы C4_L3_LLM.

ЧТО ЭТО И ЗАЧЕМ. На MVP каждый вызов модели пишется в ClickHouse:
токены, модель, latency, исход — для аналитики расхода, прогноза OPEX и
накопления против триггера №2 ADR-200. На прототипе **ClickHouse не развёрнут**,
и его роль занимает таблица `usage_events` в Postgres. Prometheus при этом
остаётся: он для наблюдаемости в реальном времени (дашборд), а эта
таблица — фактовая, для произвольной нарезки (по абоненту, типу, модели), чего
счётчики Prometheus без взрыва кардинальности не дают.

ЗАПИСЬ — BEST-EFFORT. Телеметрия не имеет права уронить ответ абоненту: любой
отказ хранилища здесь — предупреждение в журнал, не исключение. Тот же принцип,
что у `SessionStore.save`.

СИНХРОННАЯ ЗАПИСЬ НА ПРОТОТИПЕ. `record` выполняет `INSERT` в том же потоке,
**после** того как ответ абоненту собран целиком, поэтому во время до первого
куска не входит. На MVP это станет асинхронной пачкой (callback в ClickHouse) —
шов проходит по этому классу, а не по вызывающему коду.

ПЕРСОНАЛЬНЫХ ДАННЫХ ЗДЕСЬ НЕТ. `subscriber_id` / `session_id` — идентификаторы,
как в `inquiries` и в аудите. `pii_entities` — только типы найденных сущностей
(из `PiiReport`, где значений не бывает по построению).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["SCHEMA_PATH", "UsageEvent", "UsageStore"]

logger = logging.getLogger(__name__)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "postgres"
    / "init"
    / "03_usage_events.sql"
)
"""Единственное место, где написана схема — как у `app/inquiries/store.py`."""


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Один вызов LLM Gateway."""

    trace_id: str
    subscriber_id: str
    session_id: str
    channel: str
    provider_alias: str
    outcome: str
    """`ok` | `blocked` | `error` — то же, что метка `assistant_responses_total`."""

    model: str = ""
    inquiry_type: str | None = None
    topic: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: int | None = None
    total_ms: int | None = None
    finish_reason: str = "stop"
    guardrail_reason: str | None = None
    pii_detected: bool = False
    pii_entities: list[str] = field(default_factory=list)


class UsageStore:
    """Запись событий использования в Postgres (роль ClickHouse на прототипе).

    :param connect: как получить соединение (функция, а не готовое соединение —
        сервис живёт дольше любого соединения). `None` — запись выключена, вызовы
        `record` молча ничего не делают.
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

    def record(self, event: UsageEvent) -> None:
        """Записать событие. Отказ хранилища — предупреждение, не исключение.

        Телеметрия не должна ломать ответ абоненту: если Postgres недоступен,
        теряется одна строка аналитики, а не запрос.
        """
        if self._connect is None:
            return
        try:
            self.ensure_schema()
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO usage_events (
                            trace_id, subscriber_id, session_id, channel,
                            inquiry_type, topic, provider_alias, model,
                            prompt_tokens, completion_tokens, ttft_ms, total_ms,
                            outcome, finish_reason, guardrail_reason,
                            pii_detected, pii_entities
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s)
                        """,
                        (
                            event.trace_id,
                            event.subscriber_id,
                            event.session_id,
                            event.channel,
                            event.inquiry_type,
                            event.topic,
                            event.provider_alias,
                            event.model,
                            event.prompt_tokens,
                            event.completion_tokens,
                            event.ttft_ms,
                            event.total_ms,
                            event.outcome,
                            event.finish_reason,
                            event.guardrail_reason,
                            event.pii_detected,
                            json.dumps(event.pii_entities, ensure_ascii=False),
                        ),
                    )
                conn.commit()
        except Exception as exc:  # noqa: BLE001 — телеметрия не роняет ответ
            logger.warning("событие использования не записано (%s)", exc)
