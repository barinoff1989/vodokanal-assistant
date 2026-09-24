-- События использования LLM Gateway — одна строка на вызов модели.
--
-- ЧТО ЭТА ТАБЛИЦА ЕСТЬ. Роль «Billing Callback» диаграммы C4_L3_LLM: события
-- каждого вызова пишутся сюда, в Postgres (ClickHouse не нужен: объём —
-- ADR-200). Prometheus при этом остаётся:
-- он для наблюдаемости в реальном времени (дашборд), а usage_events
-- — фактовая таблица для произвольной нарезки (расход по абоненту/типу/модели,
-- прогноз OPEX, накопление против триггера №2 ADR-200), чего счётчики
-- Prometheus без взрыва кардинальности не дают.
--
-- ПЕРСОНАЛЬНЫХ ДАННЫХ ЗДЕСЬ НЕТ. subscriber_id и session_id — идентификаторы,
-- как в таблице inquiries и в аудит-логе. pii_entities — только ТИПЫ найденных
-- сущностей (из PiiReport, где значений не бывает по построению).
--
-- Файл один, применяется дважды — как 01_inquiries.sql: Postgres выполняет
-- docker/postgres/init только при создании тома, поэтому его же читает
-- app/gateway/usage.py при первом обращении.

CREATE TABLE IF NOT EXISTS usage_events (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    at                TIMESTAMPTZ NOT NULL DEFAULT now(),

    trace_id          TEXT        NOT NULL,
    subscriber_id     TEXT        NOT NULL,
    session_id        TEXT        NOT NULL,
    channel           TEXT        NOT NULL,

    -- Оси классификации (ADR-300). Код, а не отображаемое название.
    inquiry_type      TEXT,
    topic             TEXT,

    -- Псевдоним — что запросили; model — что фактически ответило. Расхождение
    -- и есть срабатывание запасного провайдера (ADR-200).
    provider_alias    TEXT        NOT NULL,
    model             TEXT        NOT NULL DEFAULT '',

    -- Раздельно: цена входных и выходных токенов у провайдера разная.
    prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
    completion_tokens INTEGER     NOT NULL DEFAULT 0,

    ttft_ms           INTEGER,
    total_ms          INTEGER,

    -- ok | blocked | error — то же, что метка outcome у assistant_responses_total.
    outcome           TEXT        NOT NULL,
    finish_reason     TEXT        NOT NULL DEFAULT 'stop',

    -- Причина блокировки охранителями: pii_leak | toxicity | policy | prompt_leak.
    guardrail_reason  TEXT,

    pii_detected      BOOLEAN     NOT NULL DEFAULT false,
    pii_entities      JSONB       NOT NULL DEFAULT '[]'::JSONB
);

-- Отчёты идут «за период» и «по абоненту», всегда от свежих.
CREATE INDEX IF NOT EXISTS usage_events_at_idx
    ON usage_events (at DESC);
CREATE INDEX IF NOT EXISTS usage_events_subscriber_at_idx
    ON usage_events (subscriber_id, at DESC);
