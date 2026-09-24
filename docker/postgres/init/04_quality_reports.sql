-- Отчёты о качестве ответа — одна строка на прогон судьи по эталонному набору.
--
-- ЧТО ЭТА ТАБЛИЦА ЕСТЬ. `quality_report`: итог оценки судьёй
-- (faithfulness, answer_relevancy) пишется рядом с телеметрией
-- использования. На MVP это ClickHouse, на прототипе ClickHouse не развёрнут —
-- его место занимает эта таблица в базе `telemetry`.
--
-- ЗЕРНО — ПРОГОН, НЕ ОТВЕТ. Ночной прогон (`scripts/run_quality_eval.py`)
-- прогоняет весь эталонный набор и пишет СРЕДНИЕ по прогону плюс долю ответов
-- выше стартовых порогов. Пооответная разбивка не хранится: набор небольшой,
-- прогон воспроизводим, а сравнивают между собой именно прогоны (базовая линия
-- и регрессия).
--
-- ПРОПУСК ФИКСИРУЕТСЯ. Если судья совпал с отвечающей моделью (failover
-- генерации), оценка не выполняется — пишется строка с skipped = true и
-- причиной, а не отсутствие записи (ADR-300).
--
-- ПЕРСОНАЛЬНЫХ ДАННЫХ ЗДЕСЬ НЕТ: только агрегаты и имена моделей.
--
-- Файл применяется дважды, как 03_usage_events.sql: контейнером инициализации
-- и `app/quality/store.py` при первом обращении.

CREATE TABLE IF NOT EXISTS quality_reports (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    at                  TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Версия эталонного набора: сравнивать прогоны можно только на одной
    -- версии набора.
    golden_set_version  TEXT        NOT NULL DEFAULT '',

    -- Псевдоним и имя фактической модели-судьи и отвечающей модели.
    -- Версия судьи фиксируется — иначе оценки прогонов несравнимы.
    judge_alias         TEXT        NOT NULL,
    judge_model         TEXT        NOT NULL DEFAULT '',
    answering_alias     TEXT        NOT NULL DEFAULT '',

    -- Сколько вопросов в прогоне и по скольким судья вернул оценку.
    questions_total     INTEGER     NOT NULL DEFAULT 0,
    questions_scored    INTEGER     NOT NULL DEFAULT 0,

    -- Средние по прогону, [0, 1]. NULL, если ни одного ответа не оценено.
    faithfulness_avg    DOUBLE PRECISION,
    answer_relevancy_avg DOUBLE PRECISION,

    -- Доля ответов, прошедших ОБА стартовых порога, [0, 1].
    pass_rate           DOUBLE PRECISION,

    -- Прогон не выполнялся: судья совпал с отвечающей моделью (failover).
    skipped             BOOLEAN     NOT NULL DEFAULT false,
    skip_reason         TEXT
);

-- Отчёты смотрят «за период», всегда от свежих; тренд метрики — по времени.
CREATE INDEX IF NOT EXISTS quality_reports_at_idx
    ON quality_reports (at DESC);
