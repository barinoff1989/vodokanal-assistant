-- Оценки качества отдельных ответов — async-часть Guardrails диаграммы C4_L3_LLM.
--
-- ЧЕМ ОТЛИЧАЕТСЯ ОТ quality_reports. Та таблица — зерно «прогон»: ночной прогон
-- по эталонному набору, средние и доля выше порогов, сравнивают прогоны между
-- собой. Эта — зерно «ответ»: судья оценивает конкретный живой ответ после того,
-- как поток дошёл до абонента (правило потоковой выдачи, оценка не блокирует выдачу). Нужна
-- для разреза качества в реальном времени — по теме, по абоненту, по периоду.
--
-- ОЦЕНКА ПРЕДВАРИТЕЛЬНАЯ. На прототипе судья — локальный Qwen-7B, который слабо
-- различает: поле provisional = true стоит на каждой строке,
-- чтобы её не приняли за настоящую оценку. На MVP судья — YandexGPT против
-- отвечающего GigaChat, и флаг снимается.
--
-- НЕ КАЖДЫЙ ОТВЕТ. Вызов судьи — это вызов модели: под нагрузкой он конкурирует
-- за процессор с живыми запросами (GPU на прототипе нет). Доля оцениваемых
-- ответов — quality_sample_rate; строка пишется только по попавшим в выборку.
--
-- ПЕРСОНАЛЬНЫХ ДАННЫХ ЗДЕСЬ НЕТ: идентификаторы (как в usage_events и аудите),
-- имя модели-судьи, два числа и короткое пояснение судьи (без текста ответа и
-- без текста вопроса).
--
-- Файл применяется дважды, как 03/04: контейнером инициализации и
-- `app/quality/store.py` при первом обращении.

CREATE TABLE IF NOT EXISTS quality_assessments (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    at                TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Тот же trace_id, что в usage_events и в ответе абоненту: связывает
    -- оценку с конкретным вызовом.
    trace_id          TEXT        NOT NULL,
    subscriber_id     TEXT        NOT NULL,
    session_id        TEXT        NOT NULL,

    topic             TEXT,
    inquiry_type      TEXT,

    -- Имя фактической модели-судьи. Псевдоним — в настройке judge_provider.
    judge_model       TEXT        NOT NULL DEFAULT '',

    -- Две оценки, [0, 1]. NULL, если судья не вернул оценку.
    faithfulness      DOUBLE PRECISION,
    answer_relevancy  DOUBLE PRECISION,

    -- Оценка получена слабым прототипным судьёй — не настоящая.
    provisional       BOOLEAN     NOT NULL DEFAULT true,

    -- scored | unavailable | conflict — почему оценки может не быть:
    -- unavailable — судья не ответил или ответ не разобрать;
    -- conflict — судья совпал с отвечающей моделью (failover, ADR-300).
    outcome           TEXT        NOT NULL DEFAULT 'scored',
    detail            TEXT
);

CREATE INDEX IF NOT EXISTS quality_assessments_at_idx
    ON quality_assessments (at DESC);
CREATE INDEX IF NOT EXISTS quality_assessments_topic_at_idx
    ON quality_assessments (topic, at DESC);
