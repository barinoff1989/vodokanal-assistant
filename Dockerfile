# Backend в контейнере — Backend_Containerization_Plan.md (рабочий архив разработки).
#
# Multi-stage: стадия builder ставит зависимости (включая то, что не объявлено
# в pyproject.toml и обычно ставится вручную по Makefile — LiteLLM, поиск,
# PII-фильтр), финальная стадия содержит только результат установки и код —
# без кэша pip и промежуточных слоёв builder'а.

FROM python:3.12-slim AS builder

WORKDIR /build

# Зависимости пакета (pyproject.toml) — редко меняются, отдельный слой кэша.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

# То, что в pyproject.toml не объявлено и на хосте ставится отдельными
# командами Makefile (install-llm, install-search, install-pii): LiteLLM,
# модели поиска/переранжирования (тянут torch, ~250+ МБ CPU-колёс),
# qdrant-client, PII-фильтр. Кладём в образ целиком — контейнер должен уметь
# то же самое, что и полностью настроенная машина разработчика.
#
# python-docx — не extras "data" (это генератор синтетики), а рантайм: kb/
# содержит настоящие .docx (kb/contracts, kb/regulations), и build_knowledge_base()
# читает их через app/kb/documents.py:load_docx при каждой сборке корпуса —
# без него reindex_kb.py падает на ModuleNotFoundError (проверено сборкой).
RUN pip install --no-cache-dir \
    litellm \
    qdrant-client \
    sentence-transformers \
    psutil \
    presidio-analyzer \
    presidio-anonymizer \
    python-docx

# Русская модель spaCy — не с PyPI, та же команда, что и на хосте (Makefile,
# install-pii). Устанавливается штатным путём spaCy, не подобранным вручную
# URL колеса — версия всегда совместима с уже поставленным spaCy.
RUN python -m spacy download ru_core_news_md

# Модель эмбеддингов — скачивается один раз на этапе сборки, не в рантайме.
# app/kb/search.py по умолчанию работает с EMBEDDING_LOCAL_FILES_ONLY=true
# (падает явно, а не молча идёт в сеть, если модели нет) — тот же приём, что
# на хосте (`make install-search` качает веса при первом использовании).
# Кэш ложится в HOME (/root в этом образе) — тот же путь нужен в runtime-стадии.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('intfloat/multilingual-e5-small')"

FROM python:3.12-slim AS runtime

WORKDIR /app

# Всё, что установил builder — site-packages и консольные скрипты (uvicorn).
COPY --from=builder /usr/local /usr/local
# Кэш весов модели эмбеддингов (см. RUN выше в стадии builder).
COPY --from=builder /root/.cache /root/.cache

# Код и данные, нужные в рантайме. golden_set/ и fixtures/ сюда намеренно не
# попадают — их читают только офлайн-скрипты замеров, не сам сервис.
# scripts/reindex_kb.py нужен отдельному одноразовому сервису reindex
# (docker-compose.yml) — та же переменная image, тот же слой.
#
# docker/postgres/init/*.sql — не только для контейнера postgres (он
# применяет их сам при первом создании тома): app/gateway/usage.py,
# app/quality/store.py, app/inquiries/store.py читают эти же файлы и
# применяют схему сами при старте — идемпотентно, не только один раз на
# новом томе. reference/ — таблица тарифов (app/config.py:tariff_table_path).
# Без них соответствующие возможности молча выключаются (проверено логами
# первого запуска backend), не падают — но выключаться не должны.
COPY app ./app
COPY web ./web
COPY kb ./kb
COPY data_example ./data_example
COPY docker/postgres/init ./docker/postgres/init
COPY reference ./reference
COPY scripts/reindex_kb.py ./scripts/reindex_kb.py

EXPOSE 8000

# --host 0.0.0.0 обязателен: дефолт uvicorn (127.0.0.1) недоступен снаружи
# контейнера даже с проброшенным портом.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
