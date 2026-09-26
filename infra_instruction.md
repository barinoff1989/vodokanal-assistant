# infra_instruction — поднятие стека

Инструкция для запуска всего стека проекта: базы данных, векторное хранилище, Backend с шлюзом к модели
и демо-стендом. Стек описан в `docker-compose.yml`; отдельной папки `infra/` нет, инфраструктурные файлы
лежат в корне репозитория.

## 1. Что входит в стек

| Файл или папка | Назначение |
|---|---|
| `docker-compose.yml` | Описание всех сервисов, портов, томов и порядка запуска |
| `Dockerfile` | Образ Backend и задачи переиндексации (multi-stage: стадия сборки ставит зависимости, финальная стадия содержит только результат и код) |
| `docker/postgres/init/` | Скрипты первичной инициализации Postgres: базы и таблицы (`00_create_databases.sql`, `01_inquiries.sql`, `03_usage_events.sql`, `04_quality_reports.sql`, `05_quality_assessments.sql`) |
| `env.example.sh` | Справочник переменных окружения (команды `export`); файл `.env` приложение не читает |
| `Makefile` | Короткие цели для тех же команд (`make install`, `make test`, `make model-pull`, `make preflight`) |

Сервисы Compose:

| Сервис | Контейнер | Образ | Порты хоста | Что делает |
|---|---|---|---|---|
| `postgres` | `vk-postgres` | `postgres:16-alpine` | 5432 | Четыре базы: `billing_stub`, `lk_stub`, `inquiries_stub` (локальные заменители систем водоканала) и `telemetry` (события использования модели, отчёты качества) |
| `redis` | `vk-redis` | `redis:7-alpine` | 6379 | Счётчики лимитов запросов и состояние диалога; данные на диск не сохраняются |
| `qdrant` | `vk-qdrant` | `qdrant/qdrant:latest` | 6333 (REST), 6334 (gRPC) | Векторный поиск по базе знаний; том `qdrant_data` |
| `reindex` | `vk-reindex` | собирается из `Dockerfile` | — | Одноразовая переиндексация базы знаний в Qdrant перед стартом Backend; завершается кодом 0 |
| `backend` | `vk-backend` | собирается из `Dockerfile` | 8000 (API и демо-стенд), 9100 (метрики) | Backend, LLM Gateway (внутри процесса), обезличивание, охранители, демо-стенд `web/` |

Порядок запуска задан зависимостями: `postgres` и `redis` (готовы по healthcheck) → `qdrant` (готов) →
`reindex` (выполнился) → `backend`.

## 2. Требования

- Docker Desktop с Compose v2 (проверено на Docker 29.7.2).
- Свободные порты: 5432, 6379, 6333, 6334, 8000, 9100.
- Память: контейнер Backend занимает около 1,3 ГиБ; весь стек рассчитан на машину с 15 ГБ ОЗУ.
- Python 3.12 нужен только для тестов и скриптов на хосте (см. `README.md`), для стека он не нужен.

## 3. Запуск

Из корня репозитория:

```bash
docker compose up -d --build
```

Команда собирает образ, поднимает сервисы в порядке зависимостей и возвращает управление после старта.
Во время сборки в образ ставятся зависимости Python, русская модель spaCy (`ru_core_news_md`) и модель эмбеддингов `intfloat/multilingual-e5-small`; при запуске контейнера скачивать их не нужно.

Проверка состояния:

```bash
docker compose ps
```

Все сервисы должны быть в статусе `Up`, `postgres`, `redis` и `qdrant` — `healthy`; `reindex` после
завершения в списке не показывается. Итог переиндексации виден в его логе:

```bash
docker compose logs reindex --tail 3
```

Ожидаемая строка: `переиндексация завершена: 51 фрагментов в коллекции 'kb_faq'`.

## 4. Адреса

| Что | Адрес |
|---|---|
| Демо-стенд | http://localhost:8000 |
| Проверка живости | http://localhost:8000/v1/healthz (ответ `{"status":"ok"}`) |
| Swagger | http://localhost:8000/docs |
| Основной эндпоинт | `POST http://localhost:8000/v1/generate` (поток SSE) |
| Метрики Prometheus | http://localhost:9100/metrics |
| Qdrant | http://localhost:6333/dashboard |

Пример запроса:

```bash
curl -N -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"system":"помощник абонента водоканала","query":"Какие сейчас тарифы на холодную воду?","parameters":{"stream":true},"metadata":{"subscriber_id":"2100367945","session_id":"demo-1"}}'
```

## 5. Выбор модели-провайдера

Настройки читаются только из переменных окружения хоста: Compose подставляет их в контейнер `backend`.
Файл `.env` не используется. Полный список переменных — `env.example.sh`.

| Провайдер | Значение `LLM_PROVIDER` | Что нужно |
|---|---|---|
| Локальная модель (по умолчанию) | `local-test` | Ollama на хосте (`ollama serve`) и модель: `ollama pull qwen2.5:7b-instruct-q4_K_M`. Из контейнера Ollama доступна по `http://host.docker.internal:11434`, это уже задано в `docker-compose.yml` |
| YandexGPT | `yandexgpt` | `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` |
| GigaChat | `gigachat` | `GIGACHAT_CLIENT_ID`, `GIGACHAT_CLIENT_SECRET`, `GIGACHAT_SCOPE`; корневые сертификаты НУЦ Минцифры |

Пример для YandexGPT.

PowerShell (текущая сессия):

```powershell
$env:LLM_PROVIDER = "yandexgpt"
$env:YANDEX_API_KEY = "значение"
$env:YANDEX_FOLDER_ID = "значение"
docker compose up -d
```

bash:

```bash
export LLM_PROVIDER=yandexgpt YANDEX_API_KEY=значение YANDEX_FOLDER_ID=значение
docker compose up -d
```

При изменении переменных Compose пересоздаёт только контейнер `backend`. Диагностика подключения к
YandexGPT: `python scripts/check_yandex.py`.

Вопросы про тарифы, график отключений, регламентные формулировки, факты лицевого счёта и бланки
заявлений отвечаются без обращения к модели, поэтому демо-стенд работает на них и без ключей и без Ollama.

## 6. Управление стеком

| Действие | Команда |
|---|---|
| Логи Backend | `docker compose logs -f backend` |
| Остановить, данные сохранить | `docker compose down` |
| Остановить и удалить данные (базы Postgres, коллекция Qdrant) | `docker compose down -v` |
| Пересобрать образ после изменения `pyproject.toml` или `Dockerfile` | `docker compose build backend reindex` |
| Переиндексировать базу знаний вручную | `docker compose run --rm reindex` |
| Перезапустить только Backend | `docker compose restart backend` |

Скрипты из `docker/postgres/init/` выполняются только при первом создании тома `pgdata`. Чтобы применить
их заново, нужно выполнить `docker compose down -v` и поднять стек снова.

Каталоги `app/` и `web/` смонтированы в контейнер `backend`, Backend запущен с `--reload`: правки кода
применяются без пересборки образа.

## 7. Запуск Backend без Docker

Вариант для точечной отладки: Postgres, Redis и Qdrant поднимаются командой
`docker compose up -d postgres redis qdrant`, Backend запускается на хосте:

```bash
python -m pip install .
python -m pip install ".[pii]" && python -m spacy download ru_core_news_md
python -m uvicorn app.main:app --reload
```

По умолчанию на хосте поиск по базе знаний работает в памяти процесса (`VECTOR_STORE=memory`); чтобы
использовать Qdrant, задайте `VECTOR_STORE=qdrant`.

## 8. Целевое развёртывание

Для MVP проектом предусмотрен общий Kubernetes-кластер с личным кабинетом: namespace `platform`, `ai-app`,
`inference`, `ai-data`, `observability`. Состав, реплики и ресурсы описаны в
`AI_docs/C4_Deployment.html` и `AI_docs/ADR/ADR-500-resources-and-cost.md` (лист `Sizing` в
`AI_docs/Tech_Stack_Matrix.xlsx`). Helm-чартов в репозитории нет: для прототипа и демонстрации стек
поднимается через `docker-compose.yml`.

## 9. Типичные ситуации

| Ситуация | Что происходит и что делать |
|---|---|
| Порт хоста занят при `docker compose up` | Docker не запускает контейнер, для которого порт занят. Освободить порт; порт метрик хоста меняется переменной `METRICS_PORT` (внутри контейнера метрики остаются на 9100) |
| Порт метрик занят при запуске Backend на хосте | Backend поднимается, метрики не отдаются, в логе предупреждение «отдача метрик не запущена». Освободить порт или задать `METRICS_PORT` |
| Redis недоступен | Запросы получают ответ `503` с заголовком `Retry-After`: лимиты не проверяются молча (ADR-400). Поднять `redis` и повторить |
| Postgres недоступен | Backend поднимается, запись событий использования выключается (предупреждение в логе) |
| Ollama не запущена при `LLM_PROVIDER=local-test` | Вопросы, которым нужна модель, получают ответ `502` (`upstream-unavailable`); вопросы прямых ответчиков отвечаются как обычно |
| `reindex` завершился с ошибкой | Backend не стартует (зависит от успешного завершения `reindex`); причина — в `docker compose logs reindex`, чаще всего Qdrant ещё не готов |
