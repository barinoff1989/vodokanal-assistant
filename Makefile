# Команды запуска. На Windows выполнять из Git Bash.
# Набор растёт по шагам плана разработки; сейчас закрыт шаг 0.5.

PYTHON ?= python
LOCAL_MODEL ?= qwen2.5:7b-instruct-q4_K_M

.PHONY: help install-min install install-llm install-data install-pii model-pull preflight test test-live lint typecheck check

help:
	@echo "install-min — всё, что нужно тестам (без тяжёлого LiteLLM)"
	@echo "install     — весь пакет целиком, включая LiteLLM и Redis"
	@echo "install-llm — только тяжёлый LiteLLM, если install-min уже прошёл"
	@echo "install-data — генератор синтетики (шаг 0)"
	@echo "install-pii — защита персональных данных + русская модель (шаг 3)"
	@echo "model-pull  — скачать локальную тестовую модель в Ollama"
	@echo "preflight   — проверить, что Ollama поднята и модель на месте"
	@echo "test        — тесты без внешних зависимостей"
	@echo "test-live   — тесты на реальной модели (нужна запущенная Ollama)"
	@echo "lint        — ruff"
	@echo "typecheck   — mypy"
	@echo "check       — lint + typecheck + test"

# Установка разбита намеренно: LiteLLM тянет десятки мегабайт зависимостей и на
# нестабильном канале обрывается по таймауту. Тестам он не нужен — импортируется
# лениво, только в момент реального вызова модели.
#
# Это удобство разработки, а не свойство пакета: в `pyproject.toml` LiteLLM
# числится обязательным, потому что без него сервис не ответит.
PIP_SLOW = -m pip install --timeout 120 --retries 10

# Всё, что импортируется тестами на уровне модуля. Без этого набора `pytest`
# падает на импорте, а не пропускает проверки.
install-min:
	$(PYTHON) $(PIP_SLOW) pytest pytest-asyncio ruff mypy 		pydantic pydantic-settings httpx fastapi sse-starlette 		prometheus-client pyyaml

install: install-min
	$(PYTHON) $(PIP_SLOW) "."

install-llm:
	$(PYTHON) $(PIP_SLOW) litellm

install-data:
	$(PYTHON) $(PIP_SLOW) ".[data]"

install-pii:
	$(PYTHON) $(PIP_SLOW) ".[pii]"
	$(PYTHON) -m spacy download ru_core_news_md

model-pull:
	ollama pull $(LOCAL_MODEL)

preflight:
	$(PYTHON) -c "import asyncio; from app.gateway.local_test_backend import LocalTestBackend; \
	asyncio.run(LocalTestBackend.from_settings().preflight()); print('OK: Ollama доступна, модель загружена')"

test:
	$(PYTHON) -m pytest -q -m "not live"

test-live:
	$(PYTHON) -m pytest -q -m live

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy app

check: lint typecheck test
