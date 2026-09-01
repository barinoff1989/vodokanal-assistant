# Команды запуска. На Windows выполнять из Git Bash.
# Набор растёт по шагам плана разработки; сейчас закрыт шаг 0.5.

PYTHON ?= python
LOCAL_MODEL ?= qwen2.5:7b-instruct-q4_K_M

.PHONY: help install model-pull preflight test test-live lint typecheck check

help:
	@echo "install     — установить зависимости (включая dev)"
	@echo "model-pull  — скачать локальную тестовую модель в Ollama"
	@echo "preflight   — проверить, что Ollama поднята и модель на месте"
	@echo "test        — тесты без внешних зависимостей"
	@echo "test-live   — тесты на реальной модели (нужна запущенная Ollama)"
	@echo "lint        — ruff"
	@echo "typecheck   — mypy"
	@echo "check       — lint + typecheck + test"

install:
	$(PYTHON) -m pip install -e ".[dev]"

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
