# Команды запуска. На Windows выполнять из Git Bash.
# Набор растёт по шагам плана разработки; закрыты шаги 0.5, 1-6, 8, 8.5, 9 и часть 10.

PYTHON ?= python
LOCAL_MODEL ?= qwen2.5:7b-instruct-q4_K_M

.PHONY: help install-min install install-llm install-data install-pii install-search measure-search golden-set subscriber-directory measure-retrieval measure-reranking model-pull preflight test test-live lint typecheck check

help:
	@echo "install-min — всё, что нужно тестам (без тяжёлого LiteLLM)"
	@echo "install     — весь пакет целиком, включая LiteLLM и Redis"
	@echo "install-llm — только тяжёлый LiteLLM, если install-min уже прошёл"
	@echo "install-data — генератор синтетики (шаг 0)"
	@echo "install-pii — защита персональных данных + русская модель (шаг 3)"
	@echo "install-search — модели поиска: эмбеддинги и переранжирование (шаг 5)"
	@echo "model-pull  — скачать локальную тестовую модель в Ollama"
	@echo "preflight   — проверить, что Ollama поднята и модель на месте"
	@echo "measure-search — замер моделей поиска на этой машине (шаг 5)"
	@echo "golden-set  — собрать эталонный набор поиска из разметки"
	@echo "subscriber-directory — пересобрать справочник абонентов стенда из data_example/"
	@echo "measure-retrieval — качество поиска и цена порога (секунды)"
	@echo "measure-reranking — кросс-энкодер против косинуса (минут двадцать)"
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

# Ставится на шаге 5. Тянет torch — на процессорной машине это около 250 МБ
# колёс, плюс по 2,3 ГБ весов на каждую модель при первом запуске замера.
install-search:
	$(PYTHON) $(PIP_SLOW) sentence-transformers psutil

# Отвечает на вопрос, укладываются ли модели ADR-007 в бюджет задержки на машине
# без видеоускорителя. Числа, а не рассуждение: проект уже дважды получал
# результаты, обратные ожиданиям (раздел 51).
measure-search:
	$(PYTHON) scripts/measure_search_models.py

# Эталонный набор поиска. Сборка обязательна перед любым из двух замеров: она
# подставляет тексты вопросов из настоящих источников и падает, если источник
# сместился.
golden-set:
	$(PYTHON) scripts/build_golden_set.py

# Справочник абонентов для демо-стенда (`web/subscribers.json`) — 200 карточек
# из присланного примера Биллинга. Коммитится, как `kb/*.json`: маленький,
# воспроизводимый, источник (`data_example/*.csv`) уже в репозитории. На
# рабочей системе справочника нет — личность даёт SSO-сессия ЛК.
subscriber-directory:
	$(PYTHON) scripts/build_subscriber_directory.py

# Качество поиска и цена порога. Секунды.
measure-retrieval: golden-set
	$(PYTHON) scripts/measure_retrieval.py --json golden_set/measurement.json

# Кросс-энкодер против косинуса, вне пути абонента. Двадцать минут: 1320 пар,
# каждая проходит через модель целиком. Это офлайн-работа, медленно здесь
# ничего не стоит.
measure-reranking: golden-set
	$(PYTHON) scripts/measure_reranking.py --json golden_set/reranking.json

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
