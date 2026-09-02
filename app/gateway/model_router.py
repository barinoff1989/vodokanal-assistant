"""Маршрутизатор моделей: конфигурация становится живой.

ЧТО ЭТО ЧИНИТ.
Раздел 5.5 контекста утверждает «config.yaml, не код»: смена провайдера должна
быть правкой конфигурации. До сих пор это было неправдой — файл
`litellm_config.yaml` не читал никто, а псевдоним провайдера расшифровывался
методом шлюза. Живой вызов это и вскрыл: псевдоним уходил провайдеру как имя
модели и не распознавался.

ПОЧЕМУ МАРШРУТИЗАТОР, А НЕ ПРОКСИ ОТДЕЛЬНЫМ ПРОЦЕССОМ.
Пункт 32 сведённого TODO допускал оба пути. Прокси — это второй процесс со
своей конфигурацией, своим портом и своим жизненным циклом; на прототипе с
15 ГБ памяти и без него тесно. Маршрутизатор LiteLLM принимает ровно тот же
`model_list` и те же цепочки запасных провайдеров, что объявлены в файле, но
живёт внутри нашего процесса. Проверено: он принимает `model_list`, `fallbacks`,
`num_retries` и `timeout` — весь состав файла.

ПЕРЕКЛЮЧЕНИЕ ПРИ СБОЕ (пункт 33).
Цепочки в `router_settings.fallbacks` до сих пор были описанием: их никто не
исполнял. Требование ADR-002, «Подтверждение», пункт 1 — шлюз с минимум двумя
провайдерами и автоматическим переключением — выполнялось наполовину, и это
было честно записано в разделе 49.3. Теперь исполняется.

ПОДСТАНОВКА ЗНАЧЕНИЙ.
LiteLLM подставляет `os.environ/VAR` **только целым значением**, не внутри
строки: проверено на версии 1.99.0, `gpt://os.environ/FOLDER/...` разворачивается
в `None`. Здесь та же семантика — заменяется значение целиком либо не заменяется
вовсе. Оставлено намеренно совместимым с прокси: если его когда-нибудь введут,
файл заработает без правок.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from app.config import Settings, get_settings

__all__ = [
    "CONFIG_PATH",
    "ModelConfigError",
    "build_router",
    "fallback_chains",
    "load_config",
    "resolve_value",
]

CONFIG_PATH = Path(__file__).resolve().parent / "litellm_config.yaml"
ENV_PREFIX = "os.environ/"


class ModelConfigError(RuntimeError):
    """Конфигурация моделей не пригодна к использованию."""


def resolve_value(value: Any) -> Any:
    """Подставить значение переменной окружения, если задана ссылка на неё.

    Только целое значение: `os.environ/VAR` заменяется, `префикс/os.environ/VAR`
    — нет. Та же семантика, что у самого LiteLLM; расхождение здесь означало бы,
    что файл работает у нас и не работает у прокси.
    """
    if isinstance(value, str) and value.startswith(ENV_PREFIX):
        return os.environ.get(value[len(ENV_PREFIX) :], "")
    return value


def _prime_environment(settings: Settings) -> None:
    """Заполнить переменные, у которых источник — настройки приложения.

    Имя модели Яндекса собирается из идентификатора каталога (`app/config.py`),
    и держать его ещё и в файле окружения значило бы завести второй источник
    истины — тот самый, что уже расходился при первой правке. Поэтому: если
    переменная пуста, она заполняется вычисленным значением, и владельцем
    остаётся `config.py`.
    """
    if not os.environ.get("YANDEX_MODEL_URI"):
        os.environ["YANDEX_MODEL_URI"] = settings.yandex_model
    for name, value in (
        ("YANDEX_API_BASE", settings.yandex_api_base),
        ("YANDEX_API_KEY", settings.yandex_api_key),
        ("OLLAMA_BASE_URL", settings.ollama_base_url),
    ):
        if not os.environ.get(name) and value:
            os.environ[name] = value


def load_config(path: Path | None = None, *, settings: Settings | None = None) -> dict[str, Any]:
    """Прочитать конфигурацию моделей с подставленными значениями."""
    resolved_settings = settings if settings is not None else get_settings()
    _prime_environment(resolved_settings)

    source = path if path is not None else CONFIG_PATH
    if not source.is_file():
        raise ModelConfigError(f"конфигурация моделей не найдена: {source}")

    document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not document.get("model_list"):
        raise ModelConfigError(f"в конфигурации нет ни одной модели: {source}")

    for entry in document["model_list"]:
        params = entry.get("litellm_params", {})
        entry["litellm_params"] = {name: resolve_value(value) for name, value in params.items()}
    return dict(document)


def fallback_chains(config: dict[str, Any]) -> list[dict[str, list[str]]]:
    """Цепочки запасных провайдеров из конфигурации.

    Формат совпадает с тем, что ожидает маршрутизатор: список словарей вида
    ``{"основной": ["запасной"]}``. Совпадение не случайно — файл писался под
    прокси, и переносить его в другой формат означало бы снова развести два
    описания одного и того же.
    """
    settings = config.get("router_settings") or {}
    return list(settings.get("fallbacks") or [])


def build_router(settings: Settings | None = None, *, path: Path | None = None) -> Any:
    """Собрать маршрутизатор LiteLLM по конфигурации.

    :raises ModelConfigError: конфигурация непригодна.
    """
    resolved_settings = settings if settings is not None else get_settings()
    config = load_config(path, settings=resolved_settings)
    router_settings = config.get("router_settings") or {}

    known = {entry["model_name"] for entry in config["model_list"]}
    if resolved_settings.llm_provider not in known:
        raise ModelConfigError(
            f"провайдер {resolved_settings.llm_provider!r} не описан в конфигурации; "
            f"есть: {', '.join(sorted(known))}"
        )

    # LiteLLM не помечает Router как часть открытого интерфейса,
    # хотя он документирован и используется штатно.
    from litellm import Router  # type: ignore[attr-defined]

    return Router(
        model_list=config["model_list"],
        fallbacks=fallback_chains(config),
        num_retries=int(router_settings.get("num_retries", 2)),
        timeout=float(router_settings.get("timeout", 60)),
    )
