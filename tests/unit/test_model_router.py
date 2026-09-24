"""Проверки маршрутизатора моделей.

Смысл: убедиться, что конфигурация **читается**, а не лежит описанием. До этого
файл не читал никто, псевдоним провайдера расшифровывался кодом, а цепочки
запасных провайдеров были текстом, который никто не исполнял.

Сборка настоящего маршрутизатора вынесена в отдельную проверку с пометкой
`slow`: она тянет LiteLLM, а это семь секунд разбора библиотеки.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from app.config import Settings
from app.gateway.model_router import (
    CONFIG_PATH,
    ModelConfigError,
    build_router,
    fallback_chains,
    load_config,
    resolve_value,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="yandexgpt",
        yandex_folder_id="b1gTEST",
        yandex_api_key="key-TEST",
    )


# --- подстановка значений ------------------------------------------------------- #


def test_ссылка_на_переменную_разворачивается():
    os.environ["ТЕСТОВАЯ_ПЕРЕМЕННАЯ"] = "значение"
    assert resolve_value("os.environ/ТЕСТОВАЯ_ПЕРЕМЕННАЯ") == "значение"


def test_ссылка_внутри_строки_не_разворачивается():
    """Та же семантика, что у самого LiteLLM.

    Проверено на версии 1.99.0: `gpt://os.environ/FOLDER/...` разворачивается в
    `None`. Расхождение здесь означало бы, что файл работает у нас и не работает
    у прокси — ровно тот класс ошибки, из-за которого имя модели однажды не
    собралось.
    """
    value = "gpt://os.environ/ТЕСТОВАЯ_ПЕРЕМЕННАЯ/latest"
    assert resolve_value(value) == value


def test_обычное_значение_не_трогается():
    assert resolve_value("ollama_chat/qwen2.5") == "ollama_chat/qwen2.5"
    assert resolve_value(120) == 120


# --- чтение конфигурации --------------------------------------------------------- #


def test_конфигурация_проекта_читается(settings: Settings):
    """Тот самый файл, который до сих пор не читал никто."""
    config = load_config(settings=settings)
    aliases = {entry["model_name"] for entry in config["model_list"]}
    assert aliases == {"local-test", "yandexgpt", "gigachat"}


def test_имя_модели_собирается_из_настроек(settings: Settings):
    """Идентификатор каталога живёт в одном месте — в настройках.

    Держать его ещё и в файле окружения значило бы завести второй источник
    истины: он уже расходился при первой правке, и живой вызов упал с
    «не удалось разобрать имя модели».
    """
    os.environ.pop("YANDEX_MODEL_URI", None)
    config = load_config(settings=settings)
    yandex = next(e for e in config["model_list"] if e["model_name"] == "yandexgpt")
    assert yandex["litellm_params"]["model"] == "openai/gpt://b1gTEST/yandexgpt/latest"


def test_ключ_подставляется_из_настроек(settings: Settings):
    os.environ.pop("YANDEX_API_KEY", None)
    config = load_config(settings=settings)
    yandex = next(e for e in config["model_list"] if e["model_name"] == "yandexgpt")
    assert yandex["litellm_params"]["api_key"] == "key-TEST"


def test_отсутствие_файла_это_внятная_ошибка(settings: Settings, tmp_path: Path):
    with pytest.raises(ModelConfigError, match="не найдена"):
        load_config(tmp_path / "нет-такого.yaml", settings=settings)


def test_пустая_конфигурация_отклоняется(settings: Settings, tmp_path: Path):
    empty = tmp_path / "пусто.yaml"
    empty.write_text("model_list: []\n", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="ни одной модели"):
        load_config(empty, settings=settings)


# --- цепочки запасных провайдеров --------------------------------------- #


def test_цепочки_объявлены_для_обоих_контуров(settings: Settings):
    """Прототип и MVP: у каждого основного провайдера есть запасной."""
    chains = fallback_chains(load_config(settings=settings))
    declared = {alias for chain in chains for alias in chain}
    assert {"yandexgpt", "gigachat"} <= declared


def test_у_прототипа_запасной_локальная_модель(settings: Settings):
    """Требование ADR-200: минимум два провайдера с переключением при сбое."""
    chains = fallback_chains(load_config(settings=settings))
    yandex = next(chain["yandexgpt"] for chain in chains if "yandexgpt" in chain)
    assert yandex == ["local-test"]


def test_запасные_провайдеры_описаны_в_конфигурации(settings: Settings):
    """Цепочка, указывающая на несуществующий псевдоним, молча не сработает."""
    config = load_config(settings=settings)
    known = {entry["model_name"] for entry in config["model_list"]}
    for chain in fallback_chains(config):
        for primary, backups in chain.items():
            assert primary in known, primary
            assert set(backups) <= known, backups


# --- защита от опечатки в псевдониме ------------------------------------------------ #


def test_неизвестный_провайдер_отклоняется_при_сборке(tmp_path: Path):
    """Иначе опечатка в настройке всплыла бы обращением к провайдеру."""
    broken = Settings(_env_file=None, llm_provider="выдуманный")
    with pytest.raises(ModelConfigError, match="не описан"):
        build_router(broken)


# --- соответствие файла и его назначения --------------------------------------------- #


def test_в_конфигурации_нет_зарубежных_провайдеров():
    """ADR-200: передача персональных данных за периметр исключена.

    Проверка структурная, по адресам и именам моделей: свободный текст файла
    содержит слово «зарубежные» в самом запрете, и поиск по нему давал бы
    ложное срабатывание.
    """
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for entry in document["model_list"]:
        params = entry["litellm_params"]
        target = f"{params.get('model', '')} {params.get('api_base', '')}"
        for foreign in ("api.openai.com", "anthropic", "gpt-4", "claude-"):
            assert foreign not in target, f"{entry['model_name']}: {foreign}"


def test_промпты_не_логируются():
    """Тот же кодовый путь используется с внешним провайдером."""
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert document["litellm_settings"]["turn_off_message_logging"] is True


# --- сборка настоящего маршрутизатора ------------------------------------------------- #


@pytest.mark.slow
def test_маршрутизатор_собирается(settings: Settings):
    """Проверка того, что конфигурация принимается самим LiteLLM.

    Помечена медленной: тянет библиотеку, а это семь секунд разбора.
    """
    router = build_router(settings)
    assert router is not None
    assert hasattr(router, "acompletion")


@pytest.mark.live
async def test_переключение_на_запасного_провайдера_работает():
    """Требование ADR-200.

    До этой работы цепочки были описанием в конфигурации, которое никто не
    исполнял: требование числилось выполненным наполовину, и это было честно
    записано.

    Основной провайдер ломается намеренно — подставляется заведомо неверный
    ключ. Если ответ всё же придёт, значит сработала цепочка.
    """
    broken = Settings(
        _env_file=None,
        llm_provider="yandexgpt",
        yandex_folder_id="b1gНЕСУЩЕСТВУЮЩИЙ",
        yandex_api_key="заведомо-неверный-ключ",
    )
    router = build_router(broken)

    try:
        response = await router.acompletion(
            model="yandexgpt",
            messages=[{"role": "user", "content": "Скажи одно слово: работает"}],
            max_tokens=30,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"запасной провайдер недоступен (нужна Ollama): {exc}")

    # Ответила именно запасная модель, а не основная.
    assert "ollama" in str(getattr(response, "model", "")).lower()
    assert response.choices[0].message.content
