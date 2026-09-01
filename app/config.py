"""Настройки приложения из переменных окружения.

Минимальная версия под шаг 0.5 — нужны только выбор провайдера и параметры
локальной тестовой модели. Шаг 1 плана разработки расширяет этот модуль
(таксономия, пороги поиска, лимиты квот) — здесь намеренно нет ничего лишнего.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "production"]


class Settings(BaseSettings):
    """Настройки, читаемые из .env и переменных окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Environment = "local"

    # Псевдоним провайдера из app/gateway/litellm_config.yaml:
    # local-test | yandexgpt | gigachat.
    # Смена этого значения — единственное, что требуется для перехода
    # локальная модель <-> управляемый API (ADR-002, вариант C).
    #
    # Генерация на прототипе — yandexgpt, на MVP — gigachat: последний требует
    # корневых сертификатов НУЦ Минцифры, которых на прототипе не будет
    # (контекст, раздел 46).
    llm_provider: str = "local-test"

    # Локальная тестовая модель (шаг 0.5)
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen2.5:7b-instruct-q4_K_M"
    local_timeout_seconds: float = 120.0

    # YandexGPT — генерация на прототипе, судья качества на MVP.
    # Штатного провайдера в LiteLLM нет, поэтому адрес задаётся явно.
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_api_base: str = "https://llm.api.cloud.yandex.net/v1"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Настройки читаются один раз за процесс."""
    return Settings()
