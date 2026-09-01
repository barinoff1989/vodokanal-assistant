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

    # Псевдоним провайдера из app/gateway/litellm_config.yaml.
    # Смена этого значения — единственное, что требуется для перехода
    # локальная модель <-> управляемый API (ADR-002, вариант C).
    llm_provider: str = "local-test"

    # Локальная тестовая модель (шаг 0.5)
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen2.5:7b-instruct-q4_K_M"
    local_timeout_seconds: float = 120.0

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Настройки читаются один раз за процесс."""
    return Settings()
