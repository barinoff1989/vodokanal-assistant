"""Настройки приложения из переменных окружения.

Значения, которые меняются от развёртывания к развёртыванию: выбор провайдера,
параметры поиска, лимиты, подключение к локальным слепкам систем.

Чего здесь намеренно НЕТ:

* длины ответа модели — она живёт в `GenerationParameters` (`app/models.py`),
  чтобы у неё был один источник, а не два расходящихся;
* самой таксономии — она в `app/taxonomy.py`, потому что это справочник, а не
  настройка: типы обращений нельзя переопределить переменной окружения, иначе
  вернётся рассинхрон, из-за которого переписывается прототип (раздел 11.1).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
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

    # --- Поиск по базе знаний (раздел 8.2) ---------------------------------- #
    # Числа не выдуманы: top_k=20 -> top_n=3 и порог 0.7 закреплены разделом 8.2
    # и показаны на диаграмме последовательностей Этапа 1.
    vector_top_k: int = 20
    """Сколько кандидатов возвращает векторный поиск до переранжирования."""

    rerank_top_n: int = 3
    """Сколько фрагментов уходит в промпт после переранжирования."""

    score_threshold: float = 0.7
    """Ниже этого значения контекст считается не найденным и включается
    запасной путь без модели (раздел 8.2). Порог влияет на долю ответов
    в запасном режиме — её видно в метрике Fallback Rate (раздел 37.2)."""

    # --- Лимиты (раздел 35.1) ------------------------------------------------ #
    # Квоты считаются на абонента и сессию плюс общий предел сервиса.
    # Квот по департаментам нет намеренно: департаменты не пользователи
    # чат-интерфейса, и эта ошибка уже была исправлена в разделе 35.1.
    quota_subscriber_per_minute: int = 20
    quota_session_per_minute: int = 10
    quota_service_per_minute: int = 300
    quota_retry_after_seconds: int = 30
    """Значение заголовка `Retry-After` при отказе по лимиту — обязательно
    вместе с кодом 429 (правило 4.5, раздел 7.4).

    > **[ПРЕДПОЛОЖЕНИЕ]** Сами лимиты подобраны как разумные для прототипа, а не
    > выведены из нагрузки: она известна только оценочно и с разбросом в 25 раз
    > (раздел 42.1). Пересчитать после подтверждения допущений A1–A3."""

    # --- Счётчики лимитов и состояние сессии (шаг 4) ------------------------- #
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # --- Локальные слепки внешних систем (шаг 0) ----------------------------- #
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "vodokanal"
    postgres_password: str = "vodokanal_local"
    billing_db: str = "billing_stub"
    lk_db: str = "lk_stub"
    inquiries_db: str = "inquiries_stub"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def redis_url(self) -> str:
        """Адрес хранилища счётчиков и состояния сессии."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def dsn(self, database: str) -> str:
        """Строка подключения к одному из локальных слепков.

        Собирается здесь, а не в каждом адаптере: иначе три места будут знать
        про пароль и разойдутся при первой же смене порта.
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{database}"
        )

    @model_validator(mode="after")
    def _local_model_is_not_for_production(self) -> Settings:
        """Локальная тестовая модель не отвечает абонентам.

        Она нужна для офлайн-проверок и не проходила ни оценки качества, ни
        замера задержки под нагрузкой (ADR-007, раздел 39.2). Ошибиться легко:
        достаточно выкатить конфигурацию разработчика — поэтому проверка здесь,
        а не в инструкции.
        """
        if self.is_production and self.llm_provider == "local-test":
            raise ValueError(
                "провайдер local-test недопустим в production: локальная тестовая "
                "модель не предназначена для ответов абонентам (ADR-007)"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Настройки читаются один раз за процесс."""
    return Settings()
