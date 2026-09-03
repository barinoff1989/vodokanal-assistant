"""Проверки настроек.

Смысл не в том, что значения читаются, а в том, что закреплены числа, взятые
из решений проекта, и что настройка, способная навредить абоненту, не проходит
молча.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def _settings(**overrides) -> Settings:
    """Настройки без чтения .env: иначе тесты зависели бы от машины."""
    return Settings(_env_file=None, **overrides)


# --- числа из решений проекта ------------------------------------------------ #


def test_параметры_поиска_совпадают_с_принятыми():
    """Раздел 8.2: 20 кандидатов, 3 фрагмента в промпт, порог 0.7.

    Тест сторожит не арифметику, а то, что числа не разъедутся с диаграммой
    последовательностей и с описанием поиска.
    """
    settings = _settings()
    assert settings.vector_top_k == 20
    assert settings.rerank_top_n == 3
    assert settings.score_threshold == 0.7


def test_в_промпт_уходит_меньше_чем_нашлось():
    """Переранжирование сужает выборку; обратное означало бы ошибку настройки."""
    settings = _settings()
    assert settings.rerank_top_n < settings.vector_top_k


def test_лимит_сессии_не_больше_лимита_абонента():
    """У абонента может быть несколько сессий, обратное соотношение бессмысленно."""
    settings = _settings()
    assert settings.quota_session_per_minute <= settings.quota_subscriber_per_minute


def test_общий_лимит_сервиса_самый_большой():
    settings = _settings()
    assert settings.quota_service_per_minute > settings.quota_subscriber_per_minute


def test_задано_время_повтора_для_отказа_по_лимиту():
    """Правило 4.5: код 429 без `Retry-After` недопустим."""
    assert _settings().quota_retry_after_seconds > 0


# --- подключение к локальным слепкам ----------------------------------------- #


def test_три_базы_слепка_различны():
    """Три независимые системы — три базы, а не одна на всех."""
    settings = _settings()
    assert len({settings.billing_db, settings.lk_db, settings.inquiries_db}) == 3


def test_строка_подключения_собирается():
    settings = _settings()
    dsn = settings.dsn(settings.inquiries_db)
    assert dsn.startswith("postgresql://")
    assert dsn.endswith("/inquiries_stub")
    assert f":{settings.postgres_port}/" in dsn


# --- защита от опасной настройки --------------------------------------------- #


def test_локальная_модель_запрещена_в_production():
    """Она не проходила ни оценки качества, ни замера задержки (ADR-007)."""
    with pytest.raises(ValidationError):
        _settings(app_env="production", llm_provider="local-test")


def test_локальная_модель_разрешена_вне_production():
    assert _settings(app_env="local", llm_provider="local-test").llm_provider == "local-test"


def test_управляемый_провайдер_в_production_проходит():
    assert _settings(app_env="production", llm_provider="yandexgpt").is_production is True


def test_неизвестное_окружение_отклоняется():
    with pytest.raises(ValidationError):
        _settings(app_env="staging")


def test_модель_эмбеддингов_задана_настройкой():
    """ADR-014: на прототипе e5-small, на MVP BGE-M3 по ADR-007.

    Значение обязано быть настройкой, а не константой в коде: оно ложится в
    payload каждого вектора и сверяется при поиске. Расхождение модели индекса
    и модели запроса — самая дорогая ошибка в поиске, а моделей теперь две.
    """
    assert _settings().embedding_model
    assert "/" in _settings().embedding_model


def test_число_фрагментов_в_промпте_меньше_числа_кандидатов():
    """`top_k = 20 -> top_n = 3` (ADR-006).

    На прототипе переранжирования между ними нет (ADR-014), но соотношение
    сохраняется: порог отсеивает, а в промпт уходит тройка.
    """
    settings = _settings()
    assert settings.rerank_top_n < settings.vector_top_k
    assert 0.0 < settings.score_threshold <= 1.0
