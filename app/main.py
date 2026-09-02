"""Точка входа: сборка приложения с реальными зависимостями.

Запуск:

    python -m uvicorn app.main:app --reload

Стенд открывается на http://localhost:8000, API — там же по путям `/v1/...`.
"""

from __future__ import annotations

import logging
import os

from app.config import get_settings
from app.gateway.llm_gateway import LlmGateway
from app.gateway.quota import QuotaManager

logger = logging.getLogger(__name__)


def _build_gateway() -> LlmGateway:
    """Собрать шлюз с учётом лимитов, если хранилище доступно.

    Подключение к Redis создаётся лениво и не проверяется здесь: по ADR-008
    недоступность хранилища — это ответ `503` на конкретный запрос, а не отказ
    сервиса подняться. Иначе перезапуск Redis требовал бы перезапуска и шлюза.
    """
    settings = get_settings()
    try:
        import redis

        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
        quota = QuotaManager(
            client,
            subscriber_limit=settings.quota_subscriber_per_minute,
            session_limit=settings.quota_session_per_minute,
            service_limit=settings.quota_service_per_minute,
            store_retry_after=settings.quota_retry_after_seconds,
        )
    except ImportError:  # pragma: no cover — клиент есть в зависимостях шага 4
        logger.warning("клиент Redis не установлен: лимиты не проверяются")
        quota = None

    return LlmGateway(quota=quota, settings=settings)


def create() -> object:
    """Собрать приложение. Вынесено функцией ради проверок."""
    from app.api import create_app
    from app.metrics import prometheus as metrics

    # Метрики отдаются отдельным портом, а не адресом основного приложения:
    # правило 4.4 разрешает ровно два адреса, и расширять контракт ради
    # служебной надобности не следует. Их читает система сбора, абоненту они
    # не показываются.
    try:
        metrics.serve(port=int(os.getenv("METRICS_PORT", "9100")))
    except OSError as exc:  # порт занят — сервис всё равно должен подняться
        logger.warning("отдача метрик не запущена: %s", exc)

    return create_app(_build_gateway())


app = create()
