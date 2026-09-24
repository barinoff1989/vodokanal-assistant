"""Опциональная загрузка секретов из HashiCorp Vault в переменные окружения.

Включается только присутствием `VAULT_ADDR` — без него это no-op, и Codespaces,
CI, прод и любое окружение без Vault читают переменные окружения как раньше,
без единой лишней строчки кода.

**Dev-режим Vault сам по себе не безопаснее обычной переменной окружения**
(один root-токен, всё в памяти, без TLS) — это репетиция интеграции для MVP
(там — Vault Agent Injector, ADR по инфраструктуре не написан), а не замена контроля доступа на
прототипе.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_SECRET_PATH = "secret/data/vodokanal/yandexgpt"

# Переменные окружения, которые эта функция способна заполнить, и их ключи
# внутри секрета Vault (KV v2: {"data": {"data": {...}}}).
_MANAGED_KEYS = (
    ("YANDEX_API_KEY", "api_key"),
    ("YANDEX_FOLDER_ID", "folder_id"),
)


def load_into_environ() -> None:
    """Подтягивает секреты YandexGPT из Vault, если он настроен через `VAULT_ADDR`.

    Не переопределяет уже заданные переменные окружения — Codespaces Secrets,
    CI-секреты и переменная, заданная явно в оболочке, всегда важнее Vault.
    Любая ошибка (Vault не поднят, сети нет, секрета нет, токена нет) —
    предупреждение в лог и работа дальше без него, тем же приёмом, что
    остальной проект обходится без недоступных опциональных зависимостей
    (`_build_usage`, `_build_live_judge`).
    """
    vault_addr = os.environ.get("VAULT_ADDR")
    if not vault_addr:
        return

    vault_token = os.environ.get("VAULT_TOKEN")
    if not vault_token:
        logger.warning(
            "VAULT_ADDR задан (%s), но VAULT_TOKEN — нет: секреты Vault пропущены",
            vault_addr,
        )
        return

    secret_path = os.environ.get("VAULT_SECRET_PATH", _DEFAULT_SECRET_PATH)

    try:
        import httpx

        response = httpx.get(
            f"{vault_addr.rstrip('/')}/v1/{secret_path}",
            headers={"X-Vault-Token": vault_token},
            timeout=2.0,
        )
        response.raise_for_status()
        secret_data = response.json()["data"]["data"]
    except Exception as exc:  # noqa: BLE001 — сбой Vault не должен ронять запуск
        logger.warning("не удалось прочитать секреты из Vault (%s): %s", vault_addr, exc)
        return

    loaded = []
    for env_name, vault_key in _MANAGED_KEYS:
        if env_name in os.environ:
            continue
        value = secret_data.get(vault_key)
        if value:
            os.environ[env_name] = value
            loaded.append(env_name)

    if loaded:
        logger.info("из Vault (%s) подтянуты: %s", vault_addr, ", ".join(loaded))
