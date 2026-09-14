"""Проверки опциональной загрузки секретов из Vault (app/secrets_vault.py).

Vault здесь всегда подменён — живой Vault для юнит-теста не нужен, живёт
только сама логика: включение по VAULT_ADDR, приоритет уже заданных
переменных, отказоустойчивость при любой ошибке Vault.
"""

from __future__ import annotations

import os

import httpx
import pytest

from app import secrets_vault


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _kv_v2_payload(api_key: str, folder_id: str) -> dict:
    return {"data": {"data": {"api_key": api_key, "folder_id": folder_id}}}


def test_без_vault_addr_ничего_не_делает(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VAULT_ADDR", raising=False)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("httpx.get не должен вызываться без VAULT_ADDR")

    monkeypatch.setattr(httpx, "get", fail_if_called)

    secrets_vault.load_into_environ()


def test_без_токена_пропускает_с_предупреждением(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "http://127.0.0.1:8200")
    monkeypatch.delenv("VAULT_TOKEN", raising=False)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("httpx.get не должен вызываться без VAULT_TOKEN")

    monkeypatch.setattr(httpx, "get", fail_if_called)

    secrets_vault.load_into_environ()


def test_успешно_подтягивает_переменные(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "http://127.0.0.1:8200")
    monkeypatch.setenv("VAULT_TOKEN", "root-local-dev")
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: _FakeResponse(_kv_v2_payload("key-из-vault", "folder-из-vault")),
    )

    try:
        secrets_vault.load_into_environ()
        assert os.environ["YANDEX_API_KEY"] == "key-из-vault"
        assert os.environ["YANDEX_FOLDER_ID"] == "folder-из-vault"
    finally:
        os.environ.pop("YANDEX_API_KEY", None)
        os.environ.pop("YANDEX_FOLDER_ID", None)


def test_не_перезаписывает_уже_заданную_переменную(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "http://127.0.0.1:8200")
    monkeypatch.setenv("VAULT_TOKEN", "root-local-dev")
    monkeypatch.setenv("YANDEX_API_KEY", "уже-задан-снаружи")
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: _FakeResponse(_kv_v2_payload("key-из-vault", "folder-из-vault")),
    )

    try:
        secrets_vault.load_into_environ()
        assert os.environ["YANDEX_API_KEY"] == "уже-задан-снаружи"
        assert os.environ["YANDEX_FOLDER_ID"] == "folder-из-vault"
    finally:
        os.environ.pop("YANDEX_FOLDER_ID", None)


def test_ошибка_vault_не_роняет_запуск(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "http://127.0.0.1:8200")
    monkeypatch.setenv("VAULT_TOKEN", "root-local-dev")
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)

    def raise_connect_error(*_args: object, **_kwargs: object) -> None:
        raise httpx.ConnectError("Vault недоступен")

    monkeypatch.setattr(httpx, "get", raise_connect_error)

    secrets_vault.load_into_environ()

    assert "YANDEX_API_KEY" not in os.environ
    assert "YANDEX_FOLDER_ID" not in os.environ
