"""Проверки готового бланка как файла: рендер HTML и хранилище ссылок.

`app/documents/artifact.py` — дубль объектного хранилища на прототипе. Проверки
сторожат: экранирование данных абонента, круговой путь put → get, срок жизни
ссылки и отказ отдавать файл по кривому токену (обход каталога).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.documents.artifact import TOKEN_RE, ArtifactStore, render_html


def test_render_html_экранирует_данные_абонента():
    html = render_html("Заявление", "ФИО: Петров <тест> & Ко", disclaimer="образец")
    assert "&lt;тест&gt;" in html
    assert "<тест>" not in html
    assert "образец" in html
    # Кнопка печати и верстка под печать на месте.
    assert "window.print()" in html
    assert "@media print" in html


def test_put_get_круговой_путь(tmp_path: Path):
    store = ArtifactStore(root=tmp_path, ttl_seconds=3600)
    url = store.put(render_html("Бланк", "тело бланка"))
    assert url is not None
    token = url.rsplit("/", 1)[-1]
    assert TOKEN_RE.match(token)
    page = store.get(token)
    assert page is not None
    assert "тело бланка" in page


def test_просроченная_ссылка_не_отдаётся(tmp_path: Path):
    store = ArtifactStore(root=tmp_path, ttl_seconds=1)
    url = store.put(render_html("Бланк", "тело"))
    assert url is not None
    token = url.rsplit("/", 1)[-1]
    path = tmp_path / f"{token}.html"
    old = time.time() - 10
    os.utime(path, (old, old))
    assert store.get(token) is None


def test_неизвестный_и_кривой_токен_дают_none(tmp_path: Path):
    store = ArtifactStore(root=tmp_path, ttl_seconds=3600)
    assert store.get("нетакого0000000000") is None
    # Обход каталога отсекается регулярным выражением токена до файловой системы.
    assert store.get("../config") is None
    assert store.get("..%2Fconfig") is None
    assert not TOKEN_RE.match("../config")


def test_уборка_удаляет_просроченные_при_записи(tmp_path: Path):
    store = ArtifactStore(root=tmp_path, ttl_seconds=1)
    first = store.put(render_html("A", "a"))
    assert first is not None
    stale = tmp_path / (first.rsplit("/", 1)[-1] + ".html")
    old = time.time() - 10
    os.utime(stale, (old, old))

    store.put(render_html("B", "b"))  # запись подметает каталог
    assert not stale.exists()


def test_токены_разные_у_двух_бланков(tmp_path: Path):
    store = ArtifactStore(root=tmp_path, ttl_seconds=3600)
    a = store.put(render_html("A", "a"))
    b = store.put(render_html("B", "b"))
    assert a is not None and b is not None
    assert a != b
