"""Проверки корпуса базы знаний прототипа.

Корпус — не тестовая выдумка, а то, по чему ассистент будет искать ответ
(ADR-013: база знаний прототипа собирается из открытых источников). Поэтому он
проверяется как данные, а не как фикстура: испорченный корпус даёт не падение
тестов, а тихо неверные ответы абоненту.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[2] / "kb" / "faq_voronezh.json"


@pytest.fixture(scope="module")
def corpus() -> list[dict[str, str]]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_корпус_непустой(corpus):
    """Двадцать пар на момент выгрузки. Порог мягкий: FAQ живой."""
    assert len(corpus) >= 15


def test_у_каждой_пары_есть_вопрос_ответ_и_источник(corpus):
    """Источник показывается абоненту (`SourceRef`), и пустой обесценивает ответ:
    проверить его будет негде."""
    for item in corpus:
        assert item["chunk_id"]
        assert item["question"]
        assert item["answer"]
        assert item["source_title"]
        assert item["source_url"].startswith("https://")


def test_опознаватели_фрагментов_не_повторяются(corpus):
    """`chunk_id` уходит в ссылку на источник; совпадение свело бы два разных
    ответа к одной ссылке."""
    ids = [item["chunk_id"] for item in corpus]
    assert len(ids) == len(set(ids))


def test_разметка_не_просочилась_в_текст(corpus):
    """Теги в тексте попали бы и в промпт, и в ответ абоненту."""
    for item in corpus:
        for field in ("question", "answer"):
            assert "<" not in item[field], f"{item['chunk_id']}: разметка в {field}"
            assert "&nbsp" not in item[field]


def test_списки_в_ответах_сохранены(corpus):
    """Три ответа из двадцати первая редакция разбора теряла целиком.

    Ответы со списками содержат вложенные `<li>`, и разбор обрывался на них.
    Потеря прошла молча — сторож считал только общее число пар. Здесь
    проверяется, что многострочные ответы вообще есть: их одиннадцать.
    """
    multiline = [item for item in corpus if "\n" in item["answer"]]
    assert len(multiline) >= 8


def test_корпус_воронежский(corpus):
    """Прототип обслуживает один филиал. Чужой контакт в ответе — это чужой
    телефон в руках абонента."""
    text = " ".join(item["answer"] for item in corpus)
    for other in ("Краснодар", "Архангельск", "Сахалин", "Оренбург"):
        assert other not in text
