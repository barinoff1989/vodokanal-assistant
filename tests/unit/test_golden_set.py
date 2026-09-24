"""Проверки эталонного набора поиска.

Набор проверяется как **данные**, а не как фикстура: испорченная разметка не
роняет тесты, а тихо портит числа замера — и портит их в приятную сторону,
потому что ошибиться легче всего в сторону завышения.

Главная охраняемая здесь граница — **происхождение вопроса**. Настоящие вопросы
берутся из выгрузки обращений по номеру строки, и текст в разметке не хранится;
выдуманным разрешено быть только посторонним вопросам подмножества `control`,
которые проверяют порог. Если выдуманный вопрос окажется в подмножестве `real`,
замер начнёт мерить нашу способность перефразировать корпус — ровно ту ошибку,
на которой обезличиватель прошёл проверку и провалился на настоящих текстах.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "golden_set" / "labels.json"
CORPUS = ROOT / "kb" / "faq_voronezh.json"
INQUIRIES = ROOT / "fixtures" / "inquiries_voronezh.csv"

SUBSETS = {"corpus", "real", "no_answer", "outage", "control"}
EXPECT_HITS = {"corpus", "real"}
"""Подмножества, где ожидается найденный фрагмент. В остальных ожидается пусто."""


@pytest.fixture(scope="module")
def items() -> list[dict]:
    return json.loads(LABELS.read_text(encoding="utf-8"))["items"]


@pytest.fixture(scope="module")
def faq_ids() -> set[str]:
    """Опознаватели корпуса FAQ — он в репозитории и есть всегда."""
    return {item["chunk_id"] for item in json.loads(CORPUS.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def chunk_ids() -> set[str]:
    """Опознаватели ВСЕГО индекса: FAQ плюс разделы документов Word.

    Документы генерируются скриптом и в репозиторий не идут (`kb/*` исключён),
    поэтому на чистой копии их нет. Проверки, которым они нужны, пропускаются с
    указанием, что запустить, — молчаливого прохода тут быть не должно.
    """
    from app.kb.build import collect_items

    return {item.chunk_id for item in collect_items()}


def _need_documents(chunk_ids: set[str], faq_ids: set[str]) -> None:
    if chunk_ids <= faq_ids:
        pytest.skip(
            "разделов документов нет в индексе; собрать: "
            "python scripts/generate_kb_docs.py"
        )


def test_набор_нужного_размера(items):
    """Цель — 50–100 вопросов.

    Верхняя граница не формальность: набор размечен вручную, и вырасти он может
    только ручной работой — молча удвоившийся набор означал бы, что вопросы
    откуда-то нагенерированы."""
    assert 50 <= len(items) <= 100


def test_опознаватели_не_повторяются(items):
    idents = [item["id"] for item in items]
    assert len(idents) == len(set(idents))


def test_подмножества_известны(items):
    for item in items:
        assert item["subset"] in SUBSETS, item["id"]


def test_ожидаемые_фрагменты_существуют(items, chunk_ids, faq_ids):
    """Опечатка в идентификаторе фрагмента дала бы вечный промах, неотличимый от
    настоящего промаха поиска."""
    _need_documents(chunk_ids, faq_ids)
    for item in items:
        for chunk_id in item["expected"]:
            assert chunk_id in chunk_ids, f"{item['id']}: нет фрагмента {chunk_id}"


def test_ожидание_соответствует_подмножеству(items):
    """`real` и `corpus` обязаны чего-то ждать, остальные — ничего.

    Пустое ожидание в `real` засчиталось бы промахом навсегда; непустое в
    `no_answer` сделало бы вопрос без ответа вопросом с ответом."""
    for item in items:
        if item["subset"] in EXPECT_HITS:
            assert item["expected"], f"{item['id']}: ждёт пусто"
        else:
            assert not item["expected"], f"{item['id']}: ждёт фрагмент"


def test_выдуманные_вопросы_только_среди_посторонних(items):
    """Единственная запись, где текст лежит в разметке, — посторонний вопрос.

    Это и есть сторож происхождения: настоящий вопрос обязан ссылаться на
    строку выгрузки, и подменить его текстом нельзя."""
    for item in items:
        invented = item["source"]["kind"] == "invented"
        has_text = "text" in item["source"]
        assert invented == has_text, f"{item['id']}: текст без пометки о выдуманности"
        if invented:
            assert item["subset"] == "control", f"{item['id']}: выдуман вне control"


def test_настоящих_вопросов_подавляющее_большинство(items):
    """Выдуманные нужны для проверки порога и не должны становиться набором."""
    invented = sum(1 for item in items if item["source"]["kind"] == "invented")
    assert invented <= 5
    assert invented / len(items) < 0.1


def test_каждую_пару_faq_кто_то_ждёт(items, faq_ids):
    """Иначе часть корпуса не проверяется вовсе, и её поломка пройдёт незаметно.

    **Проверяется FAQ, а не весь индекс.** Разделы документов ожидаются не все и
    не должны: у инструкции есть служебные части — «Область применения», «Чего в
    ответе быть не должно», — которые не отвечают ни на один вопрос абонента.
    Требовать ожидания и от них значило бы вписать в разметку то, чего в ней
    быть не может."""
    expected = {chunk for item in items for chunk in item["expected"]}
    assert faq_ids <= expected


@pytest.mark.skipif(
    not INQUIRIES.exists(),
    reason="выгрузка обращений не хранится в репозитории: тексты настоящие",
)
def test_каждая_ссылка_на_обращение_разрешается():
    """Сдвиг строк в выгрузке молча переразметил бы весь набор."""
    with INQUIRIES.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    items = json.loads(LABELS.read_text(encoding="utf-8"))["items"]
    for item in items:
        if item["source"]["kind"] != "inquiry":
            continue
        row = item["source"]["row"]
        assert 1 <= row <= len(rows), f"{item['id']}: строки {row} нет"
        assert rows[row - 1]["text_request"].strip(), f"{item['id']}: пустой вопрос"
