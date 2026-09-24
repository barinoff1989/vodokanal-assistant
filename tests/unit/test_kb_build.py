"""Проверки приёма корпуса базы знаний (ADR-400).

Здесь сторожится **гарантия, переехавшая ко входу**. Прежде контекст чистился на
каждом ответе, и это портило текст: «Перерасчёт» превращался в «⟨ФИО⟩». ADR-400
убрал чистку с пути абонента и поставил проверку при приёме — а значит вся
тяжесть требования 152-ФЗ теперь лежит на этом файле.
"""

from __future__ import annotations

import pytest

from app.kb.build import PersonalDataInCorpusError, check_no_personal_data
from app.kb.search import KbItem


def item(body: str, *, title: str = "Раздел", chunk_id: str = "doc#1") -> KbItem:
    return KbItem(
        chunk_id=chunk_id,
        title=title,
        body=body,
        source_title="Документ",
        source_url=None,
        synthetic=True,
    )


@pytest.mark.parametrize(
    ("body", "entity"),
    [
        ("Пример заявления: л/с 9998887770, прошу перерасчёт", "ACCOUNT_NUMBER"),
        ("Телефон заявителя 8-920-555-33-11", "PHONE_NUMBER"),
        ("Почта для ответа: ivanov.petr@example.com", "EMAIL_ADDRESS"),
    ],
)
def test_документ_с_персональными_данными_роняет_приём(body: str, entity: str):
    """Ошибка, а не предупреждение.

    Предупреждение в журнале — способ не заметить: одна такая дыра в проекте уже
    заведена осознанно (пустая база знаний), и заводить вторую нельзя. Документ с
    персональными данными в базе знаний — это 152-ФЗ, а не вопрос качества.
    """
    with pytest.raises(PersonalDataInCorpusError) as excinfo:
        check_no_personal_data([item(body)])

    assert entity in excinfo.value.entities
    assert excinfo.value.chunk_id == "doc#1"


def test_ошибка_называет_документ_и_что_делать():
    """Сообщение читает человек, собирающий корпус, а не отладчик."""
    with pytest.raises(PersonalDataInCorpusError) as excinfo:
        check_no_personal_data([item("л/с 9998887770", chunk_id="regl#7")])

    text = str(excinfo.value)
    assert "regl#7" in text
    assert "ADR-400" in text
    assert "пересоберите" in text


@pytest.mark.parametrize(
    "body",
    [
        "Перерасчёт выполняется по заявлению абонента.",
        "Основание: Поверка · Документы: Свидетельство о поверке, акт",
        "Прилагаю акт обследования. Согласно Постановлению Правительства РФ.",
        "Плата вносится до десятого числа месяца, следующего за расчётным.",
    ],
)
def test_обычный_текст_регламента_приём_проходит(body: str):
    """Первая редакция проверки звала полный обезличиватель — и отказалась
    принимать наш собственный корпус, найдя «данные» в словах «Перерасчёт» и
    «Документы».

    То есть ложные срабатывания, ради избавления от которых принято решение ADR-400,
    вернулись бы на другом конце: вместо порчи текста — отказ собрать базу
    знаний. Поэтому разбор языка при приёме выключен, а остались собственные
    распознаватели с контрольными суммами.
    """
    check_no_personal_data([item(body)])


def test_данные_на_стыке_заголовка_и_тела_тоже_видны():
    """Разрезание на части происходит позже, и номер мог бы оказаться на стыке."""
    with pytest.raises(PersonalDataInCorpusError):
        check_no_personal_data([item(title="Лицевой счёт", body="9998887770")])


def test_проверяются_все_записи_а_не_первая():
    """Выход по первой находке — правильно; выход по первой ЗАПИСИ — нет."""
    good = item("Перерасчёт выполняется по заявлению.", chunk_id="ok#1")
    bad = item("Телефон 8-920-555-33-11", chunk_id="bad#2")

    with pytest.raises(PersonalDataInCorpusError) as excinfo:
        check_no_personal_data([good, bad])

    assert excinfo.value.chunk_id == "bad#2"


def test_пустой_корпус_приём_проходит():
    """Пустая база знаний — отдельный случай с отдельным решением,
    и проверка ПДн его не касается."""
    check_no_personal_data([])
