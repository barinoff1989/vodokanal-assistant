"""Проверки поиска отключений по адресу.

Адреса и даты берутся из настоящего файла владельца, а не придумываются: смысл
поиска в том, чтобы вопрос абонента совпал с записью источника, а совпадение
проверяется только на настоящих написаниях.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from app.outages.store import DEFAULT_MAX_AGE, OutageStore

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "outage_schedule.txt"
LOADED_AT = datetime(2026, 8, 1, 9, 0)


@pytest.fixture(scope="module")
def store() -> OutageStore:
    return OutageStore.from_file(FIXTURE, year=2026, loaded_at=LOADED_AT)


# --- поиск ------------------------------------------------------------------- #


def test_адрес_находится(store: OutageStore):
    found = store.find("Переверткина", "1/1")
    assert found
    assert all(o.street == "переверткина" for o in found)


@pytest.mark.parametrize(
    "street",
    ["Димитрова", "ул. Димитрова", "  УЛ. ДИМИТРОВА  ", "ул Димитрова"],
)
def test_написание_улицы_в_вопросе_не_мешает(store: OutageStore, street: str):
    """Абонент пишет адрес как придётся, а в файле он записан четырьмя способами.

    Если нормализация вопроса и нормализация источника разойдутся, поиск не
    найдёт ничего и промолчит — самый неприятный вид отказа, потому что он
    выглядит как «отключений нет».
    """
    assert store.find(street, "2")


@pytest.mark.parametrize("house", ["10а", "10 а", "10А"])
def test_написание_дома_не_мешает(store: OutageStore, house: str):
    assert store.find("Переверткина", house)


def test_улица_с_названием_как_у_типа_находится(store: OutageStore):
    """«Набережная» — та самая улица, которую нормализация однажды уничтожила."""
    assert store.find("Набережная", "1")


def test_несуществующий_адрес_даёт_пусто(store: OutageStore):
    """Пустой ответ — это «отключений нет», и он должен быть именно пустым,
    а не приблизительным совпадением по соседней улице."""
    assert store.find("Выдуманная", "1") == []
    assert store.find("Переверткина", "99999") == []


# --- дата -------------------------------------------------------------------- #


def test_поиск_на_дату_отсекает_чужие_интервалы(store: OutageStore):
    inside = store.find("Переверткина", "1/1", on=date(2026, 8, 25))
    outside = store.find("Переверткина", "1/1", on=date(2026, 8, 1))
    assert inside
    assert all(o.starts_on <= date(2026, 8, 25) <= o.ends_on for o in inside)
    assert outside == []


def test_без_даты_возвращаются_все_интервалы(store: OutageStore):
    """Вопрос «когда отключат» шире вопроса «отключено ли сейчас»."""
    assert len(store.find("Переверткина", "1/1")) >= len(
        store.find("Переверткина", "1/1", on=date(2026, 8, 25))
    )


def test_интервалы_идут_от_ближайшего(store: OutageStore):
    found = store.find("Димитрова", "2")
    assert found == sorted(found, key=lambda o: (o.starts_on, o.ends_on))


# --- противоречия источника -------------------------------------------------- #


def test_адрес_в_двух_периодах_отдаёт_оба(store: OutageStore):
    """Сорок адресов файла числятся в двух пересекающихся периодах.

    Выбрать между ними не на чем, и выбор без основания хуже, потому что не
    виден абоненту (ADR-013). Показываются оба.
    """
    found = store.find("Димитрова", "2")
    assert len({(o.starts_on, o.ends_on) for o in found}) > 1


def test_район_сужает_поиск(store: OutageStore):
    """Четыре адреса числятся в двух районах сразу — это дефект файла.

    Район позволяет развести их там, где спрашивающий знает свой район.
    """
    both = store.find("Димитрова", "2")
    one = store.find("Димитрова", "2", district="Левобережный")
    assert one
    assert len(one) < len(both) or {o.district for o in both} == {"Левобережный"}
    assert all(o.district == "Левобережный" for o in one)


# --- актуальность ------------------------------------------------------------ #


def test_свежий_график_не_считается_устаревшим(store: OutageStore):
    assert store.is_stale(LOADED_AT + timedelta(days=1)) is False


def test_старый_график_считается_устаревшим(store: OutageStore):
    """Без этой проверки ассистент однажды ответит по месячному файлу — и так же
    уверенно, как в день его получения."""
    assert store.is_stale(LOADED_AT + DEFAULT_MAX_AGE + timedelta(seconds=1)) is True


def test_возраст_измеряется_от_чтения_а_не_от_дат_в_файле(store: OutageStore):
    """Даты в графике говорят о воде, а не о свежести самого графика.

    Файл с давно прошедшими периодами может быть только что полученным, и
    наоборот — вчерашний файл может описывать следующий месяц.
    """
    assert store.age(LOADED_AT + timedelta(hours=5)) == timedelta(hours=5)


def test_время_чтения_хранится_рядом_с_данными(store: OutageStore):
    """Отметка не вычисляется на лету: единственное, что система знает о
    свежести файла, — когда он был прочитан."""
    assert store.loaded_at == LOADED_AT
