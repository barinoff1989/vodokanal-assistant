"""Проверки ответа об отключении — разбор адреса и путь мимо модели."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.models import GenerateRequest, GenerationParameters, RequestMetadata
from app.outages.answer import OutageResponder, parse_address
from app.outages.store import DEFAULT_MAX_AGE, OutageStore
from app.taxonomy import Topic

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "outage_schedule.txt"
LOADED_AT = datetime(2026, 8, 1, 9, 0)
NOW = datetime(2026, 8, 5, 10, 0)


@pytest.fixture(scope="module")
def responder() -> OutageResponder:
    return OutageResponder(
        OutageStore.from_file(FIXTURE, year=2026, loaded_at=LOADED_AT)
    )


def ask(address: str | None, topic: Topic | None = Topic.OUTAGE) -> GenerateRequest:
    return GenerateRequest(
        system="помощник абонента",
        query="почему нет воды?",
        parameters=GenerationParameters(stream=False),
        metadata=RequestMetadata(
            subscriber_id="sub-1", session_id="sess-1", topic=topic, address=address
        ),
    )


# --- разбор адреса ЛК -------------------------------------------------------- #


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        # Так адрес ведёт личный кабинет — ровно в этом виде он есть в выгрузке.
        ("г. Тестовый, ул. Набережная, д. 65, кв. 156", ("набережная", "65")),
        ("г. Тестовый, ул. Димитрова, д. 2, кв. 5", ("димитрова", "2")),
        ("г. Тестовый, ул. Переверткина, д. 1/1", ("переверткина", "1/1")),
        # Дом без слова «д.».
        ("ул. Ленинский пр-т, 223", ("ленинский", "223")),
        # Без города.
        ("ул. Мопра, д. 19/1", ("мопра", "19/1")),
    ],
)
def test_адрес_личного_кабинета_разбирается(address: str, expected):
    assert parse_address(address) == expected


def test_дом_не_путается_с_деревней():
    """Настоящая ошибка, пойманная сквозным прогоном.

    Сокращение «д.» означает и «дом», и «деревню». Первая редакция отбрасывала
    «д. 2» как населённый пункт — адрес переставал разбираться целиком, и
    ассистент молча уходил отвечать моделью.
    """
    assert parse_address("г. Тестовый, ул. Димитрова, д. 2, кв. 5") == ("димитрова", "2")


@pytest.mark.parametrize(
    "address",
    ["г. Тестовый", "", "просто текст", "кв. 5"],
)
def test_неразобранный_адрес_даёт_none(address: str):
    """Неразобранный адрес честнее неверно разобранного: первый виден, а второй
    ответит про чужую улицу."""
    assert parse_address(address) is None


# --- когда ответчик молчит --------------------------------------------------- #


def test_чужая_тема_не_наш_случай(responder: OutageResponder):
    """Молчание означает «идите обычным путём», а не «ответа нет»."""
    assert responder.answer(ask("г. Тестовый, ул. Димитрова, д. 2", Topic.GENERAL), now=NOW) is None
    assert responder.answer(ask("г. Тестовый, ул. Димитрова, д. 2", None), now=NOW) is None


def test_без_адреса_ответчик_молчит(responder: OutageResponder):
    """Адрес приходит из профиля ЛК; без него отвечать не на чем."""
    assert responder.answer(ask(None), now=NOW) is None
    assert responder.answer(ask("г. Тестовый"), now=NOW) is None


# --- когда отвечает ---------------------------------------------------------- #


def test_отключение_находится(responder: OutageResponder):
    answer = responder.answer(ask("г. Тестовый, Набережная, д. 1"), now=NOW)
    assert answer is not None
    assert "28.07" in answer.text and "13.08" in answer.text


def test_отсутствие_отключений_это_ответ_а_не_молчание(responder: OutageResponder):
    """Ключевое отличие от «не мой случай».

    Пустой результат поиска — законный ответ «отключений нет». Промолчи мы
    здесь, за нас ответила бы модель, которой график неизвестен, — и выдумала
    бы срок.
    """
    answer = responder.answer(ask("г. Тестовый, ул. Выдуманная, д. 7"), now=NOW)
    assert answer is not None
    assert "нет" in answer.text.lower()


def test_несколько_периодов_показываются_все(responder: OutageResponder):
    """Сорок адресов файла числятся в двух пересекающихся периодах.

    Выбор без основания хуже, потому что не виден абоненту (ADR-013).
    """
    answer = responder.answer(ask("г. Тестовый, ул. Димитрова, д. 2"), now=NOW)
    assert answer is not None
    assert answer.text.count("с ") >= 2
    assert "несколько периодов" in answer.text


def test_прошедшие_отключения_не_предлагаются(responder: OutageResponder):
    """Вопрос «почему нет воды» — про сейчас и дальше, а не про август прошлого
    года. Закончившийся период в ответе выглядел бы как действующий."""
    answer = responder.answer(
        ask("г. Тестовый, Набережная, д. 1"), now=datetime(2026, 12, 1, 10, 0)
    )
    assert answer is not None
    assert "нет" in answer.text.lower()


# --- оговорка об актуальности ------------------------------------------------ #


def test_в_ответе_всегда_есть_дата_получения_графика(responder: OutageResponder):
    answer = responder.answer(ask("г. Тестовый, Набережная, д. 1"), now=NOW)
    assert answer is not None
    assert answer.disclaimer is not None
    assert "01.08.2026" in answer.disclaimer


def test_устаревший_график_отвечает_с_предупреждением(responder: OutageResponder):
    """Без оговорки ассистент ответит по месячному файлу так же уверенно, как в
    день его получения."""
    stale_now = LOADED_AT + DEFAULT_MAX_AGE + timedelta(days=1)
    answer = responder.answer(ask("г. Тестовый, Набережная, д. 1"), now=stale_now)
    assert answer is not None
    assert answer.disclaimer is not None
    assert "устареть" in answer.disclaimer


def test_свежий_график_отвечает_без_предупреждения(responder: OutageResponder):
    answer = responder.answer(ask("г. Тестовый, Набережная, д. 1"), now=NOW)
    assert answer is not None
    assert answer.disclaimer is not None
    assert "устареть" not in answer.disclaimer
