"""Проверки тарифов: таблица, поиск по дате и ответ без модели.

Цена ошибки здесь выше обычной: неверный тариф — это неверная сумма в ответе
про деньги, и абонент сверит её с квитанцией. Поэтому проверяется не только
«нашёлся ли», но и «промолчал ли там, где данных нет».
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.config import get_settings
from app.models import GenerateRequest, RequestMetadata
from app.tariffs.answer import TariffResponder
from app.tariffs.store import Tariff, TariffStore
from app.taxonomy import Topic

ROOT = Path(__file__).resolve().parents[2]


def request(topic: Topic | None, query: str = "какой тариф на воду") -> GenerateRequest:
    return GenerateRequest(
        system="помощник абонента",
        query=query,
        metadata=RequestMetadata(subscriber_id="21000001", session_id="s-1", topic=topic),
    )


@pytest.fixture
def store() -> TariffStore:
    return TariffStore(
        [
            Tariff("питьевая вода", date(2026, 1, 1), date(2026, 9, 30), Decimal("36.55")),
            Tariff("питьевая вода", date(2026, 10, 1), date(2026, 12, 31), Decimal("40.94")),
            Tariff("водоотведение", date(2026, 1, 1), date(2026, 9, 30), Decimal("38.32")),
        ],
        fetched_on=date(2026, 9, 4),
        source_url="https://example/rates",
    )


@pytest.fixture
def responder(store: TariffStore) -> TariffResponder:
    return TariffResponder(store)


NOW = datetime(2026, 9, 5, 12, 0)


def test_на_свою_тему_отвечает_без_модели(responder: TariffResponder):
    answer = responder.answer(request(Topic.TARIFF), now=NOW)
    assert answer is not None
    assert "36,55" in answer.text
    assert "38,32" in answer.text


def test_на_чужую_тему_молчит(responder: TariffResponder):
    for topic in (Topic.OUTAGE, Topic.WATER_QUALITY, Topic.GENERAL, None):
        assert responder.answer(request(topic), now=NOW) is None


def test_в_пробеле_таблицы_молчит(store: TariffStore):
    """Загрузчик отбрасывает строки с противоречивыми периодами, и на их месте
    остаётся пробел.

    Промолчать там честно — вопрос уйдёт обычным путём. Назвать цену соседнего
    периода значило бы назвать сумму, которой в этот день не было."""
    responder = TariffResponder(store)
    assert responder.answer(request(Topic.TARIFF), now=datetime(2027, 1, 1)) is None


def test_называется_цена_для_населения_а_не_без_ндс(responder: TariffResponder):
    """Абонент платит по строке квитанции; вторая цифра дала бы повод сверять не то."""
    answer = responder.answer(request(Topic.TARIFF), now=NOW)
    assert answer is not None
    assert "НДС" in answer.text
    assert "29,96" not in answer.text  # цена без НДС того же периода


def test_ближайшее_изменение_называется(responder: TariffResponder):
    """«Почему выросла сумма» — самая частая тема обращений.

    Предупредить об изменении дешевле, чем потом объяснять его."""
    answer = responder.answer(request(Topic.TARIFF), now=NOW)
    assert answer is not None
    assert "01.10.2026" in answer.text
    assert "40,94" in answer.text


def test_изменение_без_изменения_цены_не_объявляется():
    """Периоды идут подряд по полугодиям, и цена в них часто совпадает.

    Сказать «тариф изменится» и назвать то же число — заставить абонента сверять
    две одинаковые цифры."""
    store = TariffStore(
        [
            Tariff("питьевая вода", date(2026, 1, 1), date(2026, 6, 30), Decimal("36.55")),
            Tariff("питьевая вода", date(2026, 7, 1), date(2026, 12, 31), Decimal("36.55")),
        ],
        fetched_on=date(2026, 6, 1),
    )
    answer = TariffResponder(store).answer(
        request(Topic.TARIFF), now=datetime(2026, 6, 1)
    )
    assert answer is not None
    assert "изменится" not in answer.text


# --- настоящая таблица --------------------------------------------------------- #


def test_настоящая_таблица_читается_и_не_содержит_пересечений():
    """Пересечение периодов означало бы два ответа на один день.

    На странице такой дефект **есть** — «с 01.01.2023 по 30.06.2030», — и
    загрузчик его отбрасывает. Проверка сторожит, что отброс не отключили."""
    path = ROOT / get_settings().tariff_table_path
    if not path.exists():
        pytest.skip("таблица тарифов не загружена")

    store = TariffStore.from_file(path)
    assert len(store) > 0
    for service in store.services:
        rows = sorted(
            (t for t in store._tariffs if t.service == service),
            key=lambda t: t.starts_on,
        )
        for previous, current in zip(rows, rows[1:], strict=False):
            assert current.starts_on > previous.ends_on, (
                f"{service}: период {current.starts_on} налезает на {previous.ends_on}"
            )


def test_в_настоящей_таблице_обе_услуги():
    """Перепутать воду с водоотведением значит назвать абоненту чужую цену."""
    path = ROOT / get_settings().tariff_table_path
    if not path.exists():
        pytest.skip("таблица тарифов не загружена")

    services = TariffStore.from_file(path).services
    assert "питьевая вода" in services
    assert "водоотведение" in services


def test_отброшенная_строка_видна_в_файле_как_пробел():
    """Дефект источника не заглажен: в таблице остаётся дыра, а не подмена.

    Если однажды загрузчик начнёт «чинить» такие строки, эта проверка упадёт и
    потребует объяснить, откуда взялось значение."""
    path = ROOT / get_settings().tariff_table_path
    if not path.exists():
        pytest.skip("таблица тарифов не загружена")

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["services"]["питьевая вода"]
    starts = [date.fromisoformat(r["from"]) for r in rows]
    gaps = [
        (a, b)
        for a, b in zip(starts, starts[1:], strict=False)
        if (b - a).days > 200  # полугодовой шаг; больше — пропущенный период
    ]
    assert gaps, "пробел от отброшенной строки исчез — проверьте загрузчик"


def test_называются_все_дорожающие_услуги_а_не_первая():
    """Первая редакция сообщала только про воду, хотя с той же даты дорожало и
    водоотведение — сильнее.

    Умолчать о большем подорожании в ответе про деньги хуже, чем не сказать
    ничего: абонент решит, что знает всю сумму. Нашлось сверкой ответа с файлом,
    а не тестом."""
    store = TariffStore(
        [
            Tariff("питьевая вода", date(2026, 1, 1), date(2026, 9, 30), Decimal("36.55")),
            Tariff("питьевая вода", date(2026, 10, 1), date(2026, 12, 31), Decimal("40.94")),
            Tariff("водоотведение", date(2026, 1, 1), date(2026, 9, 30), Decimal("38.95")),
            Tariff("водоотведение", date(2026, 10, 1), date(2026, 12, 31), Decimal("46.73")),
        ],
        fetched_on=date(2026, 9, 4),
    )
    answer = TariffResponder(store).answer(request(Topic.TARIFF), now=NOW)
    assert answer is not None
    assert "40,94" in answer.text
    assert "46,73" in answer.text, "умолчали о подорожании водоотведения"
