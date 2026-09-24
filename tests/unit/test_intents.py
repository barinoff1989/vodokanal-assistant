"""Проверки детектора намерений: реплика делится на задачи, лишнего не режется.

Ошибка детектора бывает двух видов, и оба вредны: **недорез** — вторая задача
молча теряется (так и было до детектора), **перерез** — одна задача превращается в
две и абонент получает два ответа на один вопрос. Поэтому рядом лежат и те, и
другие случаи.
"""

from __future__ import annotations

import pytest

from app.agents import intents
from app.agents.intents import Condition, PartKind, split_intents
from app.billing import answer as billing_answer


@pytest.mark.parametrize(
    ("query", "signatures"),
    [
        (
            "Какой у меня долг и когда поверка счётчика?",
            [("account", "debt"), ("account", "verification")],
        ),
        (
            "Когда отключат воду на моей улице и сколько стоит кубометр воды?",
            [("outage", ""), ("tariff", "")],
        ),
        (
            "Хочу заказать поверку счётчика и справку об отсутствии задолженности",
            [("type", "meter_verification"), ("type", "certificate")],
        ),
        (
            "Заказать поверку и опломбировку счётчика, пожалуйста",
            [("type", "meter_verification"), ("type", "meter_sealing")],
        ),
        (
            "Какой у меня долг? Когда поверка счётчика?",
            [("account", "debt"), ("account", "verification")],
        ),
    ],
)
def test_две_задачи_делятся(query, signatures):
    parts = split_intents(query)
    assert [p.signature for p in parts] == signatures


@pytest.mark.parametrize(
    "query",
    [
        "Хочу подать заявку на поверку счётчика",
        "Что такое повышающий коэффициент и когда он применяется?",
        "Вода из-под крана, холодная и горячая, пахнет хлоркой",
        "Здравствуйте, подскажите, пожалуйста, как получить справку об отсутствии задолженности",
        "Когда поверка счётчика?",
        "Сколько я должен за август и когда нужно оплатить?",
    ],
)
def test_одна_задача_не_режется(query):
    """Перерез вреднее недореза: два ответа на один вопрос сбивают абонента."""
    parts = split_intents(query)
    assert len(parts) == 1
    assert parts[0].kind is PartKind.ANSWER


def test_вопрос_как_передать_показания_не_операция():
    """«Как передать» — вопрос о порядке, на него отвечает база знаний, а не отказ."""
    queries = (
        "Как передать показания?",
        "Где можно передать показания?",
        "Можно ли передать показания по телефону?",
    )
    for query in queries:
        parts = split_intents(query)
        assert all(p.kind is not PartKind.UNSUPPORTED for p in parts), query


@pytest.mark.parametrize(
    ("query", "action"),
    [
        ("Хочу передать показания 1234", "передача показаний"),
        ("Передайте показания 1234", "передача показаний"),
        ("Оплатите мой счёт", "оплата"),
        ("Измените мой адрес", "изменение данных абонента"),
    ],
)
def test_просьба_выполнить_операцию_узнаётся(query, action):
    parts = split_intents(query)
    assert len(parts) == 1
    assert parts[0].kind is PartKind.UNSUPPORTED
    assert parts[0].action == action


def test_условная_часть_привязана_к_долгу():
    query = "Передайте показания 1234 и скажите, нет ли долга. Если нет — закажите поверку"
    parts = split_intents(query)
    assert [p.kind for p in parts] == [PartKind.UNSUPPORTED, PartKind.ANSWER, PartKind.CONDITIONAL]
    conditional = parts[2]
    assert conditional.condition is Condition.NO_DEBT
    assert conditional.text == "закажите поверку"
    assert conditional.signature == ("type", "meter_verification")


def test_условие_про_долг_названное_прямо():
    parts = split_intents("Скажите, какой у меня долг. Если долга нет, закажите справку")
    assert parts[-1].kind is PartKind.CONDITIONAL
    assert parts[-1].condition is Condition.NO_DEBT


def test_непроверяемое_условие_помечено_неизвестным():
    """Условие не про долг проверить нечем — часть не должна выполняться молча."""
    parts = split_intents("Когда поверка счётчика? Если скоро — закажите поверку")
    assert parts[-1].kind is PartKind.CONDITIONAL
    assert parts[-1].condition is Condition.UNKNOWN


def test_повторно_названная_задача_не_дублируется():
    parts = split_intents("Какой у меня долг? И сколько я должен?")
    assert len(parts) == 1


def test_слова_связки_не_остаются_в_частях():
    parts = split_intents("Заказать поверку и опломбировку счётчика, пожалуйста")
    assert parts[1].text == "опломбировку счётчика"


def test_признаки_фактов_счёта_совпадают_с_ответчиком():
    """Копия в детекторе не должна разойтись с оригиналом (import цикличен)."""
    assert intents._VERIFICATION_MARKERS == billing_answer._VERIFICATION_MARKERS
    assert intents._READINGS_MARKERS == billing_answer._READINGS_MARKERS
    assert intents._CHARGES_MARKERS == billing_answer._CHARGES_MARKERS
    assert intents._DEBT_MARKERS == billing_answer._DEBT_MARKERS
