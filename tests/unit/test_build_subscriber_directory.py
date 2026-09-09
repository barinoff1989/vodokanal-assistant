"""Проверки справочника абонентов для демо-стенда.

Справочник — не фикстура, а данные стенда: испорченная сборка не роняет тесты, а
тихо ломает демонстрацию (пустой список, отключение у всех или ни у кого,
`subscriber_id`, по которому регистрация не находит профиль).

Работает на **настоящем примере** из `data_example/` — том же, на котором живёт
прототип. Подменять его здесь нечем и незачем.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_subscriber_directory import build  # noqa: E402

from app.billing.source import CsvBillingSource  # noqa: E402

DATA = ROOT / "data_example"
TODAY = date(2026, 9, 8)


def test_справочник_собран_из_всего_примера() -> None:
    entries = build(today=TODAY)
    assert len(entries) == len(CsvBillingSource(DATA))


def test_порядок_по_фамилии() -> None:
    """Выгрузка идёт «Фамилия Имя Отчество» — сортировка по имени и есть
    сортировка по фамилии, которую ждёт поиск по абоненту."""
    entries = build(today=TODAY)
    names = [e.name.casefold() for e in entries]
    assert names == sorted(names)


def test_идентификатор_абонента_разрешается_в_биллинге() -> None:
    """`account` уходит стендом как `subscriber_id`, а регистрация обращения
    ищет по нему профиль (`app/backend/registration.py`). Номер, которого нет в
    Биллинге, означал бы абонента, которому нельзя оформить обращение."""
    billing = CsvBillingSource(DATA)
    for entry in build(today=TODAY):
        assert billing.account(entry.account) is not None, entry.account


def test_долг_показан_тогда_и_только_тогда_когда_он_есть() -> None:
    billing = CsvBillingSource(DATA)
    for entry in build(today=TODAY):
        account = billing.account(entry.account)
        assert account is not None
        assert (entry.debt is not None) == account.has_debt, entry.account


def test_баланс_в_рублях_с_запятой() -> None:
    entry = next(e for e in build(today=TODAY) if e.account == "2100202213")
    assert entry.balance == "−1 250,00 ₽"
    assert entry.debt == "1 250,00 ₽"


def test_просроченная_поверка_видна_в_карточке() -> None:
    """Счётчик 2100202213 просрочен на выгрузке — карточка должна это сказать,
    это один из частых вопросов абонента (FAQ, вопросы 3 и 10)."""
    entry = next(e for e in build(today=TODAY) if e.account == "2100202213")
    assert "просрочена" in entry.meters


def test_отключение_находится_у_части_абонентов_а_не_у_всех() -> None:
    """Оба ответа — «отключение есть» и «отключений нет» — нужны на демонстрации.
    Если график попадает во всех или ни в кого, показать второй нечем.

    Признак вычисляется тем же поиском по адресу, что и ответ абоненту
    (`app/outages`), — вписанная руками пометка однажды уже устарела молча."""
    entries = build(today=TODAY)
    with_outage = sum(1 for e in entries if e.outage)
    assert 0 < with_outage < len(entries)


def test_часть_карточек_несёт_профиль_лк() -> None:
    """Телефон и почта есть не у всех (в `ЛКК.csv` меньше строк, чем счетов) —
    но у кого-то должны быть, иначе связка со списком ЛК потерялась."""
    entries = build(today=TODAY)
    assert any(e.phone for e in entries)
    assert all((e.phone is None) == (e.email is None) for e in entries)
