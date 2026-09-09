"""Проверки доступа к данным Биллинга и личного кабинета.

Работают на **настоящем примере** из `data_example/`: реальных данных не будет
вовсе, и этот пример — то, на чём прототип живёт. Проверять его подменой значило
бы не проверить ничего.

Часть проверок сторожит не наш код, а **связность самих данных** — это прямое
требование шага 0 плана разработки: «каждый лицевой счёт в начислениях
существует в списке счетов».

Обе такие проверки некоторое время стояли с `xfail(strict=True)` — начисления не
связывались со счетами, адреса не годились для поиска отключений. Обе пометки
сняты вместе с расхождениями, и оба раза о починке сообщил сам тест: строгий
`xfail` падает, когда ожидаемая ошибка исчезает. Без строгости починка прошла бы
незамеченной, а пометка осталась бы описывать то, чего больше нет.
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from app.billing.source import CsvBillingSource

DATA = Path(__file__).resolve().parents[2] / "data_example"


CHARGED_ACCOUNT = "2100202213"
"""Счёт с долгом и начислениями за все три периода — на нём видны и разнесение
долга, и порядок периодов."""


@pytest.fixture(scope="module")
def billing() -> CsvBillingSource:
    return CsvBillingSource(DATA)


# --- чтение --------------------------------------------------------------------- #


def test_счета_читаются(billing: CsvBillingSource):
    assert len(billing) == 200


def test_перечень_всех_счетов_доступен_стенду(billing: CsvBillingSource):
    """`accounts()` не в протоколе `BillingSource` — настоящий API его не отдаст.
    Он только для справочника демо-стенда, и вернуть должен ровно то же, что
    видно через `account()`."""
    everyone = billing.accounts()
    assert len(everyone) == len(billing)
    assert billing.account(everyone[0].number) == everyone[0]


def test_счёт_находится_по_номеру(billing: CsvBillingSource):
    account = billing.account("2100202213")

    assert account is not None
    assert account.full_name == "Петров Николай Егорович"
    assert account.opened_on == date(2012, 7, 22)


def test_суммы_читаются_точно_а_не_приблизительно(billing: CsvBillingSource):
    """Деньги — `Decimal`, а не `float`.

    Суммы приходят с запятой и участвуют в ответах о задолженности. Ошибку
    округления в копейках абонент замечает первой, и доверия она стоит дороже
    любой другой.
    """
    account = billing.account("2100202213")

    assert account is not None
    assert account.debt == Decimal("1250.00")
    assert account.balance == Decimal("-1250.00")
    assert isinstance(account.debt, Decimal)


def test_неизвестный_счёт_даёт_пусто_а_не_ошибку(billing: CsvBillingSource):
    """Абонент с чужим номером должен получить «не найдено», а не пятисотую."""
    assert billing.account("0000000000") is None
    assert billing.meters("0000000000") == []
    assert billing.charges("0000000000") == []


def test_пробелы_вокруг_номера_не_мешают(billing: CsvBillingSource):
    assert billing.account("  2100202213  ") is not None


# --- счётчики ------------------------------------------------------------------- #


def test_просроченная_поверка_видна(billing: CsvBillingSource):
    """Один из самых частых вопросов и прямая причина начисления по нормативу с
    повышающим коэффициентом (FAQ, вопросы 3 и 10)."""
    meters = billing.meters("2100202213")
    overdue = [m for m in meters if m.verification_overdue(date(2026, 9, 3))]

    assert overdue
    assert overdue[0].verify_by is not None


def test_у_счёта_может_быть_несколько_счётчиков(billing: CsvBillingSource):
    """ХВС и ГВС — разные приборы, и ответ про «счётчик» в единственном числе
    был бы неполным."""
    kinds = {m.kind for m in billing.meters("2100202213")}
    assert len(kinds) > 1


# --- начисления ------------------------------------------------------------------ #


def test_начисления_идут_от_свежего_периода(billing: CsvBillingSource):
    charges = billing.charges(CHARGED_ACCOUNT)

    assert charges
    assert [c.period for c in charges] == sorted(
        (c.period for c in charges), reverse=True
    )


def test_основание_начисления_сохраняется(billing: CsvBillingSource):
    """Ради него начисления и нужны: вопрос «почему такая сумма» отвечается
    основанием, а не самой суммой."""
    bases = {c.basis for c in billing.charges(CHARGED_ACCOUNT)}
    assert bases & {"по счётчику", "по нормативу", "перерасчёт"}


# --- профиль личного кабинета ----------------------------------------------------- #


def test_профиль_связывает_пользователя_со_счётом(billing: CsvBillingSource):
    """Через него в запрос попадает адрес абонента — тот, по которому ищутся
    отключения."""
    profile = billing.profile("sub-2100303314")

    assert profile is not None
    assert profile.account == "2100303314"


# --- связность данных (требование шага 0 плана) ------------------------------------ #


def _column(filename: str, column: str) -> set[str]:
    with (DATA / filename).open(encoding="utf-8", newline="") as handle:
        return {row[column].strip() for row in csv.DictReader(handle, delimiter=";")}


def test_счета_личного_кабинета_существуют_в_биллинге():
    """Профиль без счёта означал бы абонента, которому нечего показать."""
    assert not _column("ЛКК.csv", "Лицевой счёт") - _column("ЛСФЛ.csv", "Номер лицевого счёта")


def test_счётчики_привязаны_к_существующим_счетам():
    assert not _column("счетчики.csv", "Номер счёта") - _column(
        "ЛСФЛ.csv", "Номер лицевого счёта"
    )


def test_начисления_привязаны_к_существующим_счетам():
    """Прямое требование шага 0 плана: «каждый лицевой счёт в начислениях
    существует в списке счетов».

    Некоторое время стояло `xfail`: присланный пример начислений не связывался
    со списком счетов вовсе. `strict=True` заставил тест упасть, когда файл
    перегенерировали, — и пометка снята вместе с расхождением. Ровно за этим
    строгий `xfail` и нужен: без него починка прошла бы незамеченной.
    """
    assert not _column("начисления.csv", "Номер счёта") - _column(
        "ЛСФЛ.csv", "Номер лицевого счёта"
    )


def test_адреса_годятся_для_поиска_отключений():
    """Прототип обслуживает Воронеж, и график отключений — по воронежским
    улицам.

    Второй тест, стоявший с `xfail(strict=True)`: адреса примера были
    сгенерированы обобщёнными названиями, и сценарий «по вашему адресу
    отключение» показать было не на ком. Пометка снята вместе с расхождением —
    адреса приведены `scripts/normalize_addresses.py`.
    """
    from app.outages.parser import normalize_street, parse_schedule

    schedule = parse_schedule(
        (DATA.parent / "tests" / "data" / "outage_schedule.txt")
        .read_text(encoding="utf-8")
        .splitlines(),
        year=2026,
    )
    known = {o.street for o in schedule}
    streets = {
        normalize_street(address.split(",")[1])
        for address in _column("ЛСФЛ.csv", "Адрес")
        if "," in address
    }
    assert streets <= known


def test_город_в_адресах_воронеж():
    """Прототип обслуживает один филиал (раздел 60.1).

    Город «г. Тестовый» остался бы виден на демонстрации рядом с воронежскими
    улицами графика — сочетание, которого не бывает.
    """
    cities = {address.split(",")[0].strip() for address in _column("ЛСФЛ.csv", "Адрес")}
    assert cities == {"г. Воронеж"}


def test_каждый_адрес_разбирается_на_улицу_и_дом():
    """Неразобранный адрес означает абонента, которому нельзя ответить про
    отключение: ответчик молчит, и запрос уходит к модели, которая графика не
    видела."""
    from app.outages.answer import parse_address

    for address in _column("ЛСФЛ.csv", "Адрес"):
        assert parse_address(address) is not None, address


def test_часть_адресов_попадает_в_график_а_часть_нет():
    """Если бы отключение находилось у каждого, демонстрация показывала бы не
    работу поиска, а его неспособность ответить «нет». Оба ответа настоящие и
    оба нужны.
    """
    from datetime import datetime

    from app.outages.answer import parse_address
    from app.outages.store import OutageStore

    store = OutageStore.from_file(
        DATA.parent / "tests" / "data" / "outage_schedule.txt",
        year=2026,
        loaded_at=datetime(2026, 8, 1, 9, 0),
    )
    addresses = _column("ЛСФЛ.csv", "Адрес")
    hits = sum(1 for a in addresses if store.find(*parse_address(a)))  # type: ignore[misc]

    assert 0 < hits < len(addresses)
