"""Доступ к данным Биллинга и личного кабинета.

На прототипе источник — **файлы примера** в `data_example/`. Реальных данных не
будет вовсе: на MVP они придут через API Биллинга, и заменится тогда только этот
модуль. Всё, что за ним, работает с записями, а не с файлами.

ОТСТУПЛЕНИЕ ОТ ADR-003, ЗАПИСАННОЕ ЯВНО

ADR-003 требует обращаться к чужим системам только через их API и никогда — в их
базу или файлы. На прототипе API нет: закрытая сеть, доступа не будет ни на
разработке, ни на демонстрации (раздел 39). Читать нечего, кроме примера.

Это **второе отступление того же рода**, что у графика отключений (ADR-013), и
довод тот же: шов проходит по источнику. Формы записей ниже описывают то, что
нужно ассистенту, а не то, что лежит в CSV, — иначе при переходе на API пришлось
бы переписывать всё, что стоит за чтением.

ДЕНЬГИ — `Decimal`, А НЕ `float`

Суммы приходят с запятой в качестве разделителя («1250,00») и участвуют в
ответах абоненту о задолженности. `float` здесь дал бы ошибку округления в
копейках, а такую ошибку абонент замечает первой и доверия она стоит дороже,
чем любая другая.

ЧЕГО ЗДЕСЬ НЕТ. Записи по-прежнему нет: на Этапе 1 Биллинг только читается
(раздел 5.3). Запись появится на Этапе 2 через `app/adapters/billing_adapter.py`
и через настоящий API, а не через эти файлы.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

__all__ = [
    "Account",
    "BillingSource",
    "Charge",
    "CsvBillingSource",
    "LkProfile",
    "Meter",
]

DELIMITER = ";"
ENCODING = "utf-8"


def _money(raw: str) -> Decimal:
    """Сумма из строки с запятой. Пустое значение — ноль, а не ошибка.

    Пустая клетка в выгрузке означает «не начислено», и падать на ней значило бы
    ронять ответ абоненту из-за пробела в чужих данных.
    """
    value = raw.strip().replace(" ", "").replace("\xa0", "").replace(",", ".")
    if not value:
        return Decimal("0")
    try:
        return Decimal(value)
    except InvalidOperation:
        return Decimal("0")


def _day(raw: str) -> date | None:
    """Дата в формате ISO. ``None`` — если пусто или не разобралась.

    Неразобранная дата честнее подставленной: «срок поверки неизвестен» — это
    ответ, а выдуманный срок абонент примет за настоящий.
    """
    value = raw.strip()
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Account:
    """Лицевой счёт физического лица (источник 1, файл `ЛСФЛ`)."""

    number: str
    full_name: str
    address: str
    balance: Decimal
    debt: Decimal
    opened_on: date | None

    @property
    def has_debt(self) -> bool:
        return self.debt > 0


@dataclass(frozen=True, slots=True)
class Meter:
    """Прибор учёта (источник 9, файл `счетчики`)."""

    account: str
    kind: str
    """ХВС или ГВС — как в выгрузке, без приведения к своему словарю.

    Приводить пришлось бы дважды: здесь и на MVP под настоящий API, а сойтись
    эти приведения могли бы не сразу."""

    serial: str
    installed_on: date | None
    verified_on: date | None
    verify_by: date | None
    reading: str
    reading_date: date | None

    def verification_overdue(self, today: date) -> bool:
        """Истёк ли межповерочный интервал.

        Один из самых частых вопросов абонента и прямая причина начисления по
        нормативу с повышающим коэффициентом (FAQ, вопросы 3 и 10).
        """
        return self.verify_by is not None and self.verify_by < today


@dataclass(frozen=True, slots=True)
class Charge:
    """Начисление за расчётный период (источник 1, файл `начисления`)."""

    account: str
    period: str
    service: str
    accrued: Decimal
    paid: Decimal
    debt: Decimal
    basis: str
    """«по счётчику», «по нормативу» или «перерасчёт».

    Ради этого поля начисления и нужны ассистенту: вопрос «почему такая сумма»
    отвечается основанием, а не самой суммой."""


@dataclass(frozen=True, slots=True)
class LkProfile:
    """Профиль личного кабинета (источник 4, файл `ЛКК`)."""

    user_id: str
    account: str
    phone: str
    email: str


class BillingSource(Protocol):
    """Что нужно ассистенту от Биллинга и личного кабинета.

    Это же — форма будущего адаптера к API: на MVP заменится реализация, а не
    вызовы.
    """

    def account(self, number: str) -> Account | None: ...

    def meters(self, number: str) -> list[Meter]: ...

    def charges(self, number: str) -> list[Charge]: ...

    def profile(self, user_id: str) -> LkProfile | None: ...


def _rows(path: Path) -> Iterator[dict[str, str]]:
    if not path.exists():
        return
    with path.open(encoding=ENCODING, newline="") as handle:
        yield from csv.DictReader(handle, delimiter=DELIMITER)


class CsvBillingSource:
    """Чтение примера данных из каталога.

    Всё поднимается в память при создании: двести счетов и двести счётчиков —
    это доли мегабайта, а запрос абонента не должен ждать разбора файла.
    """

    FILES = {
        "accounts": "ЛСФЛ.csv",
        "profiles": "ЛКК.csv",
        "meters": "счетчики.csv",
        "charges": "начисления.csv",
    }

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._accounts = {a.number: a for a in self._read_accounts()}
        self._profiles = {p.user_id: p for p in self._read_profiles()}
        self._meters: dict[str, list[Meter]] = {}
        for meter in self._read_meters():
            self._meters.setdefault(meter.account, []).append(meter)
        self._charges: dict[str, list[Charge]] = {}
        for charge in self._read_charges():
            self._charges.setdefault(charge.account, []).append(charge)

    def __len__(self) -> int:
        return len(self._accounts)

    def _path(self, key: str) -> Path:
        return self._directory / self.FILES[key]

    def _read_accounts(self) -> Iterator[Account]:
        for row in _rows(self._path("accounts")):
            yield Account(
                number=row["Номер лицевого счёта"].strip(),
                full_name=row["ФИО"].strip(),
                address=row["Адрес"].strip(),
                balance=_money(row["Баланс"]),
                debt=_money(row["Задолженность"]),
                opened_on=_day(row["Дата открытия"]),
            )

    def _read_profiles(self) -> Iterator[LkProfile]:
        for row in _rows(self._path("profiles")):
            yield LkProfile(
                user_id=row["Идентификатор пользователя"].strip(),
                account=row["Лицевой счёт"].strip(),
                phone=row["Телефон"].strip(),
                email=row["Электронная почта"].strip(),
            )

    def _read_meters(self) -> Iterator[Meter]:
        for row in _rows(self._path("meters")):
            yield Meter(
                account=row["Номер счёта"].strip(),
                kind=row["Тип ИПУ"].strip(),
                serial=row["Заводской номер счётчика"].strip(),
                installed_on=_day(row["Дата установки"]),
                verified_on=_day(row["Дата последней поверки"]),
                verify_by=_day(row["Срок следующей поверки"]),
                reading=row["Последнее показание"].strip(),
                reading_date=_day(row["Дата передачи показания"]),
            )

    def _read_charges(self) -> Iterator[Charge]:
        for row in _rows(self._path("charges")):
            yield Charge(
                account=row["Номер счёта"].strip(),
                period=row["Расчётный период"].strip(),
                service=row["Вид услуги"].strip(),
                accrued=_money(row["Начислено"]),
                paid=_money(row["Оплачено"]),
                debt=_money(row["Задолженность"]),
                basis=row["Основание начисления"].strip(),
            )

    # --- чтение ------------------------------------------------------------ #

    def account(self, number: str) -> Account | None:
        return self._accounts.get(number.strip())

    def meters(self, number: str) -> list[Meter]:
        return list(self._meters.get(number.strip(), ()))

    def charges(self, number: str) -> list[Charge]:
        """Начисления по счёту, от свежего периода к старому."""
        found = self._charges.get(number.strip(), ())
        return sorted(found, key=lambda c: c.period, reverse=True)

    def profile(self, user_id: str) -> LkProfile | None:
        return self._profiles.get(user_id.strip())
