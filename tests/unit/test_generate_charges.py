"""Проверки генератора начислений.

Генератор заменил присланный файл, который не связывался со списком счетов ни
одним номером. Поэтому проверяется не «работает ли код», а **выполняются ли
свойства, ради которых он написан**: связность, воспроизводимость и то, что
данные не противоречат сами себе.

Проверки идут по уже сгенерированному файлу, а не по подмене: он и есть то, на
чём живёт прототип, и подмена проверила бы не его.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_example"
CHARGES = DATA / "начисления.csv"

SEWAGE_FOR = {"ХВ": "ХВО", "ГВ": "ГВО"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _money(raw: str) -> Decimal:
    return Decimal(raw.strip().replace(",", "."))


@pytest.fixture(scope="module")
def charges() -> list[dict[str, str]]:
    return _rows(CHARGES)


@pytest.fixture(scope="module")
def accounts() -> dict[str, Decimal]:
    return {
        row["Номер лицевого счёта"]: _money(row["Задолженность"])
        for row in _rows(DATA / "ЛСФЛ.csv")
    }


# --- состав ---------------------------------------------------------------------- #


def test_начисления_за_три_периода(charges):
    assert len({row["Расчётный период"] for row in charges}) == 3


def test_начисления_у_девяноста_процентов_счетов(charges, accounts):
    """Доля задана владельцем. Оставшиеся десять процентов — не забытые, а
    правдоподобные: закрытые счета, недавно открытые, нежилые помещения."""
    covered = {row["Номер счёта"] for row in charges}
    assert round(len(covered) / len(accounts), 2) == 0.90


def test_у_счёта_несколько_услуг_за_месяц(charges):
    """Прямое требование владельца: за один месяц у счёта несколько строк."""
    per_month = defaultdict(set)
    for row in charges:
        per_month[(row["Номер счёта"], row["Расчётный период"])].add(row["Вид услуги"])
    assert max(len(services) for services in per_month.values()) > 1


def test_виды_услуг_только_из_перечня(charges):
    assert {row["Вид услуги"] for row in charges} <= {"ХВ", "ХВО", "ГВ", "ГВО", "Полив"}


def test_основания_только_из_перечня(charges):
    assert {row["Основание начисления"] for row in charges} <= {
        "по счётчику",
        "по нормативу",
        "перерасчёт",
    }


# --- водоотведение следует за своей водой ------------------------------------------ #


@pytest.mark.parametrize(("water", "sewage"), SEWAGE_FOR.items())
def test_водоотведение_один_к_одному_со_своей_водой(charges, water: str, sewage: str):
    """ХВО за ХВ, ГВО за ГВ. Строка водоотведения без своей воды означала бы
    плату за отвод того, что не подавали."""
    water_keys = {
        (r["Номер счёта"], r["Расчётный период"]) for r in charges if r["Вид услуги"] == water
    }
    sewage_keys = {
        (r["Номер счёта"], r["Расчётный период"]) for r in charges if r["Вид услуги"] == sewage
    }
    assert water_keys == sewage_keys


@pytest.mark.parametrize(("water", "sewage"), SEWAGE_FOR.items())
def test_у_воды_и_её_водоотведения_одно_основание(charges, water: str, sewage: str):
    """Пара «вода по счётчику, водоотведение по нормативу» в жизни не
    встречается, и первый же вопрос по такой строке остался бы без ответа."""
    basis = {
        (r["Номер счёта"], r["Расчётный период"], r["Вид услуги"]): r["Основание начисления"]
        for r in charges
    }
    for (account, period, service), value in basis.items():
        if service == sewage:
            assert basis[(account, period, water)] == value


def test_у_полива_нет_водоотведения(charges):
    """Вода на полив в канализацию не попадает: начислять за неё водоотведение
    значило бы брать плату за то, чего не было.

    Проверяется счётом: строк водоотведения ровно столько же, сколько строк ХВ и
    ГВ вместе. Полив в этот счёт не добавляет ничего.
    """
    kinds = [row["Вид услуги"] for row in charges]
    water = sum(1 for k in kinds if k in SEWAGE_FOR)
    sewage = sum(1 for k in kinds if k in SEWAGE_FOR.values())
    assert kinds.count("Полив") > 0
    assert water == sewage


# --- согласие с соседними файлами ---------------------------------------------------- #


def test_горячая_вода_только_у_тех_у_кого_есть_счётчик_гвс(charges):
    """Иначе абонент платил бы за услугу, прибора учёта которой у него нет."""
    with_hot = {
        row["Номер счёта"]
        for row in _rows(DATA / "счетчики.csv")
        if row["Тип ИПУ"].strip() == "ГВС"
    }
    charged_hot = {row["Номер счёта"] for row in charges if row["Вид услуги"] == "ГВ"}
    assert charged_hot <= with_hot


def test_долг_по_строкам_не_превышает_долг_счёта(charges, accounts):
    """Разнесённый долг обязан помещаться в задолженность из ЛСФЛ.

    Меньше — законно: часть долга старше трёхмесячного окна, и показывать её
    внутри окна значило бы выдумать начисления, которых в нём не было. Больше —
    расхождение, которое абонент заметит первым.
    """
    by_account: dict[str, Decimal] = defaultdict(Decimal)
    for row in charges:
        by_account[row["Номер счёта"]] += _money(row["Задолженность"])

    for account, debt in by_account.items():
        assert debt <= accounts[account], f"счёт {account}: разнесено больше долга"


def test_оплачено_и_долг_в_сумме_дают_начислено(charges):
    """Копейки обязаны сходиться в каждой строке: пропорциональное разнесение
    округляет, и без досыла остатка в последнюю строку итог разъехался бы."""
    for row in charges:
        accrued = _money(row["Начислено"])
        assert _money(row["Оплачено"]) + _money(row["Задолженность"]) == accrued


def test_свежие_периоды_не_оплачены_раньше_старых(charges, accounts):
    """Абонент, заплативший за август, но не за июнь, — признак того, что долг
    разносится не с того конца."""
    debts: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    for row in charges:
        debts[(row["Номер счёта"], row["Расчётный период"])] += _money(row["Задолженность"])

    by_account: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)
    for (account, period), debt in debts.items():
        by_account[account].append((period, debt))

    for account, months in by_account.items():
        ordered = [debt for _, debt in sorted(months)]
        paid_after_unpaid = any(
            ordered[i] > 0 and ordered[i + 1] == 0 for i in range(len(ordered) - 1)
        )
        assert not paid_after_unpaid, f"счёт {account}: старый период не оплачен, свежий оплачен"


# --- воспроизводимость ---------------------------------------------------------------- #


@pytest.mark.slow
def test_одно_зерно_даёт_тот_же_файл(tmp_path: Path):
    """Требование шага 0 плана: повторный запуск с тем же зерном даёт побайтово
    тот же файл. Случайность нужна для правдоподобия, а не для сюрпризов.
    """
    work = tmp_path / "data"
    work.mkdir()
    for name in ("ЛСФЛ.csv", "счетчики.csv"):
        (work / name).write_bytes((DATA / name).read_bytes())

    def run() -> bytes:
        subprocess.run(
            [sys.executable, "scripts/generate_charges.py", "--directory", str(work)],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return (work / "начисления.csv").read_bytes()

    assert run() == run()
