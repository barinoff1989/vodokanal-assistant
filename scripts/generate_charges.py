"""Генератор начислений по лицевым счетам примера.

Присланный пример начислений **не связывался со списком счетов**: ни один из его
тридцати счетов не встречался среди двухсот счетов ЛСФЛ, а совпадал он со
счетами обращений. Два непересекающихся набора данных, и вопрос «почему такая
сумма» показать было не на ком.

Этот скрипт заменяет тот файл: начисления строятся **из самих счетов ЛСФЛ**,
поэтому связность выполняется по построению, а не проверяется задним числом.

ЧТО ЗАДАНО ВЛАДЕЛЬЦЕМ

* вид услуги — ХВ, ГВ, полив, водоотведение холодной (ХВО) и горячей (ГВО);
* основание — по счётчику, по нормативу или перерасчёт;
* у одного счёта может быть несколько строк за месяц, по разным услугам;
* три последних месяца, для каждого выбранного счёта;
* счета берутся у 90% списка.

**Водоотведение раздельное и идёт за своей водой:** ХВО за ХВ, ГВО за ГВ, один
к одному. Это не украшение: водоотведение начисляется по тому же объёму, что и
поданная вода, поэтому у него **то же основание**, что у соответствующей услуги,
и сумма считается долей от неё, а не разыгрывается заново.

Независимый розыгрыш дал бы пары, где вода по счётчику, а её водоотведение по
нормативу, — такого не бывает, и первый же вопрос по такой строке оказался бы
без ответа. По той же причине ГВО не появляется у тех, у кого нет ГВ.

ОТКУДА БЕРУТСЯ ОСНОВАНИЯ И УСЛУГИ — ИЗ ДРУГИХ ФАЙЛОВ ПРИМЕРА

Выдумывать их независимо значило бы получить третий несвязный набор. Поэтому:

* **ГВ начисляется только тем, у кого есть счётчик ГВС.** Иначе абонент платил
  бы за услугу, прибора учёта которой у него нет.
* **Основание следует состоянию счётчика.** Есть действующий ИПУ — «по
  счётчику»; нет прибора или истёк межповерочный интервал — «по нормативу».
  Ровно это и объясняет FAQ в вопросах 3 и 10, и ровно об этом спрашивают
  абоненты.
* **Задолженность сходится с той, что стоит в ЛСФЛ.** Долг счёта разносится по
  неоплаченным начислениям, а не назначается независимо: расхождение между
  «долг 1250» в одном файле и суммой строк в другом абонент заметил бы первым.

ПОВТОРЯЕМОСТЬ. Одно зерно даёт побайтово тот же файл — требование проекта.
Случайность нужна для правдоподобия, а не для сюрпризов.

ЗАПУСК

    python scripts/generate_charges.py
    python scripts/generate_charges.py --seed 42 --as-of 2026-09-01
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

DATA = Path("data_example")
ACCOUNTS_FILE = "ЛСФЛ.csv"
METERS_FILE = "счетчики.csv"
OUTPUT_FILE = "начисления.csv"

DELIMITER = ";"
ENCODING = "utf-8"
HEADER = [
    "Номер счёта",
    "Расчётный период",
    "Вид услуги",
    "Начислено",
    "Оплачено",
    "Задолженность",
    "Основание начисления",
]

ACCOUNT_SHARE = 0.90
"""Доля счетов, по которым есть начисления. Задана владельцем.

Остальные десять процентов — не забытые, а правдоподобные: закрытые счета,
недавно открытые, нежилые помещения."""

MONTHS = 3

SERVICES = {
    "ХВ": (Decimal("180"), Decimal("650")),
    "ГВ": (Decimal("320"), Decimal("980")),
    "Полив": (Decimal("90"), Decimal("260")),
}
"""Услуга и разброс месячной суммы.

Водоотведения здесь нет, потому что оно не разыгрывается: его сумма выводится из
ХВ (см. :data:`SEWAGE_RATIO`)."""

SEWAGE_FOR = {"ХВ": "ХВО", "ГВ": "ГВО"}
"""Какой воде какое водоотведение соответствует.

**Полива здесь нет намеренно:** вода на полив в канализацию не попадает, и
начислять за неё водоотведение значило бы брать плату за то, чего не было."""

SEWAGE_RATIO = Decimal("0.48")
"""Доля от суммы соответствующей услуги. Взята из присланного примера: 198,30
при 412,50 — около сорока восьми процентов; тариф водоотведения действительно
ниже тарифа подачи."""

IRRIGATION_SHARE = 0.22
"""Доля счетов с поливом. Услуга сезонная и не у всех: в присланном примере она
встречалась примерно у каждого пятого."""

RECALCULATION_SHARE = 0.06
"""Доля строк с основанием «перерасчёт». Событие редкое, но именно оно чаще
всего и приводит абонента с вопросом."""


@dataclass(frozen=True, slots=True)
class AccountFacts:
    """Что известно о счёте из соседних файлов."""

    number: str
    debt: Decimal
    has_cold_meter: bool
    has_hot_meter: bool
    verification_overdue: bool


def _money(raw: str) -> Decimal:
    value = raw.strip().replace(" ", "").replace("\xa0", "").replace(",", ".")
    return Decimal(value) if value else Decimal("0")


def _out(value: Decimal) -> str:
    """Сумма в том же виде, в каком она пришла: запятая, две цифры."""
    return f"{value:.2f}".replace(".", ",")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding=ENCODING, newline="") as handle:
        return list(csv.DictReader(handle, delimiter=DELIMITER))


def collect_facts(directory: Path, today: date) -> list[AccountFacts]:
    """Свести счета со счётчиками — основание начисления зависит от прибора."""
    meters: dict[str, list[dict[str, str]]] = {}
    for row in _read(directory / METERS_FILE):
        meters.setdefault(row["Номер счёта"].strip(), []).append(row)

    facts = []
    for row in _read(directory / ACCOUNTS_FILE):
        number = row["Номер лицевого счёта"].strip()
        own = meters.get(number, [])
        overdue = False
        for meter in own:
            raw = meter["Срок следующей поверки"].strip()
            try:
                if date.fromisoformat(raw) < today:
                    overdue = True
            except ValueError:
                continue
        facts.append(
            AccountFacts(
                number=number,
                debt=_money(row["Задолженность"]),
                has_cold_meter=any(m["Тип ИПУ"].strip() == "ХВС" for m in own),
                has_hot_meter=any(m["Тип ИПУ"].strip() == "ГВС" for m in own),
                verification_overdue=overdue,
            )
        )
    return facts


def periods(as_of: date, count: int = MONTHS) -> list[str]:
    """Последние расчётные периоды, от старого к новому.

    Берётся месяц, **предшествующий** текущему: за незакрытый месяц начислений
    ещё нет, и показывать их значило бы показывать несуществующее.
    """
    year, month = as_of.year, as_of.month
    result = []
    for step in range(count, 0, -1):
        total = year * 12 + (month - 1) - step
        result.append(f"{total // 12:04d}-{total % 12 + 1:02d}")
    return result


def services_for(facts: AccountFacts, rng: random.Random) -> list[str]:
    """Какие услуги начисляются этому счёту.

    ГВ — только при наличии счётчика ГВС: иначе абонент платил бы за услугу,
    прибора учёта которой у него нет, и первый же вопрос по такой строке
    оказался бы без ответа.
    """
    chosen = ["ХВ"]
    if facts.has_hot_meter:
        chosen.append("ГВ")
    if rng.random() < IRRIGATION_SHARE:
        chosen.append("Полив")
    return chosen


def basis_for(facts: AccountFacts, service: str, rng: random.Random) -> str:
    """Основание начисления по состоянию прибора учёта.

    Полив считается по нормативу всегда: прибора учёта на него не ставят.
    """
    if service == "Полив":
        return "по нормативу"
    if rng.random() < RECALCULATION_SHARE:
        return "перерасчёт"
    if service == "ГВ":
        by_meter = facts.has_hot_meter and not facts.verification_overdue
        return "по счётчику" if by_meter else "по нормативу"
    if not facts.has_cold_meter or facts.verification_overdue:
        return "по нормативу"
    return "по счётчику"


def generate(facts: list[AccountFacts], as_of: date, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    chosen = sorted(facts, key=lambda f: f.number)
    take = round(len(chosen) * ACCOUNT_SHARE)
    chosen = rng.sample(chosen, take)
    chosen.sort(key=lambda f: f.number)

    rows: list[list[str]] = []
    for account in chosen:
        services = services_for(account, rng)
        month_rows: list[list[str]] = []
        for period in periods(as_of):
            for service in services:
                low, high = SERVICES[service]
                amount = Decimal(rng.randrange(int(low) * 100, int(high) * 100)) / 100
                basis = basis_for(account, service, rng)
                month_rows.append(
                    [
                        account.number,
                        period,
                        service,
                        _out(amount),
                        "",  # оплачено — проставляется ниже, когда известен долг
                        "",
                        basis,
                    ]
                )
                sewage_name = SEWAGE_FOR.get(service)
                if sewage_name is not None:
                    # Водоотведение идёт следом за своей водой и наследует её
                    # основание: пара «вода по счётчику, водоотведение по
                    # нормативу» в жизни не встречается.
                    sewage = (amount * SEWAGE_RATIO).quantize(Decimal("0.01"))
                    month_rows.append(
                        [
                            account.number,
                            period,
                            sewage_name,
                            _out(sewage),
                            "",
                            "",
                            basis,
                        ]
                    )
        rows.extend(_settle(month_rows, account.debt))
    return rows


def _settle(rows: list[list[str]], debt: Decimal) -> list[list[str]]:
    """Разнести долг счёта по начислениям, начиная со свежих периодов.

    Долг счёта из ЛСФЛ разносится по строкам, а не назначается независимо:
    расхождение между «долг 1250» в одном файле и суммой строк в другом абонент
    заметил бы первым, и объяснить его было бы нечем.

    **Разносится помесячно, а не построчно.** Построчный обход давал бы месяц,
    где холодная вода оплачена, а её водоотведение нет, — так не платят: абонент
    вносит сумму по квитанции, а не по отдельным услугам. Поэтому свежие месяцы
    не оплачены целиком, один месяц оплачен частично и **пропорционально по всем
    услугам**, старые оплачены полностью.

    **Долг может оказаться больше, чем начислено за три месяца** — тогда
    разносится столько, сколько помещается. Это не ошибка: часть задолженности
    старше окна, и показывать её внутри окна значило бы выдумать начисления,
    которых в нём не было.
    """
    by_period: dict[str, list[list[str]]] = {}
    for row in rows:
        by_period.setdefault(row[1], []).append(row)

    remaining = debt
    for period in sorted(by_period, reverse=True):
        month = by_period[period]
        total = sum(Decimal(row[3].replace(",", ".")) for row in month)
        share = min(total, remaining)
        remaining -= share

        left = share
        for index, row in enumerate(month):
            accrued = Decimal(row[3].replace(",", "."))
            if index == len(month) - 1:
                unpaid = left  # остаток целиком, чтобы копейки сошлись
            else:
                portion = (accrued * share / total) if total else Decimal("0")
                unpaid = min(portion.quantize(Decimal("0.01")), left)
            left -= unpaid
            row[4] = _out(accrued - unpaid)
            row[5] = _out(unpaid)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--directory", type=Path, default=DATA)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date(2026, 9, 1),
        help="от какой даты отсчитываются три последних периода",
    )
    args = parser.parse_args(argv)

    facts = collect_facts(args.directory, args.as_of)
    rows = generate(facts, args.as_of, args.seed)

    output = args.directory / OUTPUT_FILE
    with output.open("w", encoding=ENCODING, newline="") as handle:
        writer = csv.writer(handle, delimiter=DELIMITER)
        writer.writerow(HEADER)
        writer.writerows(rows)

    accounts = {row[0] for row in rows}
    print(f"записано: {output}")
    print(f"  строк: {len(rows)}")
    print(f"  счетов: {len(accounts)} из {len(facts)} ({100 * len(accounts) // len(facts)}%)")
    print(f"  периоды: {', '.join(periods(args.as_of))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
