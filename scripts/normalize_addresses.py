"""Привести адреса примера к воронежским — иначе отключения не показать.

Адреса в присланном примере сгенерированы обобщёнными названиями — Ленина, Мира,
Садовая, Заводская — и городом «г. Тестовый». Прототип же обслуживает Воронеж, а
график плановых отключений идёт по воронежским улицам, и совпадала из двенадцати
ровно одна, да и то случайно («Набережная»).

Следствие простое: сценарий «по вашему адресу отключение» показать было не на
ком, хотя и график, и поиск по нему работают.

ОТКУДА БЕРУТСЯ НОВЫЕ АДРЕСА — ИЗ САМОГО ГРАФИКА

Улицы не выдумываются и не берутся из справочника: они читаются из
`tests/data/outage_schedule.txt`. Любой другой источник дал бы третий несвязный
набор — ту же болезнь, которой страдали присланные начисления.

ЧАСТЬ АДРЕСОВ ПОПАДАЕТ В ГРАФИК, ЧАСТЬ — НЕТ, И ЭТО НАМЕРЕННО

Если бы отключение находилось у каждого абонента, демонстрация показывала бы не
работу поиска, а его неспособность ответить «нет». Поэтому доля
:data:`AFFECTED_SHARE` получает дом, который в графике **есть**, остальные —
дом на той же улице, которого в графике **нет**.

Оба ответа настоящие и оба нужны: «отключение с 24 по 28 августа» и «плановых
отключений по вашему адресу нет».

ЧТО НЕ ТРОГАЕТСЯ. Номер квартиры остаётся прежним: он ни на что не влияет и
менять его — лишнее движение в чужих данных.

ПОВТОРЯЕМОСТЬ. Одно зерно даёт побайтово тот же файл: адрес выводится из номера
лицевого счёта, а не из порядка строк.

ЗАПУСК

    python scripts/normalize_addresses.py
    python scripts/normalize_addresses.py --seed 42 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from app.outages.parser import parse_schedule

DATA = Path("data_example")
ACCOUNTS_FILE = "ЛСФЛ.csv"
SCHEDULE = Path("tests/data/outage_schedule.txt")

DELIMITER = ";"
ENCODING = "utf-8"
CITY = "г. Воронеж"

AFFECTED_SHARE = 0.40
"""Доля счетов, чей дом есть в графике отключений.

Не сто процентов: демонстрация обязана показывать и ответ «отключений нет» —
иначе проверить, что поиск умеет отвечать отрицательно, будет негде."""

_FLAT = re.compile(r"(кв\.\s*\d+)", re.IGNORECASE)


def schedule_addresses(path: Path, year: int = 2026) -> dict[str, list[str]]:
    """Улицы и дома из графика: где что отключают.

    Названия берутся в том виде, в каком они записаны в источнике
    (`street_raw`), а не нормализованными: в адрес абонента должно попасть
    читаемое «ул. Димитрова», а не «димитрова».
    """
    outages = parse_schedule(path.read_text(encoding=ENCODING).splitlines(), year=year)
    houses: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    for outage in outages:
        houses[outage.street].add(outage.house_raw.strip())
        # Из нескольких написаний берётся самое полное — с типом улицы.
        current = display.get(outage.street, "")
        if len(outage.street_raw) > len(current):
            display[outage.street] = outage.street_raw
    return {display[street]: sorted(found) for street, found in houses.items()}


def unaffected_house(taken: list[str], rng: random.Random) -> str:
    """Номер дома, которого на этой улице в графике нет.

    Берётся заведомо больший номер, а не случайный: случайный мог бы совпасть с
    отключаемым, и тогда «отключений нет» оказалось бы неправдой.
    """
    numbers = [int(m.group()) for h in taken if (m := re.match(r"\d+", h))]
    highest = max(numbers, default=100)
    return str(highest + rng.randrange(2, 40))


def rebuild(address: str, street: str, house: str) -> str:
    """Собрать адрес заново, сохранив номер квартиры."""
    flat = _FLAT.search(address)
    parts = [CITY, street, f"д. {house}"]
    if flat is not None:
        parts.append(flat.group(1))
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--directory", type=Path, default=DATA)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    known = schedule_addresses(args.schedule)
    if not known:
        print("график отключений не разобрался — адреса не тронуты", file=sys.stderr)
        return 1
    streets = sorted(known)

    path = args.directory / ACCOUNTS_FILE
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=DELIMITER)
        fields = reader.fieldnames or []
        rows = list(reader)

    rng = random.Random(args.seed)
    affected = 0
    for row in sorted(rows, key=lambda r: r["Номер лицевого счёта"]):
        street = rng.choice(streets)
        houses = known[street]
        if rng.random() < AFFECTED_SHARE:
            house = rng.choice(houses)
            affected += 1
        else:
            house = unaffected_house(houses, rng)
        row["Адрес"] = rebuild(row["Адрес"], street, house)

    if not args.dry_run:
        with path.open("w", encoding=ENCODING, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=DELIMITER)
            writer.writeheader()
            writer.writerows(rows)
        print(f"записано: {path}")

    print(f"  счетов: {len(rows)}")
    print(f"  с домом из графика: {affected} ({100 * affected // len(rows)}%)")
    print(f"  улиц использовано: {len({r['Адрес'].split(',')[1].strip() for r in rows})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
