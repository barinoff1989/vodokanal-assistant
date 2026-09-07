"""Сдвинуть график отключений в будущее — файл для демонстрации.

ЗАЧЕМ

Настоящий график владельца описывает 21 июля — 28 августа 2026. К сентябрю все
1939 записей оказались в прошлом, и ответчик отключений честно отвечает
«плановых отключений нет» на любой адрес. Путь работает, но показать на защите
нечего.

ПОЧЕМУ ОТДЕЛЬНЫЙ ФАЙЛ, А НЕ ПРАВКА НАСТОЯЩЕГО

`tests/data/outage_schedule.txt` — **настоящая выгрузка владельца**, и на ней
стоят 36 проверок разбора. В ней записаны дефекты источника, которые проект
измерял: 14 формулировок периода на 21 заголовок, перевёрнутый период «с 28 июля
по 10 июля», двойные типы улиц, висячие запятые. Правка дат на месте уничтожила
бы и образец, и то, что проверки про него доказывают.

Поэтому демонстрационный файл **производный**: он собирается отсюда, помечен в
первой строке и пересобирается одной командой.

ЧТО СДВИГ СОХРАНЯЕТ, А ЧТО НЕТ

Сдвиг **равномерный**, на одно и то же число дней для всех дат. Так сохраняются
длительности периодов (от 3 до 17 дней), пересечения периодов у одного адреса и
перевёрнутый период — то есть всё, на чём проверяется поведение загрузчика.

**Формулировки нормализуются** к одному виду «с D месяца по D месяца». Четырнадцать
вариантов написания — свойство настоящего файла, и проверяются они на нём;
воспроизводить их в производном файле незачем, а разбирать при сдвиге — работа
без потребителя.

**Диапазон не сжимается.** Настоящий график занимает 39 дней, а окно
06.09 — 01.10 короче. Сжать значило бы изменить длительности отключений, а это
данные, а не оформление: сдвиг ставит начало на заданную дату, конец приходится
куда придётся, и скрипт печатает получившийся диапазон.

ЗАПУСК

    python scripts/make_demo_schedule.py                 # начало 6 сентября
    python scripts/make_demo_schedule.py --start 2026-10-01
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.outages.parser import parse_period  # noqa: E402

SOURCE = ROOT / "tests" / "data" / "outage_schedule.txt"
OUTPUT = ROOT / "data_example" / "график_отключений_демо.txt"

DEFAULT_START = date(2026, 9, 6)
YEAR = 2026

MONTHS = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля", 5: "мая", 6: "июня",
    7: "июля", 8: "августа", 9: "сентября", 10: "октября", 11: "ноября",
    12: "декабря",
}

MARK = (
    "# ДЕМОНСТРАЦИОННЫЙ ФАЙЛ. Собран scripts/make_demo_schedule.py из настоящей "
    "выгрузки владельца сдвигом всех дат на {shift} дн. Улицы и дома настоящие, "
    "даты — нет. Настоящий график: tests/data/outage_schedule.txt"
)

_PERIOD_LINE = re.compile(r"^\s*[сС]\s+\d")


def shifted_line(text: str, shift: int) -> str | None:
    """Переписать строку периода со сдвинутыми датами.

    ``None`` — строка периодом не является либо не разобралась; тогда она
    переносится как есть. Молча терять период нельзя: у него остался бы список
    домов без заголовка, и загрузчик отнёс бы их к предыдущему периоду.
    """
    periods = parse_period(text, YEAR)
    if not periods:
        return None
    parts = [
        f"с {s.day} {MONTHS[s.month]} по {e.day} {MONTHS[e.month]}"
        for s, e in (
            (start + timedelta(days=shift), end + timedelta(days=shift))
            for start, end in periods
        )
    ]
    return " и ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help="на какую дату поставить начало самого раннего периода",
    )
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    lines = SOURCE.read_text(encoding="utf-8").splitlines()

    # Сдвиг считается по самой ранней дате всего файла, а не по первой строке:
    # периоды идут не по возрастанию, и «первый» не значит «самый ранний».
    starts = [
        start
        for line in lines
        if _PERIOD_LINE.match(line)
        for start, _ in parse_period(line, YEAR)
    ]
    if not starts:
        sys.exit("в источнике не нашлось ни одного периода — разбор сломан")
    shift = (args.start - min(starts)).days

    result = [MARK.format(shift=shift), ""]
    moved = 0
    for line in lines:
        if _PERIOD_LINE.match(line) and (new := shifted_line(line, shift)) is not None:
            result.append(new)
            moved += 1
        else:
            result.append(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(result) + "\n", encoding="utf-8")

    ends = [
        end
        for line in lines
        if _PERIOD_LINE.match(line)
        for _, end in parse_period(line, YEAR)
    ]
    print(f"сдвиг: {shift} дн.")
    print(f"периодов переписано: {moved}")
    print(f"было:  {min(starts)} … {max(ends)}")
    print(
        f"стало: {min(starts) + timedelta(days=shift)} … "
        f"{max(ends) + timedelta(days=shift)}"
    )
    print(f"записано: {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
