"""Забрать таблицу тарифов воронежского филиала — питьевая вода и водоотведение.

ПОЧЕМУ ТАРИФЫ НЕ ИДУТ В БАЗУ ЗНАНИЙ

Правило, выведенное ADR-100: **база знаний
отвечает на то, что объясняется; то, что проверяется по факту, отвечается точным
поиском, а не генерацией по найденному.**

Тариф — третий случай этого правила после регламентной формулировки и графика
отключений. Таблица из двадцати периодов до 2034 года — ровно тот материал, на
котором модель уверенно назовёт не тот год: соседние строки различаются только
датами и парой цифр, а цена ошибки — неверная сумма в ответе про деньги.

Общее у всех трёх случаев то же: **нужен поиск по ключу, а не по смыслу.**
Ключ здесь — дата.

ЧТО ПРОВЕРЯЕТСЯ ПРИ ЗАГРУЗКЕ, И ЗАЧЕМ

На странице **есть дефект**: одна строка объявлена как «с 01.01.2023 по
30.06.2030» — год начала опечатан, по порядку следования там 2030. Взять её как
есть значит отвечать тарифом 2030 года на вопрос про 2023–2024: интервал
накрывает семь лишних лет.

Поэтому загрузчик **сверяет периоды между собой** и отбрасывает строку, которая
налезает на предыдущую или кончается раньше, чем начинается. Отброшенная строка
оставляет пробел, а пробел честен: ответчик на него промолчит. Подставленное
значение — нет.

Это тот же выбор, что с графиком отключений, но решённый иначе, и
разница существенна: там противоречие было **свойством документа**, собранного
человеком по периодам, и терялись адреса; здесь противоречие делает **неверным
сам ответ про деньги**.

ЗАПУСК

    python scripts/fetch_tariffs.py
    python scripts/fetch_tariffs.py --check   # сверить с сохранённым, не писать
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

SOURCE_URL = "https://voronezh.rosvodokanal.ru/users/individual/rates/"
SOURCE_TITLE = "Тарифы и нормативы — ООО «РВК-Воронеж»"
DEFAULT_OUTPUT = Path("reference/tariffs_voronezh.json")

SERVICES: tuple[tuple[str, str], ...] = (
    ("питьевая вода", "питьев"),
    ("водоотведение", "водоотведени"),
)
"""Порядок таблиц на странице и признак заголовка для каждой.

Первая таблица — вода, вторая — водоотведение. **Порядок проверяется, а не
берётся на веру:** поменяй страница таблицы местами, и абоненту молча называлась
бы чужая цена.

Признак задан явно, а не выведен из названия. Первая редакция брала последнее
слово названия — «вода», — а в заголовке стоит «на питьевую **воду**»: правило,
угаданное из имени, не совпало с текстом источника при первом же прогоне."""

_TABLE = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)
_ROW = re.compile(r"<tr.*?</tr>", re.DOTALL | re.IGNORECASE)
_CELL = re.compile(r"<t[dh].*?</t[dh]>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_PERIOD = re.compile(r"с\s+(\d{2})\.(\d{2})\.(\d{4})\s+по\s+(\d{2})\.(\d{2})\.(\d{4})")
_MONEY = re.compile(r"^\d+([.,]\d+)?$")


def cell_text(fragment: str) -> str:
    text = _TAG.sub("", fragment)
    return re.sub(r"\s+", " ", html.unescape(text).replace("\xa0", " ")).strip()


def parse_table(table: str) -> list[dict[str, str]]:
    """Строки таблицы, у которых первая ячейка — разбираемый период."""
    rows: list[dict[str, str]] = []
    for row in _ROW.findall(table):
        cells = [cell_text(cell) for cell in _CELL.findall(row)]
        if len(cells) < 3:
            continue
        period = _PERIOD.search(cells[0])
        if period is None:
            continue  # заголовок и строка нумерации столбцов
        day1, month1, year1, day2, month2, year2 = period.groups()
        if not _MONEY.match(cells[2]):
            continue
        rows.append(
            {
                "from": f"{year1}-{month1}-{day1}",
                "to": f"{year2}-{month2}-{day2}",
                "without_vat": cells[1].replace(",", "."),
                "for_population": cells[2].replace(",", "."),
            }
        )
    return rows


def drop_contradictions(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    """Убрать строки, которые делают ответ неверным, и сказать какие.

    Отбрасывается строка, у которой конец раньше начала либо начало попадает
    внутрь предыдущего периода. Второе и ловит опечатку года: «с 01.01.2023 по
    30.06.2030» накрывает семь лет чужих тарифов.
    """
    kept: list[dict[str, str]] = []
    dropped: list[str] = []
    for row in rows:
        if row["to"] < row["from"]:
            dropped.append(f"{row['from']}..{row['to']} — конец раньше начала")
            continue
        if kept and row["from"] <= kept[-1]["to"]:
            dropped.append(
                f"{row['from']}..{row['to']} — начало внутри периода "
                f"{kept[-1]['from']}..{kept[-1]['to']}"
            )
            continue
        kept.append(row)
    return kept, dropped


def parse(page: str) -> dict[str, list[dict[str, str]]]:
    tables = _TABLE.findall(page)
    if len(tables) < len(SERVICES):
        sys.exit(
            f"на странице {len(tables)} таблиц, ожидалось не меньше {len(SERVICES)}: "
            "разметка изменилась, разбор остановлен"
        )

    # Заголовок таблицы ищется в тексте перед ней: перепутать воду с
    # водоотведением значит назвать абоненту чужую цену.
    result: dict[str, list[dict[str, str]]] = {}
    for (service, marker), table in zip(SERVICES, tables[: len(SERVICES)], strict=True):
        before = cell_text(page[: page.index(table)][-600:]).lower()
        if marker not in before:
            sys.exit(
                f"перед таблицей не найдено упоминание «{service}»: "
                "порядок таблиц на странице изменился, разбор остановлен"
            )
        rows, dropped = drop_contradictions(parse_table(table))
        if not rows:
            sys.exit(f"таблица «{service}» разобрана пустой: разметка изменилась")
        for note in dropped:
            print(f"  ОТБРОШЕНО ({service}): {note}")
        result[service] = rows
    return result


def fetch(url: str) -> str:
    import httpx

    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    return response.text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument(
        "--check", action="store_true", help="сверить с сохранённым, ничего не записывая"
    )
    args = parser.parse_args(argv)

    tariffs = parse(fetch(args.url))
    payload = {
        "source_title": SOURCE_TITLE,
        "source_url": args.url,
        "fetched_on": date.today().isoformat(),
        "services": tariffs,
    }

    for service, rows in tariffs.items():
        print(f"{service}: {len(rows)} периодов, {rows[0]['from']} … {rows[-1]['to']}")

    if args.check:
        if not args.output.exists():
            print(f"нет сохранённого файла {args.output}")
            return 1
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        # Дата загрузки меняется каждый прогон и расхождением не является.
        same = saved.get("services") == payload["services"]
        print("совпадает с сохранённым" if same else "РАСХОЖДЕНИЕ с сохранённым")
        return 0 if same else 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"записано: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
