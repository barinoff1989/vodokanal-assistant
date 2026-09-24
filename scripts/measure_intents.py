"""Доля реплик с несколькими задачами по детектору намерений — на выгрузке обращений.

Меряет **срабатывание детектора**, а не долю многозадачных обращений: разметки
числа задач людьми на этой выгрузке нет. Число годится как ориентир и для поиска
перереза (примеры печатаются), но триггер 4 перехода на LangGraph (доля выше 5%,
ADR-200) решается по разметке, а не по этому скрипту.

ЗАПУСК

    python scripts/measure_intents.py
    python scripts/measure_intents.py --examples 15
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from app.agents.intents import PartKind, split_intents

SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "inquiries_voronezh.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=10, help="сколько примеров печатать")
    args = parser.parse_args(argv)

    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r["text_request"].strip() for r in csv.DictReader(handle, delimiter=";")]
    rows = [r for r in rows if r]

    sizes: Counter[int] = Counter()
    multi: list[tuple[str, list[str]]] = []
    unsupported = 0
    for text in rows:
        parts = split_intents(text)
        sizes[len(parts)] += 1
        if any(p.kind is PartKind.UNSUPPORTED for p in parts):
            unsupported += 1
        if len(parts) >= 2:
            multi.append((text, [f"{p.kind.value}:{p.signature}" for p in parts]))

    total = len(rows)
    print(f"реплик: {total}")
    for size in sorted(sizes):
        print(f"  частей {size}: {sizes[size]} ({sizes[size] / total:.0%})")
    print(f"с двумя и более задачами (по детектору): {len(multi)} ({len(multi) / total:.1%})")
    print(f"с операцией без выполнения: {unsupported}")
    for text, signatures in multi[: args.examples]:
        print("-" * 70)
        print(text[:300].replace("\n", " "))
        print("  ->", signatures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
