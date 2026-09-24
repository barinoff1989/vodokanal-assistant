"""Замер запасного режима классификации на настоящих обращениях.

Запасной режим определяет тип обращения по ключевым словам, когда модель
недоступна. До получения выгрузки перечень слов был
предположением — так и помечен в модуле. Здесь он получает число.

ГЛАВНОЕ, ЧТО НАДО ЗНАТЬ ПЕРЕД ЧТЕНИЕМ ЛЮБОЙ ЦИФРЫ ОТСЮДА

**`bname` — не разметка текста.** Оператор классифицировал случай, зная больше,
чем написано: вложения, телефонный разговор, историю лицевого счёта. Проверка:
**только 28% текстов содержат хоть одно слово из названия собственного типа**, а
у восьми типов из шестнадцати — ни один текст из девяти.

Отсюда два следствия, без которых замер вводит в заблуждение:

1. **У точности есть потолок, и задаёт его не классификатор.** «Отмена
   обращения» — это, судя по текстам («ПУ 1шт ГВ», «первичная», «Тестовая»),
   скорее статус обращения, чем его тема; вывести такую метку из текста нельзя
   никаким способом.
2. **«Другое» — свалка**: под ним лежат обращения, у которых есть
   собственный тип. Считать попадание в «Другое» успехом нельзя, поэтому эти
   строки из точности исключены и показаны отдельно.

Поэтому скрипт печатает не одно число, а четыре, и главное из них — **охват**:
доля текстов, в которых нашёлся хоть какой-то сигнал. Точность считается только
там, где сигнал есть, и только по типам, метка которых из текста в принципе
выводима.

РАЗДЕЛЕНИЕ ВЫБОРКИ

Слова в `app/taxonomy.py` собраны по этим же текстам, поэтому замер на них
целиком показал бы не качество, а запоминание. Выборка делится по каждому типу:
две трети — на которых слова подбирались, треть — отложенная. Оба числа
печатаются рядом, и расхождение между ними и есть мера подгонки.

Девять наблюдений на тип — очень мало. Числа отсюда **указывают направление, но
не являются метрикой качества**; настоящая оценка появится на эталонном наборе
собранном ручной разметкой, а не выгрузкой как есть.

ЗАПУСК

    python scripts/measure_classifier.py <выгрузка.csv>
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.taxonomy import DISPLAY_NAMES, KEYWORDS, InquiryType, classify_by_keywords, parse

SOURCE_ENCODING = "cp1251"
DELIMITER = ";"
TEXT_COLUMN = 9
"""Вторая из одноимённых колонок `text_request` — настоящий текст абонента.

Первая пара колонок в выгрузке содержит сгенерированный шаблон и
необезличенные персональные данные; читаем по номеру, а не по
имени, потому что имена в заголовке повторяются."""

TRAIN_SHARE = 2 / 3
"""Какая часть каждого типа считается той, на которой слова подбирались."""


def label_derivable(display_name: str, text: str) -> bool:
    """Содержит ли текст хоть одно значимое слово из названия своего типа.

    Грубая мера — сравниваются шестибуквенные начала слов, — но она отвечает на
    единственный нужный вопрос: можно ли эту метку вывести из текста вообще.
    """
    generic = {
        "заказать", "узла", "воды", "эксплуатацию", "прибора", "учета",
        "денежных", "обращения", "повторная", "приемка", "направить",
    }
    roots = [w[:6] for w in re.findall(r"[а-яё]{4,}", display_name.lower()) if w not in generic]
    lowered = text.lower()
    return bool(roots) and any(root in lowered for root in roots)


@dataclass
class Split:
    """Итоги по одной части выборки."""

    name: str
    total: int = 0
    with_signal: int = 0
    correct: int = 0
    ambiguous: int = 0
    scored: int = 0
    """Строк, участвовавших в точности: сигнал есть и тип не «Другое»."""

    def line(self) -> str:
        coverage = f"{self.with_signal}/{self.total}"
        share = 100 * self.with_signal // self.total if self.total else 0
        accuracy = (
            f"{self.correct}/{self.scored} ({100 * self.correct // self.scored}%)"
            if self.scored
            else "—"
        )
        return (
            f"  {self.name:12} строк {self.total:4}   охват {coverage:>8} ({share:3}%)"
            f"   точность {accuracy:>14}   неоднозначных {self.ambiguous}"
        )


@dataclass
class Report:
    train: Split = field(default_factory=lambda: Split("обучающая"))
    held_out: Split = field(default_factory=lambda: Split("отложенная"))
    derivable: int = 0
    total: int = 0
    dumped: int = 0
    """Строк с типом «Другое» — исключены из точности."""
    per_type: dict[str, tuple[int, int]] = field(default_factory=dict)


def measure(rows: list[tuple[str, str]]) -> Report:
    report = Report()
    by_type: dict[str, list[str]] = defaultdict(list)
    for name, text in rows:
        by_type[name].append(text)

    for name, texts in sorted(by_type.items()):
        cut = max(1, round(len(texts) * TRAIN_SHARE))
        try:
            expected: InquiryType | None = parse(name)
        except Exception:  # noqa: BLE001 — тип из выгрузки может быть незнакомым
            expected = None

        derivable_here = 0
        for index, text in enumerate(texts):
            report.total += 1
            if label_derivable(name, text):
                report.derivable += 1
                derivable_here += 1

            split = report.train if index < cut else report.held_out
            split.total += 1

            codes = classify_by_keywords(text)
            if codes:
                split.with_signal += 1
            if len(codes) > 1:
                split.ambiguous += 1

            if expected is InquiryType.OTHER:
                report.dumped += 1
                continue
            if not codes:
                continue
            split.scored += 1
            if expected in codes:
                split.correct += 1

        report.per_type[name] = (derivable_here, len(texts))

    return report


def read(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding=SOURCE_ENCODING, newline="") as handle:
        rows = list(csv.reader(handle, delimiter=DELIMITER))
    return [
        (row[0].strip(), row[TEXT_COLUMN])
        for row in rows[1:]
        if any(cell.strip() for cell in row) and len(row) > TEXT_COLUMN
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="выгрузка обращений (cp1251)")
    args = parser.parse_args(argv)

    rows = read(args.source)
    report = measure(rows)

    print(f"текстов: {report.total}\n")
    print("ПОТОЛОК, ЗАДАННЫЙ ДАННЫМИ")
    share = 100 * report.derivable // report.total if report.total else 0
    print(f"  метка выводится из текста: {report.derivable}/{report.total} ({share}%)")
    print("  ниже — по типам; ноль означает, что метку из текста получить нельзя\n")
    for name, (hits, total) in sorted(report.per_type.items(), key=lambda kv: kv[1][0]):
        mark = "  <-- метка не в тексте" if hits == 0 else ""
        print(f"    {hits:2}/{total:2}  {name[:52]:52}{mark}")

    print("\nЗАПАСНОЙ РЕЖИМ")
    print(report.train.line())
    print(report.held_out.line())
    print(f"\n  строк с типом «Другое», исключённых из точности: {report.dumped}")
    print(
        "\n  Точность считается только там, где сигнал есть: классификатор без сигнала\n"
        "  не ошибается, а молчит, и смешивать эти случаи нельзя. Расхождение между\n"
        "  обучающей и отложенной частью — мера подгонки под выборку."
    )
    unused = sorted(c.value for c in InquiryType if c not in KEYWORDS)
    print(f"\n  типы без ключевых слов (осознанно): {', '.join(unused)}")
    print(f"  из них с названием: {', '.join(DISPLAY_NAMES[InquiryType(c)] for c in unused)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
