"""Очистка выгрузки обращений: убрать шаблонные колонки, проверить остаток.

Выгрузка `test44.csv` пришла обезличенной **не полностью**. Обезличивание
прошло по структурным колонкам (`fname`, `address`, `email`, `ls_refid`,
`login_lkk`), но не затронуло текст: в заголовке `text_request` и `answer_msg`
встречаются **дважды**, и первая пара — сгенерированный шаблон вида
«Вопрос от ⟨ФИО⟩ (л/с ⟨номер⟩): подскажите порядок обращения», где ФИО и номер
взяты не из обезличенных колонок, а из настоящего источника. Обращения
настоящие, значит это данные реальных людей.

Замер до очистки: ФИО в 112 строках из 136, номера лицевых счетов в 126, среди
них 119 различных. Фамилия из обезличенного `fname` не встречается в шаблонном
тексте **ни разу**.

ЧТО ДЕЛАЕТ СКРИПТ

0. Приводит выгрузку к **одному филиалу** — Воронежу.

   Прототип обслуживает один филиал (решение 3 сентября 2026), а в выгрузке
   перемешаны четыре: «РВК-Сахалин», «РВК-Архангельск», «Краснодар Водоканал» и
   «РВК-Воронеж». Причём названия разного строения: краснодарское идёт как
   «⟨город⟩ Водоканал», остальные — как «РВК-⟨город⟩».

   Оставить как есть нельзя: база знаний берётся с воронежского сайта, график
   отключений — по воронежским улицам, а ответы ссылались бы на сахалинскую
   организацию. Абонент получил бы чужие телефоны и чужие адреса центров
   обслуживания — ошибка того же рода, что назвать чужую улицу в отключениях.

1. Отбрасывает первую пару колонок целиком. Не чистит, а именно отбрасывает:
   ценности в них нет — это заполнитель одинаковой формы для всех тем, — а
   очистка оставила бы вопрос, всё ли вычищено.
2. Прогоняет оставшиеся тексты через **наш собственный** обезличиватель
   (`app.gateway.pii_filter`) и сообщает, что он нашёл.

Второй пункт — не перестраховка. До сих пор обезличиватель проверялся на
придуманных примерах; здесь настоящий язык абонентов. Три остатка, найденные
разбором вручную (ФИО в ответе строки 26, адрес почты в запросе строки 56,
телефон в ответе строки 87), служат мишенью: если наш фильтр их не находит —
это дефект фильтра, обнаруженный настоящими данными, а не выдуманным тестом.

ЗАПУСК

    python scripts/sanitize_inquiries.py <входной.csv> [-o fixtures/inquiries_sample.csv]

Вход читается в `cp1251` (кодировка выгрузки), выход пишется в `utf-8`.

КУДА КЛАДЁТСЯ РЕЗУЛЬТАТ

По умолчанию — в `fixtures/`, который исключён из репозитория. Тексты обращений
настоящие, и даже после очистки решение о внесении их в историю git принимает
владелец данных, а не скрипт.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Кодировка выгрузки. Не UTF-8: файл приходит из системы под Windows.
SOURCE_ENCODING = "cp1251"
DELIMITER = ";"

BRANCH_CITY = "Воронеж"
BRANCH_ORG = 'ООО "РВК-Воронеж"'

CITY = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
"""Город — любое слово с заглавной, а не перечень.

Первая редакция перечисляла филиалы по именам и **пропустила Оренбург**: в
выгрузке нашлось «ООО "Оренбург Водоканал"», которого не было ни в одном списке.
Перечень филиалов не закрыт, и полагаться на него нельзя."""
QUOTES = "[«»\"'`]*"
"""Кавычки берутся любые и в любом числе: в выгрузке есть и «ООО " Краснодар
Водоканал"» с лишним пробелом, и «ООО "Краснодар Водоканал,» — с потерянной
закрывающей."""

# Две формы названия организации, обе встречаются в выгрузке: краснодарская идёт
# как «<город> Водоканал», остальные — как «РВК-<город>».
_ORG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"ООО\s*{QUOTES}\s*РВК-{CITY}\s*{QUOTES}"),
    re.compile(rf"ООО\s*{QUOTES}\s*{CITY}\s+Водоканал\s*{QUOTES}"),
)
"""Без `IGNORECASE`: заглавная буква в названии города — часть признака, и без
неё шаблон начал бы цеплять обычный текст со словом «водоканал»."""

_CITY_IN_ADDRESS = re.compile(rf"\bг\.\s*(?!Воронеж){CITY}")
"""Город в адресе. Воронеж исключён намеренно: он уже нужный, а замена сломала
бы падеж — «в г. Воронеже» стало бы «в г. Воронеж»."""


def to_branch(text: str) -> str:
    """Свести любое упоминание филиала к воронежскому.

    Название организации приводится **к одной форме целиком**, а не заменой
    города внутри прежней: «ООО "Краснодар Водоканал"» с подстановкой города
    дало бы «ООО "Воронеж Водоканал"» — организации с таким именем нет.

    Слово «Тестовая» отдельной строкой (в выгрузке под типом «Отмена обращения»
    такая запись есть) не трогается: шаблон требует «г.» перед названием, а это
    не город, а служебная пометка оператора.
    """
    for pattern in _ORG_PATTERNS:
        text = pattern.sub(BRANCH_ORG, text)
    return _CITY_IN_ADDRESS.sub(f"г. {BRANCH_CITY}", text)



# Имена, встречающиеся в заголовке дважды. Первое вхождение каждого — шаблон,
# второе — настоящий текст. Порядок в файле именно такой, и он проверяется:
# если выгрузка изменится, скрипт остановится, а не отбросит нужные данные.
DUPLICATED = ("text_request", "answer_msg")


@dataclass
class Report:
    """Что нашлось в текстах после отбрасывания шаблонных колонок."""

    rows: int = 0
    dropped_columns: list[str] = field(default_factory=list)
    findings: dict[str, int] = field(default_factory=dict)
    rows_with_findings: set[int] = field(default_factory=set)
    analyzer_available: bool = False

    def add(self, row_number: int, entity_type: str) -> None:
        self.findings[entity_type] = self.findings.get(entity_type, 0) + 1
        self.rows_with_findings.add(row_number)


def duplicate_positions(header: list[str]) -> dict[str, list[int]]:
    """Где именно стоят одноимённые колонки."""
    positions: dict[str, list[int]] = {}
    for index, name in enumerate(header):
        positions.setdefault(name, []).append(index)
    return positions


def columns_to_drop(header: list[str]) -> list[int]:
    """Номера колонок-шаблонов — первые вхождения одноимённых пар.

    :raises ValueError: заголовок не такой, как в разобранной выгрузке. Лучше
        остановиться, чем угадать и отбросить настоящие тексты.
    """
    positions = duplicate_positions(header)
    drop: list[int] = []
    for name in DUPLICATED:
        found = positions.get(name, [])
        if len(found) != 2:
            raise ValueError(
                f"ожидались ровно два вхождения колонки {name!r}, найдено {len(found)}. "
                "Формат выгрузки изменился — проверьте заголовок вручную, "
                "прежде чем что-либо отбрасывать."
            )
        drop.append(found[0])
    return sorted(drop)


def text_columns(header: list[str], dropped: list[int]) -> list[int]:
    """Колонки, которые надо проверить обезличивателем.

    Проверяются оставшиеся текстовые — то есть вторые вхождения пары. Числовые
    и служебные колонки уже обезличены и к тексту отношения не имеют.
    """
    positions = duplicate_positions(header)
    return [positions[name][1] for name in DUPLICATED if positions[name][1] not in dropped]


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding=SOURCE_ENCODING, newline="") as handle:
        rows = list(csv.reader(handle, delimiter=DELIMITER))
    if not rows:
        raise ValueError(f"файл пуст: {path}")
    header, *data = rows
    return header, [row for row in data if any(cell.strip() for cell in row)]


def scan(rows: list[list[str]], columns: list[int], report: Report) -> None:
    """Прогнать тексты через наш обезличиватель и записать находки."""
    from app.gateway.pii_filter import PiiSanitizer

    sanitizer = PiiSanitizer()
    report.analyzer_available = sanitizer.language_model_available
    for number, row in enumerate(rows, start=1):
        for index in columns:
            if index >= len(row):
                continue
            for span in sanitizer.find(row[index]):
                report.add(number, span.entity_type)


def write_rows(
    path: Path, header: list[str], rows: list[list[str]], dropped: list[int]
) -> None:
    keep = [i for i in range(len(header)) if i not in dropped]
    rows = [[to_branch(cell) for cell in row] for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=DELIMITER)
        writer.writerow([header[i] for i in keep])
        for row in rows:
            writer.writerow([row[i] if i < len(row) else "" for i in keep])


def describe(report: Report) -> Iterator[str]:
    yield f"строк обработано: {report.rows}"
    yield f"филиал приведён к одному: {BRANCH_ORG}"
    yield f"отброшено колонок: {', '.join(report.dropped_columns)}"
    if not report.analyzer_available:
        yield (
            "ВНИМАНИЕ: языковая модель обезличивателя недоступна — работали только "
            "правила по образцу (номера, телефоны, почта). ФИО без модели не находятся: "
            "поставьте её командой `make install-pii`, иначе проверка неполна."
        )
    if not report.findings:
        yield "обезличиватель ничего не нашёл в оставшихся текстах"
        return
    yield f"обезличиватель нашёл в {len(report.rows_with_findings)} строках:"
    for entity_type, count in sorted(report.findings.items(), key=lambda kv: -kv[1]):
        yield f"    {entity_type}: {count}"
    yield "строки с находками: " + ", ".join(str(n) for n in sorted(report.rows_with_findings))
    yield (
        "Это остаток, не снятый исходным обезличиванием. Проверьте и вычистите "
        "перед тем, как вносить файл куда-либо."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="исходная выгрузка (cp1251)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("fixtures/inquiries_sample.csv"),
        help="куда положить очищенный файл (по умолчанию fixtures/, вне репозитория)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="только проверить и показать отчёт, ничего не записывая",
    )
    args = parser.parse_args(argv)

    header, rows = read_rows(args.source)
    dropped = columns_to_drop(header)

    report = Report(rows=len(rows), dropped_columns=[f"[{i}] {header[i]}" for i in dropped])
    scan(rows, text_columns(header, dropped), report)

    if not args.dry_run:
        write_rows(args.output, header, rows, dropped)
        print(f"записано: {args.output}")

    for line in describe(report):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
