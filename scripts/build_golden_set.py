"""Сборка эталонного набора поиска из разметки и настоящих источников.

ЗАЧЕМ ОТДЕЛЬНЫЙ ШАГ СБОРКИ, А НЕ ФАЙЛ С ВОПРОСАМИ

Текст вопроса здесь **никогда не пишется руками**. `golden_set/labels.json`
хранит только ссылку на источник — номер строки выгрузки обращений или
идентификатор фрагмента корпуса, — а текст подставляется этим скриптом. Если
источник исчез или сместился, сборка падает.

Причина не в аккуратности. Проект уже получил дорогой урок: обезличиватель
проходил проверку на придуманных примерах и провалился на настоящих текстах,
потому что придуманные данные не содержали ни юридических ссылок, ни отраслевых
сокращений, ни местоимений с заглавной буквы (журнал, раздел 56.10). Вывод был
записан общим: **набора тестов недостаточно, если данные в нём придуманы теми
же, кто писал код.** Эталонный набор — ровно тот случай, где эту ошибку легче
всего повторить: достаточно перефразировать вопросы корпуса и получить отличный
Recall@3, ничего при этом не измерив.

Три выдуманных вопроса в наборе всё же есть — посторонние, для проверки порога.
Они помечены источником `invented`, и это единственный вид записи, где текст
лежит в разметке. Отчёт сборки называет их число отдельно.

ЧТО РАЗМЕТКА НЕ ЯВЛЯЕТСЯ

Соответствие «вопрос → фрагмент» проставлено нами, а не владельцем данных.
Вопросы настоящие, разметка — наша (пункт 49 сведённого TODO). Числа,
полученные на этом наборе, показывают направление, а не качество, пока разметку
не подтвердит владелец.

Отдельно: `bname` выгрузки (тип обращения) разметкой **не является и здесь не
используется** — метка выводится из текста лишь в 28% случаев, у восьми типов
из шестнадцати ни разу (журнал, раздел 56.11). Здесь размечается другое: какой
фрагмент базы знаний отвечает на вопрос.

ЗАПУСК

    python scripts/build_golden_set.py
    python scripts/build_golden_set.py --check   # только проверить, не писать
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "golden_set" / "labels.json"
OUTPUT = ROOT / "golden_set" / "retrieval.json"
INQUIRIES = ROOT / "fixtures" / "inquiries_voronezh.csv"
CORPUS = ROOT / "kb" / "faq_voronezh.json"

DELIMITER = ";"
"""Разделитель выгрузки. Не запятая: файл приходит из системы под Windows."""


@dataclass(frozen=True, slots=True)
class Question:
    """Готовый вопрос набора: текст подставлен, происхождение сохранено."""

    id: str
    subset: str
    text: str
    expected: tuple[str, ...]
    origin: str
    note: str


def _normalize(text: str) -> str:
    """Свести пробелы: в выгрузке встречаются переносы внутри одного вопроса.

    Смысл не меняется, а сравнивать и печатать становится возможно. Ничего,
    кроме пробельных знаков, здесь не трогается — обрезать текст абонента
    значило бы измерять не тот вопрос, который он задал.
    """
    return re.sub(r"\s+", " ", text).strip()


def load_inquiries() -> list[dict[str, str]]:
    if not INQUIRIES.exists():
        sys.exit(
            f"нет выгрузки обращений: {INQUIRIES.relative_to(ROOT)}\n"
            "Она не хранится в репозитории (тексты настоящие). Собрать:\n"
            "    python scripts/sanitize_inquiries.py <исходный.csv>"
        )
    with INQUIRIES.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=DELIMITER))


def load_corpus() -> dict[str, dict[str, str]]:
    items = json.loads(CORPUS.read_text(encoding="utf-8"))
    return {item["chunk_id"]: item for item in items}


def build() -> list[Question]:
    """Собрать набор, падая на первом же расхождении с источником."""
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = load_inquiries()
    corpus = load_corpus()

    questions: list[Question] = []
    seen: set[str] = set()

    for item in labels["items"]:
        ident = item["id"]
        if ident in seen:
            sys.exit(f"{ident}: идентификатор встречается дважды")
        seen.add(ident)

        source = item["source"]
        kind = source["kind"]

        if kind == "inquiry":
            row = source["row"]
            if not 1 <= row <= len(rows):
                sys.exit(f"{ident}: строки {row} нет в выгрузке ({len(rows)} строк)")
            text = _normalize(rows[row - 1]["text_request"])
            origin = f"обращение, строка {row}"
        elif kind == "faq":
            chunk_id = source["chunk_id"]
            if chunk_id not in corpus:
                sys.exit(f"{ident}: фрагмента {chunk_id} нет в корпусе")
            text = _normalize(corpus[chunk_id]["question"])
            origin = f"вопрос корпуса {chunk_id}"
        elif kind == "invented":
            text = _normalize(source["text"])
            origin = "выдуман"
        else:
            sys.exit(f"{ident}: неизвестный вид источника {kind!r}")

        if not text:
            sys.exit(f"{ident}: текст вопроса пуст")

        for chunk_id in item["expected"]:
            if chunk_id not in corpus:
                sys.exit(f"{ident}: ожидается фрагмент {chunk_id}, которого нет в корпусе")

        questions.append(
            Question(
                id=ident,
                subset=item["subset"],
                text=text,
                expected=tuple(item["expected"]),
                origin=origin,
                note=item.get("note", ""),
            )
        )

    return questions


def report(questions: list[Question]) -> None:
    by_subset = Counter(q.subset for q in questions)
    invented = sum(1 for q in questions if q.origin == "выдуман")

    print(f"вопросов в наборе: {len(questions)}")
    for subset, count in sorted(by_subset.items()):
        print(f"  {subset:<10} {count:>3}")
    print(f"из них настоящих: {len(questions) - invented}, выдуманных: {invented}")

    # Ни один вопрос не должен ожидать фрагмент, которого не ждёт никто другой,
    # незаметно для глаза: печать покрытия корпуса показывает перекос набора.
    covered = Counter(chunk for q in questions for chunk in q.expected)
    corpus = load_corpus()
    unused = sorted(set(corpus) - set(covered))
    print(f"фрагментов корпуса под ожиданием: {len(covered)} из {len(corpus)}")
    if unused:
        print(f"  ни разу не ожидается: {', '.join(unused)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="только проверить разметку против источников, файл не писать",
    )
    args = parser.parse_args()

    questions = build()
    report(questions)

    if args.check:
        return

    payload = {
        "_": (
            "Собрано scripts/build_golden_set.py из golden_set/labels.json. "
            "Руками не править: правки затрутся при следующей сборке."
        ),
        "questions": [
            {
                "id": q.id,
                "subset": q.subset,
                "text": q.text,
                "expected": list(q.expected),
                "origin": q.origin,
                "note": q.note,
            }
            for q in questions
        ],
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"записано: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
