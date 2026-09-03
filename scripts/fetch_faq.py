"""Забрать корпус базы знаний прототипа — FAQ воронежского филиала.

База знаний прототипа собирается из открытых источников (ADR-013): закрытых
документов водоканала на прототипе не будет. Из открытого лучший материал —
раздел «Часто задаваемые вопросы»: там вопрос уже сформулирован так, как его
задаёт абонент, а это ровно то, с чем сопоставляется запрос при поиске.

ПОЧЕМУ СКРИПТ, А НЕ ФАЙЛ РУКАМИ. Двадцать пар можно было бы перепечатать, но
тогда корпус разошёлся бы с сайтом при первой же правке, и никто бы не заметил.
Здесь он пересобирается одной командой, а расхождение видно как разница файлов.

РАЗБОР НАМЕРЕННО ГРУБЫЙ. Берутся два класса разметки, `faq-list__text` и
`faq-list__answer-text`, без разборщика HTML. Полноценный разбор здесь не нужен:
структура простая, а зависимость ради двадцати элементов не окупается. Если
разметка сайта изменится, скрипт остановится с понятной ошибкой, а не отдаст
пустой корпус — это проверяется явно.

ЗАПУСК

    python scripts/fetch_faq.py                 # в kb/faq_voronezh.json
    python scripts/fetch_faq.py --check         # только сверить с сохранённым
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

SOURCE_URL = "https://voronezh.rosvodokanal.ru/users/general_info/faq/"
SOURCE_TITLE = "Часто задаваемые вопросы — ООО «РВК-Воронеж»"
DEFAULT_OUTPUT = Path("kb/faq_voronezh.json")

_ITEM_START = '<li class="faq-list__item">'
"""Начало карточки. Конец не ищется по `</li>` намеренно.

Первая редакция брала `<li class="faq-list__item">(.*?)</li>` — и **потеряла три
пары из двадцати**: ответы со списками содержат вложенные `<li>`, на закрывающем
теге которых разбор обрывался, а карточка без ответа отбрасывалась.

Потеря прошла молча: сторож проверял только «пар не меньше пятнадцати», и
семнадцать его устроили. Порог по абсолютному числу такого не ловит — поэтому
теперь сверяется **число разобранных пар с числом вопросов на странице**."""

_QUESTION = re.compile(r'<span class="faq-list__text">(.*?)</span>', re.DOTALL)
_ANSWER = re.compile(r'<span class="faq-list__answer-text">(.*?)</span>\s*</div>', re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def to_text(fragment: str) -> str:
    """Разметку — в текст, сохранив разбиение на абзацы.

    Абзацы и переводы строк значимы: ответы содержат перечни способов подать
    заявку, и склеенные в одну строку они станут нечитаемыми и для абонента, и
    для поиска.
    """
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", fragment, flags=re.IGNORECASE)
    text = _TAG.sub("", text)
    text = html.unescape(text).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def split_items(page: str) -> list[str]:
    """Разбить страницу на карточки по началу каждой, а не по концу.

    Конец карточки определяется началом следующей: вложенные `<li>` внутри
    ответов делают поиск закрывающего тега ненадёжным.
    """
    parts = page.split(_ITEM_START)
    return parts[1:]


def parse(page: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for number, block in enumerate(split_items(page), start=1):
        question = _QUESTION.search(block)
        answer = _ANSWER.search(block)
        if question is None or answer is None:
            continue
        items.append(
            {
                "chunk_id": f"faq-{number:02d}",
                "question": to_text(question.group(1)),
                "answer": to_text(answer.group(1)),
                "source_title": SOURCE_TITLE,
                "source_url": SOURCE_URL,
            }
        )
    return [item for item in items if item["question"] and item["answer"]]


def fetch(url: str) -> str:
    import httpx

    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    return response.text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument(
        "--check",
        action="store_true",
        help="сверить сайт с сохранённым корпусом, ничего не записывая",
    )
    args = parser.parse_args(argv)

    page = fetch(args.url)
    items = parse(page)
    questions = len(_QUESTION.findall(page))
    if len(items) != questions:
        print(
            f"на странице {questions} вопросов, а разобрано {len(items)} пар — "
            "часть ответов потеряна при разборе. Корпус не тронут.",
            file=sys.stderr,
        )
        return 1

    payload = json.dumps(items, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        if not args.output.exists():
            print(f"сохранённого корпуса нет: {args.output}", file=sys.stderr)
            return 1
        same = args.output.read_text(encoding="utf-8") == payload
        print("совпадает с сайтом" if same else "РАСХОДИТСЯ с сайтом")
        return 0 if same else 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    lengths = sorted(len(i["answer"]) for i in items)
    print(f"записано: {args.output}")
    print(f"  пар вопрос-ответ: {len(items)}")
    print(f"  длина ответа: от {lengths[0]} до {lengths[-1]}, медиана {lengths[len(lengths) // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
