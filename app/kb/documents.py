"""Приём документов Word в базу знаний: разбор по разделам.

Источники 2, 5 и 8 каталога — регламенты, инструкции и реестр договоров —
приходят файлами Word. На прототипе настоящих нет (закрытая сеть), и вместо них
`scripts/generate_kb_docs.py` собирает синтетические той же формы.

ЕДИНИЦА НАРЕЗКИ — РАЗДЕЛ, А НЕ СТРАНИЦА И НЕ ЧИСЛО ЗНАКОВ

Заголовок в таком документе несёт то же, что вопрос в паре FAQ: он говорит, о
чём раздел, словами, близкими к вопросу абонента. Резать мимо него — терять
единственную подпись, которая у текста есть.

Отсюда фрагмент = заголовок + всё до следующего заголовка того же или более
высокого уровня. Вложенный раздел («2.1» внутри «2») становится отдельным
фрагментом, а его заголовок несёт **путь**: «2. Основания начисления → 2.1.
Переход между способами». Путь нужен затем, что «2.1» сам по себе не значит
ничего, а с родителем — значит.

ЧТО ПРОИСХОДИТ С ТАБЛИЦАМИ

Таблица разворачивается в строки вида «Основание: … · Документы: … · Срок: …» —
заголовок столбца приклеивается к значению. Без этого строка «Изменение числа
проживающих · Заявление · Расчётный период» теряет смысл: непонятно, что здесь
основание, а что срок.

ПОМЕТКА О СИНТЕТИЧНОСТИ ПЕРЕЖИВАЕТ РАЗБОР

Документ помечен в трёх местах, и разбор **обязан** донести пометку до
фрагмента: иначе синтетическая процедура попадёт в ответ абоненту неотличимо от
настоящей. Сам абзац-предупреждение при этом в текст фрагмента не идёт — это не
знание, а служебная надпись, и в поиске она только мешала бы.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.taxonomy import InquiryType

__all__ = ["DocumentSection", "SYNTHETIC_CATEGORY", "load_docx", "load_directory"]

SYNTHETIC_CATEGORY = "СИНТЕТИКА"
"""Значение `core_properties.category`, которым помечает генератор.

Проверяется именно свойство файла, а не текст первого абзаца: абзац может быть
переписан при правке, свойство переживает редактирование в Word."""

_HEADING = re.compile(r"^Heading (\d+)$")
_TITLE_STYLE = "Title"
"""Название документа. В Word это отдельный стиль, а не «Heading 0».

Первая редакция искала только `Heading N`, и название в разбор не попадало —
а вместе с ним терялся корень пути заголовков. У пяти файлов реестра договоров
разделы называются одинаково («1. Когда применяется»), и без названия документа
их фрагменты становились неразличимы."""


def _level_of(style_name: str) -> int | None:
    """Уровень заголовка по имени стиля; ``None`` — не заголовок."""
    if style_name == _TITLE_STYLE:
        return 0
    match = _HEADING.match(style_name)
    return int(match.group(1)) if match else None
_SYNTHETIC_TEXT = re.compile(r"СИНТЕТИЧЕСКИЙ ДОКУМЕНТ", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """Раздел документа — готовый фрагмент базы знаний.

    :param chunk_id: опознаватель вида ``instrukciya-nachisleniya#3``.
    :param heading: путь заголовков от верхнего к текущему.
    :param body: текст раздела без заголовка и без служебных надписей.
    :param source_title: название документа.
    :param source_path: путь файла — на MVP заменится адресом в хранилище.
    :param synthetic: собран ли документ генератором.
    :param inquiry_type: тип обращения, к которому относится документ
        (значение `InquiryType`), из свойства файла `subject`; ``None`` —
        документ общий (регламент, инструкция) и подходит под любой тип.
    """

    chunk_id: str
    heading: str
    body: str
    source_title: str
    source_path: str
    synthetic: bool
    inquiry_type: str | None = None


def _table_lines(table: object) -> list[str]:
    """Таблицу — в строки «столбец: значение», по строке на запись."""
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]  # type: ignore[attr-defined]
    if not rows:
        return []
    header, *body = rows
    if not body:
        return [" · ".join(c for c in header if c)]

    # Таблица из двух столбцов — это пары «свойство → значение», и заголовок в
    # ней служебный («Условие | Значение»). Приклеивать его к каждой строке
    # значило бы получить «Условие: Срок рассмотрения · Значение: 5 дней»
    # вместо «Срок рассмотрения: 5 дней».
    if len(header) == 2:
        return [f"{row[0]}: {row[1]}" for row in body if row[0] and row[1]]

    lines = []
    for row in body:
        pairs = [
            f"{name}: {value}"
            for name, value in zip(header, row, strict=False)
            if value and name
        ]
        if pairs:
            lines.append(" · ".join(pairs))
    return lines


def _blocks(document: object) -> list[tuple[str, str]]:
    """Тело документа по порядку: ('heading', текст) | ('text', текст).

    Абзацы и таблицы идут в том порядке, в каком лежат в файле. `python-docx`
    отдаёт их двумя раздельными списками, поэтому порядок восстанавливается по
    XML-дереву — иначе таблица из раздела 3 приклеилась бы к последнему разделу.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parent = document.element.body  # type: ignore[attr-defined]
    out: list[tuple[str, str]] = []
    for child in parent.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)  # type: ignore[arg-type]
            text = paragraph.text.strip()
            if not text or _SYNTHETIC_TEXT.search(text):
                continue
            style = paragraph.style
            level = _level_of(style.name or "" if style else "")
            out.append(("heading" if level is not None else "text", text))
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)  # type: ignore[arg-type]
            for line in _table_lines(table):
                out.append(("text", line))
    return out


def _heading_levels(document: object) -> dict[str, int]:
    """Уровень для каждого текста-заголовка. Заголовок 0 — название документа."""
    levels: dict[str, int] = {}
    for paragraph in document.paragraphs:  # type: ignore[attr-defined]
        style = paragraph.style
        level = _level_of(style.name or "" if style else "")
        if level is not None and paragraph.text.strip():
            levels[paragraph.text.strip()] = level
    return levels


def load_docx(path: Path) -> list[DocumentSection]:
    """Разобрать один файл Word на разделы."""
    from docx import Document

    document = Document(str(path))
    properties = document.core_properties
    title = (properties.title or path.stem).strip()
    synthetic = (properties.category or "").strip() == SYNTHETIC_CATEGORY
    # Тип — свойство файла, а не догадка по тексту: тот же довод, что у пометки
    # синтетичности выше. Неизвестное значение — не тип: фильтровать по
    # опечатке значило бы молча потерять документ из поиска.
    subject = (properties.subject or "").strip()
    inquiry_type = subject if subject in {t.value for t in InquiryType} else None

    levels = _heading_levels(document)
    sections: list[DocumentSection] = []
    stack: list[tuple[int, str]] = []
    current: list[str] = []
    heading_path = ""

    def flush() -> None:
        nonlocal current
        text = "\n".join(current).strip()
        if heading_path and text:
            sections.append(
                DocumentSection(
                    chunk_id=f"{path.stem}#{len(sections) + 1}",
                    heading=heading_path,
                    body=text,
                    source_title=title,
                    source_path=str(path),
                    synthetic=synthetic,
                    inquiry_type=inquiry_type,
                )
            )
        current = []

    for kind, text in _blocks(document):
        if kind == "text":
            current.append(text)
            continue

        flush()
        level = levels.get(text, 1)
        if level == 0:
            # Название документа — не раздел, но **корень пути**. Без него у
            # пяти файлов реестра договоров заголовки совпадают дословно
            # («1. Когда применяется»), и фрагменты становятся неразличимы:
            # вектор заголовка у всех пяти один и тот же.
            stack = [(0, text)]
            heading_path = ""
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        heading_path = " → ".join(name for _, name in stack)

    flush()
    return sections


def load_directory(root: Path) -> list[DocumentSection]:
    """Разобрать все файлы Word в каталоге и его подкаталогах.

    Порядок обхода отсортирован: опознаватели фрагментов должны быть
    воспроизводимы, иначе повторная индексация меняет их без причины.
    """
    if not root.exists():
        return []
    sections: list[DocumentSection] = []
    for path in sorted(root.rglob("*.docx")):
        if path.name.startswith("~$"):  # временный файл Word
            continue
        sections.extend(load_docx(path))
    return sections
