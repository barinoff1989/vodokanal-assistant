"""Детектор намерений: разбор реплики абонента на отдельные задачи.

ЗАЧЕМ. Прототип обрабатывал одну задачу на реплику: тема брала первую сработавшую
по правилам, тип — первый найденный, остальное молча терялось («Какой у меня долг
и когда поверка?» отвечало только про поверку). Здесь реплика делится на части,
каждая из которых получает свой ответ (`Orchestrator`, путь `multi`).

КАК УСТРОЕНО — ПРАВИЛА, БЕЗ МОДЕЛИ, В ТРИ ШАГА.

1. **Жёсткие границы.** Конец предложения (`.`, `!`, `?`), `;` и слова «а также»,
   «потом», «затем» отделяют части безусловно.
2. **Мягкие границы.** Внутри предложения кандидатами служат « и », «, а », «, но »
   и запятая. Разрез по ним принимается, **только если каждая сторона несёт
   собственную задачу** — то есть по ней узнаётся тема ответа, тип обращения или
   операция, которую помощник не выполняет. Кусок без сигнала («скажите»,
   «пожалуйста») приклеивается к соседнему. Поэтому «поверка и опломбировка»
   делится, а «вода из-под крана, холодная и горячая» — нет.
3. **Условия.** Часть вида «Если нет — закажите поверку» превращается в условную:
   выполняется, только если условие по предыдущей части выполнено. Проверяется
   единственное условие, которое можно проверить по данным Биллинга, — наличие
   долга; любое другое остаётся непроверенным, и часть не выполняется (и об этом
   говорится абоненту).

**Что здесь намеренно не делается.** Разбор идёт правилами на тех же признаках,
что и классификация (`app/agents/triage.py`, `app/taxonomy.py`), — второй словарь
рядом разошёлся бы с первым. Модель не зовётся: она стоила бы времени на каждой
реплике, а её решение о разбиении нечем проверить. Замер разбора на живых
репликах не проводился: только на выгрузке обращений без разметки числа задач
(`scripts/measure_intents.py`) — доля срабатываний, а не точность.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.agents.triage import Triage
from app.taxonomy import InquiryType, Topic, classify_by_keywords

__all__ = [
    "ACTION_NAMES",
    "MAX_QUERY_CHARS",
    "Condition",
    "Part",
    "PartKind",
    "split_intents",
]


class PartKind(StrEnum):
    """Что делать с частью реплики."""

    ANSWER = "answer"
    """Обычная задача: ответить (правилами или через модель)."""

    UNSUPPORTED = "unsupported"
    """Операция, которую помощник не выполняет («передайте показания», «оплатите»)."""

    CONDITIONAL = "conditional"
    """Задача с условием: выполняется, только если условие по данным подтверждено."""


class Condition(StrEnum):
    """Условие, при котором выполняется условная часть."""

    NO_DEBT = "no_debt"
    """«Если долга нет» / «если нет»: по счёту нет задолженности."""

    HAS_DEBT = "has_debt"
    """«Если есть долг»: по счёту есть задолженность."""

    UNKNOWN = "unknown"
    """Условие нельзя проверить по данным помощника."""


@dataclass(frozen=True, slots=True)
class Part:
    """Одна задача из реплики."""

    text: str
    """Текст части (для условной — только действие, без «если…»)."""

    signature: tuple[str, str]
    """Что за задача. Две части с одной подписью — одна задача."""

    kind: PartKind = PartKind.ANSWER
    condition: Condition | None = None
    condition_text: str = ""
    action: str = ""
    """Название операции для `UNSUPPORTED`."""


# --- операции, которых помощник не выполняет ------------------------------------ #

_UNSUPPORTED: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "передача показаний",
        re.compile(
            r"(?:передайте|передай|передать|внесите|внести|отправьте|отправить|"
            r"подайте|подать)(?![\w-])[^.?!,;]{0,20}показани"
        ),
    ),
    (
        "оплата",
        re.compile(
            r"(?:оплатите|оплати|оплатить|заплатите|заплатить|погасите|погасить)(?![\w-])"
            r"[^.?!,;]{0,15}"
            r"(?:сч[её]т|квитанц|долг|задолж|услуг)"
        ),
    ),
    (
        "изменение данных абонента",
        re.compile(
            r"(?:измените|изменить|поменяйте|поменять|обновите|обновить)(?![\w-])[^.?!,;]{0,15}"
            r"(?:адрес|телефон|почт|email|фио|фамили)"
        ),
    ),
)
ACTION_NAMES: tuple[str, ...] = tuple(name for name, _ in _UNSUPPORTED)

# «Как передать показания?» — вопрос о порядке, а не просьба выполнить: отвечает база знаний.
_HOW_TO = re.compile(
    r"(?<![\w-])(?:как|где|куда|каким образом|можно ли|можно|могу ли|нужно ли|до какого|"
    r"в каком порядке|зачем|почему|когда)(?![\w-])"
)

# --- границы ---------------------------------------------------------------------- #

_HARD = re.compile(
    r"(?<=[.!?])\s+"
    r"|\s*;\s*"
    r"|\s+(?:а также|а еще|а ещё|и еще|и ещё|потом|затем|после этого)\s+",
    re.IGNORECASE,
)
_SOFT = re.compile(r"\s*,\s*(?:а|но)\s+|\s+и\s+|\s*,\s*", re.IGNORECASE)

# «Если нет — закажите поверку», «если долга нет, оформите справку»
_CONDITIONAL = re.compile(
    r"^\s*(?:а\s+)?если\s+(?P<cond>[^,—–\-]{1,40}?)\s*(?:,|—|–|-|\bто\b)\s*(?P<act>\S.*)$",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(r"\b(?:нет|нету|не\s+(?:будет|имеется|числится)|отсутствует|ноль)\b")
_POSITIVE = re.compile(r"\b(?:есть|имеется|числится|будет)\b")
_DEBT_WORD = re.compile(r"долг|задолжен")


# Признаки фактов счёта — копия `app/billing/answer.py` (импортировать оттуда нельзя:
# ответчик импортирует оркестратор, а тот — этот модуль). Совпадение с оригиналом
# сторожит `tests/unit/test_intents.py`.
_VERIFICATION_MARKERS: tuple[str, ...] = ("поверк", "межповероч")
_READINGS_MARKERS: tuple[str, ...] = ("показани",)
_CHARGES_MARKERS: tuple[str, ...] = ("начисл", "списал", "списан", "квитанц")
_DEBT_MARKERS: tuple[str, ...] = (
    "долг",
    "должен",
    "должна",
    "задолжен",
    "к оплате",
    "оплатить",
    "платить",
    "остаток по сч",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("ё", "е"))


_REQUEST = re.compile(
    r"(?<![\w-])(?:хочу|хотел\w*|хотим|прошу|просим|нужн\w*|необходим\w*|надо|можно|"
    r"подскажите|скажите|сообщите|объясните|разъясните|заказать|закажите|оформить|"
    r"оформите|получить|подать|подайте|записать|запишите|узнать|как|где|когда|сколько|"
    r"почему|зачем|какой|какая|какие|куда|есть ли|нет ли|имеется ли)(?![\w-])|\?"
)
"""Признак просьбы или вопроса.

Фрагмент с одним лишь признаком типа обращения («…акт поверки…», «…квитанции…»)
может быть подробностью к просьбе, а не второй задачей. Тип считается
самостоятельной задачей, только если фрагмент — просьба или вопрос либо просьба
прозвучала раньше в том же предложении («Хочу заказать поверку и справку»)."""


def _is_request(text: str) -> bool:
    return _REQUEST.search(_norm(text)) is not None


_FILLER = re.compile(
    r"^(?:(?:скажите|подскажите|пожалуйста|и|а|также|ещё|еще|мне)(?![\w-])[\s,]*)+"
    r"|(?:[\s,]*(?<![\w-])(?:пожалуйста|скажите|подскажите|и|а))+[\s.,?!]*$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Снять с края части слова-связки («скажите», «пожалуйста», «и»)."""
    previous = None
    while previous != text:
        previous = text
        text = _FILLER.sub("", text.strip()).strip(" ,")
    return text


def _account_fact(lowered: str) -> str:
    """Какой факт счёта спрашивают — в том же приоритете, что у ответчика счёта."""
    if any(m in lowered for m in _VERIFICATION_MARKERS):
        return "verification"
    if any(m in lowered for m in _READINGS_MARKERS):
        return "readings"
    if any(m in lowered for m in _CHARGES_MARKERS):
        return "charges"
    if any(m in lowered for m in _DEBT_MARKERS):
        return "debt"
    return "account"


def _signal(
    fragment: str, requested: bool = True
) -> tuple[tuple[str, str], PartKind, str] | None:
    """Подпись задачи во фрагменте, её вид и название операции — либо `None`.

    Порядок: операция, которую не выполняем, → тема ответа → тип обращения.
    """
    lowered = _norm(fragment)
    if not lowered:
        return None
    if not _HOW_TO.search(lowered):
        for name, pattern in _UNSUPPORTED:
            if pattern.search(lowered):
                return ("action", name), PartKind.UNSUPPORTED, name
    topic = Triage._topic_by_rules(fragment)  # noqa: SLF001 — те же признаки, что у классификации
    if topic is Topic.ACCOUNT:
        if not requested:
            return None  # «…переданных показаний…» в рассказе — не вопрос о счёте
        return ("account", _account_fact(lowered)), PartKind.ANSWER, ""
    if topic is not None and topic is not Topic.GENERAL:
        return (topic.value, ""), PartKind.ANSWER, ""
    codes = [c for c in classify_by_keywords(fragment) if c is not InquiryType.OTHER]
    if codes and requested:
        return ("type", codes[0].value), PartKind.ANSWER, ""
    return None


def _fragments(clause: str) -> list[str]:
    """Разрезать предложение по мягким границам на все кандидаты."""
    pieces = [p for p in _SOFT.split(clause) if p and p.strip()]
    return pieces or [clause]


def _split_clause(clause: str) -> list[tuple[str, tuple[str, str], PartKind, str]]:
    """Разбить одно предложение на части с собственными задачами."""
    fragments = _fragments(clause)
    signals = []
    requested = False
    for fragment in fragments:
        requested = requested or _is_request(fragment)
        signals.append(_signal(fragment, requested))
    if sum(s is not None for s in signals) < 2:
        one = _signal(clause, _is_request(clause))
        return [(_clean(clause), one[0], one[1], one[2])] if one else []

    parts: list[tuple[str, tuple[str, str], PartKind, str]] = []
    pending = ""  # фрагменты без сигнала до первой задачи
    for fragment, signal in zip(fragments, signals, strict=True):
        if signal is None:
            if parts:
                text, sig, kind, action = parts[-1]
                parts[-1] = (f"{text} {fragment.strip()}", sig, kind, action)
            else:
                pending = f"{pending} {fragment.strip()}".strip()
            continue
        text = f"{pending} {fragment.strip()}".strip() if pending else fragment.strip()
        pending = ""
        if parts and parts[-1][1] == signal[0]:
            prev_text, sig, kind, action = parts[-1]
            parts[-1] = (f"{prev_text} {text}", sig, kind, action)
        else:
            parts.append((text, signal[0], signal[1], signal[2]))
    return [(_clean(t), sig, kind, action) for t, sig, kind, action in parts]


MAX_QUERY_CHARS = 300
"""Реплики длиннее не разбираются: это письма, а не вопросы чата.

Длинное письмо («Прошу вернуть переплату. Реквизиты прикрепляю. Банк: …») несёт
одну просьбу и подробности к ней, а признаки типов в каждой фразе дают несколько
«задач». Замер на выгрузке обращений показал такой перерез у длинных текстов;
чат-реплика короткая, поэтому предел разумен. Значение — выбор автора, на живых
репликах чата не подбиралось."""


def split_intents(query: str) -> list[Part]:
    """Разобрать реплику на задачи.

    Одна задача или ни одной — список из одной части: вызывающий идёт прежним
    путём (текст реплики целиком), кроме случая, когда эта единственная часть —
    операция, которую помощник не выполняет (`PartKind.UNSUPPORTED`).
    """
    if len(query) > MAX_QUERY_CHARS:
        return [Part(text=query.strip(), signature=("none", ""))]
    clauses = [c for c in _HARD.split(query.strip()) if c and c.strip()]
    parts: list[Part] = []
    for clause in clauses:
        match = _CONDITIONAL.match(clause)
        if match and parts:
            act = match.group("act").strip()
            cond_text = match.group("cond").strip()
            lowered = _norm(cond_text)
            if _DEBT_WORD.search(lowered):
                condition = Condition.NO_DEBT if _NEGATIVE.search(lowered) else Condition.HAS_DEBT
            elif any(p.signature == ("account", "debt") for p in parts):
                # «Если нет» после вопроса о долге относится к долгу.
                if _NEGATIVE.search(lowered):
                    condition = Condition.NO_DEBT
                elif _POSITIVE.search(lowered):
                    condition = Condition.HAS_DEBT
                else:
                    condition = Condition.UNKNOWN
            else:
                condition = Condition.UNKNOWN
            acts = _split_clause(act) or [(act, ("cond", act), PartKind.ANSWER, "")]
            for text, sig, _kind, action in acts:
                parts.append(
                    Part(
                        text=text,
                        signature=sig,
                        kind=PartKind.CONDITIONAL,
                        condition=condition,
                        condition_text=cond_text,
                        action=action,
                    )
                )
            continue
        for text, sig, kind, action in _split_clause(clause):
            if any(p.signature == sig and p.kind is not PartKind.CONDITIONAL for p in parts):
                continue  # та же задача, названная второй раз
            parts.append(Part(text=text, signature=sig, kind=kind, action=action))
    if parts:
        return parts
    return [Part(text=query.strip(), signature=("none", ""))]
