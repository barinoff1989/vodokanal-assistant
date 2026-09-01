"""Обезличивание запроса до отправки провайдеру.

Правило 4.2: персональные данные абонента не покидают периметр. Фильтр стоит в
шлюзе, после проверки лимитов и до обращения к модели (раздел 41.3), и это
единственное место, где сырые данные превращаются в метки.

УСТРОЙСТВО. Распознаватели разделены на две группы, и это не вкусовщина:

* **Собственные** — СНИЛС, ИНН, лицевой счёт, телефон. Чистые функции без
  Presidio: работают мгновенно, проверяются без загрузки языковой модели, а
  главное — считают контрольные суммы. У СНИЛС и ИНН они есть, и без них любые
  одиннадцать цифр подряд выглядели бы как СНИЛС.
* **Presidio** — ФИО и почта. Здесь нужен разбор языка, регулярным выражением
  фамилию не поймать.

ЧТО ПОКАЗАЛА ПРОВЕРКА БАЗОВОГО НАБОРА (до написания этого модуля):

* лицевой счёт `1234567890` определялся как телефон — своего распознавателя
  нет, а под шаблон телефона он подходит. Отсюда контекстные слова: цифры
  считаются счётом, только если рядом стоит «лицевой счёт», «л/с» или «ЛС»
  (раздел 11.5 прямо предупреждал, что `\\b\\d{9,10}\\b` слишком широк);
* одна и та же почта получала три метки сразу — EMAIL, ORGANIZATION и URL.
  Отсюда явный перечень интересующих нас сущностей и разрешение перекрытий:
  из двух пересекающихся находок остаётся та, что длиннее и увереннее.

В журнал уходит только :class:`PiiReport` — перечень типов найденного, без
единого значения (правило 4.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, cast

from app.models import PiiReport

__all__ = [
    "PLACEHOLDERS",
    "PiiSanitizer",
    "PiiSpan",
    "find_account_numbers",
    "find_inn",
    "find_phones",
    "find_snils",
    "is_valid_inn",
    "is_valid_snils",
    "sanitize",
]


@dataclass(frozen=True, slots=True)
class PiiSpan:
    """Найденный фрагмент персональных данных.

    Само значение здесь не хранится намеренно — только границы и тип. Объект
    может попасть в журнал при отладке, и тогда он не должен ничего раскрыть.
    """

    start: int
    end: int
    entity_type: str
    score: float = 1.0

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: PiiSpan) -> bool:
        return self.start < other.end and other.start < self.end


PLACEHOLDERS: dict[str, str] = {
    "PERSON": "<ФИО>",
    "SNILS": "<СНИЛС>",
    "INN": "<ИНН>",
    "ACCOUNT_NUMBER": "<ЛИЦЕВОЙ_СЧЁТ>",
    "PHONE_NUMBER": "<ТЕЛЕФОН>",
    "EMAIL_ADDRESS": "<ПОЧТА>",
    "LOCATION": "<АДРЕС>",
}
"""Метки на русском: они уходят в промпт, и модель должна понимать, что перед
ней скрытое поле, а не случайный набор символов."""

PRESIDIO_ENTITIES: tuple[str, ...] = ("PERSON", "EMAIL_ADDRESS", "LOCATION")
"""Что берём у Presidio. Перечень закрытый: без него на почту приходили ещё
метки ORGANIZATION и URL, которые к персональным данным отношения не имеют и
только создавали перекрытия."""


# --- Общий приём: цифры рядом со словом-подсказкой ---------------------------- #

_HINT_WINDOW = 40
"""Насколько далеко от подсказки может стоять номер. Сорок символов — это
примерно «лицевой счёт для оплаты услуг: 1234567890»; больше брать опасно,
начнут попадать числа из соседнего предложения."""


def _find_near_hint(
    text: str,
    hint: re.Pattern[str],
    digits: re.Pattern[str],
    entity_type: str,
    score: float,
) -> list[PiiSpan]:
    """Найти числа, стоящие рядом со словом-подсказкой.

    Подсказка в тексте — более сильное свидетельство, чем совпадение формата:
    «л/с 9876543210» это лицевой счёт, даже если число случайно проходит
    проверку контрольной суммы ИНН. Поэтому находки по подсказке получают
    более высокую оценку и побеждают при разрешении перекрытий.
    """
    ends = [m.end() for m in hint.finditer(text)]
    if not ends:
        return []
    return [
        PiiSpan(m.start(), m.end(), entity_type, score)
        for m in digits.finditer(text)
        if any(0 <= m.start() - end <= _HINT_WINDOW for end in ends)
    ]


# --- СНИЛС ------------------------------------------------------------------- #

_SNILS_RE = re.compile(r"\b(\d{3})[-\s]?(\d{3})[-\s]?(\d{3})[-\s]?(\d{2})\b")


def is_valid_snils(digits: str) -> bool:
    """Проверить контрольную сумму СНИЛС.

    Без неё распознавателем оказался бы любой набор из одиннадцати цифр —
    например, номер заявки или дата с временем. Контрольная сумма отсекает
    примерно 99 из 100 случайных совпадений.
    """
    if len(digits) != 11 or not digits.isdigit():
        return False
    # Номера до 001-001-998 контрольную сумму не имеют.
    if int(digits[:9]) <= 1001998:
        return True

    checksum = sum(int(digit) * (9 - index) for index, digit in enumerate(digits[:9]))
    if checksum in (100, 101):
        checksum = 0
    elif checksum > 101:
        checksum %= 101
        if checksum in (100, 101):
            checksum = 0
    return checksum == int(digits[9:])


_SNILS_HINT = re.compile(r"\bснилс\b", re.IGNORECASE)


def find_snils(text: str) -> list[PiiSpan]:
    """Найти СНИЛС по контрольной сумме либо по слову-подсказке рядом.

    Вторая ветка нужна для номеров с опечаткой: сумма не сойдётся, но если
    рядом написано «СНИЛС», это всё равно персональные данные и уходить
    провайдеру они не должны.
    """
    spans = [
        PiiSpan(m.start(), m.end(), "SNILS", 0.95)
        for m in _SNILS_RE.finditer(text)
        if is_valid_snils("".join(m.groups()))
    ]
    spans.extend(_find_near_hint(text, _SNILS_HINT, _SNILS_RE, "SNILS", 0.9))
    # Один номер может найтись обеими ветками сразу; наружу отдаём одну
    # находку, иначе вызывающий код увидит дубль там, где данные единственные.
    return _resolve_overlaps(spans)


# --- ИНН --------------------------------------------------------------------- #

_INN_RE = re.compile(r"\b(\d{10}|\d{12})\b")

_INN_WEIGHTS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _inn_check_digit(digits: str, weights: tuple[int, ...]) -> int:
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 % 10


def is_valid_inn(digits: str) -> bool:
    """Проверить контрольную сумму ИНН — десятизначного или двенадцатизначного.

    Десятизначный принадлежит организации, двенадцатизначный — человеку. Для
    задачи проекта важен второй, но отсекать первый не нужно: он тоже не должен
    уходить провайдеру.
    """
    if not digits.isdigit():
        return False
    if len(digits) == 10:
        return _inn_check_digit(digits[:9], _INN_WEIGHTS_10) == int(digits[9])
    if len(digits) == 12:
        return _inn_check_digit(digits[:10], _INN_WEIGHTS_11) == int(digits[10]) and (
            _inn_check_digit(digits[:11], _INN_WEIGHTS_12) == int(digits[11])
        )
    return False


_INN_HINT = re.compile(r"\bинн\b", re.IGNORECASE)


def find_inn(text: str) -> list[PiiSpan]:
    """Найти ИНН по контрольной сумме либо по слову-подсказке рядом."""
    spans = [
        PiiSpan(m.start(), m.end(), "INN", 0.85)
        for m in _INN_RE.finditer(text)
        if is_valid_inn(m.group())
    ]
    spans.extend(_find_near_hint(text, _INN_HINT, _INN_RE, "INN", 0.9))
    # Один номер может найтись обеими ветками сразу; наружу отдаём одну
    # находку, иначе вызывающий код увидит дубль там, где данные единственные.
    return _resolve_overlaps(spans)


# --- Лицевой счёт ------------------------------------------------------------- #

_ACCOUNT_CONTEXT = re.compile(
    r"(?:сч[её]т\w*|л\s*/\s*с|\bлс\b|\bл\.с\.|\baccount\b)",
    re.IGNORECASE,
)
"""Подсказки, рядом с которыми длинное число считается лицевым счётом.

Слово «счёт» в любой форме, а не только сочетание «лицевой счёт». Проверка на
выходе показала дыру: модель в ответе естественно пишет «задолженность **по
счёту** 1234567890», и при более узкой подсказке такая утечка проходила
насквозь. Ложных срабатываний это почти не добавляет — рядом должно стоять
именно девяти- или десятизначное число, а суммы в рублях столько цифр не имеют.
"""

_ACCOUNT_DIGITS = re.compile(r"\b\d{9,10}\b")


def find_account_numbers(text: str) -> list[PiiSpan]:
    """Найти лицевые счета — только рядом со словом-подсказкой.

    Раздел 11.5 прямо предупреждал: шаблон из девяти-десяти цифр слишком широк.
    Без подсказки под него попадут номер телефона без кода, идентификатор заявки
    и любое длинное число, а обезличивание превратится в порчу текста.

    Оценка 0.9 выше, чем у ИНН по контрольной сумме (0.85), и это осознанно:
    десятизначный номер рядом со словом «л/с» может случайно пройти проверку
    ИНН, но подсказка в тексте — свидетельство сильнее совпадения формата.
    """
    return _find_near_hint(text, _ACCOUNT_CONTEXT, _ACCOUNT_DIGITS, "ACCOUNT_NUMBER", 0.9)


# --- Телефон ------------------------------------------------------------------ #

# Исправленный шаблон из раздела 11.4: в прототипе он был синтаксически неверен
# (`(?\d{3})?` — незакрытая группа), то есть телефоны не находились вовсе.
_PHONE_RE = re.compile(
    r"(?:\+7|\b8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b",
)


def find_phones(text: str) -> list[PiiSpan]:
    return [PiiSpan(m.start(), m.end(), "PHONE_NUMBER", 0.8) for m in _PHONE_RE.finditer(text)]


# --- Сведение находок --------------------------------------------------------- #


def _resolve_overlaps(spans: list[PiiSpan]) -> list[PiiSpan]:
    """Из пересекающихся находок оставить одну.

    Побеждает более длинная, при равной длине — более уверенная. Иначе почта,
    помеченная сразу тремя типами, была бы заменена трижды, и текст развалился
    бы: замены накладывались бы друг на друга.
    """
    ordered = sorted(spans, key=lambda s: (-s.length, -s.score, s.start))
    kept: list[PiiSpan] = []
    for span in ordered:
        if not any(span.overlaps(other) for other in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


_UNSET = object()
"""Часовой: отличает «движок не передали» от «передали None намеренно».
Второе означает работу без разбора языка, и это допустимый режим."""


class _Finding(Protocol):
    """Форма находки Presidio — ровно те поля, которые мы читаем."""

    start: int
    end: int
    entity_type: str
    score: float


class _Analyzer(Protocol):
    """Минимум, который нужен от Presidio. Ради подмены в тестах."""

    def analyze(self, text: str, language: str, entities: list[str]) -> list[_Finding]: ...


@lru_cache(maxsize=1)
def _presidio_analyzer() -> _Analyzer | None:
    """Собрать движок Presidio один раз за процесс.

    Загрузка русской языковой модели занимает секунды, поэтому кэшируется. Если
    Presidio или модель не установлены, фильтр не падает, а работает только на
    собственных распознавателях — но об этом обязан узнать вызывающий код,
    см. :attr:`PiiSanitizer.language_model_available`.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "ru", "model_name": "ru_core_news_md"}],
            }
        )
        engine = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=["ru"]
        )
        # Presidio не типизирован под наш протокол, но по форме совпадает:
        # у его находок есть start, end, entity_type и score.
        return cast("_Analyzer", engine)
    except Exception:  # noqa: BLE001 — причина неважна, важен факт недоступности
        return None


class PiiSanitizer:
    """Обезличиватель: заменяет персональные данные метками.

    :param analyzer: движок разбора языка. По умолчанию берётся Presidio; можно
        передать свой или ``None``, чтобы работали только собственные
        распознаватели — так устроены быстрые тесты.
    """

    def __init__(self, analyzer: _Analyzer | None | object = _UNSET) -> None:
        resolved = _presidio_analyzer() if analyzer is _UNSET else analyzer
        self._analyzer: _Analyzer | None = resolved  # type: ignore[assignment]

    @property
    def language_model_available(self) -> bool:
        """Доступен ли разбор языка.

        Если нет — ФИО и адреса не распознаются. Это не повод падать, но повод
        поднять тревогу: в рабочем контуре такое состояние недопустимо.
        """
        return self._analyzer is not None

    def find(self, text: str) -> list[PiiSpan]:
        """Найти все персональные данные, разрешив перекрытия."""
        spans = [
            *find_snils(text),
            *find_inn(text),
            *find_account_numbers(text),
            *find_phones(text),
        ]
        if self._analyzer is not None:
            spans.extend(
                PiiSpan(r.start, r.end, r.entity_type, r.score)
                for r in self._analyzer.analyze(
                    text=text, language="ru", entities=list(PRESIDIO_ENTITIES)
                )
            )
        return _resolve_overlaps(spans)

    def sanitize(self, text: str) -> tuple[str, PiiReport]:
        """Вернуть обезличенный текст и отчёт.

        Отчёт содержит только типы найденного — значения в него не попадают ни
        при каких условиях (правило 4.2, проверяется схемой :class:`PiiReport`).
        """
        spans = self.find(text)
        if not spans:
            return text, PiiReport()

        pieces: list[str] = []
        cursor = 0
        for span in spans:
            pieces.append(text[cursor : span.start])
            pieces.append(PLACEHOLDERS.get(span.entity_type, f"<{span.entity_type}>"))
            cursor = span.end
        pieces.append(text[cursor:])

        entity_types = sorted({span.entity_type for span in spans})
        return "".join(pieces), PiiReport(pii_detected=True, entities=entity_types)


def sanitize(text: str) -> tuple[str, PiiReport]:
    """Обезличить текст обезличивателем по умолчанию."""
    return PiiSanitizer().sanitize(text)
