"""Проверки справочника типов обращений.

Половина тестов здесь воспроизводит конкретные дефекты прошлого прототипа
(контекст, разделы 11.1, 11.6, 11.8). Смысл не в покрытии, а в том, чтобы
ошибка, уже сломавшая проект однажды, не смогла вернуться незамеченной.
"""

from __future__ import annotations

import pytest

from app.taxonomy import (
    DISPLAY_NAMES,
    KEYWORDS,
    LEGACY_ALIASES,
    InquiryType,
    UnknownInquiryTypeError,
    coerce,
    display_name,
    parse,
)

# --- целостность самого справочника ---------------------------------------- #


def test_у_каждого_кода_есть_отображаемое_название():
    """Иначе интерфейс покажет абоненту машинный код."""
    assert set(DISPLAY_NAMES) == set(InquiryType)


def test_отображаемые_названия_не_совпадают_с_кодами():
    """Название и код — разные сущности; совпадение означает путаницу ролей."""
    for code, name in DISPLAY_NAMES.items():
        assert name != code.value


def test_коды_машиночитаемы():
    """Требование раздела 9.2: без пробелов, в нижнем регистре, латиницей."""
    for code in InquiryType:
        assert code.value == code.value.strip().lower()
        assert " " not in code.value
        assert code.value.isascii()


def test_ключевые_слова_ссылаются_только_на_существующие_коды():
    """Ровно тот рассинхрон, что сломал прототип: ключи разъехались с кодами."""
    assert set(KEYWORDS) <= set(InquiryType)


def test_у_прочего_нет_ключевых_слов():
    """`other` — исход по умолчанию, а не распознаваемый тип."""
    assert InquiryType.OTHER not in KEYWORDS


def test_ключевые_слова_нормализованы():
    for code, words in KEYWORDS.items():
        assert words, f"пустой список слов у {code}"
        for word in words:
            assert word == word.strip().lower()


def test_псевдонимы_нормализованы_и_ведут_к_существующим_кодам():
    for alias, code in LEGACY_ALIASES.items():
        assert alias == alias.strip().lower(), f"псевдоним {alias!r} не нормализован"
        assert code in InquiryType


# --- разбор значений: дефекты прошлой версии -------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        "recalculation",
        "  recalculation  ",  # хвостовые пробелы, раздел 11.1
        "RECALCULATION",
        "\tRecalculation\n",
    ],
)
def test_пробелы_и_регистр_не_мешают_разбору(raw: str):
    """Главный дефект прототипа: 'перерасчёт ' с пробелом ломал фильтр молча."""
    assert parse(raw) is InquiryType.RECALCULATION


def test_значение_с_пробелом_равно_значению_без_него():
    """Смысл нормализации: два написания дают один и тот же код."""
    assert parse("debt ") is parse("debt")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("перерасчёт ", InquiryType.RECALCULATION),  # раздел 11.1
        ("перерасчет", InquiryType.RECALCULATION),  # без буквы ё
        ("поверка_счётчика ", InquiryType.METER_VERIFICATION),
        ("задолженность", InquiryType.DEBT),
        ("prochee", InquiryType.OTHER),  # раздел 11.8
        ("Прочее", InquiryType.OTHER),
    ],
)
def test_значения_прошлых_версий_переводятся(raw: str, expected: InquiryType):
    """Старые записи диалогов иначе молча превратились бы в «прочее»."""
    assert parse(raw) is expected


def test_код_передаётся_насквозь():
    assert parse(InquiryType.EMERGENCY) is InquiryType.EMERGENCY


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_пустое_значение_это_ошибка(raw):
    """Пустой тип — признак поломки выше по стеку, а не повод молча подставить."""
    with pytest.raises(UnknownInquiryTypeError):
        parse(raw)


def test_неизвестное_значение_это_ошибка():
    with pytest.raises(UnknownInquiryTypeError):
        parse("выдуманный_тип")


def test_текст_ошибки_содержит_само_значение():
    """Без него отладка рассинхрона снова превращается в угадывание."""
    with pytest.raises(UnknownInquiryTypeError, match="выдуманный_тип"):
        parse("выдуманный_тип")


# --- мягкий разбор ---------------------------------------------------------- #


def test_мягкий_разбор_подставляет_прочее():
    """Абонент не должен получать отказ из-за неожиданного ответа модели."""
    assert coerce("выдуманный_тип") is InquiryType.OTHER
    assert coerce(None) is InquiryType.OTHER


def test_мягкий_разбор_не_портит_корректные_значения():
    """Подмена на `other` должна касаться только неопознанного."""
    assert coerce(" DEBT ") is InquiryType.DEBT


# --- отображение ------------------------------------------------------------ #


def test_название_выдаётся_по_любому_написанию_кода():
    assert display_name(" debt ") == "Задолженность"
    assert display_name(InquiryType.EMERGENCY) == "Авария"


def test_название_для_неизвестного_кода_это_ошибка():
    """Молчаливое «Прочее» в интерфейсе скрыло бы поломку классификации."""
    with pytest.raises(UnknownInquiryTypeError):
        display_name("выдуманный_тип")


# --- пригодность к фильтру поиска ------------------------------------------- #


def test_код_уходит_в_фильтр_без_преобразований():
    """Дефект 11.6: фильтр не находил документы из-за несовпадения значений.

    `StrEnum` даёт равенство со строкой, поэтому код можно класть в метаданные
    и в условие поиска как есть.
    """
    metadata = {"inquiry_type": InquiryType.RECALCULATION}
    assert metadata["inquiry_type"] == "recalculation"
    assert f"{InquiryType.RECALCULATION}" == "recalculation"
