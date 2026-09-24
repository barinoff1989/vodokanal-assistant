"""Проверки справочника типов обращений.

Половина тестов здесь воспроизводит конкретные дефекты прошлого прототипа.
Смысл не в покрытии, а в том, чтобы
ошибка, уже сломавшая проект однажды, не смогла вернуться незамеченной.
"""

from __future__ import annotations

import pytest

from app.taxonomy import (
    DISPLAY_NAMES,
    KEYWORDS,
    LEGACY_ALIASES,
    SYNTHETIC_CODES,
    TOPIC_DISPLAY_NAMES,
    InquiryType,
    Topic,
    UnknownInquiryTypeError,
    UnknownTopicError,
    classify_by_keywords,
    coerce,
    coerce_topic,
    display_name,
    parse,
    parse_topic,
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
    """Требование: без пробелов, в нижнем регистре, латиницей."""
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
        "accrual_recalculation",
        "  accrual_recalculation  ",  # хвостовые пробелы
        "ACCRUAL_RECALCULATION",
        "\tAccrual_Recalculation\n",
    ],
)
def test_пробелы_и_регистр_не_мешают_разбору(raw: str):
    """Главный дефект прототипа: 'перерасчёт ' с пробелом ломал фильтр молча."""
    assert parse(raw) is InquiryType.ACCRUAL_RECALCULATION


def test_значение_с_пробелом_равно_значению_без_него():
    """Смысл нормализации: два написания дают один и тот же код."""
    assert parse("debt ") is parse("debt")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("перерасчёт ", InquiryType.ACCRUAL_RECALCULATION),  # хвостовой пробел
        ("перерасчет", InquiryType.ACCRUAL_RECALCULATION),  # без буквы ё
        ("поверка_счётчика ", InquiryType.METER_VERIFICATION),
        ("задолженность", InquiryType.DEBT),
        ("prochee", InquiryType.OTHER),  # латиницей
        ("Прочее", InquiryType.OTHER),
        # Код версии 4.20: название реального типа шире перерасчёта.
        ("recalculation", InquiryType.ACCRUAL_RECALCULATION),
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
    """Дефект прошлого прототипа: фильтр не находил документы из-за несовпадения значений.

    `StrEnum` даёт равенство со строкой, поэтому код можно класть в метаданные
    и в условие поиска как есть.
    """
    metadata = {"inquiry_type": InquiryType.ACCRUAL_RECALCULATION}
    assert metadata["inquiry_type"] == "accrual_recalculation"
    assert f"{InquiryType.ACCRUAL_RECALCULATION}" == "accrual_recalculation"


# --- состав справочника (ADR-300) -------------------------------------------- #


REFERENCE_NAMES = (
    "Возврат денежных средств",
    "Задолженность",
    "Обследование",
    "Оплата",
    "Заказать опломбировку",
    "Пени",
    "Снятие пломбы",
    "Справка",
    "Заказать поверку",
    "Направить документы",
    "Начисления/Перерасчет",
    "Заказать установку прибора учета",
    "Отмена обращения",
    "Другое",
    "Повторная приемка узла учета воды в эксплуатацию (без авто)",
    "Повторная приемка узла учета воды в эксплуатацию (с авто)",
)
"""Справочник владельца системы обращений, дословно. Шестнадцать типов."""


def test_справочник_воспроизведён_дословно():
    """Сторожит не арифметику, а то, что типы не добавили и не убрали молча.

    Число уже один раз оказалось не тем, каким его считали: по присланному
    списку названий типов было тринадцать, а в выгрузке нашлось шестнадцать —
    «Другое» и обе «Повторные приемки» в список не попали.
    """
    ours = {DISPLAY_NAMES[code] for code in InquiryType if code is not InquiryType.EMERGENCY}
    assert ours == set(REFERENCE_NAMES)


def test_авария_единственный_код_сверх_справочника():
    """Аварий в справочнике нет — подтверждено владельцем (ADR-100).

    Код существует, потому что заявка регистрируется в Диспетчерской, и
    адаптеру нужно что-то отклонять. Появление здесь второго кода означает,
    что кто-то расширил регистрационную ось без решения.
    """
    beyond = {c for c in InquiryType if DISPLAY_NAMES[c] not in REFERENCE_NAMES}
    assert beyond == {InquiryType.EMERGENCY}


def test_все_коды_справочника_помечены_синтетическими():
    """Пока владелец не прислал настоящие коды, каждый из них — наш.

    Без пометки синтетический код однажды примут за настоящий при
    согласовании — та же ошибка, что недостоверные цифры.
    """
    assert set(InquiryType) - {InquiryType.EMERGENCY} == SYNTHETIC_CODES


def test_название_с_хвостовым_пробелом_разбирается():
    """Не гипотеза: в выгрузке владельца у этого типа хвостовой пробел.

    Дефект прошлого прототипа сломал прототип и живёт в данных реальной системы прямо
    сейчас. Нормализация на входе обязательна.
    """
    raw = "Повторная приемка узла учета воды в эксплуатацию (без авто) "
    assert parse(raw) is InquiryType.METER_REACCEPTANCE


def test_две_приемки_различаются():
    """«С авто» и «без авто» — разные типы справочника.

    Что означают суффиксы, неизвестно, но схлопывать их в один
    код нельзя: владелец ведёт их порознь.
    """
    assert InquiryType.METER_REACCEPTANCE is not InquiryType.METER_REACCEPTANCE_AUTO


# --- несравнимость осей (ADR-300) -------------------------------------------- #


def test_значения_осей_не_пересекаются():
    """Главная защита от возврата дефекта прошлого прототипа в новом виде.

    Оси описывают разное: во что регистрируют и о чём спрашивают. Пересечение
    значений позволило бы сравнить их между собой, и сравнение прошло бы
    молча — ровно так прототип и ломался.
    """
    assert {c.value for c in InquiryType}.isdisjoint({t.value for t in Topic})


def test_тема_не_разбирается_как_тип_обращения():
    """И наоборот. Перепутанная ось должна давать ошибку, а не тихий результат."""
    with pytest.raises(UnknownInquiryTypeError):
        parse(Topic.OUTAGE.value)
    with pytest.raises(UnknownTopicError):
        parse_topic(InquiryType.DEBT.value)


def test_у_каждой_темы_есть_название():
    assert set(TOPIC_DISPLAY_NAMES) == set(Topic)


def test_мягкий_разбор_темы_даёт_обычный_путь():
    """Тема без особого пути ответа отвечается по базе знаний."""
    assert coerce_topic("выдуманная_тема") is Topic.GENERAL
    assert coerce_topic(None) is Topic.GENERAL


def test_разбор_темы_терпит_пробелы_и_регистр():
    assert parse_topic("  OUTAGE  ") is Topic.OUTAGE


def test_коды_тем_машиночитаемы():
    for topic in Topic:
        assert topic.value == topic.value.strip().lower()
        assert " " not in topic.value
        assert topic.value.isascii()


# --- запасной режим классификации ------------------------------------------- #
#
# Примеры ниже — настоящие тексты обращений из выгрузки владельца, а не
# придуманные. Разница существенна: язык оператора не совпадает с названиями
# типов, и перечень слов, собранный по названиям, на этих текстах не работал
# (охват 46%). Слова собраны по наблюдённому словарю, охват стал 73%.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # «Обследование» во всех девяти случаях записано так. Слова
        # «обследование» нет ни в одном из них.
        ("снятие кп", InquiryType.INSPECTION),
        # «Справка» в этой системе — справка для сделки купли-продажи.
        ("снять контрольные показания для продажи 28.08.2026 сделка", InquiryType.CERTIFICATE),
        ("Снятие пломбы ИПУ ХВС", InquiryType.METER_UNSEALING),
        ("Заявка на опломбировку;", InquiryType.METER_SEALING),
        ("ввод ИПУ ХВС и ГВС (в+к) норматив", InquiryType.METER_SEALING),
        (
            "Здравствуйте! Направляю поверку счетчика холодной воды.",
            InquiryType.METER_VERIFICATION,
        ),
        ("прошу убрать из квитанций сумму пени в размере 331, 78р.", InquiryType.PENALTY),
        ("Откуда взялась задолженность, если плачу вовремя?", InquiryType.DEBT),
        (
            "Здравствуйте,заменили прибор учета,прошу принять новые данные",
            InquiryType.METER_INSTALLATION,
        ),
        ("Прошу отменить заявку А022815", InquiryType.INQUIRY_CANCELLATION),
        ("Добрый вечер. По адресу неделю нет воды. Поясните причину.", InquiryType.EMERGENCY),
    ],
)
def test_запасной_режим_узнаёт_настоящие_тексты(text: str, expected: InquiryType):
    assert expected in classify_by_keywords(text)


def test_запасной_режим_возвращает_все_подошедшие_коды():
    """Неоднозначность — свойство текста, и прятать её выбором первого нельзя.

    Каждый пятый настоящий текст даёт больше одного кода. Вызывающий код должен
    видеть это и решать сам — спросить уточнение или отдать в «Другое».
    """
    codes = classify_by_keywords("Прошу пересчитать задолженность и убрать пени")
    assert len(codes) > 1
    assert InquiryType.PENALTY in codes
    assert InquiryType.DEBT in codes


def test_запасной_режим_молчит_а_не_угадывает():
    """Отсутствие сигнала — не ошибка классификации, а её отсутствие.

    Текст «ПУ 1шт ГВ» лежит в выгрузке под типом «Отмена обращения»: оператор
    знал больше, чем написано. Выводить отсюда тип нечем, и придумывать его
    нельзя.
    """
    assert classify_by_keywords("ПУ 1шт ГВ") == []
    assert classify_by_keywords("") == []


def test_у_прочего_и_приемки_с_авто_нет_ключевых_слов():
    """Пустой перечень честнее угаданного.

    `other` — исход по умолчанию. У «приемки (с авто)» одно наблюдение, и что
    означает «с авто», неизвестно: слова пришлось бы выдумать.
    """
    assert InquiryType.OTHER not in KEYWORDS
    assert InquiryType.METER_REACCEPTANCE_AUTO not in KEYWORDS
