"""Единый справочник типов обращений.

Это самый маленький модуль проекта и одновременно самый ответственный.
Правило 4.8 требует, чтобы значение `inquiry_type` было одинаковым во всех
компонентах: классификация, фильтр поиска, маршрутизация агента, выбор шаблона
документа, аналитика. Прототип прошлой версии на этом сломался — в нём
одновременно жили русские значения с хвостовыми пробелами («перерасчёт »),
русские без пробелов и код `prochee`, из-за чего фильтр поиска не находил
ничего (контекст, разделы 11.1, 11.8).

Отсюда устройство модуля:

* канонические коды — единственная форма, в которой тип живёт внутри системы;
* отображаемые названия вынесены отдельно и в сравнениях не участвуют;
* любое значение извне проходит через :func:`parse` или :func:`coerce`, где
  снимаются пробелы и регистр;
* ключевые слова запасного режима лежат здесь же, рядом с кодами, — иначе они
  снова разъедутся, как в прошлый раз.

> **[ОТКРЫТО — TODO 4а]** Совпадает ли справочник реальной БД обращений с этими
> десятью кодами, неизвестно: вопрос задан владельцу системы (`Data_Request.md`).
> Если номенклатура своя, понадобится таблица соответствия — отдельным модулем,
> не правкой этого перечня: канонические коды остаются внутренним языком системы.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DISPLAY_NAMES",
    "KEYWORDS",
    "LEGACY_ALIASES",
    "InquiryType",
    "UnknownInquiryTypeError",
    "coerce",
    "display_name",
    "parse",
]


class InquiryType(StrEnum):
    """Канонические коды типов обращений (контекст, раздел 9.3).

    `StrEnum`, а не обычный `Enum`: значение должно без преобразований уходить
    в фильтр векторного поиска и в метаданные запроса, где ожидается строка.
    """

    RECALCULATION = "recalculation"
    METER_VERIFICATION = "meter_verification"
    RECONCILIATION = "reconciliation"
    CONTRACT_SIGNING = "contract_signing"
    CONTRACT_TERMINATION = "contract_termination"
    DEBT = "debt"
    WATER_QUALITY = "water_quality"
    EMERGENCY = "emergency"
    TARIFF_CHANGE = "tariff_change"
    OTHER = "other"


class UnknownInquiryTypeError(ValueError):
    """Значение не является каноническим кодом и не имеет известного псевдонима."""


# Отображаемые названия. Живут отдельно от кодов сознательно: как только
# название попадёт в сравнение или в фильтр, вернётся ровно тот дефект, из-за
# которого переписывается прототип.
DISPLAY_NAMES: dict[InquiryType, str] = {
    InquiryType.RECALCULATION: "Перерасчёт",
    InquiryType.METER_VERIFICATION: "Поверка счётчика",
    InquiryType.RECONCILIATION: "Сверка расчётов",
    InquiryType.CONTRACT_SIGNING: "Заключение договора",
    InquiryType.CONTRACT_TERMINATION: "Расторжение договора",
    InquiryType.DEBT: "Задолженность",
    InquiryType.WATER_QUALITY: "Качество воды",
    InquiryType.EMERGENCY: "Авария",
    InquiryType.TARIFF_CHANGE: "Смена тарифа",
    InquiryType.OTHER: "Прочее",
}


# Псевдонимы прошлых версий. Нужны не для красоты: старые записи диалогов и
# разметка, сделанные до нормализации, содержат именно эти значения, и без
# перевода они молча превратились бы в «прочее».
# Ключи хранятся уже приведёнными к нижнему регистру и без пробелов.
LEGACY_ALIASES: dict[str, InquiryType] = {
    # Русские значения прототипа (раздел 11.1)
    "перерасчёт": InquiryType.RECALCULATION,
    "перерасчет": InquiryType.RECALCULATION,
    "поверка_счётчика": InquiryType.METER_VERIFICATION,
    "поверка_счетчика": InquiryType.METER_VERIFICATION,
    "поверка счётчика": InquiryType.METER_VERIFICATION,
    "поверка счетчика": InquiryType.METER_VERIFICATION,
    "сверка": InquiryType.RECONCILIATION,
    "задолженность": InquiryType.DEBT,
    "долг": InquiryType.DEBT,
    "качество_воды": InquiryType.WATER_QUALITY,
    "качество воды": InquiryType.WATER_QUALITY,
    "авария": InquiryType.EMERGENCY,
    "смена_тарифа": InquiryType.TARIFF_CHANGE,
    "прочее": InquiryType.OTHER,
    # Транслитерация из запасной ветки прототипа (раздел 11.8)
    "prochee": InquiryType.OTHER,
}


# Ключевые слова запасного режима классификации: используются, когда модель
# недоступна и тип определяется без неё (контекст, раздел 6.1, шаг 3).
#
# > **[ПРЕДПОЛОЖЕНИЕ]** Слова подобраны по формулировкам разделов 9 и 11.2, а не
# > по реальным обращениям абонентов — их тексты ещё не получены
# > (`Data_Request.md`, пункт 1). После получения выборки перечень нужно
# > пересобрать по частотности, а не дополнять на глаз.
KEYWORDS: dict[InquiryType, tuple[str, ...]] = {
    InquiryType.RECALCULATION: ("перерасчёт", "перерасчет", "пересчит", "перерасч"),
    InquiryType.METER_VERIFICATION: (
        "поверк",
        "счётчик",
        "счетчик",
        "прибор учёта",
        "прибор учета",
    ),
    InquiryType.RECONCILIATION: ("сверк", "акт сверки", "сверить"),
    InquiryType.CONTRACT_SIGNING: ("заключ", "договор", "подключ", "технолог"),
    InquiryType.CONTRACT_TERMINATION: ("расторж", "растор", "закрыть договор", "отключ"),
    InquiryType.DEBT: ("задолж", "долг", "неоплат", "пеня", "пени", "квитанц"),
    InquiryType.WATER_QUALITY: ("качеств", "мутн", "запах", "ржав", "цвет воды"),
    InquiryType.EMERGENCY: ("авари", "прорыв", "утечк", "нет воды", "порыв", "затоп"),
    InquiryType.TARIFF_CHANGE: ("тариф", "норматив", "ставк"),
    # Для `other` ключевых слов нет и быть не должно: это исход по умолчанию,
    # а не тип, который можно распознать.
}


def _clean(value: str) -> str:
    """Снять пробелы и регистр — ровно то, чего не хватало прототипу.

    Хвостовой пробел в значении «перерасчёт » ломал фильтр поиска молча:
    ошибки не возникало, просто ничего не находилось (раздел 11.1).
    """
    return value.strip().lower()


def parse(value: str | InquiryType | None) -> InquiryType:
    """Привести значение к каноническому коду, иначе — ошибка.

    Строгий разбор для мест, где неизвестный тип означает поломку: чтение
    справочника, загрузка эталонного набора, разбор ответа классификатора.

    :raises UnknownInquiryTypeError: значение пустое или не опознано.
    """
    if isinstance(value, InquiryType):
        return value
    if value is None:
        raise UnknownInquiryTypeError("тип обращения не задан")

    cleaned = _clean(value)
    if not cleaned:
        raise UnknownInquiryTypeError("тип обращения пуст")

    try:
        return InquiryType(cleaned)
    except ValueError:
        pass

    alias = LEGACY_ALIASES.get(cleaned)
    if alias is not None:
        return alias

    raise UnknownInquiryTypeError(f"неизвестный тип обращения: {value!r}")


def coerce(value: str | InquiryType | None) -> InquiryType:
    """То же, но неопознанное значение становится `other`, а не ошибкой.

    Для пользовательского пути: абонент не должен получить отказ из-за того,
    что классификатор вернул неожиданное значение. Ответственность за то, чтобы
    поток `other` не рос незаметно, лежит на метрике доли `other` в разбивке
    трафика (дашборд наблюдаемости, раздел 37.1).
    """
    try:
        return parse(value)
    except UnknownInquiryTypeError:
        return InquiryType.OTHER


def display_name(value: str | InquiryType) -> str:
    """Название для показа абоненту. Внутри системы не используется."""
    return DISPLAY_NAMES[parse(value)]
