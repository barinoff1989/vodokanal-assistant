"""Две оси классификации обращения: во что регистрируют и о чём спрашивают.

Правило 4.8 требует, чтобы значение типа было одинаковым во всех компонентах.
Прототип прошлой версии на этом сломался — в нём одновременно жили русские
значения с хвостовыми пробелами («перерасчёт »), русские без пробелов и код
`prochee`, из-за чего фильтр поиска не находил ничего (разделы 11.1, 11.8).

ПОЧЕМУ ОСЕЙ ДВЕ (ADR-011).
Реальный справочник типов обращений получен 2 сентября 2026 и с придуманным
перечнем раздела 9.3 не совпал: `parse()` распознавал 1 значение из 13, шесть
кодов из десяти не имели соответствия вовсе. Расхождение оказалось **в оси**, а
не в названиях: наш перечень строился по теме обращения, реальный — по
заказываемой услуге и документообороту.

* :class:`InquiryType` — **регистрационная** ось. Во что заводится обращение в
  системе водоканала. Нужна на пути записи (Этап 2).
* :class:`Topic` — **тематическая** ось. О чём спрашивает абонент. Нужна для
  выбора пути ответа и фильтра поиска (Этап 1).

Правило 4.8 в новой редакции: у каждой оси один справочник и одно назначение, а
**значения разных осей никогда не сравниваются между собой**. Это не возврат к
дефекту 11.1: там было два словаря для одного и того же, здесь — две оси для
разного. Несравнимость сторожится тестом, иначе однажды сравнят молча.

КОДЫ СИНТЕТИЧЕСКИЕ.
Владелец передал названия, но не коды (пункт 2 `Data_Request.md`). Коды ниже
назначены нами и помечены :data:`SYNTHETIC_CODES`. Это не формальность: проект
уже пострадал от выдуманных чисел, поданных как факт (раздел 26.4), и
синтетический код, принятый за настоящий, — та же ошибка в другом материале.
Колонка реального кода добавляется сюда же, без правки чего-либо ещё.

НОРМАЛИЗАЦИЯ ОБЯЗАТЕЛЬНА НА ВХОДЕ.
В выгрузке реальной системы у типа «Повторная приемка узла учета воды в
эксплуатацию (без авто) » **хвостовой пробел**. Дефект 11.1 не гипотетический —
он живёт в данных владельца прямо сейчас.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DISPLAY_NAMES",
    "TOPIC_DISPLAY_NAMES",
    "KEYWORDS",
    "LEGACY_ALIASES",
    "SYNTHETIC_CODES",
    "InquiryType",
    "Topic",
    "UnknownInquiryTypeError",
    "UnknownTopicError",
    "coerce",
    "coerce_topic",
    "display_name",
    "parse",
    "parse_topic",
    "topic_display_name",
]


class InquiryType(StrEnum):
    """Во что регистрируется обращение (ADR-011, регистрационная ось).

    Шестнадцать значений реального справочника плюс :attr:`EMERGENCY`.

    `StrEnum`, а не обычный `Enum`: значение должно без преобразований уходить
    в метаданные запроса и в вызов чужого API, где ожидается строка.
    """

    # --- шестнадцать типов реального справочника ---
    REFUND = "refund"
    DEBT = "debt"
    INSPECTION = "inspection"
    PAYMENT = "payment"
    METER_SEALING = "meter_sealing"
    PENALTY = "penalty"
    METER_UNSEALING = "meter_unsealing"
    CERTIFICATE = "certificate"
    METER_VERIFICATION = "meter_verification"
    DOCUMENT_SUBMISSION = "document_submission"
    ACCRUAL_RECALCULATION = "accrual_recalculation"
    METER_INSTALLATION = "meter_installation"
    INQUIRY_CANCELLATION = "inquiry_cancellation"
    METER_REACCEPTANCE = "meter_reacceptance"
    METER_REACCEPTANCE_AUTO = "meter_reacceptance_auto"
    OTHER = "other"

    # --- сверх справочника ---
    EMERGENCY = "emergency"
    """Аварийная заявка. В справочнике обращений её нет, и это подтверждено
    владельцем систем: аварии ведутся в Диспетчерской (ADR-009).

    Значение сохранено потому, что заявка всё-таки **регистрируется** — просто
    в другой системе. `app/adapters/inquiry_service.py` отклоняет её явно, и
    без этого кода отклонять было бы нечего.

    Не путать с :attr:`Topic.OUTAGE`: там вопрос абонента об отсутствии воды,
    здесь заявка на устранение аварии. Разные оси, разные пути."""


class Topic(StrEnum):
    """О чём спрашивает абонент (ADR-011, тематическая ось).

    Определяет **путь ответа**, а не место регистрации. Состав растёт на
    код-шаге 5 из состава базы знаний; сейчас здесь только те темы, для которых
    путь ответа уже решён.

    Значения намеренно не пересекаются со значениями :class:`InquiryType` —
    сравнение осей между собой вернуло бы дефект 11.1 в новом виде.
    """

    GENERAL = "general"
    """Обычный вопрос: ответ собирается по базе знаний. Исход по умолчанию."""

    OUTAGE = "outage"
    """Вопрос об отсутствии воды. Отвечается точным поиском по графику
    плановых отключений, без обращения к модели (ADR-013).

    В выгрузке обращений такой вопрос встречается фактически: «По адресу
    неделю нет холодной воды. Поясните причину» — под типом «Другое»."""

    WATER_QUALITY = "water_quality"
    """Вопрос о качестве воды. Отвечается утверждённой формулировкой о
    стандарте качества, дословно и без обращения к модели (ADR-012).

    Тема живёт здесь, а не на регистрационной оси: фиксированный ответ — это
    путь чтения. Обращения такого типа в справочнике владельца нет."""


class UnknownInquiryTypeError(ValueError):
    """Значение не является кодом типа и не имеет известного псевдонима."""


class UnknownTopicError(ValueError):
    """Значение не является кодом темы."""


SYNTHETIC_CODES: frozenset[InquiryType] = frozenset(
    {
        InquiryType.REFUND,
        InquiryType.DEBT,
        InquiryType.INSPECTION,
        InquiryType.PAYMENT,
        InquiryType.METER_SEALING,
        InquiryType.PENALTY,
        InquiryType.METER_UNSEALING,
        InquiryType.CERTIFICATE,
        InquiryType.METER_VERIFICATION,
        InquiryType.DOCUMENT_SUBMISSION,
        InquiryType.ACCRUAL_RECALCULATION,
        InquiryType.METER_INSTALLATION,
        InquiryType.INQUIRY_CANCELLATION,
        InquiryType.METER_REACCEPTANCE,
        InquiryType.METER_REACCEPTANCE_AUTO,
        InquiryType.OTHER,
    }
)
"""Коды, назначенные нами, а не взятые из системы владельца (пункт 36 TODO).

Пометка существует, чтобы синтетический код не приняли за настоящий при
согласовании с владельцем. Когда реальные коды придут, здесь останутся только
те, для которых соответствие не найдено, — а не пустое множество молча."""


DISPLAY_NAMES: dict[InquiryType, str] = {
    InquiryType.REFUND: "Возврат денежных средств",
    InquiryType.DEBT: "Задолженность",
    InquiryType.INSPECTION: "Обследование",
    InquiryType.PAYMENT: "Оплата",
    InquiryType.METER_SEALING: "Заказать опломбировку",
    InquiryType.PENALTY: "Пени",
    InquiryType.METER_UNSEALING: "Снятие пломбы",
    InquiryType.CERTIFICATE: "Справка",
    InquiryType.METER_VERIFICATION: "Заказать поверку",
    InquiryType.DOCUMENT_SUBMISSION: "Направить документы",
    InquiryType.ACCRUAL_RECALCULATION: "Начисления/Перерасчет",
    InquiryType.METER_INSTALLATION: "Заказать установку прибора учета",
    InquiryType.INQUIRY_CANCELLATION: "Отмена обращения",
    InquiryType.METER_REACCEPTANCE: (
        "Повторная приемка узла учета воды в эксплуатацию (без авто)"
    ),
    InquiryType.METER_REACCEPTANCE_AUTO: (
        "Повторная приемка узла учета воды в эксплуатацию (с авто)"
    ),
    InquiryType.OTHER: "Другое",
    InquiryType.EMERGENCY: "Авария",
}
"""Названия дословно как в системе владельца — включая написание «Перерасчет»
без «ё» и пробел перед скобкой там, где он есть в выгрузке.

Правка написания «для красоты» разошлась бы со справочником, а сверять по нему
будут именно эти строки. Исключение — `emergency`: его в справочнике нет,
название наше."""


TOPIC_DISPLAY_NAMES: dict[Topic, str] = {
    Topic.GENERAL: "Общий вопрос",
    Topic.OUTAGE: "Отключение воды",
    Topic.WATER_QUALITY: "Качество воды",
}


LEGACY_ALIASES: dict[str, InquiryType] = {
    # Русские значения прототипа (раздел 11.1). Нужны не для красоты: старые
    # записи диалогов и разметка содержат именно их, и без перевода они молча
    # превратились бы в «Другое».
    "перерасчёт": InquiryType.ACCRUAL_RECALCULATION,
    "перерасчет": InquiryType.ACCRUAL_RECALCULATION,
    "поверка_счётчика": InquiryType.METER_VERIFICATION,
    "поверка_счетчика": InquiryType.METER_VERIFICATION,
    "поверка счётчика": InquiryType.METER_VERIFICATION,
    "поверка счетчика": InquiryType.METER_VERIFICATION,
    "задолженность": InquiryType.DEBT,
    "долг": InquiryType.DEBT,
    "авария": InquiryType.EMERGENCY,
    "прочее": InquiryType.OTHER,
    # Транслитерация из запасной ветки прототипа (раздел 11.8).
    "prochee": InquiryType.OTHER,
    # Код версии 4.20. Реальное название шире перерасчёта и включает начисления,
    # поэтому код переименован, а прежний остаётся псевдонимом.
    "recalculation": InquiryType.ACCRUAL_RECALCULATION,
    # Названия из справочника — на случай, если тип придёт названием, а не кодом.
    # Ключи уже приведены к нижнему регистру и без крайних пробелов.
    **{name.strip().lower(): code for code, name in DISPLAY_NAMES.items()},
}
"""Псевдонимы. Ключи хранятся приведёнными к нижнему регистру и без пробелов.

Коды `reconciliation`, `contract_signing`, `contract_termination` и
`tariff_change` удалены вместе с их псевдонимами: соответствия в реальном
справочнике у них нет, а в данных они не встречаются — прототип оперировал
русскими значениями, а эти коды жили только в наших документах (ADR-011)."""


# Ключевые слова запасного режима: используются, когда модель недоступна и тип
# определяется без неё (раздел 6.1, шаг 3).
#
# > **[ПРЕДПОСЫЛКА НЕ ПОДТВЕРДИЛАСЬ — пункт 42 TODO]** Прежний перечень был
# > помечен как предположение, подобранное по формулировкам разделов 9 и 11.2.
# > Замер на 136 настоящих обращениях: **54% текстов не распознаются вовсе**,
# > 11 строк дают два-три кода разом. Перечень ниже приведён к новым кодам, но
# > **по частотности реальных текстов ещё не пересобран** — это отдельная
# > работа, и до неё запасной режим не заменяет модель, а лишь удерживает
# > сервис от полного отказа.
KEYWORDS: dict[InquiryType, tuple[str, ...]] = {
    InquiryType.ACCRUAL_RECALCULATION: ("перерасчёт", "перерасчет", "пересчит", "начислен"),
    InquiryType.METER_VERIFICATION: ("поверк",),
    InquiryType.DEBT: ("задолж", "долг"),
    InquiryType.PENALTY: ("пеня", "пени", "пеню", "пеней"),
    InquiryType.PAYMENT: ("оплатил", "оплата", "оплаты", "оплату", "платёж", "платеж"),
    InquiryType.REFUND: ("возврат", "вернуть деньги", "перенаправить"),
    InquiryType.METER_SEALING: ("опломб", "пломбир"),
    InquiryType.METER_UNSEALING: ("снятие пломб", "снять пломб", "срыв пломб"),
    InquiryType.METER_INSTALLATION: ("установку прибор", "установить прибор", "замена прибор"),
    InquiryType.INSPECTION: ("обследован",),
    InquiryType.CERTIFICATE: ("справк",),
    InquiryType.DOCUMENT_SUBMISSION: ("направляю", "прикрепля", "во вложении", "высылаю"),
    InquiryType.INQUIRY_CANCELLATION: ("отменить обращение", "отмена обращения"),
    InquiryType.METER_REACCEPTANCE: ("приемк", "приёмк", "принять на расчет"),
    InquiryType.EMERGENCY: ("авари", "прорыв", "утечк", "порыв", "затоп"),
    # Для `other` ключевых слов нет и быть не должно: это исход по умолчанию.
    # Для `meter_reacceptance_auto` их тоже нет — отличие от `meter_reacceptance`
    # в том, что означает «с авто», а это неизвестно (пункт 41 TODO).
}


def _clean(value: str) -> str:
    """Снять крайние пробелы и регистр.

    Хвостовой пробел ломал фильтр поиска молча: ошибки не возникало, просто
    ничего не находилось (раздел 11.1). В выгрузке реального справочника такой
    пробел есть — у типа «…(без авто) ».
    """
    return value.strip().lower()


def parse(value: str | InquiryType | None) -> InquiryType:
    """Привести значение к коду типа обращения, иначе — ошибка.

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
    что классификатор вернул неожиданное значение.

    `other` соответствует реальному типу «Другое» и принимающей системой
    допустим — гейт записи не нужен (ADR-011). Но в выгрузке «Другое» работает
    свалкой: под ним лежат обращения, у которых есть собственный тип. Поэтому
    попадание сюда **не считается результатом классификации** при оценке
    качества, а доля `other` вынесена в метрику с порогом (раздел 37.1).
    """
    try:
        return parse(value)
    except UnknownInquiryTypeError:
        return InquiryType.OTHER


def parse_topic(value: str | Topic | None) -> Topic:
    """Привести значение к коду темы, иначе — ошибка.

    :raises UnknownTopicError: значение пустое или не опознано.
    """
    if isinstance(value, Topic):
        return value
    if value is None:
        raise UnknownTopicError("тема не задана")

    cleaned = _clean(value)
    if not cleaned:
        raise UnknownTopicError("тема пуста")

    try:
        return Topic(cleaned)
    except ValueError:
        raise UnknownTopicError(f"неизвестная тема: {value!r}") from None


def coerce_topic(value: str | Topic | None) -> Topic:
    """То же, но неопознанное значение становится `general`.

    Исход по умолчанию — обычный путь через базу знаний: тема, для которой не
    заведён особый путь ответа, отвечается поиском по знаниям.
    """
    try:
        return parse_topic(value)
    except UnknownTopicError:
        return Topic.GENERAL


def display_name(value: str | InquiryType) -> str:
    """Название типа для показа абоненту. Внутри системы не используется."""
    return DISPLAY_NAMES[parse(value)]


def topic_display_name(value: str | Topic) -> str:
    """Название темы для показа абоненту."""
    return TOPIC_DISPLAY_NAMES[parse_topic(value)]
