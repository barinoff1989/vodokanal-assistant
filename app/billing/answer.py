"""Ответ о фактах лицевого счёта — точным чтением Биллинга, без модели.

Тема `account` (ADR-011) уводит вопрос сюда. Ответ собирается из данных
Биллинга по `subscriber_id`, а не генерируется: задолженность, начисление за
месяц, срок поверки, последние показания — это факты, и пересказ факта моделью
ровно там и ошибается, где абонент проверит ответ по квитанции.

**Четвёртый случай одного правила** после регламентной формулировки (ADR-012),
графика отключений (ADR-013) и тарифов: база знаний отвечает на то, что
объясняется; то, что проверяется по факту, отвечается поиском по ключу. Ключ
здесь — лицевой счёт, для начислений ещё и период.

ТОЛЬКО НЕЙТРАЛЬНЫЙ ЗАПРОС ФАКТА. На 136 настоящих обращениях почти каждый
вопрос о долге — это спор: «откуда задолженность, если я плачу», «на каком
основании начислено», «прошу пересчитать». На такой вопрос ответить суммой из
Биллинга хуже, чем промолчать: абонент спрашивал не сколько, а почему. Спор
отсекается стражем (`_DISPUTE_MARKERS`), и запрос уходит обычным путём — в базу
знаний и к оператору.

ПРЕДМЕТНАЯ ЛОГИКА ЖИВЁТ ЗДЕСЬ, А НЕ В ОРКЕСТРАЦИИ. Backend знает только, что у
него может быть прямой ответчик. Тот же шов, которым подключены ответчик
отключений, реестр регламентных ответов и ответчик тарифов.

ЧТО К ЭТОМУ ПУТИ ПРИМЕНЯЕТСЯ. Лимиты — да: ответ без модели ничего не стоит нам,
но обработка запроса стоит. Обезличивание — нет: данные лицевого счёта наружу
не уходят, всё остаётся в ответе абоненту, который и есть их владелец.
Охранители — нет: проверять сумму из Биллинга нашими правилами не на чем.

ПРИЗНАКИ ВОПРОСА — ПРЕДПОЛОЖЕНИЕ. Наблюдённого словаря чистых запросов факта
нет: в выгрузке обращений их почти не встречается. Формы взяты по языку вопроса
и помечены как предположение — так же, как маркеры тем `tariff` и
`water_quality`. Пересобрать замером, когда такие обращения появятся.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from app.backend.orchestrator import DirectAnswer
from app.billing.source import BillingSource
from app.models import GenerateRequest
from app.taxonomy import Topic

__all__ = ["AccountResponder", "resolve_period"]



# Спор, а не справка. Проверяется до всего остального: если вопрос про «почему»,
# отвечать «сколько» бессмысленно, даже когда в тексте есть слово «задолженность».
_DISPUTE_MARKERS: tuple[str, ...] = (
    "почему",
    "на каком основании",
    "откуда",
    "пересчит",
    "перерасчёт",
    "перерасчет",
    "оспор",
    "не согласен",
    "не согласна",
    "разберитесь",
    "разобраться",
    "неверн",
    "ошибочн",
    "необоснованн",
    "завышен",
    "не правильно",
    "неправильно",
)


# Признаки подвопроса — корни, а не готовые фразы. Точность «это вопрос факта, а
# не спор» уже обеспечил Triage (`ACCOUNT_PATTERNS`) и страж выше; ответчику
# осталось выбрать один из четырёх. Порядок проверки — от частного к общему:
# денежный вопрос («долг», «к оплате») ловится последним, он самый широкий.
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

_MONTHS: dict[str, int] = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
    "июн": 6,
    "июл": 7,
    "август": 8,
}
_MAY = re.compile(r"\bма[йяе]\b")
_NUMERIC_PERIOD = re.compile(
    r"\b(?:(?P<y1>20\d\d)[-./](?P<m1>0[1-9]|1[0-2])"
    r"|(?P<m2>0[1-9]|1[0-2])[-./](?P<y2>20\d\d))\b"
)
_BARE_MONTH_NUM = re.compile(r"\bза\s+(?P<m>0[1-9]|1[0-2])\b")
_YEAR = re.compile(r"\b(20\d\d)\b")

_MONTH_NAMES_RU = (
    "",
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("ё", "е"))


def _money(value: Decimal) -> str:
    """Сумма в вид, привычный по квитанции: разряды пробелом, копейки запятой."""
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def _period_ru(period: str) -> str:
    """`2026-06` -> `июнь 2026`."""
    try:
        year, month = period.split("-")
        return f"{_MONTH_NAMES_RU[int(month)]} {year}"
    except (ValueError, IndexError):
        return period


def _month_from_query(lowered: str) -> int | None:
    for stem, number in _MONTHS.items():
        if stem in lowered:
            return number
    if _MAY.search(lowered):
        return 5
    return None


def resolve_period(query: str, *, available: list[str], today: date) -> str | None:
    """Определить расчётный период из вопроса.

    ``None`` — период в вопросе не назван и данных нет вовсе. Если период не
    назван, а данные есть — берётся самый свежий: «сколько начислили» без месяца
    почти всегда про последнюю квитанцию.

    Год без месяца игнорируется, месяц без года разрешается по данным: если за
    названный месяц начисление есть — берётся оно, иначе текущий год.
    """
    lowered = _norm(query)

    numeric = _NUMERIC_PERIOD.search(lowered)
    if numeric is not None:
        year = numeric.group("y1") or numeric.group("y2")
        return f"{year}-{int(numeric.group('m1') or numeric.group('m2')):02d}"

    month: int | None = _month_from_query(lowered)
    if month is None:
        bare = _BARE_MONTH_NUM.search(lowered)
        month = int(bare.group("m")) if bare is not None else None

    if month is None:
        return available[0] if available else None

    year_match = _YEAR.search(lowered)
    if year_match is not None:
        return f"{year_match.group(1)}-{month:02d}"

    for period in available:
        if period.endswith(f"-{month:02d}"):
            return period
    return f"{today.year}-{month:02d}"


@dataclass(frozen=True, slots=True)
class AccountResponder:
    """Отвечает на нейтральный запрос факта о лицевом счёте.

    ``None`` означает «это не мой случай» — запрос идёт обычным путём. Ответчик
    молчит, если: тема не `account`; в вопросе есть признак спора; абонент по
    `subscriber_id` не найден; подвопрос не распознан.

    :param billing: источник данных Биллинга. Тот же экземпляр, что у регистрации
        обращений: одни данные, одно чтение файлов при старте.
    """

    billing: BillingSource

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:
        if request.metadata.topic is not Topic.ACCOUNT:
            return None

        lowered = _norm(request.query)
        if any(marker in lowered for marker in _DISPUTE_MARKERS):
            # Вопрос про «почему», а не «сколько». Уходит в базу знаний и к
            # оператору — там ему и место.
            return None

        number = request.metadata.subscriber_id
        account = self.billing.account(number)
        if account is None:
            return None

        today = now.date()
        if any(m in lowered for m in _VERIFICATION_MARKERS):
            return self._verification(number, today)
        if any(m in lowered for m in _READINGS_MARKERS):
            return self._readings(number)
        if any(m in lowered for m in _CHARGES_MARKERS):
            return self._charges(number, request.query, today)
        if any(m in lowered for m in _DEBT_MARKERS):
            return self._debt(number)
        return None

    # -- подвопросы --------------------------------------------------------- #

    def _debt(self, number: str) -> DirectAnswer:
        account = self.billing.account(number)
        assert account is not None  # проверено в answer
        if not account.has_debt:
            text = f"По лицевому счёту {number} задолженности нет."
            if account.balance > 0:
                text += f" На счёте переплата {_money(account.balance)} ₽."
            return DirectAnswer(text=text)

        by_period = self._debt_by_period(number)
        text = f"Задолженность по лицевому счёту {number}: {_money(account.debt)} ₽."
        if by_period:
            text += "\nВ том числе:\n" + "\n".join(by_period)
        return DirectAnswer(text=text)

    def _debt_by_period(self, number: str) -> list[str]:
        lines: list[str] = []
        for charge in self.billing.charges(number):
            if charge.debt > 0:
                lines.append(
                    f"- {_period_ru(charge.period)}, {charge.service}: "
                    f"{_money(charge.debt)} ₽"
                )
        return lines

    def _charges(self, number: str, query: str, today: date) -> DirectAnswer:
        available = sorted({c.period for c in self.billing.charges(number)}, reverse=True)
        period = resolve_period(query, available=available, today=today)
        if period is None:
            return DirectAnswer(
                text=f"По лицевому счёту {number} начислений в данных нет.",
            )

        rows = [c for c in self.billing.charges(number) if c.period == period]
        if not rows:
            have = ", ".join(_period_ru(p) for p in available[:3])
            return DirectAnswer(
                text=(
                    f"За {_period_ru(period)} начислений по лицевому счёту {number} "
                    f"в данных нет. Есть за: {have}."
                ),
            )

        lines = [
            f"- {c.service}: начислено {_money(c.accrued)} ₽, "
            f"оплачено {_money(c.paid)} ₽ ({c.basis})"
            for c in rows
        ]
        total = sum((c.accrued for c in rows), Decimal("0"))
        text = (
            f"Начисления за {_period_ru(period)} по лицевому счёту {number}:\n"
            + "\n".join(lines)
            + f"\nВсего начислено: {_money(total)} ₽."
        )
        return DirectAnswer(text=text)

    def _verification(self, number: str, today: date) -> DirectAnswer:
        meters = self.billing.meters(number)
        if not meters:
            return DirectAnswer(
                text=f"По лицевому счёту {number} счётчиков в данных нет.",
            )

        parts: list[str] = []
        for meter in meters:
            label = f"счётчик {meter.kind} №{meter.serial}"
            if meter.verify_by is None:
                parts.append(f"{label}: срок следующей поверки в данных не указан")
            elif meter.verification_overdue(today):
                parts.append(
                    f"{label}: срок поверки истёк {meter.verify_by:%d.%m.%Y} — "
                    "до новой поверки начисление идёт по нормативу"
                )
            else:
                parts.append(
                    f"{label}: поверка действительна до {meter.verify_by:%d.%m.%Y}"
                )
        return DirectAnswer(text="\n".join(parts))

    def _readings(self, number: str) -> DirectAnswer:
        meters = self.billing.meters(number)
        if not meters:
            return DirectAnswer(
                text=f"По лицевому счёту {number} счётчиков в данных нет.",
            )

        parts: list[str] = []
        for meter in meters:
            if not meter.reading:
                continue
            line = f"счётчик {meter.kind} №{meter.serial}: последнее показание {meter.reading}"
            if meter.reading_date is not None:
                line += f", передано {meter.reading_date:%d.%m.%Y}"
            parts.append(line)

        if not parts:
            return DirectAnswer(
                text=f"По лицевому счёту {number} переданных показаний в данных нет.",
            )
        return DirectAnswer(text="\n".join(parts))
