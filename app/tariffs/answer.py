"""Ответ о тарифе — точным поиском по дате, без обращения к модели.

Тема `tariff` (ADR-300) уводит вопрос сюда. Ответ собирается из опубликованной
таблицы, а не генерируется: соседние периоды различаются только датами и парой
цифр, и пересказ такой таблицы — ровно то, в чём модель ошибается убедительно.
Третий случай правила после регламентной формулировки и графика отключений.

ПРЕДМЕТНАЯ ЛОГИКА ЖИВЁТ ЗДЕСЬ, А НЕ В ОРКЕСТРАЦИИ. Backend знает только, что у
него может быть прямой ответчик, и ничего — про воду, тарифы и НДС. Тот же шов,
которым подключены ответчик отключений и реестр регламентных ответов.

ЧТО К ЭТОМУ ПУТИ ПРИМЕНЯЕТСЯ. Лимиты — да: ответ без модели ничего не стоит нам,
но обработка запроса стоит, и путь в обход лимитов стал бы способом бесплатно
давить сервис. Обезличивание — нет, наружу ничего не уходит. Охранители — нет,
проверять опубликованную регулятором цифру нашими правилами не на чем.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from app.backend.orchestrator import DirectAnswer
from app.models import GenerateRequest
from app.tariffs.store import Tariff, TariffStore
from app.taxonomy import Topic

__all__ = ["TariffResponder"]


def _money(value: object) -> str:
    """Число — в вид, привычный по квитанции: две цифры после запятой."""
    return f"{value:.2f}".replace(".", ",")


@dataclass(frozen=True, slots=True)
class TariffResponder:
    """Отвечает на вопрос о цене куба."""

    store: TariffStore

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:
        if request.metadata.topic is not Topic.TARIFF:
            return None

        today = now.date()
        current = self.store.on(today)
        if not current:
            # Пробел в таблице: загрузчик отбросил строку с противоречивым
            # периодом. Молчание значит «не мой случай», и вопрос уйдёт обычным
            # путём — это честнее, чем назвать цену соседнего периода.
            return None

        lines = [
            f"Тариф для населения на {today.strftime('%d.%m.%Y')} "
            f"(за 1 куб. м, с НДС):"
        ]
        lines += [
            f"- {tariff.service}: {_money(tariff.for_population)} руб."
            for tariff in current
        ]

        if changes := self._next_changes(current, today):
            starts_on = changes[0].starts_on.strftime("%d.%m.%Y")
            listed = "; ".join(
                f"{c.service} — {_money(c.for_population)} руб" for c in changes
            )
            lines.append(f"С {starts_on} тариф изменится: {listed}.")

        return DirectAnswer(text="\n".join(lines))

    def _next_changes(self, current: list[Tariff], today: date) -> list[Tariff]:
        """Все услуги, дорожающие в ближайшую дату изменения.

        **По одной услуге называть нельзя.** Первая редакция брала ближайшую
        строку и сообщала только про воду — а с той же даты дорожало и
        водоотведение, причём сильнее (46,73 против 40,94). Умолчать о большем
        подорожании в ответе про деньги хуже, чем не сказать ничего: абонент
        решит, что знает всю сумму.

        Нашлось не тестом, а сверкой ответа с файлом.

        Периоды идут подряд по полугодиям, и в части из них цена совпадает с
        нынешней. Такие не объявляются: сказать «тариф изменится» и назвать то
        же число — заставить сверять две одинаковые цифры.
        """
        first = self.store.next_change(today)
        if first is None:
            return []

        same = {t.service: t.for_population for t in current}
        return [
            tariff
            for tariff in self.store.on(first.starts_on)
            if same.get(tariff.service) != tariff.for_population
        ]
