"""Таблица тарифов и поиск по дате.

Тариф — факт, действующий в свой период, а не текст, который объясняют. Поэтому
он живёт здесь, а не в базе знаний: пересказ таблицы моделью — ровно тот случай,
где она ошибается убедительно (ADR-012, ADR-013, обобщение раздела 57.5).

ХРАНИТСЯ ТО, ЧТО ОПУБЛИКОВАНО, И ОТМЕТКА, КОГДА ПРОЧИТАНО. Второе не украшение:
тариф меняется решением регулятора, и ответ по файлу полугодовой давности звучит
так же уверенно, как в день загрузки.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

__all__ = ["Tariff", "TariffStore"]


@dataclass(frozen=True, slots=True)
class Tariff:
    """Тариф одной услуги на один период.

    Деньги — :class:`~decimal.Decimal`, как и в Биллинге (журнал, раздел 63.2):
    `float` даёт ошибку округления в копейках, а её абонент замечает первой.
    """

    service: str
    starts_on: date
    ends_on: date
    for_population: Decimal
    """Цена за куб для населения, с НДС. Именно она стоит в квитанции.

    Цена без НДС в файле тоже есть и намеренно **не** попадает в ответ: абонент
    платит по строке квитанции, и назвать ему вторую цифру значит дать повод
    сверять не то."""


class TariffStore:
    """Проиндексированная по периодам таблица.

    :param tariffs: строки таблицы.
    :param fetched_on: когда страница прочитана.
    :param source_url: откуда.
    """

    def __init__(
        self,
        tariffs: list[Tariff],
        *,
        fetched_on: date,
        source_url: str = "",
    ) -> None:
        self._tariffs = tuple(tariffs)
        self.fetched_on = fetched_on
        self.source_url = source_url

    def __len__(self) -> int:
        return len(self._tariffs)

    @property
    def services(self) -> tuple[str, ...]:
        """Услуги в порядке появления — он же порядок ответа."""
        seen: list[str] = []
        for tariff in self._tariffs:
            if tariff.service not in seen:
                seen.append(tariff.service)
        return tuple(seen)

    @classmethod
    def from_file(cls, path: Path) -> TariffStore:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tariffs = [
            Tariff(
                service=service,
                starts_on=date.fromisoformat(row["from"]),
                ends_on=date.fromisoformat(row["to"]),
                for_population=Decimal(row["for_population"]),
            )
            for service, rows in payload["services"].items()
            for row in rows
        ]
        return cls(
            tariffs,
            fetched_on=date.fromisoformat(payload["fetched_on"]),
            source_url=payload.get("source_url", ""),
        )

    def on(self, day: date) -> list[Tariff]:
        """Тарифы, действующие в этот день, по одному на услугу.

        **Пустой список — законный ответ.** Загрузчик отбрасывает строки с
        противоречивыми периодами (`scripts/fetch_tariffs.py`), и на месте
        отброшенной остаётся пробел. Промолчать там честно; подставить соседний
        период — назвать цену, которой в этот день не было.
        """
        return [t for t in self._tariffs if t.starts_on <= day <= t.ends_on]

    def next_change(self, after: date) -> Tariff | None:
        """Ближайший период, начинающийся позже указанного дня.

        Нужен, чтобы ответ называл не только сегодняшнюю цену, но и дату её
        изменения: «почему выросла сумма» — самая частая тема обращений, и
        предупредить дешевле, чем потом объяснять.
        """
        future = [t for t in self._tariffs if t.starts_on > after]
        return min(future, key=lambda t: t.starts_on) if future else None
