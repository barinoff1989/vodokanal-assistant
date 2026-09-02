"""Хранилище графика отключений и поиск по адресу.

Отвечает на единственный вопрос: **отключена ли вода по этому адресу и когда**.
Это поиск по ключу, а не по смыслу, поэтому база знаний здесь не при чём
(ADR-013): адрес не несёт смысла для векторного сходства, а пересказ списка из
шестидесяти домов модель делает убедительно и неверно.

ПОЧЕМУ В ПАМЯТИ, А НЕ В ТАБЛИЦЕ. На прототипе источник — файл, и класть его в
Postgres значило бы завести таблицу, в которую никто не пишет, ради одного
чтения. Проект уже разбирал этот случай с другой стороны: метрика, объявленная в
коде и не записываемая ничем (раздел 54.1). Таблица появится тогда, когда
появится то, что в неё пишет, — то есть на MVP вместе с онлайн-доступом.

**Форма записи при этом уже как в будущей БД** — адрес, признак отключения,
интервал (см. :class:`~app.outages.parser.Outage`). Меняется место хранения, а
не устройство: запрос и путь ответа переживут переход без правок.

ОТМЕТКА АКТУАЛЬНОСТИ ОБЯЗАТЕЛЬНА. Единственное, что система знает о свежести
файла, — когда он прочитан. Без :attr:`OutageStore.loaded_at` и проверки возраста
ассистент однажды скажет «воды не будет до 13 августа», прочитав месячный файл,
и скажет это так же уверенно, как в день получения.

**Устаревание и недоступность — разные случаи, и различать их обязательно.** В
первом система знает, что данные старые; во втором — что не знает ничего.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from app.outages.parser import Outage, normalize_house, normalize_street, parse_schedule

__all__ = ["DEFAULT_MAX_AGE", "OutageStore"]

DEFAULT_MAX_AGE = timedelta(days=7)
"""После какого возраста данные считаются устаревшими.

Неделя — не измеренная величина, а осознанное допущение: периоды в графике
длятся от трёх до семнадцати дней, и файл старше недели заведомо не описывает
происходящее сегодня. Числом это станет, когда владелец назовёт периодичность
выпуска (пункт 45 сведённого TODO)."""


@dataclass(frozen=True, slots=True)
class OutageStore:
    """Прочитанный график и время его чтения."""

    outages: tuple[Outage, ...]
    loaded_at: datetime

    @classmethod
    def from_lines(
        cls, lines: Iterable[str], *, year: int, loaded_at: datetime
    ) -> OutageStore:
        return cls(tuple(parse_schedule(lines, year=year)), loaded_at)

    @classmethod
    def from_file(cls, path: Path, *, year: int, loaded_at: datetime) -> OutageStore:
        return cls.from_lines(
            path.read_text(encoding="utf-8").splitlines(), year=year, loaded_at=loaded_at
        )

    def age(self, now: datetime) -> timedelta:
        return now - self.loaded_at

    def is_stale(self, now: datetime, max_age: timedelta = DEFAULT_MAX_AGE) -> bool:
        """Старше ли график допустимого возраста.

        Вызывающий код обязан спросить это до того, как отдать ответ: устаревший
        график отвечает так же уверенно, как свежий.
        """
        return self.age(now) > max_age

    def find(
        self,
        street: str,
        house: str,
        *,
        district: str | None = None,
        on: date | None = None,
    ) -> list[Outage]:
        """Отключения по адресу, от ближайшего.

        Адрес нормализуется так же, как при разборе, — иначе «ул. Димитрова» из
        вопроса абонента не совпала бы с «Димитрова» из файла.

        Возвращаются **все** совпавшие интервалы, а не один. Сорок адресов
        числятся в двух пересекающихся периодах, и выбрать между ними не на чем
        (ADR-013): выбор без основания хуже, потому что не виден.

        :param district: сузить до района. Нужен там, где название улицы
            повторяется: четыре адреса файла числятся в двух районах сразу.
        :param on: оставить только интервалы, накрывающие эту дату. ``None`` —
            вернуть все, включая прошедшие и будущие.
        """
        wanted_street = normalize_street(street)
        wanted_house = normalize_house(house)
        found = [
            outage
            for outage in self.outages
            if outage.street == wanted_street
            and outage.house == wanted_house
            and (district is None or outage.district == district)
            and (on is None or outage.starts_on <= on <= outage.ends_on)
        ]
        return sorted(found, key=lambda o: (o.starts_on, o.ends_on))
