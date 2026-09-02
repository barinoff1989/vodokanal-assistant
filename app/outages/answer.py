"""Ответ об отключении воды — точным поиском, без обращения к модели.

Тема `outage` (ADR-011) уводит вопрос сюда. Ответ собирается из графика по
адресу абонента, а не генерируется: график протухает за дни, адрес не несёт
смысла для векторного поиска, а пересказ списка домов модель делает убедительно
и неверно (ADR-013).

АДРЕС БЕРЁТСЯ ИЗ ПРОФИЛЯ ЛИЧНОГО КАБИНЕТА, а не из текста вопроса. ЛК его знает;
извлекать из формулировки абонента — лишний шаг, на котором теряются как раз
самые нужные случаи («у нас третий день нет воды» адреса не содержит вовсе).

ПРЕДМЕТНАЯ ЛОГИКА ЖИВЁТ ЗДЕСЬ, А НЕ В ШЛЮЗЕ. Шлюз знает только, что у него может
быть прямой ответчик, и ничего — про водоканал, адреса и отключения. Иначе
утверждение из его же описания («здесь нет ничего про водоканал») перестало бы
быть правдой при первой же теме.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.gateway.llm_gateway import DirectAnswer
from app.models import GenerateRequest
from app.outages.parser import normalize_house, normalize_street
from app.outages.store import DEFAULT_MAX_AGE, OutageStore
from app.taxonomy import Topic

__all__ = ["OutageResponder", "parse_address"]

# Населённый пункт: «г. Тестовый», «пос. Дубовое». Сокращения «д.» (деревня) и
# «с.» (село) сюда НЕ входят намеренно — в адресе «д.» почти всегда дом, и на
# «г. Тестовый, ул. Димитрова, д. 2» разбор терял именно номер дома.
# После сокращения требуется буква, а не цифра: это и отличает «д. 2» от «д. Ивановка».
_CITY = re.compile(r"^(?:г|гор|город|пос|посёлок|поселок)\.?\s+[^\W\d]", re.IGNORECASE)
_FLAT = re.compile(r"^(?:кв|квартира|оф|офис|пом|помещение)\.?\s*\d", re.IGNORECASE)
_HOUSE = re.compile(r"^(?:д|дом|влд|владение)\.?\s*(?P<number>\S.*)$", re.IGNORECASE)
_BARE_HOUSE = re.compile(r"^\d+\S*$")


def parse_address(address: str) -> tuple[str, str] | None:
    """Вытащить из адреса ЛК улицу и дом. ``None`` — если не разобрался.

    Личный кабинет ведёт адрес одной строкой («г. Тестовый, ул. Набережная,
    д. 65, кв. 156»), и разложить её приходится нам.

    Разбор намеренно грубый: город и квартира отбрасываются по слову-признаку,
    из оставшегося берутся улица и дом. Угадывать сверх этого не следует —
    неразобранный адрес честнее неверно разобранного, потому что первый виден,
    а второй ответит про чужую улицу.
    """
    parts = [p.strip() for p in address.split(",") if p.strip()]
    parts = [p for p in parts if not _CITY.match(p) and not _FLAT.match(p)]

    street_raw: str | None = None
    house_raw: str | None = None
    for part in parts:
        house_match = _HOUSE.match(part)
        if house_match is not None:
            house_raw = house_match.group("number")
        elif _BARE_HOUSE.match(part):
            house_raw = part
        elif street_raw is None:
            street_raw = part

    if street_raw is None or house_raw is None:
        return None

    street = normalize_street(street_raw)
    house = normalize_house(house_raw)
    if not street or not house:
        return None
    return street, house


def _plural_days(interval: tuple[date, date]) -> str:
    start, end = interval
    return f"с {start:%d.%m} по {end:%d.%m}"


@dataclass(frozen=True, slots=True)
class OutageResponder:
    """Отвечает на вопрос об отключении, если может.

    ``None`` означает «это не мой случай» — тогда запрос идёт обычным путём.
    Ответчик молчит в двух случаях: тема не про отключения либо адрес не
    получен. **Пустой результат поиска молчанием не является:** «плановых
    отключений нет» — это ответ, и отдать его должны мы, а не модель, которая
    его выдумает.

    :param store: прочитанный график.
    :param max_age: после какого возраста данных ответ идёт с оговоркой.
    """

    store: OutageStore
    max_age: timedelta = DEFAULT_MAX_AGE

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:
        if request.metadata.topic is not Topic.OUTAGE:
            return None
        if not request.metadata.address:
            return None

        parsed = parse_address(request.metadata.address)
        if parsed is None:
            return None

        street, house = parsed
        outages = self.store.find(street, house, on=None)
        upcoming = [o for o in outages if o.ends_on >= now.date()]

        if not upcoming:
            text = (
                f"По адресу {request.metadata.address} плановых отключений воды "
                "в графике нет."
            )
        else:
            periods = "; ".join(_plural_days((o.starts_on, o.ends_on)) for o in upcoming)
            text = (
                f"По адресу {request.metadata.address} плановое отключение воды: "
                f"{periods}."
            )
            if len(upcoming) > 1:
                # Сорок адресов файла числятся в двух пересекающихся периодах.
                # Выбрать между ними не на чем, и молчаливый выбор хуже: он не
                # виден абоненту (ADR-013).
                text += " В графике указано несколько периодов — приводим все."

        return DirectAnswer(text=text, disclaimer=self._disclaimer(now))

    def _disclaimer(self, now: datetime) -> str:
        """Когда прочитан график — и оговорка, если он стар.

        Без отметки ассистент отвечает по месячному файлу так же уверенно, как в
        день его получения.
        """
        loaded = f"Данные графика получены {self.store.loaded_at:%d.%m.%Y}."
        if self.store.is_stale(now, self.max_age):
            return loaded + " Они могли устареть — уточните в диспетчерской службе."
        return loaded
