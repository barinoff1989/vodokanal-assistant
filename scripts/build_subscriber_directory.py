"""Справочник абонентов для демо-стенда — поиск по фамилии и лицевому счёту.

ЗАЧЕМ ОН ЕСТЬ И ЧЕГО В НЁМ НЕТ

На рабочей системе справочника абонентов у виджета нет и быть не должно: абонент
уже вошёл в личный кабинет, и `Session Context Provider` (C3 виджета) берёт его
`subscriber_id` и `session_id` из SSO-сессии — искать некого. Справочник нужен
только демо-стенду: показывающий должен уметь войти любым из двухсот примерных
абонентов, чтобы показать разные сценарии (долг и переплата, отключение и его
отсутствие, просроченная поверка) и вручную воспроизвести проверки S1–S2 плана
тестирования — войти одним абонентом и попробовать подтвердить черновик другого.

Поэтому файл кладётся в `web/` рядом со стендом, а не в `app/`, и в контракт API
(правило 4.4) ничего не добавляет: стенд читает его как статический файл.

ОТКУДА БЕРУТСЯ ДАННЫЕ

Из присланного примера Биллинга (`data_example/`), тем же кодом, что читает его
сервис, — `app.billing.source.CsvBillingSource`. Второй разборщик тех же CSV
разошёлся бы с первым при первой же правке формата.

Это **пример владельца, а не настоящие абоненты**: ФИО, адреса и счётчики
синтетические. Пометка стоит в самом файле (`_meta`) и на стенде.

ПРИЗНАК ОТКЛЮЧЕНИЯ СЧИТАЕТСЯ, А НЕ ВПИСЫВАЕТСЯ РУКАМИ

Прежняя редакция стенда держала двух абонентов с припиской «у первой отключение
в графике есть». К моменту, когда демо-график пересобрали
(`scripts/make_demo_schedule.py`), приписка устарела молча: ни одной из двух
улиц в новом графике уже не было. Здесь признак `outage` вычисляется тем же
поиском по адресу, что и ответ абоненту (`app.outages`), по тому же файлу
графика, что грузит сервис, — устареть он не может.

ЗАПУСК

    python scripts/build_subscriber_directory.py
    python scripts/build_subscriber_directory.py --check   # не писать файл
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_example"
SCHEDULE = DATA / "график_отключений_демо.txt"
OUTPUT = ROOT / "web" / "subscribers.json"


@dataclass(frozen=True, slots=True)
class Entry:
    """Одна карточка справочника.

    `account` служит и идентификатором абонента: стенд отправляет его как
    `subscriber_id`, а регистрация обращения ищет по нему профиль в Биллинге
    (`app/backend/registration.py`). Отдельного `subscriber_id` тут нет
    намеренно — второе значение для одного и того же разошлось бы.
    """

    account: str
    name: str
    address: str
    balance: str
    debt: str | None
    meters: str
    phone: str | None
    email: str | None
    outage: str | None


def _money(value: Decimal) -> str:
    """Сумма в рублях: разряды пробелом, дробь запятой, минус — типографский.

    Формат тот же, что показывал стенд руками («−331,78 ₽»), — карточка не
    должна выглядеть иначе оттого, что её собрал скрипт.
    """
    text = f"{abs(value):,.2f}".replace(",", " ").replace(".", ",")
    sign = "−" if value < 0 else ""
    return f"{sign}{text} ₽"


def _meters(source: object, number: str, today: date) -> str:
    from app.billing.source import CsvBillingSource

    assert isinstance(source, CsvBillingSource)
    parts: list[str] = []
    for meter in source.meters(number):
        if meter.verify_by is None:
            note = "срок поверки неизвестен"
        elif meter.verification_overdue(today):
            note = f"поверка просрочена ({meter.verify_by:%d.%m.%Y})"
        else:
            note = f"поверка до {meter.verify_by:%d.%m.%Y}"
        parts.append(f"{meter.kind} № {meter.serial}, {note}")
    return "; ".join(parts) if parts else "нет данных о счётчике"


def _outage_hint(store: object, address: str, today: date) -> str | None:
    """Есть ли по адресу абонента предстоящее отключение в демо-графике.

    Тот же путь, что у ответчика отключений: разбор адреса ЛК на улицу и дом,
    поиск по графику, отсев прошедших интервалов (`app/outages/answer.py`).
    """
    from app.outages.answer import parse_address
    from app.outages.store import OutageStore

    assert isinstance(store, OutageStore)
    parsed = parse_address(address)
    if parsed is None:
        return None
    street, house = parsed
    upcoming = [o for o in store.find(street, house, on=None) if o.ends_on >= today]
    if not upcoming:
        return None
    return "; ".join(f"с {o.starts_on:%d.%m} по {o.ends_on:%d.%m}" for o in upcoming)


def build(*, today: date | None = None) -> list[Entry]:
    """Собрать справочник из примера Биллинга, падая, если примера нет."""
    from app.billing.source import CsvBillingSource

    day = today or date.today()

    source = CsvBillingSource(DATA)
    if not len(source):
        sys.exit(
            f"нет примера Биллинга: {DATA.relative_to(ROOT)}\n"
            "Справочник собирается из него; без файлов ЛСФЛ.csv и др. собирать нечего."
        )

    store: object | None = None
    if SCHEDULE.exists():
        from app.outages.store import OutageStore

        store = OutageStore.from_file(SCHEDULE, year=day.year, loaded_at=datetime.now())
    else:
        print(f"график отключений не найден ({SCHEDULE.relative_to(ROOT)}): признак outage пуст")

    entries: list[Entry] = []
    for account in source.accounts():
        profile = source.profile(f"sub-{account.number}")
        entries.append(
            Entry(
                account=account.number,
                name=account.full_name,
                address=account.address,
                balance=_money(account.balance),
                debt=_money(account.debt) if account.has_debt else None,
                meters=_meters(source, account.number, day),
                phone=profile.phone if profile and profile.phone else None,
                email=profile.email if profile and profile.email else None,
                outage=_outage_hint(store, account.address, day) if store is not None else None,
            )
        )

    # Порядок — по ФИО: выгрузка идёт «Фамилия Имя Отчество», значит это и есть
    # сортировка по фамилии, которую ждёт поиск по абоненту.
    entries.sort(key=lambda e: (e.name.casefold(), e.account))
    return entries


def report(entries: list[Entry]) -> None:
    with_debt = sum(1 for e in entries if e.debt)
    with_outage = sum(1 for e in entries if e.outage)
    with_phone = sum(1 for e in entries if e.phone)

    print(f"абонентов в справочнике: {len(entries)}")
    print(f"  с задолженностью: {with_debt}")
    print(f"  с предстоящим отключением: {with_outage}")
    print(f"  с профилем ЛК (телефон): {with_phone}")

    # Оба ответа — «отключение есть» и «отключений нет» — нужны на демонстрации.
    # Если график попадает во всех или ни в кого, показать второй нечем.
    if entries and not 0 < with_outage < len(entries):
        print("  ВНИМАНИЕ: отключение находится у всех или ни у кого — проверьте демо-график")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="собрать и проверить, файл не писать",
    )
    args = parser.parse_args()

    entries = build()
    report(entries)

    if args.check:
        return

    payload = {
        "_meta": {
            "источник": "data_example/ — пример Биллинга от владельца; "
            "абоненты синтетические, не настоящие",
            "собрано": "scripts/build_subscriber_directory.py",
            "правка": "руками не править: пересобирается скриптом",
            "абонентов": len(entries),
        },
        "subscribers": [
            {key: value for key, value in asdict(entry).items() if value is not None}
            for entry in entries
        ],
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"записано: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
