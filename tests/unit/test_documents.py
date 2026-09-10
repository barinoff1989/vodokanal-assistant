"""Проверки образцов заявлений и их выдачи абоненту.

Реестр `app/documents/templates.py` — образцы, собранные по публичным формам и
нормативной базе (Постановления № 644, 776). Проверки сторожат: подстановку
известных полей, отделение «дай бланк» от «оформи заявку», и что путь идёт мимо
модели.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.backend.sessions import SessionStore
from app.billing.source import CsvBillingSource
from app.config import Settings
from app.documents.responder import TemplateResponder
from app.documents.templates import (
    BLANK,
    FROM_INQUIRY_TYPE,
    TEMPLATES,
    match_template,
    render_template,
)
from app.models import GenerateRequest, GenerationParameters, RequestMetadata
from app.taxonomy import InquiryType, Topic


class _MemRedis:
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._d.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._d[key] = value

    def delete(self, key: str) -> None:
        self._d.pop(key, None)

DATA = Path(__file__).resolve().parents[2] / "data_example"
NOW = datetime(2026, 9, 8, 10, 0)
SUB = "2100202213"


@pytest.fixture(scope="module")
def billing() -> CsvBillingSource:
    return CsvBillingSource(DATA)


@pytest.fixture(scope="module")
def responder(billing: CsvBillingSource) -> TemplateResponder:
    return TemplateResponder(billing=billing, settings=Settings(_env_file=None))


def ask(
    query: str,
    *,
    topic: Topic | None = Topic.TEMPLATE,
    subscriber: str = SUB,
) -> GenerateRequest:
    return GenerateRequest(
        system="помощник",
        query=query,
        parameters=GenerationParameters(stream=False),
        metadata=RequestMetadata(
            subscriber_id=subscriber, session_id="s-1", topic=topic
        ),
    )


# --- реестр образцов ---------------------------------------------------- #


def test_семь_образцов_и_у_каждого_основание():
    assert len(TEMPLATES) == 7
    for template in TEMPLATES.values():
        assert template.rule
        assert template.title


@pytest.mark.parametrize("inquiry_type", list(FROM_INQUIRY_TYPE))
def test_каждый_тип_обращения_ведёт_на_существующий_образец(inquiry_type: InquiryType):
    assert FROM_INQUIRY_TYPE[inquiry_type] in TEMPLATES


def test_подстановка_известных_полей_и_прочерк_для_остального():
    text = render_template(
        "meter_verification",
        prefill={
            "full_name": "Петров Николай Егорович",
            "account_number": SUB,
            "address": "г. Воронеж, ул. Зои Космодемьянской, д. 50",
            "date": "08.09.2026",
        },
        company_phone="8 (473) 206-77-06",
    )
    assert "Петров Николай Егорович" in text
    assert SUB in text
    assert "08.09.2026" in text
    assert "8 (473) 206-77-06" in text
    # Поля, которых система не знает, остаются прочерком.
    assert BLANK in text
    assert "не официальная форма" in text


def test_образец_без_данных_весь_в_прочерках():
    text = render_template("connection")
    assert text.count(BLANK) > 5
    assert "договора холодного водоснабжения" in text


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("бланк на поверку", "meter_verification"),
        ("образец на опломбировку", "meter_sealing"),
        ("заявление на снятие пломбы", "meter_unsealing"),
        ("шаблон на установку счётчика", "meter_installation"),
        ("образец справки об отсутствии задолженности", "certificate"),
        ("бланк на обследование", "inspection"),
        ("заявление о подключении к водоснабжению", "connection"),
    ],
)
def test_образец_узнаётся_по_вопросу(query: str, expected: str):
    assert match_template(query.lower()) == expected


# --- выдача абоненту -------------------------------------------------- #


def test_чужая_тема_молчание(responder: TemplateResponder):
    assert responder.answer(ask("бланк на поверку", topic=Topic.GENERAL), now=NOW) is None


def test_бланк_приходит_с_данными_абонента(responder: TemplateResponder):
    answer = responder.answer(ask("дай бланк заявления на поверку"), now=NOW)
    assert answer is not None
    assert "ЗАЯВЛЕНИЕ" in answer.text
    assert "поверк" in answer.text.lower()
    # ФИО из примера Биллинга подставлено.
    assert "Петров" in answer.text


def test_без_уточнения_показывает_список(responder: TemplateResponder):
    answer = responder.answer(ask("дайте какой-нибудь бланк заявления"), now=NOW)
    assert answer is not None
    assert "какой бланк нужен" in answer.text
    assert "поверк" in answer.text.lower()


def test_тема_та_но_вопрос_не_про_бланк_подсказывает(responder: TemplateResponder):
    """`match_template` нашёл тип, но слова «бланк/образец» не было — ответчик
    всё равно отдаёт образец, добавив пояснение."""
    answer = responder.answer(ask("заявление на поверку счётчика как заполнить"), now=NOW)
    assert answer is not None
    assert "нужен бланк" in answer.text.lower() or "образец" in answer.text.lower()


# --- заполнение из Биллинга ------------------------------------------- #


def _one_meter(billing: CsvBillingSource) -> str:
    return next(n for n, m in billing._meters.items() if len(m) == 1)  # noqa: SLF001


def _two_meters(billing: CsvBillingSource) -> str:
    return next(n for n, m in billing._meters.items() if len(m) >= 2)  # noqa: SLF001


def test_один_счётчик_подставляется_в_бланк(
    responder: TemplateResponder, billing: CsvBillingSource
):
    sub = _one_meter(billing)
    meter = billing.meters(sub)[0]
    answer = responder.answer(ask("бланк на поверку", subscriber=sub), now=NOW)
    assert answer is not None
    assert f"назначение (ХВС / ГВС): {meter.kind}" in answer.text
    assert f"заводской номер: {meter.serial}" in answer.text
    assert "__________" in answer.text  # место установки и дата — всё равно прочерк


def test_два_счётчика_поля_прибора_остаются_прочерком(
    responder: TemplateResponder, billing: CsvBillingSource
):
    sub = _two_meters(billing)
    answer = responder.answer(ask("бланк на поверку", subscriber=sub), now=NOW)
    assert answer is not None
    assert "назначение (ХВС / ГВС): __________" in answer.text
    # но оба счётчика перечислены в справочном блоке
    for meter in billing.meters(sub):
        assert meter.serial in answer.text


def test_справочный_блок_под_бланком(
    responder: TemplateResponder, billing: CsvBillingSource
):
    sub = _two_meters(billing)
    answer = responder.answer(ask("бланк на снятие пломбы", subscriber=sub), now=NOW)
    assert answer is not None
    assert "Данные из вашего лицевого счёта" in answer.text
    account = billing.account(sub)
    assert account is not None
    if account.has_debt:
        assert "Задолженность" in answer.text
    else:
        assert "Задолженности нет" in answer.text


# --- реестр: плейсхолдеры ------------------------------------------- #


def test_каждый_плейсхолдер_шаблона_имеет_источник():
    """Тело шаблона не должно ссылаться на поле, которого нет ни в базовых, ни
    в собираемых. `__post_init__` уже проверяет это при импорте — тест
    фиксирует требование явно."""
    for template in TEMPLATES.values():
        import re

        refs = set(re.findall(r"\{(\w+)\}", template.body))
        assert refs <= template.field_names(), template.template_id


# --- сбор полей в диалоге ------------------------------------------- #


@pytest.fixture
def dialog(billing: CsvBillingSource) -> TemplateResponder:
    """Ответчик с хранилищем сессий — для многорепликового сбора."""
    return TemplateResponder(
        billing=billing,
        settings=Settings(_env_file=None),
        sessions=SessionStore(_MemRedis(), ttl_seconds=1800),
    )


def _say(
    responder: TemplateResponder, text: str, topic: Topic | None, sub: str = SUB
) -> str:
    request = GenerateRequest(
        system="помощник",
        query=text,
        parameters=GenerationParameters(stream=False),
        metadata=RequestMetadata(subscriber_id=sub, session_id="dlg", topic=topic),
    )
    answer = responder.answer(request, now=NOW)
    return answer.text if answer is not None else ""


def test_сбор_полей_по_репликам_даёт_заполненный_бланк(
    dialog: TemplateResponder, billing: CsvBillingSource
):
    sub = _one_meter(billing)
    assert "Задам 3 вопрос" in _say(dialog, "дай бланк на поверку", Topic.TEMPLATE, sub)
    assert "удобно" in _say(dialog, "под ванной", Topic.GENERAL, sub).lower()
    assert "снятия" in _say(dialog, "13 сентября утром", Topic.GENERAL, sub).lower()
    final = _say(dialog, "на месте", Topic.GENERAL, sub)
    assert "ЗАЯВЛЕНИЕ" in final
    assert "место установки: под ванной" in final
    assert "Удобная дата и время посещения: 13 сентября утром" in final
    assert "Способ поверки: на месте" in final
    # счётчик подставлен из Биллинга
    assert billing.meters(sub)[0].serial in final


def test_пропустить_оставляет_прочерк(dialog: TemplateResponder, billing: CsvBillingSource):
    sub = _one_meter(billing)
    _say(dialog, "бланк на поверку", Topic.TEMPLATE, sub)
    _say(dialog, "пропустить", Topic.GENERAL, sub)
    _say(dialog, "пропустить", Topic.GENERAL, sub)
    final = _say(dialog, "не знаю", Topic.GENERAL, sub)
    assert "ЗАЯВЛЕНИЕ" in final
    assert f"место установки: {BLANK}" in final


def test_отмена_прерывает_сбор(dialog: TemplateResponder):
    _say(dialog, "бланк на снятие пломбы", Topic.TEMPLATE)
    assert "не заполняем" in _say(dialog, "отмена", Topic.GENERAL)
    # после отмены свободная реплика уже не перехватывается
    assert _say(dialog, "просто текст", Topic.GENERAL) == ""


def test_другая_тема_во_время_сбора_не_перехватывается(dialog: TemplateResponder):
    _say(dialog, "бланк на поверку", Topic.TEMPLATE)
    # вопрос про счёт — не ответ на поле бланка
    assert _say(dialog, "какая у меня задолженность", Topic.ACCOUNT) == ""
    # сбор при этом продолжается — следующий свободный ответ принимается
    assert "удобно" in _say(dialog, "под ванной", Topic.GENERAL).lower()


def test_без_хранилища_сессий_бланк_отдаётся_сразу(responder: TemplateResponder):
    """У `responder` (модульная фикстура) sessions=None — сбор не ведётся."""
    answer = responder.answer(ask("дай бланк на поверку"), now=NOW)
    assert answer is not None
    assert "ЗАЯВЛЕНИЕ" in answer.text
    assert "Задам" not in answer.text


# --- готовый бланк как файл ---------------------------------------- #


def test_с_хранилищем_бланк_приходит_ещё_и_ссылкой(
    billing: CsvBillingSource, tmp_path: Path
):
    from app.documents.artifact import ArtifactStore

    store = ArtifactStore(root=tmp_path, ttl_seconds=3600)
    responder = TemplateResponder(
        billing=billing, settings=Settings(_env_file=None), artifacts=store
    )
    answer = responder.answer(ask("дай бланк на поверку"), now=NOW)
    assert answer is not None
    assert answer.document_url is not None
    assert "ЗАЯВЛЕНИЕ" in answer.text  # текст в ленте остаётся
    assert "ссылке" in answer.text.lower()

    token = answer.document_url.rsplit("/", 1)[-1]
    page = store.get(token)
    assert page is not None
    account = billing.account(SUB)
    assert account is not None
    assert account.full_name in page  # данные абонента попали в лист для печати
    assert "window.print()" in page


def test_отказ_хранилища_не_ломает_выдачу_бланка(
    billing: CsvBillingSource, tmp_path: Path
):
    from app.documents.artifact import ArtifactStore

    class _BrokenStore(ArtifactStore):
        def put(self, html_text: str) -> str | None:
            return None

    responder = TemplateResponder(
        billing=billing,
        settings=Settings(_env_file=None),
        artifacts=_BrokenStore(root=tmp_path, ttl_seconds=3600),
    )
    answer = responder.answer(ask("дай бланк на поверку"), now=NOW)
    assert answer is not None
    assert answer.document_url is None
    assert "ЗАЯВЛЕНИЕ" in answer.text  # текст пришёл, путь не сломался
    assert "ссылке" not in answer.text.lower()
