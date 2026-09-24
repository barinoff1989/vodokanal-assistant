"""Публичные контакты организации не вырезаются из ответов.

Это не ложные срабатывания обезличивателя: номер распознан **верно**, он
действительно телефон. Но телефон организации, опубликованный ею самой,
персональными данными не является, а фильтр вырезал его наравне с личным — и
ответ «позвоните по ⟨ТЕЛЕФОН⟩» бесполезен ровно там, где он нужен.

Поэтому отсев по форме (`looks_like_pii`) здесь неприменим по построению: он
снимает находки, непохожие на данные, а эти на данные похожи и ими являются.
Разделять приходится списком, и главная проверка тут — **что список не отстал от
корпуса**.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import get_settings
from app.gateway.pii_filter import (
    PUBLIC_EMAILS,
    PUBLIC_PHONES,
    PiiSanitizer,
    is_public_information,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "kb" / "faq_voronezh.json"

_PHONE = re.compile(r"(?:\+7|\b8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")


@pytest.fixture
def sanitizer() -> PiiSanitizer:
    """Обезличиватель без разбора языка.

    Телефоны и почту находят собственные распознаватели, а поднимать языковую
    модель ради этих проверок — четыре секунды на пустом месте.
    """
    return PiiSanitizer(analyzer=None)


@pytest.mark.parametrize(
    "written",
    [
        "8 (473) 206-77-06",
        "8-903-411-86-76",
        "8-800-200-36-46",
        "88612170333",
        "+7 (473) 206-77-06",
    ],
)
def test_публичный_номер_остаётся_в_тексте(sanitizer: PiiSanitizer, written: str):
    """Один и тот же номер записан по-разному, и сравнение строк пропустило бы половину.

    В корпусе он стоит как «8 (473) 206-77-06», в обращениях — как
    «8-903-411-86-76»; междугородний код пишут и через 8, и через +7."""
    text = f"Позвоните по телефону {written} в рабочее время."
    cleaned, report = sanitizer.sanitize(text)
    assert written in cleaned
    assert "<ТЕЛЕФОН>" not in cleaned
    assert report.pii_detected is False


@pytest.mark.slow
def test_публичный_адрес_почты_остаётся():
    """Настоящий случай из сквозного прогона: ответ про справку отправлял
    абонента «на адрес ⟨ПОЧТА⟩».

    Почту находит только разбор языка (`EMAIL_ADDRESS` из Presidio), поэтому
    здесь нужен настоящий анализатор: с `analyzer=None` проверка прошла бы, ничего
    не проверив."""
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    text = "Направьте обращение на адрес spravka@rosvodokanal.ru с документами."
    cleaned, _ = sanitizer.sanitize(text)
    assert "spravka@rosvodokanal.ru" in cleaned


def test_личный_номер_по_прежнему_скрывается(sanitizer: PiiSanitizer):
    """Список не должен превратиться в дыру: всё, чего в нём нет, скрывается."""
    text = "Мой телефон 8-952-118-44-21, перезвоните."
    cleaned, report = sanitizer.sanitize(text)
    assert "8-952-118-44-21" not in cleaned
    assert "<ТЕЛЕФОН>" in cleaned
    assert report.pii_detected is True


@pytest.mark.slow
def test_подставная_почта_абонента_скрывается():
    """`test@test.ru` из выгрузки стоит на месте настоящего адреса абонента.

    Внести её в исключения означало бы перестать скрывать сам этот случай."""
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    cleaned, _ = sanitizer.sanitize("Прошу присылать квитанции на test@test.ru")
    assert "test@test.ru" not in cleaned


def test_у_счёта_снилс_и_инн_публичного_варианта_не_бывает():
    """Если такой номер встретится в ответе — его и надо скрыть."""
    for entity in ("ACCOUNT_NUMBER", "SNILS", "INN"):
        assert is_public_information(entity, "84732067706") is False


@pytest.mark.slow
def test_адрес_офиса_остаётся_в_ответе():
    """Третий случай того же дефекта, найденный сквозным прогоном.

    «Пеше-Стрелецкая» стоит в половине ответов корпуса, и абонент получал
    «приходите в офис по адресу ⟨АДРЕС⟩, д. 90» — бесполезно ровно там, где
    адрес и нужен."""
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    text = "Обратитесь в офис по адресу ул. Пеше-Стрелецкая, д. 90, каб. 113."
    cleaned, _ = sanitizer.sanitize(text)
    assert "Пеше-Стрелецкая" in cleaned


def test_чужой_адрес_по_прежнему_скрывается():
    """Список публичных мест не должен стать дырой для адреса абонента."""
    assert is_public_information("LOCATION", "Ленинский проспект") is False
    assert is_public_information("PERSON", "Пантелеева") is False


def test_каждый_контакт_корпуса_внесён_в_исключения():
    """**Главная проверка этого файла.**

    Корпус пополняется скриптом с публичного сайта, и новый телефон в нём иначе
    снова начал бы вырезаться из ответов молча — ровно так этот дефект и возник.
    Проверка связывает список с корпусом в ту сторону, в какую он расходится.
    """
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    text = " ".join(f"{c['question']} {c['answer']}" for c in corpus)

    missing_phones = [
        found
        for found in _PHONE.findall(text)
        if not is_public_information("PHONE_NUMBER", found)
    ]
    missing_emails = [
        found
        for found in _EMAIL.findall(text)
        if not is_public_information("EMAIL_ADDRESS", found)
    ]
    assert not missing_phones, f"контакты корпуса не в исключениях: {missing_phones}"
    assert not missing_emails, f"адреса корпуса не в исключениях: {missing_emails}"


@pytest.mark.slow
def test_в_корпусе_обезличиватель_не_находит_ничего():
    """**Сильнейшая проверка этого файла, и она структурная.**

    Корпус собран с публичного сайта — персональных данных в нём нет **по
    построению**. Значит любая находка обезличивателя здесь ложная, и правильное
    их число — ноль, а не «мало».

    До правки находок было тринадцать: адрес офиса девять раз плюс «СберОнлайн»,
    «Госуслугах» и путь меню банковского приложения. Половина ответов корпуса
    теряла адрес, по которому абонента туда и зовут.
    """
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    leftovers = []
    for chunk in corpus:
        chunk_text = "\n".join((chunk["question"], chunk["answer"]))
        for span in sanitizer.find(chunk_text):
            leftovers.append(
                f"{chunk['chunk_id']} {span.entity_type}: "
                f"{chunk_text[span.start : span.end]!r}"
            )
    assert not leftovers, "обезличиватель режет публичный корпус: " + "; ".join(leftovers)


def test_номер_контакт_центра_из_настроек_внесён():
    """Тот же номер живёт в `app/config.py` и подставляется в отказ консультанта.

    Два экземпляра одного значения расходятся при первой правке — этот класс
    ошибки проект проходил трижды. Здесь они связаны проверкой."""
    assert is_public_information("PHONE_NUMBER", get_settings().contact_center_phone)


def test_списки_нормализованы():
    """Хранить «8 (473) 206-77-06» в списке цифр — значит не найти его никогда."""
    assert all(phone.isdigit() for phone in PUBLIC_PHONES)
    assert all(email == email.lower() for email in PUBLIC_EMAILS)
