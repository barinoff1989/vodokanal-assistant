"""Проверки обезличивания.

Быстрые тесты работают без разбора языка: собственные распознаватели — чистые
функции, и загрузка языковой модели ради них была бы тратой секунд на каждый
прогон. Тесты, которым нужен Presidio, вынесены в конец и помечены `slow`.

Отдельно проверяется то, ради чего модуль существует: значения персональных
данных не должны попадать ни в отчёт, ни в обезличенный текст.
"""

from __future__ import annotations

import pytest

from app.gateway.pii_filter import (
    PiiSanitizer,
    PiiSpan,
    find_account_numbers,
    find_inn,
    find_phones,
    find_snils,
    is_valid_inn,
    is_valid_snils,
    looks_like_pii,
    sanitize,
)


@pytest.fixture
def sanitizer() -> PiiSanitizer:
    """Обезличиватель без разбора языка — только собственные распознаватели."""
    return PiiSanitizer(analyzer=None)


# --- СНИЛС: контрольная сумма ------------------------------------------------- #

# Номер с заведомо верной контрольной суммой, посчитанной по алгоритму.
VALID_SNILS = "11223344595"


def test_снилс_с_верной_суммой_принимается():
    assert is_valid_snils(VALID_SNILS)


def test_снилс_с_испорченной_суммой_отвергается():
    """Ровно ради этого сумма и считается: иначе годился бы любой набор цифр."""
    broken = VALID_SNILS[:-2] + "00"
    assert not is_valid_snils(broken)


@pytest.mark.parametrize("digits", ["1234567890", "123456789012", "abc", ""])
def test_не_снилс_отвергается(digits: str):
    assert not is_valid_snils(digits)


@pytest.mark.parametrize(
    "text",
    [
        f"СНИЛС {VALID_SNILS}",
        f"СНИЛС {VALID_SNILS[:3]}-{VALID_SNILS[3:6]}-{VALID_SNILS[6:9]} {VALID_SNILS[9:]}",
    ],
)
def test_снилс_находится_в_любой_записи(text: str):
    """Разделители у номера бывают разные, а тип данных один и тот же."""
    assert [s.entity_type for s in find_snils(text)] == ["SNILS"]


def test_случайные_одиннадцать_цифр_не_считаются_снилсом():
    """Номер заявки или дата со временем не должны обезличиваться как СНИЛС."""
    assert find_snils("Заявка 20240115123 принята") == []


# --- ИНН: контрольная сумма ---------------------------------------------------- #

VALID_INN_10 = "7707083893"  # десятизначный, организации
VALID_INN_12 = "500100732259"  # двенадцатизначный, человека


def test_инн_обеих_длин_принимается():
    assert is_valid_inn(VALID_INN_10)
    assert is_valid_inn(VALID_INN_12)


def test_инн_с_испорченной_суммой_отвергается():
    assert not is_valid_inn(VALID_INN_10[:-1] + "0")
    assert not is_valid_inn(VALID_INN_12[:-1] + "0")


def test_инн_находится_в_тексте():
    assert [s.entity_type for s in find_inn(f"ИНН {VALID_INN_12}")] == ["INN"]


# --- Лицевой счёт: только рядом с подсказкой ----------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Мой лицевой счёт 1234567890",
        "лицевого счёта 1234567890",
        "л/с 1234567890",
        "ЛС 1234567890",
        "Номер счёта 1234567890",
    ],
)
def test_счёт_находится_рядом_с_подсказкой(text: str):
    assert [s.entity_type for s in find_account_numbers(text)] == ["ACCOUNT_NUMBER"]


def test_длинное_число_без_подсказки_не_трогается():
    """Раздел 11.5: шаблон из девяти-десяти цифр слишком широк сам по себе.

    Без этой проверки обезличивание портило бы текст, затирая номера заявок,
    суммы и любые длинные числа.
    """
    assert find_account_numbers("Заявка 1234567890 выполнена") == []


def test_подсказка_действует_на_ограниченном_расстоянии():
    """Иначе слово из начала абзаца пометило бы число из соседнего предложения."""
    far = "лицевой счёт указан в договоре, который был подписан давно. Заявка 1234567890"
    assert find_account_numbers(far) == []


# --- Телефон ------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "+7 916 123-45-67",
        "+79161234567",
        "8 (916) 123-45-67",
        "8-916-123-45-67",
    ],
)
def test_телефон_находится_в_разных_записях(text: str):
    """В прототипе шаблон был синтаксически неверен, телефоны не находились вовсе."""
    assert [s.entity_type for s in find_phones(text)] == ["PHONE_NUMBER"]


# --- перекрытия ----------------------------------------------------------------- #


def test_перекрывающиеся_находки_сводятся_к_одной(sanitizer: PiiSanitizer):
    """Проверка базового набора показала: почта получала сразу три метки.

    Если не разрешить перекрытия, замены наложатся друг на друга и текст
    развалится.
    """
    spans = sanitizer.find(f"лицевой счёт {VALID_INN_10}")
    starts = [s.start for s in spans]
    assert len(starts) == len(set(starts))
    for first, second in zip(spans, spans[1:], strict=False):
        assert not first.overlaps(second)


def test_из_двух_находок_остаётся_длинная():
    short = PiiSpan(10, 14, "SHORT")
    long = PiiSpan(8, 20, "LONG")
    from app.gateway.pii_filter import _resolve_overlaps

    assert [s.entity_type for s in _resolve_overlaps([short, long])] == ["LONG"]


# --- обезличивание целиком ------------------------------------------------------ #


def test_значения_не_остаются_в_тексте(sanitizer: PiiSanitizer):
    """Главное требование правила 4.2, проверенное напрямую."""
    text = f"Лицевой счёт 1234567890, телефон +7 916 123-45-67, СНИЛС {VALID_SNILS}"
    clean, report = sanitizer.sanitize(text)

    assert "1234567890" not in clean
    assert "916" not in clean
    assert VALID_SNILS not in clean
    assert report.pii_detected is True
    assert set(report.entities) == {"ACCOUNT_NUMBER", "PHONE_NUMBER", "SNILS"}


def test_в_отчёт_попадают_только_типы(sanitizer: PiiSanitizer):
    """Схема отчёта запрещает значения, но проверить стоит и здесь."""
    _, report = sanitizer.sanitize("л/с 1234567890")
    assert report.entities == ["ACCOUNT_NUMBER"]
    assert all(not any(ch.isdigit() for ch in entity) for entity in report.entities)


def test_метки_на_русском(sanitizer: PiiSanitizer):
    """Метка уходит в промпт, и модель должна понимать, что это скрытое поле."""
    clean, _ = sanitizer.sanitize("л/с 1234567890")
    assert "<ЛИЦЕВОЙ_СЧЁТ>" in clean


def test_текст_без_данных_не_меняется(sanitizer: PiiSanitizer):
    text = "Как часто нужно поверять счётчик воды?"
    clean, report = sanitizer.sanitize(text)
    assert clean == text
    assert report.pii_detected is False
    assert report.entities == []


def test_окружающий_текст_сохраняется(sanitizer: PiiSanitizer):
    """Обезличивание не должно съедать слова вокруг находки."""
    clean, _ = sanitizer.sanitize("Прошу перерасчёт, л/с 1234567890, спасибо")
    assert clean.startswith("Прошу перерасчёт, л/с ")
    assert clean.endswith(", спасибо")


def test_несколько_находок_подряд(sanitizer: PiiSanitizer):
    clean, report = sanitizer.sanitize("л/с 1234567890 и л/с 9876543210")
    assert clean.count("<ЛИЦЕВОЙ_СЧЁТ>") == 2
    assert report.entities == ["ACCOUNT_NUMBER"]


def test_находка_не_хранит_значение():
    """Объект находки может попасть в журнал — раскрыть он ничего не должен."""
    span = PiiSpan(0, 10, "ACCOUNT_NUMBER")
    assert "1234567890" not in repr(span)


def test_отсутствие_языковой_модели_видно_вызывающему(sanitizer: PiiSanitizer):
    """Без разбора языка ФИО не находятся — это состояние нельзя не заметить."""
    assert sanitizer.language_model_available is False


# --- с разбором языка (медленно) ------------------------------------------------ #


@pytest.mark.slow
def test_фио_находится_разбором_языка():
    """Регулярным выражением фамилию не поймать — здесь нужен Presidio."""
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    clean, report = sanitizer.sanitize("Иванов Иван Петрович просит перерасчёт")
    assert "Иванов" not in clean
    assert "PERSON" in report.entities


@pytest.mark.slow
def test_почта_не_даёт_трёх_меток():
    """Проверка базового набора давала EMAIL, ORGANIZATION и URL на одну строку."""
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    clean, report = sanitizer.sanitize("почта ivanov@example.com")
    assert "ivanov@example.com" not in clean
    assert clean.count("<") == 1
    assert "ORGANIZATION" not in report.entities
    assert "URL" not in report.entities


# --- отсев ложных находок разбора языка ----------------------------------------- #
#
# Все случаи ниже — не выдуманные, а взятые из настоящих обращений абонентов
# (136 текстов, раздел 56.10 контекста). До отсева обезличиватель давал на них
# 63 находки, из которых персональными данными были пять.
#
# Синтетика такого класса ошибок не показывала и показать не могла: в ней
# очевидные персональные данные и нет ни юридических ссылок, ни отраслевых
# сокращений, ни местоимений с заглавной буквы — то есть ровно того, из чего
# состоит настоящее обращение. Это пятый случай закономерности раздела 54.4 и
# новая её разновидность: прежние четыре нашла живая проверка, здесь
# подменялись **данные**.


@pytest.mark.parametrize(
    ("entity_type", "fragment"),
    [
        # Показания счётчика, помеченные как ФИО. Самый опасный случай: обезличивание
        # уничтожало число, ради которого задан вопрос.
        ("PERSON", "ХВ 142,858м3"),
        # Юридические ссылки.
        ("LOCATION", "РФ"),
        ("LOCATION", "Российской Федерации"),
        # Отраслевые сокращения — в обоих регистрах.
        ("LOCATION", "ИПУ"),
        ("LOCATION", "ХВ"),
        ("LOCATION", "хв"),
        ("LOCATION", "ипу"),
        # Местоимения вежливого обращения с заглавной буквы.
        ("PERSON", "Вами"),
        ("PERSON", "Ваше"),
        ("LOCATION", "Вашей"),
        ("LOCATION", "Вашему адресу"),
        # Слова в начале предложения.
        ("PERSON", "Пункту"),
        ("LOCATION", "Сама"),
        ("PERSON", "Ещё"),
        ("LOCATION", "Прилагаю"),
        ("LOCATION", "Водой"),
        # Слипшиеся слова — признак неверно найденных границ.
        ("PERSON", "день!Объясните"),
        # Роли сторон и месяцы.
        ("PERSON", "Контрагенту"),
        ("PERSON", "Август"),
    ],
)
def test_ложные_находки_разбора_языка_отсеиваются(entity_type: str, fragment: str):
    assert looks_like_pii(entity_type, fragment) is False


@pytest.mark.parametrize(
    ("entity_type", "fragment"),
    [
        ("PERSON", "Пантелеева"),
        ("PERSON", "Дарья Сергеевна"),
        ("PERSON", "Иванов Иван Петрович"),
        # Выгрузки биллинга пишут ФИО целиком заглавными — под шаблон
        # сокращения они не подходят, каждое слово длиннее четырёх букв.
        ("PERSON", "АЛЕКСАНДРОВ ИГОРЬ АЛЕКСАНДРОВИЧ"),
        ("LOCATION", "Ленинский проспект"),
        # Фамилия через дефис: дефис — не знак препинания внутри имени.
        ("PERSON", "Салтыков-Щедрин"),
    ],
)
def test_настоящие_данные_отсев_переживают(entity_type: str, fragment: str):
    assert looks_like_pii(entity_type, fragment) is True


def test_отсев_не_касается_почты():
    """У почты жёсткая форма, ложных срабатываний она не давала.

    Проверка формы сняла бы её саму: в адресе есть и точка, и собака.
    """
    assert looks_like_pii("EMAIL_ADDRESS", "ivanov@example.com") is True


def test_отсев_не_касается_собственных_распознавателей():
    """СНИЛС, ИНН, счёт и телефон состоят из цифр и проверку формы не прошли бы.

    Они и не должны её проходить: у них контрольные суммы и слова-подсказки,
    то есть свидетельство сильнее, чем у разбора языка.
    """
    for entity_type in ("SNILS", "INN", "ACCOUNT_NUMBER", "PHONE_NUMBER"):
        assert looks_like_pii(entity_type, "1234567890") is True


@pytest.mark.slow
def test_юридическая_ссылка_переживает_обезличивание():
    """Сквозная проверка на настоящем предложении из обращения.

    До отсева получалось «Согласно <ФИО> 84 Постановления Правительства
    <АДРЕС>», и правильный ответ по такому тексту невозможен в принципе.
    """
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    text = "Согласно Пункту 84 Постановления Правительства РФ от 06.05.2011 № 354"
    clean, report = sanitizer.sanitize(text)
    assert clean == text
    assert report.pii_detected is False


@pytest.mark.slow
def test_показания_счётчика_переживают_обезличивание():
    """Число, ради которого задан вопрос, заменялось на метку ФИО."""
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    text = "передавала показания ИПУ ХВ 142,858м3, а в программе внесены 143 м3"
    clean, _ = sanitizer.sanitize(text)
    assert "142,858м3" in clean
    assert "<ФИО>" not in clean


@pytest.mark.slow
def test_настоящее_фио_по_прежнему_скрывается():
    """Отсев не должен был ослабить фильтр — проверяется на том же тексте,
    из которого взяты ложные находки."""
    sanitizer = PiiSanitizer()
    if not sanitizer.language_model_available:
        pytest.skip("Presidio или русская модель не установлены")

    clean, report = sanitizer.sanitize(
        "подрядная организация ИП Пантелеева Дарья Сергеевна выполняет доставку"
    )
    assert "Пантелеева" not in clean
    assert "Дарья" not in clean
    assert "PERSON" in report.entities


# --- одиночное слово в начале предложения (пункт 71) -------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Перерасчёт не делают, если есть техническая возможность установки.",
        "Пункту 84 Постановления соответствует обследование.",
        "Прилагаю акт обследования помещения.",
        "Показания переданы вовремя. Начисления пересчитаны.",
    ],
)
def test_первое_слово_предложения_не_считается_именем(text: str):
    """Разбор языка помечает именем любое слово с заглавной, а в начале
    предложения с заглавной стоит каждое.

    Случай не выдуман: на «Перерасчёт не делают…» охранители блокировали ответ
    целиком, и абонент вместо объяснения получал «Не могу показать этот ответ».
    """
    _, report = sanitize(text)
    assert not report.pii_detected, f"ложная находка: {report.entities}"


@pytest.mark.parametrize(
    "text",
    [
        "Доставку осуществляет ИП Пантелеева Дарья Сергеевна.",
        "Пантелеева Дарья Сергеевна отвечает за доставку.",
    ],
)
def test_настоящее_фио_правило_не_снимает(text: str):
    """Правило снимает **одно слово**. Настоящее ФИО стоит несколькими — и
    находится независимо от того, начинает оно предложение или нет."""
    _, report = sanitize(text)
    assert report.pii_detected
    assert "PERSON" in report.entities


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("Счёт 2100707718 закрыт.", "ACCOUNT_NUMBER"),
        ("Телефон 8-920-555-33-11 для связи.", "PHONE_NUMBER"),
        ("Почта ivanov.petr@example.com для ответа.", "EMAIL_ADDRESS"),
    ],
)
def test_правило_не_касается_собственных_распознавателей(text: str, entity: str):
    """Отсев применяется только к находкам разбора языка.

    СНИЛС, ИНН, счёт и телефон опознаются контрольной суммой и словом-подсказкой:
    свидетельство сильнее, и ослаблять его нельзя — даже если находка стоит
    первым словом предложения."""
    _, report = sanitize(text)
    assert report.pii_detected
    assert entity in report.entities


def test_то_же_слово_внутри_предложения_остаётся_находкой():
    """Правило узкое по построению: оно про положение, а не про слово.

    Иначе оно превратилось бы в ещё один словарь исключений — тот самый, который
    и требовалось перестать пополнять."""
    assert looks_like_pii("PERSON", "Иванов", before="Заявление подал ")
    assert not looks_like_pii("PERSON", "Иванов", before="Всё готово. ")
