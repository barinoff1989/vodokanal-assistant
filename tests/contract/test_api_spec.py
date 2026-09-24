"""Проверки спецификации API.

Главная — на устаревание: файл пересобирается из моделей и сравнивается с тем,
что лежит в репозитории. Если модель изменили, а спецификацию не пересобрали,
проверка падает. Без неё спецификация стала бы четвёртым случаем расходящихся
источников истины — три предыдущих проект уже пережил.

Остальные проверки сторожат конкретные решения: семь проблем прежней редакции и
пять решений, принятых по ним. Каждое из них однажды уже было принято неверно
или не принято вовсе, и без теста ничто не мешает вернуться к прежнему виду.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "AI_docs" / "API_Spec.yaml"
BUILDER = ROOT / "scripts" / "build_openapi.py"


@pytest.fixture(scope="module")
def spec() -> dict:
    if not SPEC.is_file():
        pytest.fail(f"спецификация не собрана: выполните python {BUILDER.name}")
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


# --- устаревание --------------------------------------------------------------- #


def test_спецификация_не_отстала_от_моделей():
    """Пересборка не должна ничего менять.

    Если проверка упала — модели изменились, а спецификация нет. Чинится
    запуском `python scripts/build_openapi.py`, а не правкой файла руками.
    """
    before = SPEC.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(BUILDER)], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr
    after = SPEC.read_text(encoding="utf-8")
    assert before == after, (
        "спецификация устарела относительно моделей; "
        "пересоберите: python scripts/build_openapi.py"
    )


def test_файл_помечен_как_собираемый():
    """Иначе кто-нибудь поправит его руками, и правку затрёт следующая сборка."""
    head = SPEC.read_text(encoding="utf-8")[:300]
    assert "собран скриптом" in head
    assert "Править руками" in head


# --- состав адресов ------------------------------------------------ #


def test_ровно_два_адреса(spec: dict):
    """Адрес `/models` удалён из контракта сознательно."""
    assert set(spec["paths"]) == {"/v1/generate", "/v1/healthz"}


def test_метрик_в_контракте_нет(spec: dict):
    """Они на отдельном порту: их читает система сбора, а не виджет абонента."""
    assert not any("metric" in path for path in spec["paths"])


def test_проверка_доступности_без_авторизации(spec: dict):
    """Её выполняет оркестратор, у которого внутреннего токена нет."""
    assert spec["paths"]["/v1/healthz"]["get"]["security"] == []


def test_генерация_требует_токена(spec: dict):
    assert spec["security"] == [{"InternalToken": []}]


# --- проблема 1: рабочий контур не localhost ------------------------------------- #


def test_рабочий_контур_не_машина_разработчика(spec: dict):
    """Прежняя редакция помечала localhost как рабочий контур."""
    production = spec["servers"][0]
    assert "localhost" not in production["url"]
    assert "cluster.local" in production["url"]


def test_машина_разработчика_помечена_как_таковая(spec: dict):
    developer = next(s for s in spec["servers"] if "localhost" in s["url"])
    assert "не является" in developer["description"]


# --- проблема 2 и решение 3: источники возвращаются ------------------------------- #


def test_источники_есть_в_ответе(spec: dict):
    """Поля источника принимались на входе, но не возвращались.

    При том что критерии готовности прямо требуют «ответ содержит sources».
    """
    assert "sources" in spec["components"]["schemas"]["GenerateResponse"]["properties"]


def test_источники_есть_и_в_событии_метаданных(spec: dict):
    """В потоке ответ приходит частями; источники — отдельным событием."""
    assert "sources" in spec["components"]["schemas"]["MetadataEvent"]["properties"]


# --- проблема 3 и решение 4: у события ошибки есть схема --------------------------- #


def test_у_события_ошибки_есть_схема(spec: dict):
    assert "ErrorEvent" in spec["components"]["schemas"]


def test_ошибка_в_потоке_и_обычная_имеют_одно_тело(spec: dict):
    """Два формата на два пути пришлось бы поддерживать порознь."""
    error_event = spec["components"]["schemas"]["ErrorEvent"]
    assert "ProblemDetail" in str(error_event)


def test_событие_потока_различает_виды(spec: dict):
    """Потребитель должен понимать, что пришло, не гадая по составу полей."""
    stream = spec["components"]["schemas"]["StreamEvent"]
    assert stream["discriminator"]["propertyName"] == "event"
    assert len(stream["oneOf"]) == 4


# --- проблема 4 и решение 2: департамента нет ------------------------------------- #


def test_департамента_в_метаданных_нет(spec: dict):
    """Решение в обратную сторону от того, что предлагала прежняя редакция.

    Департаменты не пользователи чат-интерфейса; по этой же причине из учёта
    лимитов убрали квоты по департаментам.
    """
    assert "department" not in spec["components"]["schemas"]["RequestMetadata"]["properties"]


# --- проблемы 6 и 7: метаданные под абонента --------------------------------------- #


def test_идентификатор_назван_по_абоненту(spec: dict):
    """Пользователь — абонент, а не сотрудник."""
    properties = spec["components"]["schemas"]["RequestMetadata"]["properties"]
    assert "subscriber_id" in properties
    assert "user_id" not in properties


def test_тип_обращения_есть_в_метаданных(spec: dict):
    assert "inquiry_type" in spec["components"]["schemas"]["RequestMetadata"]["properties"]


def test_перечень_каналов_задан(spec: dict):
    channel = spec["components"]["schemas"]["Channel"]
    assert set(channel["enum"]) == {"lk_web", "lk_mobile"}


# --- решение 5: код 503 (ADR-400) ---------------------------------------------------- #


def test_отказ_хранилища_это_отдельный_код(spec: dict):
    """Код 429 означал бы «вы превысили лимит» и возлагал причину на абонента."""
    responses = spec["paths"]["/v1/generate"]["post"]["responses"]
    assert "503" in responses
    assert "429" in responses


@pytest.mark.parametrize("status", ["429", "503"])
def test_у_отказов_объявлен_заголовок_повтора(spec: dict, status: str):
    """RFC 7807 требует его для 429; ADR-400 — для 503."""
    response = spec["paths"]["/v1/generate"]["post"]["responses"][status]
    assert "Retry-After" in response["headers"]


def test_все_ошибки_в_формате_rfc7807(spec: dict):
    """Ошибки по RFC 7807, без исключений для отдельных кодов."""
    responses = spec["paths"]["/v1/generate"]["post"]["responses"]
    for status, body in responses.items():
        if status == "200":
            continue
        assert "application/problem+json" in body["content"], status


# --- ограничения запроса -------------------------------------------------- #


def test_длина_запроса_ограничена(spec: dict):
    request = spec["components"]["schemas"]["GenerateRequest"]["properties"]
    assert request["query"]["maxLength"] == 2000


def test_системная_часть_обязательна(spec: dict):
    """Без неё модель отвечает вне роли."""
    schema = spec["components"]["schemas"]["GenerateRequest"]
    assert "system" in schema["required"]
    assert schema["properties"]["system"]["minLength"] == 1


def test_поток_включён_по_умолчанию(spec: dict):
    """Правило потоковой выдачи: ответ доходит до абонента постепенно."""
    parameters = spec["components"]["schemas"]["GenerationParameters"]["properties"]
    assert parameters["stream"]["default"] is True


def test_длина_ответа_ограничена_бюджетом(spec: dict):
    """Замер показал: три секунды критериев готовности — это 150–200 токенов."""
    parameters = spec["components"]["schemas"]["GenerationParameters"]["properties"]
    assert parameters["max_tokens"]["default"] <= 200


# --- согласие с работающим сервисом ------------------------------------------------------ #


def test_коды_ошибок_совпадают_с_теми_что_отдаёт_шлюз(spec: dict):
    """Спецификация описывает то, что сервис действительно делает.

    Расхождение здесь опаснее отсутствия спецификации: ей поверят.
    """
    from app.gateway.llm_gateway import LlmGateway

    gateway = LlmGateway(quota=None)
    declared = {int(code) for code in spec["paths"]["/v1/generate"]["post"]["responses"]}
    for status in (429, 502, 503, 504):
        problem = gateway.problem("x", "X", status, "trace-1")
        assert problem.status in declared
