"""Проверки агентов Этапа 1: классификатор и консультант.

Проверки взяты из шага 6 плана разработки, а не придуманы:

* классификатор в обоих режимах возвращает значение из справочника;
* регрессия дефекта с посторонним значением в запасном пути (11.8);
* консультант не вызывает генерацию при пустом контексте.

Агентов `form` и `draft_validator` план тоже называет, но они про слоты и
черновики, то есть про Этап 2. На прототипе писать их было бы кодом без
потребителя.
"""

from __future__ import annotations

import pytest

from app.agents.advisor import NO_CONTEXT_ANSWER, Advisor
from app.agents.triage import Triage
from app.models import ContextChunk
from app.taxonomy import InquiryType, Topic


@pytest.fixture
def triage() -> Triage:
    return Triage()


def _chunk(chunk_id: str = "faq-01") -> ContextChunk:
    return ContextChunk(
        chunk_id=chunk_id,
        text="Показания передаются до 25 числа.",
        source_title="FAQ",
        source_url="https://example.test/faq/",
        relevance_score=0.9,
    )


# --- значения только из справочника (требование шага 6) ------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "Почему нет воды?",
        "Какое качество воды?",
        "Как получить справку?",
        "",
        "   ",
        "asdfghjkl",
        "Как приготовить борщ?",
    ],
)
def test_ни_один_путь_не_даёт_значения_вне_справочника(triage: Triage, query: str):
    """«Готово, когда: ни один путь возврата типа обращения не выдаёт значения
    вне справочника» — формулировка шага 6 плана.

    Это же закрывает дефект 11.8: запасная ветка прототипа отдавала
    транслитерацию `prochee` вместо кода, и фильтр поиска молча не находил
    ничего.
    """
    result = triage.classify(query)
    assert result.topic in Topic
    assert result.inquiry_type in InquiryType


def test_посторонний_тип_извне_приводится_к_справочнику(triage: Triage):
    """Через приведение проходит и переданное снаружи.

    Иначе значение от классификатора модели на Этапе 2 попадало бы в метаданные
    как есть — и разошлось бы со справочником при первом же неожиданном ответе.
    """
    result = triage.classify("вопрос", inquiry_type="выдуманный_тип")  # type: ignore[arg-type]
    assert result.inquiry_type is InquiryType.OTHER


# --- тема решает, звать ли модель ----------------------------------------------- #


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Почему нет воды третий день?", Topic.OUTAGE),
        ("По адресу неделю нет холодной воды. Поясните причину.", Topic.OUTAGE),
        ("Когда включат воду?", Topic.OUTAGE),
        ("Отключили воду без предупреждения", Topic.OUTAGE),
        ("Какое качество воды у вас в городе?", Topic.WATER_QUALITY),
        ("Из крана идёт мутная вода", Topic.WATER_QUALITY),
        ("Как передать показания счётчика?", Topic.GENERAL),
        ("Почему начисляются пени?", Topic.GENERAL),
    ],
)
def test_тема_узнаётся_по_признакам(triage: Triage, query: str, expected: Topic):
    assert triage.classify(query).topic is expected


def test_вопрос_про_аварию_на_счётчике_не_уходит_на_график(triage: Triage):
    """Настоящий вопрос из базы знаний, на котором ломается очевидное правило.

    «Авария» выглядит признаком отключения, но в FAQ есть вопрос «Что делать,
    если прибор учета вышел из строя из-за аварии на внутренних сетях» — это про
    счётчик. По одному слову его увело бы на график отключений, где ответа нет.
    """
    query = "Что делать, если прибор учета вышел из строя из-за аварии на внутренних сетях?"
    assert triage.classify(query).topic is Topic.GENERAL


def test_тема_из_метаданных_не_переопределяется(triage: Triage):
    """Пришедшее снаружи значение осведомлённее разбора строки.

    Личный кабинет или Этап 2 могут знать тему точнее, чем видно из текста, и
    перебивать их разбором значило бы терять эти сведения.
    """
    result = triage.classify("как передать показания", topic=Topic.OUTAGE)
    assert result.topic is Topic.OUTAGE
    assert result.matched_by == "metadata"


def test_общая_тема_идёт_к_модели_а_особые_нет(triage: Triage):
    """`needs_model` и есть развилка: два пути минуют генерацию."""
    assert triage.classify("как передать показания").needs_model is True
    assert triage.classify("почему нет воды").needs_model is False
    assert triage.classify("какое качество воды").needs_model is False


def test_чем_распознано_видно_снаружи(triage: Triage):
    """Доля `default` показывает, как часто классификация ничего не дала.

    Без разбивки она неотличима от доли настоящих общих вопросов — и
    деградация классификатора выглядела бы как рост обычных обращений.
    """
    assert triage.classify("почему нет воды").matched_by == "rule"
    assert triage.classify("как передать показания").matched_by == "default"


# --- консультант ---------------------------------------------------------------- #


def test_консультант_не_вызывает_генерацию_при_пустом_контексте():
    """Дословное требование шага 6 плана.

    Модель без контекста не молчит — она отвечает связно и уверенно на общих
    сведениях, которых в регламентах водоканала может не быть. Отличить такой
    ответ от настоящего не сможет ни абонент, ни мы.
    """
    advice = Advisor(contact_phone="8 (473) 206-77-06").advise([])

    assert advice.needs_model is False
    assert advice.answer is not None


def test_при_наличии_контекста_отвечает_модель():
    advice = Advisor(contact_phone="8 (473) 206-77-06").advise([_chunk()])

    assert advice.needs_model is True
    assert advice.answer is None


def test_отказ_называет_контакт_центр():
    """«Не знаю» без продолжения оставляет абонента там же, где он был."""
    advice = Advisor(contact_phone="8 (473) 206-77-06").advise([])

    assert advice.answer is not None
    assert "8 (473) 206-77-06" in advice.answer


def test_телефон_подставляется_а_не_зашит():
    """Тот же номер есть в базе знаний, и два источника одного значения
    разойдутся при первой правке — этот класс ошибки проект проходил трижды."""
    assert "{phone}" in NO_CONTEXT_ANSWER
    other = Advisor(contact_phone="8 (800) 000-00-00").advise([])
    assert other.answer is not None
    assert "8 (800) 000-00-00" in other.answer
