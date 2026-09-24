"""Проверки агентов Этапа 1: классификатор и консультант.

Проверки взяты из требований проекта, а не придуманы:

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


# --- значения только из справочника (требование проекта) ------------------------- #


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
    вне справочника» — формулировка требования.

    Это же закрывает дефект прошлого прототипа: запасная ветка прототипа отдавала
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


@pytest.mark.parametrize(
    "query",
    [
        "Какого качества у вас питьевая вода?",
        "Вода из крана пахнет хлоркой, это нормально?",
        "Почему вода ржавая?",
        "Какая вода течёт из крана, она мутная",
        "У воды появился привкус",
    ],
)
def test_качество_воды_узнаётся_парой_вода_и_свойство(triage: Triage, query: str):
    """Плоского списка не хватало, и это нашёл живой прогон, а не тест.

    Признаки в `TOPIC_MARKERS` были взяты из названия темы («качество воды»), а
    абонент так не пишет: «качества у вас питьевая вода» не содержит этой пары
    подряд, а «пахнет хлоркой» не покрыто ничем. Оба вопроса уходили к модели, и
    та отвечала выдумкой по фрагментам про нулевую квитанцию и повышающий
    коэффициент — ровно то, от чего защищает ADR-100.
    """
    assert triage.classify(query).topic is Topic.WATER_QUALITY


@pytest.mark.parametrize(
    "query",
    [
        "Сколько стоит питьевая вода по тарифу?",
        "Прошу произвести перерасчет за воду",
        "Прошу опломбировать прибор учета холодной воды",
        "Как заказать поверку счетчика горячей воды?",
    ],
)
def test_вода_без_свойства_на_регламентный_ответ_не_уводит(triage: Triage, query: str):
    """Пара, а не одиночное слово: «питьевая вода» одна увела бы вопрос о тарифе
    на регламентный ответ о качестве, где ответа нет.

    Замер на 136 настоящих обращениях и 20 вопросах корпуса: ложных срабатываний
    ноль."""
    assert triage.classify(query).topic is not Topic.WATER_QUALITY


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


def test_чем_распознан_тип_видно_снаружи_отдельно_от_темы(triage: Triage):
    """Тема и тип — разные оси: у обеих свой matched_by, независимо друг от друга."""
    result = triage.classify("прошу опломбировать счётчик")
    assert result.matched_by == "default"  # тема — общий вопрос
    assert result.type_matched_by == "rule"  # тип нашёлся по ключевому слову

    given = triage.classify("что угодно", inquiry_type=InquiryType.CERTIFICATE)
    assert given.type_matched_by == "metadata"

    nothing = triage.classify("asdfghjkl")
    assert nothing.type_matched_by == "default"


# --- запасной классификатор типа обращения (app/agents/inquiry_type_fallback.py) #


class FakeTypeFallback:
    """Дубль запасного классификатора типа — отдаёт заданное значение без модели."""

    def __init__(self, inquiry_type: InquiryType | None) -> None:
        self._type = inquiry_type
        self.calls = 0

    async def classify(self, query: str) -> InquiryType | None:  # noqa: ARG002
        self.calls += 1
        return self._type


async def test_правила_не_нашли_тип_но_модель_помогла():
    """Ключевые слова молчат на этом тексте (нет ни одного стема из KEYWORDS) —
    запасной классификатор отвечает вместо `other`."""
    fallback = FakeTypeFallback(InquiryType.METER_VERIFICATION)
    triage = Triage(type_fallback=fallback)
    result = await triage.classify_async("asdfghjkl")

    assert result.inquiry_type is InquiryType.METER_VERIFICATION
    assert result.type_matched_by == "model"
    assert fallback.calls == 1


async def test_модель_не_нашла_тип_путь_как_раньше():
    fallback = FakeTypeFallback(None)
    triage = Triage(type_fallback=fallback)
    result = await triage.classify_async("asdfghjkl")

    assert result.inquiry_type is InquiryType.OTHER
    assert result.type_matched_by == "default"
    assert fallback.calls == 1


async def test_тип_найден_правилами_модель_не_зовётся():
    """Ключевые слова уже дали ответ — запасной классификатор лишний."""
    fallback = FakeTypeFallback(InquiryType.CERTIFICATE)
    triage = Triage(type_fallback=fallback)
    result = await triage.classify_async("прошу опломбировать счётчик")

    assert result.type_matched_by == "rule"
    assert fallback.calls == 0


async def test_тип_передан_снаружи_модель_не_зовётся():
    fallback = FakeTypeFallback(InquiryType.CERTIFICATE)
    triage = Triage(type_fallback=fallback)
    result = await triage.classify_async(
        "что угодно", inquiry_type=InquiryType.METER_SEALING
    )

    assert result.inquiry_type is InquiryType.METER_SEALING
    assert result.type_matched_by == "metadata"
    assert fallback.calls == 0


async def test_без_запасного_классификатора_типа_поведение_прежнее():
    triage = Triage()
    result = await triage.classify_async("asdfghjkl")
    assert result.inquiry_type is InquiryType.OTHER
    assert result.type_matched_by == "default"


# --- консультант ---------------------------------------------------------------- #


def test_консультант_не_вызывает_генерацию_при_пустом_контексте():
    """Дословное требование проекта.

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
