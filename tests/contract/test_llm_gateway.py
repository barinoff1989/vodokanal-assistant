"""Проверки шлюза к модели.

План разработки требует на этом шаге двух проверок: контрактной на
`/v1/generate` и на `429` с `Retry-After` при превышении лимита. Третья добавлена
сверх плана — `503` при недоступности хранилища: без неё решение ADR-008
осталось бы текстом.

Провайдер и хранилище подменяются. Иначе каждый прогон стоил бы обращений к
платному внешнему API и требовал поднятого Redis, а проверять надо порядок
проверок и перевод ошибок, а не работу чужих сервисов.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.gateway.guardrails import SAFE_FALLBACK, SyncGuardrails
from app.gateway.llm_gateway import LlmGateway, ProviderTimeoutError
from app.gateway.pii_filter import PiiSanitizer
from app.gateway.quota import QuotaManager
from app.models import (
    ContextChunk,
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    MetadataEvent,
    ProblemDetail,
    RequestMetadata,
    TokenEvent,
)

# --- подмены ------------------------------------------------------------------ #


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def incr(self, name: str) -> int:
        self.values[name] = self.values.get(name, 0) + 1
        return self.values[name]

    def expire(self, name: str, time: int) -> bool:
        return True


class BrokenRedis:
    def incr(self, name: str) -> int:
        raise ConnectionError("соединение отсутствует")

    def expire(self, name: str, time: int) -> bool:
        raise ConnectionError("соединение отсутствует")


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)
        self.delta = _Message(content)


class _Response:
    """Форма ответа LiteLLM — ровно те поля, которые читает шлюз."""

    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = type("U", (), {"prompt_tokens": 120, "completion_tokens": 40})()


def fake_completion(answer: str = "Поверку проводят раз в шесть лет."):
    """Подменённый провайдер: отдаёт заданный ответ целиком или по кускам."""

    async def _call(*, stream: bool, **kwargs: Any) -> Any:
        if not stream:
            return _Response(answer)

        async def _pieces() -> AsyncIterator[Any]:
            for word in answer.split(" "):
                yield _Response(word + " ")

        return _pieces()

    return _call


def failing_completion(exc: Exception):
    async def _call(**kwargs: Any) -> Any:
        raise exc

    return _call


class CapturingUsage:
    """Дубль хранилища событий использования — копит записанное."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)


def _gateway(**overrides: Any) -> LlmGateway:
    sanitizer = PiiSanitizer(analyzer=None)
    defaults: dict[str, Any] = {
        "sanitizer": sanitizer,
        "guardrails": SyncGuardrails(sanitizer=sanitizer),
        "completion": fake_completion(),
        "settings": Settings(_env_file=None, llm_provider="local-test"),
        "quota": QuotaManager(
            FakeRedis(), subscriber_limit=100, session_limit=100, service_limit=100
        ),
    }
    return LlmGateway(**{**defaults, **overrides})


def _request(**overrides: Any) -> GenerateRequest:
    defaults: dict[str, Any] = {
        "system": "Ты помощник абонента водоканала.",
        "query": "Как часто нужно поверять счётчик?",
        "metadata": RequestMetadata(subscriber_id="sub-1", session_id="sess-1"),
    }
    return GenerateRequest(**{**defaults, **overrides})


async def _collect(gateway: LlmGateway, request: GenerateRequest) -> list[Any]:
    return [event async for event in gateway.stream(request)]


# --- контракт ответа (проверка 1 плана) ---------------------------------------- #


async def test_ответ_содержит_поля_контракта():
    response = await _gateway().generate(_request())
    assert isinstance(response, GenerateResponse)
    assert response.answer
    assert response.trace_id.startswith("trace-")
    assert response.finish_reason is FinishReason.STOP


async def test_ответ_содержит_источники():
    """Раздел 18 требует источники; раньше поля принимались, но не возвращались."""
    request = _request(
        context=[
            ContextChunk(
                chunk_id="c1",
                text="Поверка проводится раз в шесть лет",
                source_title="Регламент поверки",
                relevance_score=0.91,
            )
        ]
    )
    response = await _gateway().generate(request)
    assert isinstance(response, GenerateResponse)
    assert [s.chunk_id for s in response.sources] == ["c1"]
    assert response.sources[0].source_title == "Регламент поверки"


async def test_расход_токенов_возвращается():
    response = await _gateway().generate(_request())
    assert isinstance(response, GenerateResponse)
    assert response.usage.total_tokens == 160


async def test_поток_идёт_кусками_и_завершается():
    """Правило 4.3: ответ доходит до абонента постепенно."""
    events = await _collect(_gateway(), _request())
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) > 1
    assert isinstance(events[-2], MetadataEvent)
    assert isinstance(events[-1], DoneEvent)


async def test_у_потока_и_ответа_целиком_один_формат_ошибки():
    """Решение 4 в моделях: два формата ошибки пришлось бы поддерживать порознь."""
    gateway = _gateway(completion=failing_completion(RuntimeError("провайдер молчит")))
    whole = await gateway.generate(_request())
    events = await _collect(gateway, _request())
    assert isinstance(whole, ProblemDetail)
    assert isinstance(events[0], ErrorEvent)
    assert whole.status == events[0].problem.status == 502


# --- лимиты (проверка 2 плана) -------------------------------------------------- #


async def test_превышение_лимита_даёт_429():
    gateway = _gateway(
        quota=QuotaManager(FakeRedis(), subscriber_limit=1, session_limit=1, service_limit=1)
    )
    first = await gateway.generate(_request())
    second = await gateway.generate(_request())
    assert isinstance(first, GenerateResponse)
    assert isinstance(second, ProblemDetail)
    assert second.status == 429


async def test_у_отказа_по_лимиту_есть_время_повтора():
    """Дефект раздела 35.1: заголовок `Retry-After` был пропущен вовсе."""
    gateway = _gateway(
        quota=QuotaManager(FakeRedis(), subscriber_limit=1, session_limit=1, service_limit=1)
    )
    await gateway.generate(_request())
    problem = await gateway.generate(_request())
    assert isinstance(problem, ProblemDetail)
    assert "повторите через" in (problem.detail or "")


async def test_лимит_проверяется_до_обращения_к_модели():
    """Порядок из раздела 41.3: отклонить дешевле, чем обезличивать и вызывать."""
    calls: list[int] = []

    async def counting(**kwargs: Any) -> Any:
        calls.append(1)
        return _Response("ответ")

    gateway = _gateway(
        completion=counting,
        quota=QuotaManager(FakeRedis(), subscriber_limit=0, session_limit=0, service_limit=0),
    )
    await gateway.generate(_request())
    assert calls == []


# --- отказ хранилища, ADR-008 (проверка 3, сверх плана) -------------------------- #


async def test_недоступное_хранилище_даёт_503_а_не_429():
    """Код 429 возлагал бы причину на абонента, который ни при чём."""
    gateway = _gateway(
        quota=QuotaManager(BrokenRedis(), subscriber_limit=10, session_limit=10, service_limit=10)
    )
    problem = await gateway.generate(_request())
    assert isinstance(problem, ProblemDetail)
    assert problem.status == 503


async def test_сообщение_об_отказе_не_раскрывает_причину():
    """Абоненту — «временно недоступен», без упоминания хранилища счётчиков."""
    gateway = _gateway(
        quota=QuotaManager(BrokenRedis(), subscriber_limit=10, session_limit=10, service_limit=10)
    )
    problem = await gateway.generate(_request())
    assert isinstance(problem, ProblemDetail)
    lowered = (problem.detail or "").lower()
    assert "временно недоступен" in lowered
    assert "redis" not in lowered
    assert "счётчик" not in lowered


async def test_отказ_хранилища_отличим_от_превышения_и_в_потоке():
    gateway = _gateway(
        quota=QuotaManager(BrokenRedis(), subscriber_limit=10, session_limit=10, service_limit=10)
    )
    events = await _collect(gateway, _request())
    assert isinstance(events[0], ErrorEvent)
    assert events[0].problem.status == 503


# --- обезличивание и охрана ------------------------------------------------------ #


async def test_данные_абонента_не_уходят_провайдеру():
    """Правило 4.2, проверенное на том, что реально ушло в вызов."""
    seen: dict[str, Any] = {}

    async def capturing(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        seen["messages"] = messages
        return _Response("ответ")

    gateway = _gateway(completion=capturing)
    await gateway.generate(_request(query="Мой л/с 1234567890, помогите с перерасчётом"))
    assert "1234567890" not in str(seen["messages"])


async def test_контекст_уходит_модели_как_есть():
    """Обратное прежнему поведению — и это решение ADR-015, а не послабление.

    Контекст больше не обезличивается: обезличиватель сделан для текста
    неизвестного происхождения, а корпус собрали мы. На вопрос про перерасчёт
    модель получала «3. ⟨ФИО⟩ выполняется по заявлению абонента», где ⟨ФИО⟩ —
    слово «Перерасчёт».

    Гарантия не исчезла, а переехала ко входу: документ с персональными данными
    не попадает в корпус вовсе (`check_no_personal_data`), и проверка на это
    стоит в `tests/unit/test_kb_build.py`.
    """
    seen: dict[str, Any] = {}

    async def capturing(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        seen["messages"] = messages
        return _Response("ответ")

    request = _request(
        context=[
            ContextChunk(
                chunk_id="c1",
                text="Перерасчёт выполняется по заявлению. Документы: заявление, акт",
                source_title="Инструкция",
                relevance_score=0.8,
            )
        ]
    )
    await _gateway(completion=capturing).generate(request)
    sent = str(seen["messages"])
    assert "Перерасчёт выполняется по заявлению" in sent
    assert "Документы: заявление, акт" in sent
    assert "<ФИО>" not in sent and "<АДРЕС>" not in sent


async def test_вопрос_абонента_обезличивается_по_прежнему():
    """ADR-015 сузил правило 4.2, а не отменил его."""
    seen: dict[str, Any] = {}

    async def capturing(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        seen["messages"] = messages
        return _Response("ответ")

    request = _request(
        query="Мой лицевой счёт 9998887770, прошу перерасчёт",
        context=[
            ContextChunk(
                chunk_id="c1",
                text="Перерасчёт выполняется по заявлению абонента.",
                source_title="Инструкция",
                relevance_score=0.8,
            )
        ],
    )
    await _gateway(completion=capturing).generate(request)
    assert "9998887770" not in str(seen["messages"])


async def test_опасный_ответ_модели_не_доходит_до_абонента():
    gateway = _gateway(completion=fake_completion("Задолженность по счёту 1234567890 — 900 рублей"))
    response = await gateway.generate(_request())
    assert isinstance(response, GenerateResponse)
    assert "1234567890" not in response.answer
    assert response.finish_reason is FinishReason.GUARDRAIL


async def test_поток_обрывается_на_опасном_ответе():
    gateway = _gateway(completion=fake_completion("Ваш лицевой счёт 1234567890 в долгах"))
    events = await _collect(gateway, _request())
    text = "".join(e.delta for e in events if isinstance(e, TokenEvent))
    assert "1234567890" not in text
    assert SAFE_FALLBACK in text
    done = [e for e in events if isinstance(e, DoneEvent)][0]
    assert done.finish_reason is FinishReason.GUARDRAIL


# --- ошибки провайдера ------------------------------------------------------------ #


async def test_таймаут_провайдера_даёт_504():
    gateway = _gateway(completion=failing_completion(ProviderTimeoutError("не ответил")))
    problem = await gateway.generate(_request())
    assert isinstance(problem, ProblemDetail)
    assert problem.status == 504


async def test_ошибка_провайдера_даёт_502():
    gateway = _gateway(completion=failing_completion(RuntimeError("сломался")))
    problem = await gateway.generate(_request())
    assert isinstance(problem, ProblemDetail)
    assert problem.status == 502


async def test_у_ошибки_есть_опознаватель_запроса():
    """Без него жалобу абонента невозможно связать с записью в журнале."""
    gateway = _gateway(completion=failing_completion(RuntimeError("сломался")))
    problem = await gateway.generate(_request())
    assert isinstance(problem, ProblemDetail)
    assert (problem.trace_id or "").startswith("trace-")


# --- живая проверка на настоящем провайдере ---------------------------------------- #


@pytest.mark.live
async def test_живой_вызов_яндекса():
    """Сквозная проверка на реальном провайдере. Запускать: pytest -m live."""
    settings = Settings()
    if not settings.yandex_api_key:
        pytest.skip("YANDEX_API_KEY не задан")

    gateway = LlmGateway(
        sanitizer=PiiSanitizer(analyzer=None),
        settings=Settings(llm_provider="yandexgpt"),
    )
    events = await _collect(gateway, _request())
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert not errors, errors[0].problem if errors else None
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) > 1, "ответ пришёл одним куском — это не поток"


# --- находки живых проверок: защита от возврата ---------------------------------- #


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError(
            "litellm.APIConnectionError: Ollama_chatException - Cannot connect to host "
            "localhost:11434 ssl:<ssl.SSLContext object at 0x0000021CF65B55D0>"
        ),
        ProviderTimeoutError("HTTPSConnectionPool(host='llm.api.cloud.yandex.net', port=443)"),
    ],
)
async def test_внутренний_текст_ошибки_не_доходит_до_абонента(failure: Exception):
    """Найдено живой проверкой: абоненту показывался адрес, порт и адрес объекта
    в памяти. И непонятно, и раскрывает устройство системы.

    Существующая проверка сообщения касалась отказа хранилища лимитов (503);
    ошибки провайдера (502 и 504) оставались незащищёнными.
    """
    gateway = _gateway(completion=failing_completion(failure))
    problem = await gateway.generate(_request())

    assert isinstance(problem, ProblemDetail)
    detail = (problem.detail or "").lower()
    for internal in ("localhost", "sslcontext", "litellm", "0x0000", "port=443", "host="):
        assert internal not in detail, f"внутренняя подробность в тексте: {internal}"
    assert "недоступен" in detail


async def test_внутренний_текст_не_доходит_и_в_потоке():
    """Тот же путь, второй режим: ошибка внутри уже начатого потока."""
    gateway = _gateway(
        completion=failing_completion(RuntimeError("Cannot connect to host localhost:11434"))
    )
    events = await _collect(gateway, _request())
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert errors
    assert "localhost" not in (errors[0].problem.detail or "").lower()


def test_прогрев_поднимает_тяжёлые_зависимости():
    """Найдено живой проверкой: восемь секунд до первого куска ответа.

    Семь из них — ленивый импорт библиотеки внутри первого запроса, ещё четыре —
    загрузка языковой модели. Обе платил первый абонент.

    Проверяется сам факт прогрева, а не его длительность: время зависит от машины,
    и порог сделал бы проверку хрупкой. Важно, что метод есть и отрабатывает без
    обращения к модели — тратить токены на прогрев незачем.
    """
    calls: list[int] = []

    async def counting(**kwargs: Any) -> Any:
        calls.append(1)
        return _Response("ответ")

    gateway = _gateway(completion=counting)
    gateway.warmup()

    assert calls == [], "прогрев не должен обращаться к модели"


def test_интерфейс_прогревает_шлюз_при_запуске():
    """Прогрев бесполезен, если его никто не вызывает.

    Без этой проверки метод остался бы написанным и не подключённым — а именно
    так и выглядела ошибка до находки.
    """
    from app.api import create_app

    warmed: list[str] = []

    class TrackingGateway:
        """Минимальная заглушка: важен только вызов прогрева при запуске."""

        def warmup(self) -> None:
            warmed.append("да")

        async def generate(self, request: GenerateRequest) -> Any:
            return GenerateResponse(answer="", model="m", trace_id="t")

        async def stream(self, request: GenerateRequest) -> AsyncIterator[Any]:
            yield DoneEvent(trace_id="t")

    with TestClient(create_app(TrackingGateway())):
        pass

    assert warmed == ["да"], "приложение не прогрело шлюз при запуске"


# --- синтетический источник помечается (пункт 67) ----------------------------- #


def _chunk(chunk_id: str, *, synthetic: bool) -> ContextChunk:
    return ContextChunk(
        chunk_id=chunk_id,
        text="текст фрагмента",
        source_title="документ",
        relevance_score=0.9,
        synthetic=synthetic,
    )


@pytest.mark.asyncio
async def test_ответ_по_синтетике_помечен_оговоркой():
    """Ответ по выдуманной процедуре обязан быть отличим от ответа по
    настоящему регламенту — иначе на демонстрации его примут за второе."""
    gateway = _gateway()
    request = _request(context=[_chunk("doc#1", synthetic=True)])

    response = await gateway.generate(request)

    assert isinstance(response, GenerateResponse)
    assert response.disclaimer
    assert response.sources[0].synthetic is True


@pytest.mark.asyncio
async def test_ответ_по_настоящему_источнику_оговорки_не_несёт():
    """Иначе оговорка обесценится: она стоит на каждом ответе и её перестают
    читать."""
    gateway = _gateway()
    request = _request(context=[_chunk("faq-01", synthetic=False)])

    response = await gateway.generate(request)

    assert isinstance(response, GenerateResponse)
    assert response.disclaimer is None
    assert response.sources[0].synthetic is False


@pytest.mark.asyncio
async def test_хватает_одного_синтетического_фрагмента():
    """Смесь настоящего и выдуманного — самый опасный случай: ответ выглядит
    обоснованным, а часть его построена на составленной нами процедуре."""
    gateway = _gateway()
    request = _request(
        context=[
            _chunk("faq-01", synthetic=False),
            _chunk("doc#2", synthetic=True),
        ]
    )

    response = await gateway.generate(request)

    assert isinstance(response, GenerateResponse)
    assert response.disclaimer


# --- учёт токенов в потоке (пункт 70) ----------------------------------------- #


class _UsageOnly:
    """Последний кусок провайдера: расход есть, текста нет.

    Форма снята с настоящего ответа, а не придумана: и YandexGPT, и локальная
    Qwen через Ollama присылают `usage` **отдельным последним куском** при
    `stream_options={"include_usage": True}`. Текстовые куски расхода не несут.
    """

    def __init__(self, prompt: int, completion: int) -> None:
        self.choices = []
        self.usage = type(
            "U", (), {"prompt_tokens": prompt, "completion_tokens": completion}
        )()
        self.model = "stub"


class _TextOnly:
    """Обычный кусок потока: текст есть, расхода нет."""

    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = None
        self.model = "stub"


def usage_completion(answer: str, *, prompt: int, completion: int):
    """Провайдер, ведущий себя как настоящий: расход отдельным куском в конце."""

    async def _call(*, stream: bool, **kwargs: Any) -> Any:
        if not stream:
            return _Response(answer)

        async def _pieces() -> AsyncIterator[Any]:
            for word in answer.split(" "):
                yield _TextOnly(word + " ")
            yield _UsageOnly(prompt, completion)

        return _pieces()

    return _call


@pytest.mark.asyncio
async def test_расход_из_последнего_куска_доходит_до_события_завершения():
    """Кусок с расходом не несёт текста, и его легко потерять.

    В цикле стоит `continue` для пустой дельты; сними расход после него — и учёт
    останется нулевым, причём незаметно: событие `done` придёт как обычно."""
    gateway = _gateway(completion=usage_completion("два слова", prompt=120, completion=7))
    events = await _collect(gateway, _request())

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].usage.prompt_tokens == 120
    assert done[0].usage.completion_tokens == 7


@pytest.mark.asyncio
async def test_потоковый_путь_запрашивает_расход_у_провайдера():
    """Без `stream_options` провайдер расход не присылает вовсе — проверено на
    обоих. Параметр обязан уходить именно в потоковом вызове."""
    seen: dict[str, Any] = {}

    async def capturing(*, stream: bool, **kwargs: Any) -> Any:
        seen.update(kwargs)
        seen["stream"] = stream
        return await usage_completion("ответ", prompt=1, completion=1)(
            stream=stream, **kwargs
        )

    gateway = _gateway(completion=capturing)
    await _collect(gateway, _request())

    assert seen["stream"] is True
    assert seen.get("stream_options") == {"include_usage": True}


@pytest.mark.asyncio
async def test_молчание_провайдера_о_расходе_не_ломает_поток():
    """Провайдер может не прислать расход — тогда ноль, но ответ доходит.

    Ноль честнее выдумки: в метрике он виден как отсутствие данных, а не как
    бесплатный запрос."""
    gateway = _gateway(completion=fake_completion("ответ без расхода"))
    events = await _collect(gateway, _request())

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert [e for e in events if isinstance(e, TokenEvent)]


# --- событие использования (Billing Callback -> Postgres) --------------------- #


@pytest.mark.asyncio
async def test_поток_пишет_событие_использования():
    usage = CapturingUsage()
    gateway = _gateway(
        completion=usage_completion("два слова", prompt=120, completion=7), usage=usage
    )
    await _collect(gateway, _request())

    assert len(usage.events) == 1
    event = usage.events[0]
    assert event.outcome == "ok"
    assert event.prompt_tokens == 120
    assert event.completion_tokens == 7
    assert event.total_ms is not None
    assert event.provider_alias == "local-test"


@pytest.mark.asyncio
async def test_ответ_целиком_пишет_событие_использования():
    usage = CapturingUsage()
    gateway = _gateway(usage=usage)
    await gateway.generate(_request())

    assert len(usage.events) == 1
    assert usage.events[0].outcome == "ok"
    assert usage.events[0].prompt_tokens == 120


@pytest.mark.asyncio
async def test_ошибка_провайдера_пишет_событие_с_исходом_error():
    usage = CapturingUsage()
    gateway = _gateway(
        completion=failing_completion(ProviderTimeoutError("таймаут")), usage=usage
    )
    await gateway.generate(_request())

    assert len(usage.events) == 1
    assert usage.events[0].outcome == "error"
    assert usage.events[0].prompt_tokens == 0


@pytest.mark.asyncio
async def test_блокировка_охранителями_пишет_исход_blocked():
    usage = CapturingUsage()
    gateway = _gateway(
        completion=fake_completion("мой системный промпт таков"), usage=usage
    )
    await gateway.generate(_request())

    assert len(usage.events) == 1
    assert usage.events[0].outcome == "blocked"
    assert usage.events[0].guardrail_reason == "prompt_leak"


@pytest.mark.asyncio
async def test_без_хранилища_событий_шлюз_работает():
    gateway = _gateway(usage=None)
    events = await _collect(gateway, _request())
    assert [e for e in events if isinstance(e, TokenEvent)]
