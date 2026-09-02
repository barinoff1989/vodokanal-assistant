"""Метрики: имена и метки, на которые ссылается дашборд наблюдаемости.

Это **контракт, а не заготовка**. Спецификация дашборда (`AI_docs/Приложения/
Дашборды.html`) уже написана и содержит готовые запросы; если код эмитит другие
имена, каждый виджет в бою окажется пустым, и заметят это не сразу, а в момент,
когда метрика понадобится для разбора происшествия.

Имена ниже выписаны из самих запросов дашборда, а не придуманы заново:

===================================== ==========================================
  ``assistant_ttft_seconds``            время до первого куска ответа
  ``assistant_response_seconds``        полное время ответа
  ``assistant_responses_total``         поток ответов с разбивкой
  ``guardrails_checked_total``          сколько ответов проверено
  ``guardrails_blocked_total``          сколько заблокировано, **с причиной**
  ``llm_gateway_tokens_total``          расход токенов
  ``pii_sanitizer_requests_total``      сколько запросов прошло обезличивание
  ``pii_sanitizer_entities_detected``   что именно найдено, по типам
  ``http_requests_total``               ответы интерфейса по кодам
  ``vector_db_pending_queries``         глубина очереди поиска
===================================== ==========================================

ПОЧЕМУ РАЗБИВКА ПО ПРИЧИНЕ БЛОКИРОВКИ ОБЯЗАТЕЛЬНА.
Дашборд выделяет утечку персональных данных в отдельный миссия-критичный виджет
и намеренно не смешивает её с общим счётчиком блокировок: утечка — прямое
нарушение 152-ФЗ, а не просто плохой ответ. Без метки ``reason`` такой виджет
построить не на чем (раздел 37.2, пункт 22 сведённого TODO).

ПОЧЕМУ МЕТРИКИ НА ОТДЕЛЬНОМ ПОРТУ.
Правило 4.4 разрешает ровно два адреса, и `/metrics` среди них нет. Добавить его
к тому же приложению значило бы расширить контракт, который проект сознательно
держит минимальным. Метрики — не часть контракта с виджетом, их читает система
сбора; поэтому они отдаются отдельным портом и наружу, к абоненту, не выходят.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram, start_http_server

__all__ = [
    "ASSISTANT_RESPONSES",
    "ASSISTANT_RESPONSE_SECONDS",
    "ASSISTANT_TTFT_SECONDS",
    "GUARDRAILS_BLOCKED",
    "GUARDRAILS_CHECKED",
    "HTTP_REQUESTS",
    "LLM_TOKENS",
    "PII_ENTITIES_DETECTED",
    "PII_REQUESTS",
    "VECTOR_DB_PENDING",
    "record_guardrail",
    "record_pii",
    "record_response",
    "record_tokens",
    "serve",
    "track_response",
]

# Границы гистограмм выбраны вокруг целевых значений критериев готовности:
# время до первого куска — 500 мс, полный ответ — 3 секунды (раздел 18).
# Без точек рядом с целью квантиль на графике «прыгает» через порог и не
# показывает, насколько именно система от него отстоит.
TTFT_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0)
RESPONSE_BUCKETS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0)

ASSISTANT_TTFT_SECONDS = Histogram(
    "assistant_ttft_seconds",
    "Время от запроса до первого куска ответа",
    buckets=TTFT_BUCKETS,
)

ASSISTANT_RESPONSE_SECONDS = Histogram(
    "assistant_response_seconds",
    "Полное время ответа абоненту",
    buckets=RESPONSE_BUCKETS,
)

ASSISTANT_RESPONSES = Counter(
    "assistant_responses_total",
    "Ответы абонентам",
    labelnames=("channel", "inquiry_type", "outcome"),
)
"""`outcome` различает исходы, которые иначе слились бы в один:

* ``ok`` — обычный ответ;
* ``fallback`` — ответ без модели (низкая уверенность либо пустой контекст);
* ``blocked`` — оборван охранителем;
* ``error`` — сбой провайдера или отказ по лимиту.

Дашборд считает по нему долю запасного пути, а без разделения ``blocked`` и
``error`` блокировка выглядела бы как обычный сбой.
"""

GUARDRAILS_CHECKED = Counter(
    "guardrails_checked_total", "Ответов проверено охранителями"
)

GUARDRAILS_BLOCKED = Counter(
    "guardrails_blocked_total",
    "Ответов заблокировано охранителями",
    labelnames=("reason",),
)

LLM_TOKENS = Counter(
    "llm_gateway_tokens_total",
    "Расход токенов",
    labelnames=("kind",),
)
"""`kind` — ``prompt`` либо ``completion``. Разделение нужно для сметы: цена у
входных и выходных токенов разная, и суммарный счётчик не позволил бы посчитать
расход в деньгах."""

PII_REQUESTS = Counter(
    "pii_sanitizer_requests_total", "Запросов прошло через обезличивание"
)

PII_ENTITIES_DETECTED = Counter(
    "pii_sanitizer_entities_detected_total",
    "Найдено сущностей персональных данных, по типам",
    labelnames=("entity",),
)
"""Метка — **тип** сущности, никогда не значение. Та же граница, что в отчёте
об обезличивании: имена метрик уходят в систему сбора, где хранятся долго и
доступны широко (правило 4.2)."""

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Ответы интерфейса",
    labelnames=("path", "status"),
)

VECTOR_DB_PENDING = Gauge(
    "vector_db_pending_queries", "Глубина очереди запросов к векторной базе"
)
"""Поиска ещё нет (шаг 5), метрика объявлена заранее и держит ноль.

Объявить сразу дешевле, чем потом: виджет дашборда на неё уже ссылается, и без
объявления запрос возвращал бы пустоту, неотличимую от «очередь пуста»."""


def record_response(
    *,
    channel: str,
    inquiry_type: str | None,
    outcome: str,
    ttft: float | None = None,
    total: float | None = None,
) -> None:
    """Записать исход ответа и его времена."""
    ASSISTANT_RESPONSES.labels(
        channel=channel, inquiry_type=inquiry_type or "unknown", outcome=outcome
    ).inc()
    if ttft is not None:
        ASSISTANT_TTFT_SECONDS.observe(ttft)
    if total is not None:
        ASSISTANT_RESPONSE_SECONDS.observe(total)


def record_guardrail(*, blocked: bool, reason: str | None = None) -> None:
    """Записать проверку охранителей и, если было, блокировку с причиной."""
    GUARDRAILS_CHECKED.inc()
    if blocked:
        GUARDRAILS_BLOCKED.labels(reason=reason or "unknown").inc()


def record_pii(entities: Iterable[str]) -> None:
    """Записать прохождение обезличивания и типы найденного."""
    PII_REQUESTS.inc()
    for entity in entities:
        PII_ENTITIES_DETECTED.labels(entity=entity).inc()


def record_tokens(*, prompt: int = 0, completion: int = 0) -> None:
    if prompt:
        LLM_TOKENS.labels(kind="prompt").inc(prompt)
    if completion:
        LLM_TOKENS.labels(kind="completion").inc(completion)


@contextmanager
def track_response(*, channel: str, inquiry_type: str | None) -> Iterator[dict[str, float]]:
    """Замерить времена ответа и записать исход при выходе.

    Возвращает словарь, куда вызывающий код кладёт ``ttft`` в момент первого
    куска и ``outcome`` — исход. Так замер не размазывается по вызывающему коду
    и не забывается при раннем возврате.
    """
    started = time.perf_counter()
    state: dict[str, float] = {}
    try:
        yield state
    finally:
        record_response(
            channel=channel,
            inquiry_type=inquiry_type,
            outcome=str(state.get("outcome", "ok")),
            ttft=state.get("ttft"),
            total=time.perf_counter() - started,
        )


def serve(port: int = 9100) -> None:
    """Поднять отдачу метрик на отдельном порту.

    Отдельный порт, а не адрес в основном приложении: правило 4.4 разрешает ровно
    два адреса, и расширять контракт ради служебной надобности не следует.
    Метрики читает система сбора, абоненту они не показываются.
    """
    start_http_server(port)
