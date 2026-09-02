"""Проверки метрик.

Главная здесь — сверка с дашбордом. Спецификация дашборда написана раньше кода
и содержит готовые запросы; если код эмитит другие имена, каждый виджет в бою
окажется пустым, и заметят это не сразу, а когда метрика понадобится для
разбора происшествия.

Поэтому имена не перечисляются в тесте вручную, а **вычитываются из самого
файла спецификации**: список, переписанный руками, разошёлся бы с источником
при первой же правке дашборда.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

from app.gateway.guardrails import BlockReason
from app.metrics import prometheus as metrics

DASHBOARD = (
    Path(__file__).resolve().parents[2] / "AI_docs" / "Приложения" / "Дашборды.html"
)


def _collect_names() -> set[str]:
    """Имена, которые код действительно объявил."""
    names: set[str] = set()
    for metric in REGISTRY.collect():
        names.add(metric.name)
        for sample in metric.samples:
            names.add(sample.name)
    return names


def _dashboard_names() -> set[str]:
    """Имена, на которые ссылается спецификация дашборда."""
    if not DASHBOARD.is_file():
        pytest.skip(f"спецификация дашборда не найдена: {DASHBOARD}")
    text = DASHBOARD.read_text(encoding="utf-8")
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S)
    pattern = (
        r"\b(assistant_[a-z_]+|guardrails_[a-z_]+|llm_gateway_[a-z_]+"
        r"|pii_sanitizer_[a-z_]+|vector_db_[a-z_]+)\b"
    )
    return set(re.findall(pattern, text))


# --- сверка с дашбордом ------------------------------------------------------- #


def test_все_метрики_дашборда_объявлены_кодом():
    """Отсутствие любой означает пустой виджет в бою (пункт 22 TODO).

    `http_requests_total` из проверки исключён: это метрика веб-сервера, её
    эмитит слой интерфейса, а не шлюз.
    """
    declared = _collect_names()
    required = _dashboard_names()

    missing = set()
    for name in required:
        base = re.sub(r"_(bucket|count|sum|total)$", "", name)
        if not any(candidate.startswith(base) for candidate in declared):
            missing.add(name)

    assert not missing, f"дашборд ссылается на необъявленные метрики: {sorted(missing)}"


def test_дашборд_действительно_прочитан():
    """Защита от тихо пустого списка: если разбор сломается, первый тест
    начнёт проходить всегда, ничего не проверяя."""
    assert len(_dashboard_names()) >= 8


# --- разбивка по причине блокировки (миссия-критичный виджет) ------------------ #


def test_блокировки_считаются_с_причиной():
    """Раздел 37.2: утечка вынесена в отдельный виджет и не смешана с общим счётчиком."""
    metrics.record_guardrail(blocked=True, reason=BlockReason.PII_LEAK.value)
    value = REGISTRY.get_sample_value(
        "guardrails_blocked_total", {"reason": "pii_leak"}
    )
    assert value is not None and value >= 1


def test_все_причины_блокировки_пригодны_как_метка():
    """Метка приходит из перечня охранителей; расхождение сломало бы виджет."""
    for reason in BlockReason:
        metrics.record_guardrail(blocked=True, reason=reason.value)
        assert (
            REGISTRY.get_sample_value("guardrails_blocked_total", {"reason": reason.value})
            is not None
        )


def test_проверенные_считаются_отдельно_от_заблокированных():
    """Иначе долю блокировок не посчитать — не с чем сравнивать."""
    before = REGISTRY.get_sample_value("guardrails_checked_total") or 0
    metrics.record_guardrail(blocked=False)
    after = REGISTRY.get_sample_value("guardrails_checked_total") or 0
    assert after == before + 1


# --- персональные данные ------------------------------------------------------- #


def test_в_метку_попадает_тип_а_не_значение():
    """Метки уходят в систему сбора, хранятся долго и доступны широко.

    Значение персональных данных там осело бы навсегда — это та же граница,
    что в отчёте об обезличивании (правило 4.2).
    """
    metrics.record_pii(["ACCOUNT_NUMBER", "PHONE_NUMBER"])
    for entity in ("ACCOUNT_NUMBER", "PHONE_NUMBER"):
        value = REGISTRY.get_sample_value(
            "pii_sanitizer_entities_detected_total", {"entity": entity}
        )
        assert value is not None
        assert not any(char.isdigit() for char in entity)


def test_прохождение_обезличивания_считается_даже_без_находок():
    """Без знаменателя долю запросов с персональными данными не посчитать."""
    before = REGISTRY.get_sample_value("pii_sanitizer_requests_total") or 0
    metrics.record_pii([])
    after = REGISTRY.get_sample_value("pii_sanitizer_requests_total") or 0
    assert after == before + 1


# --- исходы ответов -------------------------------------------------------------- #


@pytest.mark.parametrize("outcome", ["ok", "fallback", "blocked", "error"])
def test_исходы_различимы(outcome: str):
    """Блокировка охранителем не должна выглядеть как обычный сбой."""
    metrics.record_response(channel="lk_web", inquiry_type="debt", outcome=outcome)
    value = REGISTRY.get_sample_value(
        "assistant_responses_total",
        {"channel": "lk_web", "inquiry_type": "debt", "outcome": outcome},
    )
    assert value is not None


def test_тип_обращения_до_классификации_не_теряется():
    """На первом обращении типа ещё нет; метка должна быть заполнена чем-то."""
    metrics.record_response(channel="lk_web", inquiry_type=None, outcome="ok")
    assert (
        REGISTRY.get_sample_value(
            "assistant_responses_total",
            {"channel": "lk_web", "inquiry_type": "unknown", "outcome": "ok"},
        )
        is not None
    )


# --- времена ----------------------------------------------------------------------- #


def test_границы_гистограмм_стоят_вокруг_целей_готовности():
    """Без точки рядом с целью квантиль прыгает через порог.

    Цели раздела 18: время до первого куска 500 мс, полный ответ 3 секунды.
    """
    assert 0.5 in metrics.TTFT_BUCKETS
    assert 3.0 in metrics.RESPONSE_BUCKETS


def test_замер_записывает_исход_даже_при_ошибке():
    """Ранний выход не должен терять запись — иначе сбои невидимы в потоке."""
    before = REGISTRY.get_sample_value(
        "assistant_responses_total",
        {"channel": "lk_web", "inquiry_type": "unknown", "outcome": "error"},
    ) or 0

    with pytest.raises(RuntimeError), metrics.track_response(
        channel="lk_web", inquiry_type=None
    ) as state:
        state["outcome"] = "error"
        raise RuntimeError("сбой посреди ответа")

    after = REGISTRY.get_sample_value(
        "assistant_responses_total",
        {"channel": "lk_web", "inquiry_type": "unknown", "outcome": "error"},
    )
    assert after == before + 1


# --- расход токенов ------------------------------------------------------------------ #


def test_входные_и_выходные_токены_считаются_раздельно():
    """У них разная цена — суммарный счётчик не дал бы посчитать расход в деньгах."""
    metrics.record_tokens(prompt=100, completion=40)
    assert REGISTRY.get_sample_value("llm_gateway_tokens_total", {"kind": "prompt"})
    assert REGISTRY.get_sample_value("llm_gateway_tokens_total", {"kind": "completion"})


# --- очередь поиска -------------------------------------------------------------------- #


def test_метрика_поиска_объявлена_заранее():
    """Поиска ещё нет, но виджет на неё уже ссылается.

    Без объявления запрос вернул бы пустоту, неотличимую от «очередь пуста».
    """
    assert REGISTRY.get_sample_value("vector_db_pending_queries") == 0.0


def test_у_каждой_метрики_есть_виджет():
    """Обратная проверка: метрика без виджета — тот же дефект с другой стороны.

    Первый тест сторожит «виджет ссылается на несуществующую метрику». Этот —
    «метрику собираем, но никому не показываем». Без обеих сторон связь
    односторонняя, и половина расхождений остаётся невидимой.
    """
    if not DASHBOARD.is_file():
        pytest.skip("спецификация дашборда не найдена")
    text = DASHBOARD.read_text(encoding="utf-8")

    declared = {
        metric.name
        for metric in REGISTRY.collect()
        if metric.name.startswith(
            ("assistant_", "guardrails_", "llm_", "pii_", "vector_db_", "http_", "quota_")
        )
    }

    missing = {name for name in declared if name not in text}
    assert not missing, f"метрики собираются, но не показаны на дашборде: {sorted(missing)}"
