"""Судья качества ответа (LLM-judge, ADR-007 роль 4a, код-шаг 10).

ЧТО ЭТО.
Отдельная от отвечающей модель, которая читает вопрос абонента, найденный
контекст и готовый ответ и выставляет две оценки от 0 до 1:

* **faithfulness** — ответ следует из контекста, а не додуман (галлюцинации);
* **answer_relevancy** — ответ отвечает на заданный вопрос, а не уходит в сторону.

Обе метрики reference-free (раздел 36.1): эталонный ответ не нужен. Context
Precision / Recall из того же раздела считаются по выдаче поиска, а не моделью —
они закрыты замером Recall@3 (`scripts/measure_retrieval.py`) и сюда не входят.

ПОЧЕМУ ОТДЕЛЬНАЯ МОДЕЛЬ.
Правило раздела 36.4 и ADR-007: судья не совпадает с отвечающей моделью, иначе
она оценивает сама себя и систематически завышает. На прототипе судья —
`local-test` (Qwen2.5-7B через Ollama), генерация — `yandexgpt`. В момент
**failover генерации** на `local-test` судья и отвечающая модель совпадают —
:meth:`score` в этом случае поднимает :class:`JudgeConflict`, а прогон
пропускается и факт пропуска фиксируется (ADR-007, «Подтверждение», п. 9).

ВЕРСИЯ СУДЬИ ФИКСИРУЕТСЯ.
:class:`QualityScores` несёт `judge_model` — имя модели, которое вернул
провайдер. Без него оценки двух прогонов несравнимы (раздел 36.4).

ВНЕ ПУТИ АБОНЕНТА.
Вызывается ночным прогоном (`scripts/run_quality_eval.py`), не на каждом
ответе: локальный судья на процессоре стоит секунды. `check_groundedness` в
`app/gateway/guardrails.py` остаётся дешёвой предварительной оценкой на
асинхронном выходе шлюза — она помечена `provisional=True` и настоящей оценкой
судьи не является.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings

__all__ = [
    "AnswerJudge",
    "JudgeConflict",
    "JudgeError",
    "JudgeUnavailable",
    "QualityScores",
]

logger = logging.getLogger(__name__)


class JudgeError(RuntimeError):
    """Базовая ошибка судьи."""


class JudgeConflict(JudgeError):
    """Судья совпал с отвечающей моделью — оценка не выполняется.

    Штатная ситуация при failover генерации: оценка в этом состоянии была бы
    завышенной и неотличимой от честной, поэтому прогон пропускается (ADR-007).
    """


class JudgeUnavailable(JudgeError):
    """Модель-судья не ответила или ответ не разобрать в оценки."""


@dataclass(frozen=True, slots=True)
class QualityScores:
    """Оценки одного ответа. Значения в диапазоне [0, 1].

    Уходит в `quality_report` (раздел 35.1): агрегат по прогону пишется в таблицу
    `quality_reports` базы `telemetry` (на MVP — ClickHouse).
    """

    faithfulness: float
    answer_relevancy: float
    judge_model: str
    reasoning: str = ""
    provisional: bool = False
    """`True` — оценка получена не судьёй (дешёвой эвристикой). У судьи всегда
    `False`: поле обязательное, чтобы временную величину нельзя было принять за
    настоящую (та же дисциплина, что в `GroundednessReport`)."""

    def passes(self, *, faithfulness_min: float, relevancy_min: float) -> bool:
        """Прошёл ли ответ стартовые пороги раздела 36.1."""
        return (
            self.faithfulness >= faithfulness_min
            and self.answer_relevancy >= relevancy_min
        )


_SYSTEM = (
    "Ты — оценщик качества ответов справочного помощника водоканала. Ты не "
    "отвечаешь абоненту и не исправляешь ответ, а только выставляешь две оценки. "
    "Будь откалиброван: большинство разумных ответов заслуживают 0.6–0.9, а не 0. "
    "Верни ровно один объект JSON и ничего больше."
)

_RUBRIC = """\
Оцени ОТВЕТ помощника по двум критериям. Каждая оценка — число от 0.0 до 1.0
с шагом 0.1.

1. faithfulness — есть ли в ОТВЕТЕ выдумка.
   Штрафуй ТОЛЬКО за факты, которые противоречат КОНТЕКСТУ или которых в нём
   нет: выдуманные сроки, суммы, номера статей, процедуры. НЕ штрафуй за
   осторожность, неполноту или совет обратиться в контакт-центр.
   1.0 — в ответе нет фактов сверх контекста (честный отказ — тоже 1.0).
   0.5 — одно сомнительное утверждение.
   0.0 — ответ построен на выдуманных фактах или противоречит контексту.

2. answer_relevancy — по адресу ли ОТВЕТ.
   1.0 — отвечает на заданный вопрос по существу.
   0.6 — отвечает на тему вопроса, но частично, или отсылает в контакт-центр,
         когда контекст позволял ответить полнее.
   0.3 — отвечает на соседнюю тему.
   0.0 — не про то.

Пример. Вопрос про срок поверки, контекст «поверка раз в 6 лет».
Ответ «Поверка проводится раз в 6 лет» → {{"faithfulness": 1.0, "answer_relevancy": 1.0}}.
Ответ «Уточните в контакт-центре» → {{"faithfulness": 1.0, "answer_relevancy": 0.5}}.

ВОПРОС:
{question}

КОНТЕКСТ:
{context}

ОТВЕТ:
{answer}

Верни ТОЛЬКО JSON:
{{"faithfulness": <число>, "answer_relevancy": <число>, "reasoning": "<кратко почему>"}}
"""

_NO_CONTEXT = "(контекст не найден)"


class AnswerJudge:
    """Судья: вызывает модель-оценщика и разбирает её вердикт в :class:`QualityScores`.

    :param settings: настройки приложения; из них берётся псевдоним судьи
        (`judge_provider`), если не задан явно.
    :param model_alias: псевдоним модели-судьи из `litellm_config.yaml`
        (`local-test` / `yandexgpt`). По умолчанию — `settings.judge_provider`.
    :param complete: функция обращения к модели. ``None`` — берётся
        `litellm.Router` лениво (как в шлюзе), чтобы импорт модуля не тянул
        тяжёлую зависимость.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_alias: str | None = None,
        complete: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._alias = model_alias or self._settings.judge_provider
        self._complete = complete
        self._router_instance: Any = None

    @property
    def model_alias(self) -> str:
        return self._alias

    def conflicts_with(self, answering_alias: str) -> bool:
        """Судья совпадает с моделью, которая дала ответ?

        Сравнение по псевдониму провайдера: именно на этом уровне решается
        «та же модель или другая» (ADR-007). Псевдоним отвечавшей модели прогон
        берёт из маршрутизации ответа, а не из настройки — при failover он
        отличается от запрошенного.
        """
        return answering_alias == self._alias

    async def score(
        self,
        *,
        question: str,
        answer: str,
        context: Sequence[str],
        answering_alias: str | None = None,
    ) -> QualityScores:
        """Оценить один ответ.

        :param answering_alias: псевдоним модели, которая дала ответ. Если задан
            и совпадает с судьёй — поднимается :class:`JudgeConflict` до вызова
            модели (оценка при failover не выполняется).
        :raises JudgeConflict: судья совпал с отвечающей моделью.
        :raises JudgeUnavailable: модель не ответила либо ответ не разобрать.
        """
        if answering_alias is not None and self.conflicts_with(answering_alias):
            raise JudgeConflict(
                f"судья и отвечающая модель совпали ({self._alias}): "
                "оценка при failover генерации не выполняется (ADR-007)"
            )

        joined = "\n\n".join(piece.strip() for piece in context if piece.strip())
        prompt = _RUBRIC.format(
            question=question.strip(),
            context=joined or _NO_CONTEXT,
            answer=answer.strip(),
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._call(messages)
        except Exception as exc:  # noqa: BLE001 — любая ошибка модели это «судья недоступен»
            raise JudgeUnavailable(f"модель-судья не ответила: {exc}") from exc

        text = _extract_text(raw)
        model = str(_get(raw, "model") or "") or self._alias
        return _parse_scores(text, judge_model=model)

    # -- обращение к модели -------------------------------------------------- #

    async def _call(self, messages: list[dict[str, str]]) -> Any:
        params: dict[str, Any] = {
            "model": self._alias,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 400,
            # Судья обязан вернуть JSON. `drop_params` — на случай провайдера,
            # который режим не понимает (маршрутизатор `litellm_settings` не
            # читает, поэтому передаём явно); разбор ответа всё равно рассчитан
            # и на «JSON внутри пояснения».
            "response_format": {"type": "json_object"},
            "drop_params": True,
        }
        if self._complete is not None:
            return await self._complete(**params)
        return await self._router().acompletion(**params)

    def _router(self) -> Any:
        if self._router_instance is None:
            from app.gateway.model_router import build_router

            self._router_instance = build_router(self._settings)
        return self._router_instance


# --- разбор вердикта ----------------------------------------------------------- #

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _parse_scores(text: str, *, judge_model: str) -> QualityScores:
    """Достать оценки из ответа модели.

    Модель просят вернуть чистый JSON, но локальные модели нередко оборачивают
    его в пояснение — поэтому берётся первый объект `{...}` из текста, а не весь
    ответ. Если объекта нет или в нём нет обеих оценок — это `JudgeUnavailable`:
    молча подставить ноль значило бы записать плохую оценку вместо отсутствующей.
    """
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        raise JudgeUnavailable(f"в ответе судьи нет JSON: {text[:200]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeUnavailable(f"ответ судьи не разобрать: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeUnavailable(f"ответ судьи не объект: {payload!r}")

    try:
        faithfulness = _as_unit(payload["faithfulness"])
        relevancy = _as_unit(payload["answer_relevancy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgeUnavailable(f"в вердикте судьи нет обеих оценок: {payload!r}") from exc

    reasoning = str(payload.get("reasoning", "")).strip()
    return QualityScores(
        faithfulness=faithfulness,
        answer_relevancy=relevancy,
        judge_model=judge_model,
        reasoning=reasoning,
    )


def _as_unit(value: Any) -> float:
    """Привести к числу в [0, 1]. Модель иногда отдаёт «0.8», иногда 80, иногда «80%»."""
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    number = float(value)
    if number > 1.0:
        # «80» или «80%» вместо 0.8 — частая ошибка локальной модели.
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _extract_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return str(raw.choices[0].message.content or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
