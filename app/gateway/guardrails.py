"""Охранители: проверка ответа модели до того, как его увидит абонент.

Раздел 35 разделяет проверки на две части, и разделение принципиальное:

* **Синхронная** (бюджет менее 50 мс) — персональные данные в ответе,
  недопустимый тон, разглашение системного промпта. При нарушении поток
  обрывается и вместо ответа модели уходит безопасная заглушка.
* **Асинхронная** — обоснованность ответа найденным контекстом. Считается
  после завершения потока и **не блокирует выдачу**: иначе нарушилось бы
  правило 4.3, требующее, чтобы ответ доходил до абонента постепенно.

ПОЧЕМУ ПРОВЕРКА РАБОТАЕТ НА ПОТОКЕ, А НЕ НА ГОТОВОМ ОТВЕТЕ.
Ответ отдаётся абоненту по кускам. Проверить его целиком можно только после
того, как он весь получен, — а к этому моменту абонент уже прочитал начало.
Поэтому :class:`StreamGuard` проверяет накопленный текст на каждом куске и
способен оборвать поток посередине. Это дороже, чем одна проверка в конце, и
единственный способ не показать абоненту то, чего он видеть не должен.

ПРИЧИНА БЛОКИРОВКИ ЗАПИСЫВАЕТСЯ ОТДЕЛЬНЫМ ПОЛЕМ.
Дашборд наблюдаемости (раздел 37.2) выделяет утечку персональных данных в
отдельный миссия-критичный виджет и намеренно не смешивает её с общим счётчиком
блокировок: утечка — прямое нарушение 152-ФЗ, а не просто плохой ответ. Чтобы
такой виджет был возможен, причина обязана быть машиночитаемой с самого начала
(пункт 22 сведённого TODO).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from app.gateway.pii_filter import PiiSanitizer
from app.models import FinishReason

__all__ = [
    "SAFE_FALLBACK",
    "BlockReason",
    "GroundednessReport",
    "GuardrailVerdict",
    "StreamGuard",
    "SyncGuardrails",
    "check_groundedness",
]


class BlockReason(StrEnum):
    """Почему ответ заблокирован.

    Значения совпадают с метками счётчика `guardrails_blocked_total{reason=...}`
    из раздела 37.2 — иначе виджет пришлось бы переименовывать после первого же
    прогона.
    """

    PII_LEAK = "pii_leak"
    TOXICITY = "toxicity"
    POLICY = "policy"
    PROMPT_LEAK = "prompt_leak"


SAFE_FALLBACK = (
    "Не могу показать этот ответ. Уточните, пожалуйста, вопрос — "
    "или обратитесь в контактный центр водоканала."
)
"""Что видит абонент вместо заблокированного ответа.

Формулировка нейтральная намеренно: сообщение вида «ответ заблокирован из-за
персональных данных» само по себе подсказывало бы, что именно удалось вытянуть
из системы."""


@dataclass(frozen=True, slots=True)
class GuardrailVerdict:
    """Решение синхронной проверки."""

    allowed: bool
    reason: BlockReason | None = None
    entities: tuple[str, ...] = ()
    """Типы найденных персональных данных — без значений, как в отчёте
    обезличивания (правило 4.2)."""

    @property
    def text_for_subscriber(self) -> str | None:
        """Что показать вместо ответа. `None`, если ответ разрешён."""
        return None if self.allowed else SAFE_FALLBACK

    @property
    def finish_reason(self) -> FinishReason | None:
        return None if self.allowed else FinishReason.GUARDRAIL


# --- недопустимый тон ---------------------------------------------------------- #

# > **[ПРЕДПОЛОЖЕНИЕ]** Перечень собран вручную и заведомо неполон. Полноценная
# > проверка тона требует модели, но она не укладывается в бюджет 50 мс на
# > синхронном пути. Договорённость проекта: здесь ловится грубое и очевидное,
# > тонкие случаи — на асинхронном пути через судью (раздел 36.4).
_TOXIC_PATTERNS: tuple[str, ...] = (
    r"\bидиот\w*",
    r"\bдурак\w*",
    r"\bтуп(?:ой|ая|ое|ые|ым|ых|ому|ыми|о)\b",
    r"\bсам\w*\s+винова\w*",
    r"\bотвали\w*",
    r"\bзаткни\w*",
)
_TOXIC_RE = re.compile("|".join(_TOXIC_PATTERNS), re.IGNORECASE)


# --- разглашение системного промпта -------------------------------------------- #

# Модель не должна пересказывать свои инструкции: это первый шаг к обходу
# ограничений и прямая цель попытки внедрения в промпт (раздел 36.6).
_PROMPT_LEAK_RE = re.compile(
    r"(?:системн\w*\s+(?:промпт|инструкц\w*|сообщen\w*)"
    r"|мои\s+инструкции"
    r"|мне\s+(?:было\s+)?(?:велено|приказано|указано)"
    r"|я\s+(?:—|-)?\s*языкова\w*\s+модель\s+с\s+инструкц\w*"
    r"|system\s+prompt"
    r"|my\s+instructions)",
    re.IGNORECASE,
)


# --- нарушение правил ответа ---------------------------------------------------- #

# Помощник консультирует, а не обещает от имени водоканала и не даёт правовых
# гарантий. Такие обещания — не просто плохой стиль: абонент может на них
# сослаться.
_POLICY_RE = re.compile(
    r"(?:гарантиру\w*\s+(?:перерасч[её]т|возврат|списание)"
    r"|обязательно\s+вернём"
    r"|можете\s+не\s+плати\w*"
    r"|суд\s+вы\s+точно\s+выигра\w*)",
    re.IGNORECASE,
)


class SyncGuardrails:
    """Синхронные проверки ответа. Бюджет — менее 50 мс (раздел 35.1).

    :param sanitizer: чем искать персональные данные. По умолчанию — тот же
        обезличиватель, что стоит на входе: два разных набора распознавателей
        для входа и выхода означали бы, что утечка возможна ровно в том, чего
        не знает второй набор.
    """

    def __init__(self, sanitizer: PiiSanitizer | None = None) -> None:
        self._sanitizer = sanitizer if sanitizer is not None else PiiSanitizer()

    def check(self, answer: str) -> GuardrailVerdict:
        """Проверить ответ. Порядок проверок — по тяжести последствий."""
        spans = self._sanitizer.find(answer)
        if spans:
            return GuardrailVerdict(
                allowed=False,
                reason=BlockReason.PII_LEAK,
                entities=tuple(sorted({span.entity_type for span in spans})),
            )
        if _PROMPT_LEAK_RE.search(answer):
            return GuardrailVerdict(allowed=False, reason=BlockReason.PROMPT_LEAK)
        if _TOXIC_RE.search(answer):
            return GuardrailVerdict(allowed=False, reason=BlockReason.TOXICITY)
        if _POLICY_RE.search(answer):
            return GuardrailVerdict(allowed=False, reason=BlockReason.POLICY)
        return GuardrailVerdict(allowed=True)


# --- проверка на потоке ---------------------------------------------------------- #


@dataclass(slots=True)
class StreamGuard:
    """Проверка ответа по мере его поступления.

    Копит текст и проверяет накопленное на каждом куске. Как только нарушение
    найдено, поток обрывается — абонент не увидит того, что уже пришло.

    Проверять каждый кусок целиком дороже, чем один раз в конце, и это осознанная
    плата: проверка в конце опоздала бы ровно настолько, насколько абонент успел
    прочитать начало ответа.
    """

    guardrails: SyncGuardrails = field(default_factory=SyncGuardrails)
    verdict: GuardrailVerdict = field(default_factory=lambda: GuardrailVerdict(allowed=True))
    text: str = ""

    def feed(self, delta: str) -> GuardrailVerdict:
        """Добавить очередной кусок и проверить накопленное."""
        self.text += delta
        self.verdict = self.guardrails.check(self.text)
        return self.verdict

    async def filter(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Пропустить поток через проверку, оборвав его при нарушении.

        При нарушении наружу уходит безопасная заглушка вместо остатка ответа, а
        уже отданные куски отозвать нельзя — поэтому проверка идёт до выдачи
        каждого куска, а не после.
        """
        async for delta in stream:
            if not self.feed(delta).allowed:
                yield SAFE_FALLBACK
                return
            yield delta


# --- обоснованность (асинхронно) --------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GroundednessReport:
    """Насколько ответ опирается на найденный контекст.

    Уходит в `quality_report` рядом с отчётом обезличивания (раздел 35.1) и не
    влияет на то, увидит ли абонент ответ: проверка выполняется после потока.
    """

    score: float
    grounded: bool
    provisional: bool = True
    """Оценка получена упрощённым способом, а не судьёй. Поле обязательное:
    без него временная оценка неотличима от настоящей в аналитике."""


def check_groundedness(answer: str, context: Iterable[str], *, threshold: float = 0.3) -> (
    GroundednessReport
):
    """Оценить обоснованность ответа найденным контекстом.

    > **[ВРЕМЕННОЕ РЕШЕНИЕ]** Считается доля слов ответа, встречающихся в
    > контексте. Это не оценка обоснованности в смысле раздела 36: там она
    > делается судьёй — локальной моделью на прототипе и YandexGPT на MVP
    > (ADR-007). Судья подключается на шаге 10 вместе с оценкой качества;
    > до тех пор нужна хоть какая-то величина, чтобы конвейер `quality_report`
    > существовал и проверялся целиком.
    >
    > Результат помечен как предварительный, чтобы его нельзя было принять за
    > настоящую оценку — та же дисциплина, что с непроверенными числами в
    > разделах 26.4 и 28.4.
    """
    answer_words = set(re.findall(r"\w{4,}", answer.lower()))
    if not answer_words:
        return GroundednessReport(score=0.0, grounded=False)

    context_words: set[str] = set()
    for piece in context:
        context_words |= set(re.findall(r"\w{4,}", piece.lower()))

    score = len(answer_words & context_words) / len(answer_words)
    return GroundednessReport(score=score, grounded=score >= threshold)
