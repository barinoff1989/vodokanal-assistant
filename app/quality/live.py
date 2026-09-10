"""Судья на живом пути — async-часть Guardrails диаграммы C4_L3_LLM.

ЧТО ЭТО. После того как поток ответа дошёл до абонента (правило 4.3, оценка не
блокирует выдачу), судья оценивает ответ: следует ли он из найденного контекста
(faithfulness) и отвечает ли на вопрос (answer_relevancy, раздел 36.1). Итог —
строка в `quality_assessments` для разреза качества в реальном времени.

ПОЧЕМУ ОЦЕНКА ПРЕДВАРИТЕЛЬНАЯ. На прототипе судья — локальный Qwen-7B, который
слабо различает (пункт 82 TODO). Каждая строка помечена `provisional=True`:
конвейер работает и виден дашборду, но число — ориентир, не гейт. На MVP судья —
YandexGPT против отвечающего GigaChat, и флаг снимается.

ПОЧЕМУ НЕ КАЖДЫЙ ОТВЕТ. Вызов судьи — это вызов модели. GPU на прототипе нет, и
под нагрузкой судья конкурирует за процессор с живыми запросами. `sample_rate`
задаёт долю оцениваемых ответов; `1.0` годится для демонстрации с одним
абонентом.

ПОЧЕМУ ТОЛЬКО RAG-ПУТЬ. Судья зовётся из `LlmGateway`, а туда запрос попадает
только когда модель действительно вызвана и есть контекст. Прямые ответы (пять
путей без модели) оценивать нечем и незачем.

ОШИБКА СУДЬИ — СТРОКА, НЕ МОЛЧАНИЕ. Судья не ответил (`unavailable`) или совпал с
отвечающей моделью при failover (`conflict`) — пишется строка с этим исходом и
без оценок, а не отсутствие записи. Та же дисциплина, что у ночного прогона
(ADR-007, «Подтверждение»).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

from app.models import GenerateRequest
from app.quality.judge import AnswerJudge, JudgeConflict, JudgeUnavailable
from app.quality.store import AssessmentStore, QualityAssessment

__all__ = ["LiveJudge"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveJudge:
    """Оценка живого ответа судьёй и запись в `quality_assessments`.

    :param judge: собственно судья (`AnswerJudge`).
    :param store: куда писать оценку.
    :param sample_rate: доля ответов, которые оцениваются, [0, 1].
    :param answering_alias: псевдоним отвечающей модели — для проверки «судья
        совпал с отвечающей» (ADR-007). Обычно `settings.llm_provider`.
    """

    judge: AnswerJudge
    store: AssessmentStore
    sample_rate: float
    answering_alias: str

    def would_sample(self) -> bool:
        """Попал ли следующий ответ в выборку. Вызывается до создания задачи."""
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() < self.sample_rate

    async def assess(
        self,
        request: GenerateRequest,
        answer: str,
        trace_id: str,
        *,
        context: Sequence[str],
    ) -> None:
        """Оценить ответ и записать строку. Гасит всё: оценка не роняет ничего.

        Зовётся фоновой задачей после стрима — во время до первого куска ответа
        эта работа не входит.
        """
        meta = request.metadata

        def row(outcome: str, **extra: object) -> QualityAssessment:
            return QualityAssessment(
                trace_id=trace_id,
                subscriber_id=meta.subscriber_id,
                session_id=meta.session_id,
                topic=meta.topic.value if meta.topic else None,
                inquiry_type=meta.inquiry_type.value if meta.inquiry_type else None,
                outcome=outcome,
                **extra,  # type: ignore[arg-type]
            )

        try:
            scores = await self.judge.score(
                question=request.query,
                answer=answer,
                context=context,
                answering_alias=self.answering_alias,
            )
        except JudgeConflict as exc:
            self.store.record(row("conflict", detail=str(exc)))
            return
        except JudgeUnavailable as exc:
            self.store.record(row("unavailable", detail=str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 — оценка не роняет ответ абоненту
            logger.warning("судья на живом пути не отработал (%s)", exc)
            return

        self.store.record(row(
            "scored",
            judge_model=scores.judge_model,
            faithfulness=scores.faithfulness,
            answer_relevancy=scores.answer_relevancy,
            provisional=True,
            detail=(scores.reasoning[:500] or None),
        ))
