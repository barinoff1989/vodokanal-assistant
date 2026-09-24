"""Оценка качества ответа — судья (LLM-judge) и ночной прогон по эталонному набору.

Здесь живёт то, что называют
судьёй: отдельная от отвечающей модель, которая оценивает готовый ответ по
контексту — обоснованность (faithfulness) и релевантность вопросу (answer
relevancy). Обе метрики reference-free: эталонный ответ не нужен, только вопрос,
контекст и ответ.

Оценка выполняется **вне пути абонента** — ночным прогоном (`scripts/
run_quality_eval.py`), а не на каждом ответе: локальный судья на процессоре
стоит секунды, и ставить его в поток означало бы нарушить бюджет ответа.
"""

from app.quality.judge import (
    AnswerJudge,
    JudgeConflict,
    JudgeError,
    JudgeUnavailable,
    QualityScores,
)
from app.quality.store import QualityReport, QualityStore

__all__ = [
    "AnswerJudge",
    "JudgeConflict",
    "JudgeError",
    "JudgeUnavailable",
    "QualityReport",
    "QualityScores",
    "QualityStore",
]
