"""Замер порога и разрыва запасного классификатора InquiryType на живых данных.

Тот же смысл, что у `scripts/measure_classifier.py`, но не про ключевые слова,
а про эмбеддинги (`app/agents/inquiry_type_fallback.py`): для каждого
настоящего текста семи регистрируемых типов (`fixtures/inquiries_voronezh.csv`)
считается ближайший пример из `EXAMPLES` без применения порога/разрыва — и
печатается распределение оценок отдельно для верных и неверных совпадений.

Число одно не разделяет верное от неверного (та же находка, что ADR-014
сделал для порога поиска) — разделяет **разрыв** до второго (другого) типа.
Итоговые `TYPE_MATCH_THRESHOLD`/`TYPE_MATCH_MARGIN` в модуле подобраны по
выводу этого скрипта; при правке `EXAMPLES` — перезапустить и свериться.

ЗАПУСК

    python scripts/measure_inquiry_type_fallback.py
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.inquiry_type_fallback import EXAMPLES  # noqa: E402
from app.backend.registration import REGISTRABLE  # noqa: E402
from app.kb.search import (  # noqa: E402
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    SentenceTransformerEmbedder,
    cosine,
)
from app.taxonomy import DISPLAY_NAMES  # noqa: E402

FIXTURE = ROOT / "fixtures" / "inquiries_voronezh.csv"


def load_texts_by_type() -> dict[str, list[str]]:
    """Настоящие тексты по семи регистрируемым типам, остальные девять — мимо
    (раздел заголовка `inquiry_type_fallback.py` — они не про регистрацию)."""
    reverse = {name.strip().lower(): code for code, name in DISPLAY_NAMES.items()}
    by_type: dict[str, list[str]] = {}
    with FIXTURE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            code = reverse.get(row["bname"].strip().lower())
            if code in REGISTRABLE:
                by_type.setdefault(code.value, []).append(row["text_request"].strip())
    return by_type


async def measure() -> None:
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    embedder.warm_up()
    example_vectors = embedder.encode([PASSAGE_PREFIX + e.text for e in EXAMPLES])

    correct_scores: list[float] = []
    correct_margins: list[float] = []
    wrong_scores: list[float] = []
    wrong_margins: list[float] = []
    total = 0

    for expected, texts in load_texts_by_type().items():
        for text in texts:
            total += 1
            (vector,) = embedder.encode([QUERY_PREFIX + text])
            scored = sorted(
                (
                    (cosine(vector, ev), e.inquiry_type.value)
                    for e, ev in zip(EXAMPLES, example_vectors, strict=True)
                ),
                key=lambda pair: -pair[0],
            )
            best_score, best_type = scored[0]
            runner_up = next((s for s, t in scored[1:] if t != best_type), 0.0)
            margin = best_score - runner_up
            if best_type == expected:
                correct_scores.append(best_score)
                correct_margins.append(margin)
            else:
                wrong_scores.append(best_score)
                wrong_margins.append(margin)

    print(f"Всего текстов: {total} (только семь регистрируемых типов)")
    print()
    print(f"Верные ближайшие совпадения: {len(correct_scores)}")
    if correct_scores:
        print(f"  оценка близости: {min(correct_scores):.4f} .. {max(correct_scores):.4f}")
        print(
            f"  разрыв до другого типа:  "
            f"{min(correct_margins):.4f} .. {max(correct_margins):.4f}"
        )
    print()
    print(f"Неверные ближайшие совпадения: {len(wrong_scores)}")
    if wrong_scores:
        print(f"  оценка близости: {min(wrong_scores):.4f} .. {max(wrong_scores):.4f}")
        print(f"  разрыв до другого типа:  {min(wrong_margins):.4f} .. {max(wrong_margins):.4f}")
    print()
    print(
        "Порог должен лежать не выше минимума верных; разрыв — не ниже "
        "максимума неверных. Числа для TYPE_MATCH_THRESHOLD/TYPE_MATCH_MARGIN "
        "в app/agents/inquiry_type_fallback.py подбираются отсюда, а не наоборот."
    )


if __name__ == "__main__":
    asyncio.run(measure())
