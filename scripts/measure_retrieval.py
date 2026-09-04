"""Замер поиска на эталонном наборе: попадание в тройку и граница порога.

ЧТО ИМЕННО МЕРЯЕТСЯ

Три подмножества набора отвечают на три разных вопроса, и смешивать их числа
нельзя:

* `corpus` — вопрос корпуса дословно. **Мерой качества не является:** вопрос
  совпадает с проиндексированным текстом, и попадание здесь обязано быть
  стопроцентным. Это проверка, что поиск вообще работает; если она провалилась,
  остальные числа читать бессмысленно.
* `real` — настоящие вопросы абонентов, ответ на которые в корпусе есть.
  **Единственное подмножество, измеряющее качество.**
* `no_answer`, `control` — вопросы, ответа на которые нет. Меряют не поиск, а
  **порог**: обязан ли он отсечь всё.

ПОЧЕМУ ПОРОГ СЧИТАЕТСЯ ЗДЕСЬ, А НЕ ПОДБИРАЕТСЯ ГЛАЗОМ

Действующее значение 0,80 снято на восьми вопросах по делу и трёх посторонних —
это проба, а не измеренная граница, и так и записано в `app/config.py`. Скрипт
проходит порог по сетке и печатает обе цены сразу: сколько ответов теряется
(вопрос по делу остался без контекста) и сколько мусора проходит (посторонний
вопрос получает контекст). Выбирать между ними — решение, а не расчёт, поэтому
скрипт число не назначает, а показывает цену каждого.

ЗАПУСК

    make install-search        # если модель ещё не скачана
    python scripts/build_golden_set.py
    python scripts/measure_retrieval.py

Первый прогон дольше остальных: веса модели скачиваются. Проект четырежды
принимал такую загрузку за время работы — здесь она названа отдельной строкой.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.kb.search import KnowledgeBase, SentenceTransformerEmbedder  # noqa: E402

GOLDEN = ROOT / "golden_set" / "retrieval.json"

SWEEP = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]

MEASURED_SUBSETS = ("corpus", "real")
"""Подмножества, где ожидается попадание. Остальные меряют порог, а не поиск."""


def load_questions() -> list[dict[str, Any]]:
    if not GOLDEN.exists():
        sys.exit(
            f"нет собранного набора: {GOLDEN.relative_to(ROOT)}\n"
            "Собрать: python scripts/build_golden_set.py"
        )
    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return cast("list[dict[str, Any]]", payload["questions"])


def rank_all(kb: KnowledgeBase, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Прогнать каждый вопрос и сохранить полный порядок с оценками.

    Порог здесь намеренно нулевой, а `top_n` — весь корпус: отсев по порогу
    делается потом, на уже посчитанных числах. Иначе сетку порогов пришлось бы
    считать заново на каждом значении, а модель — самая дорогая часть замера.
    """
    results: list[dict[str, Any]] = []
    for question in questions:
        found = kb.search(question["text"], top_n=len(kb), threshold=0.0)
        ranking = [(chunk.chunk_id, chunk.relevance_score) for chunk in found]
        expected = set(question["expected"])
        position = next(
            (i + 1 for i, (chunk_id, _) in enumerate(ranking) if chunk_id in expected),
            None,
        )
        results.append({**question, "ranking": ranking, "position": position})
    return results


def measure_hits(results: list[dict[str, Any]], subset: str) -> dict[str, float]:
    rows = [r for r in results if r["subset"] == subset]
    if not rows:
        return {}
    hit1 = sum(1 for r in rows if r["position"] == 1)
    hit3 = sum(1 for r in rows if r["position"] is not None and r["position"] <= 3)
    mrr = sum(1 / r["position"] for r in rows if r["position"]) / len(rows)
    return {"всего": len(rows), "hit@1": hit1, "hit@3": hit3, "MRR": mrr}


def sweep(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Цена каждого значения порога, в обе стороны.

    Потеря — вопрос по делу, у которого ожидаемый фрагмент есть в тройке, но
    порог отсёк весь контекст: абонент получит отказ вместо ответа.
    Пропуск — вопрос без ответа в корпусе, который порог пропустил: абоненту
    уйдёт чужой фрагмент, а модель ответит по нему уверенно.
    """
    answerable = [
        r for r in results if r["subset"] == "real" and r["position"] is not None
    ]
    unanswerable = [r for r in results if r["subset"] in ("no_answer", "control")]

    table: list[dict[str, Any]] = []
    for threshold in SWEEP:
        lost = sum(1 for r in answerable if r["ranking"][0][1] < threshold)
        # Попадание в тройку с учётом порога: фрагмент обязан и стоять в тройке,
        # и пройти порог — иначе он до промпта не доедет.
        kept = sum(
            1
            for r in answerable
            if r["position"] <= 3 and r["ranking"][r["position"] - 1][1] >= threshold
        )
        leaked = sum(1 for r in unanswerable if r["ranking"][0][1] >= threshold)
        table.append(
            {
                "порог": threshold,
                "ответов сохранено": kept,
                "ответов потеряно": lost,
                "мусора прошло": leaked,
                "всего по делу": len(answerable),
                "всего без ответа": len(unanswerable),
            }
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="куда сохранить числа замера")
    args = parser.parse_args()

    settings = Settings()
    questions = load_questions()

    print(f"модель эмбеддингов: {settings.embedding_model}")
    print(f"вопросов: {len(questions)}")

    started = time.perf_counter()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    embedder.warm_up()
    load_seconds = time.perf_counter() - started
    print(f"загрузка модели: {load_seconds:.1f} с")

    started = time.perf_counter()
    kb = KnowledgeBase.from_file(
        ROOT / settings.kb_corpus_path,
        embedder,
        part_max_chars=settings.kb_part_max_chars,
    )
    index_seconds = time.perf_counter() - started
    print(f"индексация {len(kb)} фрагментов: {index_seconds:.2f} с")

    started = time.perf_counter()
    results = rank_all(kb, questions)
    query_ms = (time.perf_counter() - started) / len(questions) * 1000
    print(f"поиск: {query_ms:.0f} мс на вопрос (медианой не считано, это среднее)")

    print("\n=== Попадание ожидаемого фрагмента (порог не применён) ===")
    print(f"{'подмножество':<12} {'всего':>6} {'hit@1':>8} {'hit@3':>8} {'MRR':>7}")
    for subset in MEASURED_SUBSETS:
        m = measure_hits(results, subset)
        if not m:
            continue
        total = m["всего"]
        print(
            f"{subset:<12} {total:>6} "
            f"{m['hit@1']:>4} ({m['hit@1'] / total:>3.0%}) "
            f"{m['hit@3']:>4} ({m['hit@3'] / total:>3.0%}) "
            f"{m['MRR']:>7.2f}"
        )

    print("\n=== Цена порога ===")
    print(
        f"{'порог':>6} {'ответов сохранено':>18} {'потеряно':>10} {'мусора прошло':>15}"
    )
    table = sweep(results)
    for row in table:
        mark = "  <- сейчас" if abs(row["порог"] - settings.score_threshold) < 1e-9 else ""
        print(
            f"{row['порог']:>6.2f} "
            f"{row['ответов сохранено']:>10} из {row['всего по делу']:<4} "
            f"{row['ответов потеряно']:>9} "
            f"{row['мусора прошло']:>9} из {row['всего без ответа']:<4}{mark}"
        )

    print("\n=== Промахи на настоящих вопросах ===")
    misses = [
        r
        for r in results
        if r["subset"] == "real" and (r["position"] is None or r["position"] > 3)
    ]
    if not misses:
        print("нет")
    for r in misses:
        top = ", ".join(f"{c} {s:.3f}" for c, s in r["ranking"][:3])
        place = r["position"] or "не найден"
        print(f"{r['id']}: ждали {r['expected']}, место {place}")
        print(f"    тройка: {top}")
        print(f"    вопрос: {r['text'][:110]}")

    print("\n=== Что прошло бы порог из вопросов без ответа ===")
    leaks = [
        r
        for r in results
        if r["subset"] in ("no_answer", "control")
        and r["ranking"][0][1] >= settings.score_threshold
    ]
    if not leaks:
        print(f"при пороге {settings.score_threshold} — ничего")
    for r in leaks:
        chunk_id, score = r["ranking"][0]
        print(f"{r['id']} ({r['subset']}): {chunk_id} {score:.3f} — {r['text'][:90]}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "модель": settings.embedding_model,
                    "порог_действующий": settings.score_threshold,
                    "загрузка_с": round(load_seconds, 2),
                    "индексация_с": round(index_seconds, 3),
                    "поиск_мс_среднее": round(query_ms, 1),
                    "попадание": {s: measure_hits(results, s) for s in MEASURED_SUBSETS},
                    "сетка_порогов": table,
                    "вопросы": [
                        {
                            "id": r["id"],
                            "subset": r["subset"],
                            "position": r["position"],
                            "top1": r["ranking"][0],
                        }
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nчисла сохранены: {args.json}")


if __name__ == "__main__":
    main()
