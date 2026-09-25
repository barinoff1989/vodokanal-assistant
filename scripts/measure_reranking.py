"""Офлайн-замер кросс-энкодера на эталонном наборе.

Замер отвечает на **два** вопроса, и второй появился позже первого.

ВОПРОС ПЕРВЫЙ — РАСХОЖДЕНИЕ ТРОЕК

ADR-300 снял переранжирование с пути абонента: кросс-энкодер стоит 5,8 с на
двадцати кандидатах при бюджете поиска 100 мс. Но в самом решении записано, что
довод «на маленьком корпусе реранкер бесполезен» — рассуждение, а не число, и
судьба переранжирования отдана офлайн-замеру: **как часто тройка после
переранжирования отличается от тройки векторного поиска.** Совпадение почти
всегда — переранжирование не нужно и на MVP, и ADR-300 подлежит пересмотру по
существу. Расхождение в заметной доле — вопрос о видеоускорителе становится
вопросом качества, а не удобства.

ВОПРОС ВТОРОЙ — РАЗДЕЛЯЕТ ЛИ ОЦЕНКА ВООБЩЕ

Замер порога показал, что косинусная близость **не
разделяет** вопросы, на которые корпус отвечает, и вопросы, на которые не
отвечает: распределения перекрываются почти целиком, и разделяющего значения в
сетке 0,70…0,90 нет. Причина в природе меры: косинус меряет близость темы, а
нужна пригодность фрагмента для ответа.

Кросс-энкодер видит пару «вопрос — фрагмент» целиком и на эту задачу устроен
иначе. Если его оценка разделяет — у вопроса о пороге появляется
ответ, и он важнее исходного вопроса про тройки.

ЧЕГО ЭТОТ ЗАМЕР НЕ ПОКАЗЫВАЕТ

Корпус прототипа — двадцать фрагментов, и `top_k = 20` означает, что
кросс-энкодер видит **весь корпус**. То есть здесь сравниваются две меры
близости, а не «конвейер с переранжированием против конвейера без него»: на
большом корпусе полнота первой ступени ограничивала бы вторую, и часть выигрыша
исчезла бы. Это ограничение замера, а не свойство моделей.

ЗАПУСК

    python scripts/build_golden_set.py
    python scripts/measure_reranking.py

Идёт минуты: 66 вопросов на 20 фрагментов — 1320 пар, и каждая проходит через
модель целиком. Это офлайн-работа, и медленно здесь ничего не стоит.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.kb.build import collect_items  # noqa: E402
from app.kb.search import KnowledgeBase, SentenceTransformerEmbedder  # noqa: E402

GOLDEN = ROOT / "golden_set" / "retrieval.json"
RERANKER = "BAAI/bge-reranker-v2-m3"

SWEEP = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
"""Сетка для кросс-энкодера своя.

Его оценка — сигмоида и распределена иначе, чем косинусная близость: на пробной
паре «вопрос про поверку» дал 0,995 своему фрагменту и 0,000 чужому. Брать сетку
0,70…0,90 от косинуса значило бы мерить не тот отрезок.
"""


def load_questions() -> list[dict[str, Any]]:
    if not GOLDEN.exists():
        sys.exit(
            f"нет собранного набора: {GOLDEN.relative_to(ROOT)}\n"
            "Собрать: python scripts/build_golden_set.py"
        )
    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return cast("list[dict[str, Any]]", payload["questions"])


def position_of(ranking: list[tuple[str, float]], expected: set[str]) -> int | None:
    return next(
        (i + 1 for i, (chunk_id, _) in enumerate(ranking) if chunk_id in expected), None
    )


def hits(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Попадание ожидаемого фрагмента: первым, в тройке, и обратный ранг."""
    total = len(rows)
    hit1 = sum(1 for r in rows if r[key] == 1)
    hit3 = sum(1 for r in rows if r[key] is not None and r[key] <= 3)
    mrr = sum(1 / r[key] for r in rows if r[key]) / total if total else 0.0
    return {"всего": total, "hit@1": hit1, "hit@3": hit3, "MRR": mrr}


def spread(values: list[float]) -> str:
    if not values:
        return "—"
    return (
        f"{min(values):.3f} … {max(values):.3f}, медиана {statistics.median(values):.3f}"
    )


def frontier(
    real: list[dict[str, Any]], unanswerable: list[dict[str, Any]], key: str
) -> dict[int, int]:
    """Сколько ответов удаётся сохранить при заданном числе пропущенных мимо.

    Сетка порогов у двух мер разная — косинус живёт в 0,75…0,90, сигмоида
    кросс-энкодера в 0,00…0,93, — и сравнивать их по своим сеткам нельзя:
    разница получилась бы свойством шага, а не свойством меры. Поэтому порог
    здесь перебирается по **всем наблюдённым оценкам**, и меры сравниваются в
    точках равной цены: при скольких пропущенных мимо сколько ответов выживает.
    """
    thresholds = sorted({score for r in real + unanswerable for _, score in r[key]})
    best: dict[int, int] = {}
    for threshold in thresholds:
        kept = 0
        for r in real:
            expected = set(r["expected"])
            if any(
                chunk_id in expected and score >= threshold
                for chunk_id, score in r[key][:3]
            ):
                kept += 1
        leaked = sum(1 for r in unanswerable if r[key][0][1] >= threshold)
        best[leaked] = max(best.get(leaked, 0), kept)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="куда сохранить числа замера")
    parser.add_argument("--limit", type=int, help="взять только первые N вопросов")
    parser.add_argument(
        "--model",
        default=RERANKER,
        help=f"кросс-энкодер для замера (по умолчанию {RERANKER})",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="разрешить модели исполнить свой код при загрузке (нужно не всем; включать осознанно)",
    )
    args = parser.parse_args()

    settings = Settings()
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    # Индекс собирается тем же кодом, что и в сервисе: иначе замер мерил бы
    # другую базу знаний. До выделения `app/kb/build.py` здесь стояло чтение
    # одного корпуса FAQ, и документы Word в замер не попадали вовсе.
    kb = KnowledgeBase.from_items(
        collect_items(settings), embedder, part_max_chars=settings.kb_part_max_chars
    )
    print(f"корпус: {len(kb)} фрагментов, вопросов: {len(questions)}")

    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    # dtype=float32 явно: transformers 5 берёт тип из конфигурации модели, и у
    # gte-multilingual-reranker-base это float16 — на CPU он на порядки медленнее.
    import torch

    model = CrossEncoder(
        args.model,
        device="cpu",
        trust_remote_code=args.trust_remote_code,
        model_kwargs={"dtype": torch.float32},
    )
    if args.trust_remote_code:
        from _reranker_compat import fix_gte_buffers

        print(f"буферов пересоздано у модулей: {fix_gte_buffers(model)}")
    print(f"загрузка кросс-энкодера: {time.perf_counter() - started:.1f} с")

    results: list[dict[str, Any]] = []
    pairs_total = 0
    started = time.perf_counter()

    for question in questions:
        # Векторный порядок берётся тем же кодом, что работает на пути абонента:
        # порог нулевой и `top_n` во весь корпус, отсев делается потом.
        found = kb.search(question["text"], top_n=len(kb), threshold=0.0)
        vector_ranking = [(c.chunk_id, c.relevance_score) for c in found]
        candidates = vector_ranking[: settings.vector_top_k]

        texts = {c.chunk_id: c.text for c in found}
        pairs = [(question["text"], texts[chunk_id]) for chunk_id, _ in candidates]
        scores = model.predict(pairs)
        pairs_total += len(pairs)

        scored = zip(candidates, scores, strict=True)
        reranked = sorted(
            ((chunk_id, float(score)) for (chunk_id, _), score in scored),
            key=lambda pair: -pair[1],
        )

        expected = set(question["expected"])
        results.append(
            {
                "id": question["id"],
                "subset": question["subset"],
                "text": question["text"],
                "expected": question["expected"],
                "vector": vector_ranking,
                "reranked": reranked,
                "позиция_вектор": position_of(vector_ranking, expected),
                "позиция_реранк": position_of(reranked, expected),
            }
        )

    elapsed = time.perf_counter() - started
    print(
        f"переранжировано {pairs_total} пар за {elapsed:.0f} с "
        f"({elapsed / len(questions):.1f} с на вопрос, "
        f"{elapsed / pairs_total * 1000:.0f} мс на пару)"
    )

    real = [r for r in results if r["subset"] == "real"]
    corpus = [r for r in results if r["subset"] == "corpus"]
    unanswerable = [r for r in results if r["subset"] in ("no_answer", "control")]

    print("\n=== Вопрос 1: расходятся ли тройки ===")
    for name, rows in (("corpus", corpus), ("real", real), ("без ответа", unanswerable)):
        if not rows:
            continue
        same_set = sum(
            1
            for r in rows
            if {c for c, _ in r["vector"][:3]} == {c for c, _ in r["reranked"][:3]}
        )
        same_order = sum(
            1 for r in rows if [c for c, _ in r["vector"][:3]] == [c for c, _ in r["reranked"][:3]]
        )
        same_first = sum(1 for r in rows if r["vector"][0][0] == r["reranked"][0][0])
        total = len(rows)
        print(
            f"{name:<12} состав тройки совпал {same_set:>3}/{total} ({same_set / total:.0%}), "
            f"порядок {same_order:>3}/{total} ({same_order / total:.0%}), "
            f"первый {same_first:>3}/{total} ({same_first / total:.0%})"
        )

    print("\n=== Меняется ли качество ===")
    print(f"{'подмножество':<12} {'мера':<12} {'hit@1':>10} {'hit@3':>10} {'MRR':>7}")
    for name, rows in (("corpus", corpus), ("real", real)):
        if not rows:
            continue
        for label, key in (("косинус", "позиция_вектор"), ("кросс-энк.", "позиция_реранк")):
            m = hits(rows, key)
            count = m["всего"]
            print(
                f"{name:<12} {label:<12} "
                f"{m['hit@1']:>3.0f} ({m['hit@1'] / count:>3.0%}) "
                f"{m['hit@3']:>3.0f} ({m['hit@3'] / count:>3.0%}) "
                f"{m['MRR']:>7.2f}"
            )

    print("\n=== Вопрос 2: разделяет ли оценка ===")
    answerable_top = [r["reranked"][0][1] for r in real]
    unanswerable_top = [r["reranked"][0][1] for r in unanswerable]
    print(f"с ответом в корпусе ({len(answerable_top):>2}): {spread(answerable_top)}")
    print(f"без ответа        ({len(unanswerable_top):>2}): {spread(unanswerable_top)}")
    if answerable_top and unanswerable_top:
        overlap = sum(1 for x in unanswerable_top if x >= min(answerable_top))
        print(
            f"перекрытие: у {overlap} из {len(unanswerable_top)} без ответа оценка "
            f"не ниже самой слабой среди вопросов с ответом"
        )

    print("\n=== Цена порога на оценке кросс-энкодера ===")
    print(f"{'порог':>6} {'ответов сохранено':>20} {'мусора прошло':>18}")
    table: list[dict[str, Any]] = []
    reachable = [r for r in real if r["позиция_реранк"] is not None]
    for threshold in SWEEP:
        kept = sum(
            1
            for r in reachable
            if r["позиция_реранк"] <= 3
            and r["reranked"][r["позиция_реранк"] - 1][1] >= threshold
        )
        leaked = sum(1 for r in unanswerable if r["reranked"][0][1] >= threshold)
        table.append(
            {
                "порог": threshold,
                "ответов сохранено": kept,
                "мусора прошло": leaked,
                "всего по делу": len(reachable),
                "всего без ответа": len(unanswerable),
            }
        )
        print(
            f"{threshold:>6.2f} {kept:>12} из {len(reachable):<4} "
            f"{leaked:>12} из {len(unanswerable):<4}"
        )

    print(chr(10) + "=== Две меры в точках равной цены ===")
    print("Порог перебран по всем наблюдённым оценкам: у мер разные шкалы.")
    cos_front = frontier(real, unanswerable, "vector")
    ce_front = frontier(real, unanswerable, "reranked")
    print(f"{'пропущено мимо':>16} {'косинус':>14} {'кросс-энкодер':>18}")
    for leaked in range(0, min(6, len(unanswerable) + 1)):
        a = max((v for k, v in cos_front.items() if k <= leaked), default=0)
        b = max((v for k, v in ce_front.items() if k <= leaked), default=0)
        print(f"{leaked:>16} {a:>8} из {len(real):<4} {b:>12} из {len(real):<4}")

    print("\n=== Что переранжирование исправило и что сломало ===")
    for r in real:
        before, after = r["позиция_вектор"], r["позиция_реранк"]
        was_hit = before is not None and before <= 3
        now_hit = after is not None and after <= 3
        if was_hit == now_hit:
            continue
        mark = "исправлено" if now_hit else "СЛОМАНО"
        print(f"{mark}: {r['id']} место {before} -> {after}, ждали {r['expected']}")
        print(f"    {r['text'][:100]}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "кросс_энкодер": args.model,
                    "эмбеддинги": settings.embedding_model,
                    "пар": pairs_total,
                    "секунд": round(elapsed, 1),
                    "мс_на_пару": round(elapsed / pairs_total * 1000, 1),
                    "качество": {
                        name: {
                            "косинус": hits(rows, "позиция_вектор"),
                            "кросс_энкодер": hits(rows, "позиция_реранк"),
                        }
                        for name, rows in (("corpus", corpus), ("real", real))
                        if rows
                    },
                    "сетка_порогов": table,
                    "вопросы": [
                        {
                            "id": r["id"],
                            "subset": r["subset"],
                            "позиция_вектор": r["позиция_вектор"],
                            "позиция_реранк": r["позиция_реранк"],
                            "топ1_вектор": r["vector"][0],
                            "топ1_реранк": r["reranked"][0],
                            # Полный порядок сохраняется намеренно: прогон стоит
                            # двадцать минут, и любой следующий вопрос к этим
                            # числам не должен требовать повторного прогона.
                            # Текстов здесь нет — только опознаватели и оценки.
                            "порядок_реранк": r["reranked"],
                            "порядок_вектор": r["vector"],
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
