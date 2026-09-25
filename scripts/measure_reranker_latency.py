"""Замер задержки кросс-энкодера на CPU: время переранжирования 20 кандидатов.

ЗАЧЕМ. Бюджет поиска — 100 мс на весь шаг, а `BGE-reranker-v2-m3` на CPU стоит
секунды (ADR-300). Скрипт отвечает на вопрос «какая из малых моделей и какие
приёмы (короче текст, меньше кандидатов, INT8) приближают шаг к бюджету» — на
тех же фрагментах базы знаний, что видит служба, а не на синтетике.

ЧТО МЕРЯЕТСЯ. Один вызов `predict` на пачку из N пар «вопрос — фрагмент» = один
запрос абонента. Модель прогрета (два холостых вызова), потоки torch — по числу
ядер, остальная нагрузка на машине не запущена. Разные вопросы дают разную длину
пар, поэтому берётся несколько вопросов и печатается медиана и максимум.

ВАРИАНТЫ. `full` — как сейчас (длина по умолчанию модели, 20 кандидатов);
`len256` — пары обрезаются до 256 токенов; `top10` — 10 кандидатов и 256 токенов;
`int8` — динамическая квантизация линейных слоёв (torch, без ONNX) при полной длине
и `int8+len256`.

ЗАПУСК
    python scripts/measure_reranker_latency.py --json golden_set/reranker_latency.json
    python scripts/measure_reranker_latency.py --models cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

Модели должны лежать в локальном кэше Hugging Face (скачиваются отдельно).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.kb.build import collect_items  # noqa: E402
from app.kb.search import KnowledgeBase, SentenceTransformerEmbedder  # noqa: E402

GOLDEN = ROOT / "golden_set" / "retrieval.json"
DEFAULT_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "DiTy/cross-encoder-russian-msmarco",
    "Alibaba-NLP/gte-multilingual-reranker-base",
]
REMOTE_CODE = {"Alibaba-NLP/gte-multilingual-reranker-base"}


def build_batches(n_questions: int, top: int) -> list[list[tuple[str, str]]]:
    """Пары для замера: реальные вопросы и их кандидаты векторного поиска."""
    settings = Settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    kb = KnowledgeBase.from_items(
        collect_items(settings), embedder, part_max_chars=settings.kb_part_max_chars
    )
    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    real = [q for q in payload["questions"] if q["subset"] == "real"][:n_questions]
    batches = []
    for q in real:
        found = kb.search(q["text"], top_n=len(kb), threshold=0.0)[:top]
        batches.append([(q["text"], c.text) for c in found])
    print(f"корпус: {len(kb)} фрагментов; вопросов для замера: {len(batches)}; кандидатов: {top}")
    return batches


def time_predict(model: Any, batches: list[list[tuple[str, str]]], n: int) -> list[float]:
    for warm in batches[:2]:
        model.predict(warm[:n], batch_size=n, show_progress_bar=False)
    out = []
    for b in batches:
        pairs = b[:n]
        started = time.perf_counter()
        model.predict(pairs, batch_size=n, show_progress_bar=False)
        out.append(time.perf_counter() - started)
    return out


def token_stats(
    model: Any, batches: list[list[tuple[str, str]]], max_len: int | None
) -> dict[str, float]:
    tok = model.tokenizer
    lens = []
    for b in batches:
        enc = tok(
            [q for q, _ in b],
            [t for _, t in b],
            truncation=True,
            max_length=max_len or model.max_length,
            padding=False,
        )
        lens.extend(len(x) for x in enc["input_ids"])
    return {"среднее": round(statistics.mean(lens), 0), "максимум": max(lens)}


def variant(
    rows: dict[str, Any],
    batches: list[list[tuple[str, str]]],
    model: Any,
    default_len: int,
    label: str,
    n: int,
    max_len: int | None,
) -> None:
    """Замерить один вариант (число кандидатов, длина) и записать в `rows`."""
    model.max_length = max_len or default_len
    times = time_predict(model, batches, n)
    toks = token_stats(model, [b[:n] for b in batches], max_len)
    row = {
        "кандидатов": n,
        "max_length": model.max_length,
        "токенов_на_пару": toks,
        "медиана_мс": round(statistics.median(times) * 1000),
        "максимум_мс": round(max(times) * 1000),
    }
    rows[label] = row
    tokens = f"токены ср.{toks['среднее']:.0f}/макс.{toks['максимум']}"
    print(
        f"  {label:<14} n={n:<3} len={model.max_length:<5} {tokens}  "
        f"медиана {row['медиана_мс']} мс, максимум {row['максимум_мс']} мс"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--questions", type=int, default=8, help="сколько вопросов брать в замер")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    print(f"потоки torch: {torch.get_num_threads()}")
    batches20 = build_batches(args.questions, 20)
    result: dict[str, Any] = {
        "потоки_torch": torch.get_num_threads(),
        "вопросов": len(batches20),
        "модели": {},
    }

    for name in args.models:
        print(f"\n=== {name}")
        try:
            # dtype=float32 явно: transformers 5 берёт тип из конфигурации модели, у
            # gte-multilingual-reranker-base это float16 — на CPU на порядки медленнее.
            model = CrossEncoder(
                name,
                device="cpu",
                trust_remote_code=name in REMOTE_CODE,
                model_kwargs={"dtype": torch.float32},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  не загрузилась: {exc}")
            result["модели"][name] = {"ошибка": str(exc)[:300]}
            continue
        if name in REMOTE_CODE:
            from _reranker_compat import fix_gte_buffers

            print(f"  буферов пересоздано у модулей: {fix_gte_buffers(model)}")
        default_len = model.max_length
        rows: dict[str, Any] = {"max_length_по_умолчанию": default_len}

        run = partial(variant, rows, batches20, model, default_len)
        run("full", 20, None)
        run("len256", 20, 256)
        run("top10+len256", 10, 256)

        # На месте: подмена `model.model` ломает разбор входа в sentence-transformers 6.
        torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
        run("int8", 20, None)
        run("int8+len256", 20, 256)
        run("int8+top10+len256", 10, 256)
        result["модели"][name] = rows
        del model

    if args.json:
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nчисла сохранены: {args.json}")


if __name__ == "__main__":
    main()
