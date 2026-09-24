"""Ночной прогон судьи качества по эталонному набору.

ЧТО ДЕЛАЕТ. Прогоняет настоящие вопросы эталонного набора (подмножество `real` —
единственное, что измеряет качество, см. `scripts/measure_retrieval.py`) через
живой путь: поиск по базе знаний → генерация ответа шлюзом → оценка судьёй.
Пишет СРЕДНИЕ по прогону и долю ответов выше стартовых порогов
в таблицу `quality_reports` базы `telemetry` — это `quality_report`
(Postgres).

СУДЬЯ ≠ ОТВЕЧАЮЩАЯ МОДЕЛЬ (ADR-300). Если псевдоним судьи
(`JUDGE_PROVIDER`) совпал с отвечающим (`LLM_PROVIDER`) — а это и есть failover
генерации на `local-test` — прогон не выполняется: пишется строка со
`skipped = true` и причиной, а не отсутствие записи (ADR-300).

ЗАПУСК

    make install-search                 # если модель поиска не скачана
    ollama serve                        # судья — local-test через Ollama
    python scripts/build_golden_set.py
    python scripts/run_quality_eval.py

Прогон живой: нужны и модель-судья (Ollama), и провайдер генерации
(`LLM_PROVIDER` + его ключи). Без них скрипт выходит с указанием, чего не хватает,
а не молча пишет нули.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.gateway.llm_gateway import LlmGateway  # noqa: E402
from app.kb.build import collect_items  # noqa: E402
from app.kb.search import KnowledgeBase, SentenceTransformerEmbedder  # noqa: E402
from app.models import (  # noqa: E402
    Channel,
    GenerateRequest,
    GenerationParameters,
    ProblemDetail,
    RequestMetadata,
)
from app.quality import (  # noqa: E402
    AnswerJudge,
    JudgeConflict,
    JudgeError,
    QualityReport,
    QualityScores,
    QualityStore,
)

GOLDEN = ROOT / "golden_set" / "retrieval.json"

SYSTEM_PROMPT = (
    "Ты помощник абонента водоканала. Отвечай кратко, по-русски, только по теме "
    "жилищно-коммунальных услуг водоснабжения. Не выдумывай факты."
)
"""Тот же системный промпт, что подаёт стенд (`web/app.js`): прогон обязан
мерить тот же путь, которым идёт абонент, а не свой."""

# Тип функции генерации: вопрос + контекст -> (ответ, псевдоним фактической модели).
Generator = Callable[[str, Sequence[str]], Awaitable[tuple[str, str]]]


def load_questions(subset: str = "real") -> list[dict[str, Any]]:
    if not GOLDEN.exists():
        sys.exit(
            f"нет собранного набора: {GOLDEN.relative_to(ROOT)}\n"
            "Собрать: python scripts/build_golden_set.py"
        )
    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    questions = cast("list[dict[str, Any]]", payload["questions"])
    return [q for q in questions if q["subset"] == subset]


def golden_set_version() -> str:
    """Короткий хеш содержимого набора: сравнивать прогоны можно только на одной
    версии набора, а поля версии в файле нет."""
    digest = hashlib.sha1(GOLDEN.read_bytes()).hexdigest()  # noqa: S324 — не крипто
    return digest[:12]


def _gateway_generator(gateway: LlmGateway, settings: Settings) -> Generator:
    """Генерация ответа живым шлюзом — тот же путь, что у абонента."""

    async def generate(question: str, context: Sequence[str]) -> tuple[str, str]:
        from app.models import ContextChunk

        request = GenerateRequest(
            system=SYSTEM_PROMPT,
            query=question,
            context=[
                ContextChunk(
                    chunk_id=f"golden-{i}",
                    text=piece,
                    source_title="эталонный набор",
                    relevance_score=0.9,
                )
                for i, piece in enumerate(context)
            ],
            parameters=GenerationParameters(stream=False),
            metadata=RequestMetadata(
                subscriber_id="quality-eval",
                session_id="quality-eval",
                channel=Channel.LK_WEB,
            ),
        )
        response = await gateway.generate(request)
        if isinstance(response, ProblemDetail):
            raise JudgeError(f"генерация не удалась: {response.title}")
        # Псевдоним фактической модели: при failover в routing["model"] окажется
        # имя локальной модели, хотя запрошен был другой провайдер.
        actual_model = str(response.routing.get("model", ""))
        alias = settings.llm_provider
        if settings.llm_provider != "local-test" and settings.local_model in actual_model:
            alias = "local-test"
        return response.answer, alias

    return generate


async def evaluate(
    questions: list[dict[str, Any]],
    kb: KnowledgeBase,
    generate: Generator,
    judge: AnswerJudge,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Прогнать каждый вопрос: поиск → генерация → судья."""
    rows: list[dict[str, Any]] = []
    for question in questions:
        text = question["text"]
        chunks = kb.search(
            text, top_n=settings.rerank_top_n, threshold=settings.score_threshold
        )
        context = [c.text for c in chunks]

        row: dict[str, Any] = {"id": question["id"], "text": text, "context_n": len(context)}
        try:
            answer, answering_alias = await generate(text, context)
            row["answering_alias"] = answering_alias
            scores = await judge.score(
                question=text,
                answer=answer,
                context=context,
                answering_alias=answering_alias,
            )
            row["scores"] = scores
        except JudgeConflict as exc:
            row["conflict"] = str(exc)
        except JudgeError as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


def aggregate(
    rows: list[dict[str, Any]], settings: Settings, *, version: str, judge_alias: str
) -> QualityReport:
    scored = [cast("QualityScores", r["scores"]) for r in rows if "scores" in r]
    answering = next(
        (r["answering_alias"] for r in rows if r.get("answering_alias")), ""
    )

    # Прогон целиком пропущен: ни одного ответа не оценено из-за совпадения
    # судьи с отвечающей моделью (failover генерации).
    conflicts = [r for r in rows if "conflict" in r]
    if not scored and conflicts:
        return QualityReport(
            judge_alias=judge_alias,
            answering_alias=answering,
            golden_set_version=version,
            questions_total=len(rows),
            questions_scored=0,
            skipped=True,
            skip_reason=conflicts[0]["conflict"],
        )

    if not scored:
        return QualityReport(
            judge_alias=judge_alias,
            answering_alias=answering,
            golden_set_version=version,
            questions_total=len(rows),
            questions_scored=0,
        )

    faithfulness = sum(s.faithfulness for s in scored) / len(scored)
    relevancy = sum(s.answer_relevancy for s in scored) / len(scored)
    passed = sum(
        1
        for s in scored
        if s.passes(
            faithfulness_min=settings.judge_faithfulness_min,
            relevancy_min=settings.judge_relevancy_min,
        )
    )
    return QualityReport(
        judge_alias=judge_alias,
        judge_model=scored[0].judge_model,
        answering_alias=answering,
        golden_set_version=version,
        questions_total=len(rows),
        questions_scored=len(scored),
        faithfulness_avg=round(faithfulness, 4),
        answer_relevancy_avg=round(relevancy, 4),
        pass_rate=round(passed / len(scored), 4),
    )


def _print_report(report: QualityReport, rows: list[dict[str, Any]]) -> None:
    print(f"\n=== Прогон судьи: {report.judge_alias} ({report.judge_model or '—'}) ===")
    print(f"версия набора: {report.golden_set_version}")
    print(f"отвечал: {report.answering_alias or '—'}")
    if report.skipped:
        print(f"ПРОПУЩЕН: {report.skip_reason}")
        return
    print(f"вопросов: {report.questions_total}, оценено: {report.questions_scored}")
    if report.faithfulness_avg is None or report.answer_relevancy_avg is None:
        print("судья не вернул ни одной оценки")
    else:
        print(
            f"faithfulness (среднее):     {report.faithfulness_avg:.3f}  "
            f"({_mark(report.faithfulness_avg, 0.85)})"
        )
        print(
            f"answer_relevancy (среднее): {report.answer_relevancy_avg:.3f}  "
            f"({_mark(report.answer_relevancy_avg, 0.80)})"
        )
        if report.pass_rate is not None:
            print(f"доля выше обоих порогов:    {report.pass_rate:.0%}")

    misses = [
        r
        for r in rows
        if "scores" in r
        and not cast("QualityScores", r["scores"]).passes(
            faithfulness_min=0.85, relevancy_min=0.80
        )
    ]
    if misses:
        print("\n--- ниже порога ---")
        for r in misses:
            s = cast("QualityScores", r["scores"])
            print(f"{r['id']}: f={s.faithfulness:.2f} r={s.answer_relevancy:.2f}"
                  f" ctx={r['context_n']} — {s.reasoning[:90]}")
    errors = [r for r in rows if "error" in r]
    for r in errors:
        print(f"{r['id']}: ошибка — {r['error']}")


def _mark(value: float, threshold: float) -> str:
    return "ok" if value >= threshold else "ниже порога"


async def _amain(args: argparse.Namespace) -> int:
    settings = Settings()
    judge_alias = settings.judge_provider
    version = golden_set_version()

    store: QualityStore
    if args.no_store:
        store = QualityStore(None)
    else:
        try:
            import psycopg

            store = QualityStore(lambda: psycopg.connect(settings.telemetry_dsn))
            store.ensure_schema()
        except Exception as exc:  # noqa: BLE001 — базы/драйвера может не быть
            print(f"запись отчёта выключена: {exc}")
            store = QualityStore(None)

    # Совпадение на уровне конфигурации: писать skipped и выходить, модель не
    # трогая. Это тот же случай, что failover, но виден заранее.
    if settings.llm_provider == judge_alias:
        report = QualityReport(
            judge_alias=judge_alias,
            answering_alias=settings.llm_provider,
            golden_set_version=version,
            questions_total=len(load_questions()),
            skipped=True,
            skip_reason=(
                f"судья и отвечающая модель совпали ({judge_alias}): "
                "оценка при failover генерации не выполняется (ADR-300)"
            ),
        )
        store.record(report)
        _print_report(report, [])
        return 0

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    print(f"вопросов на прогон: {len(questions)}")

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    embedder.warm_up()
    kb = KnowledgeBase.from_items(
        collect_items(settings), embedder, part_max_chars=settings.kb_part_max_chars
    )
    print(f"индекс: {len(kb)} фрагментов")

    gateway = LlmGateway(settings=settings)
    gateway.warmup()
    judge = AnswerJudge(settings=settings)

    rows = await evaluate(
        questions, kb, _gateway_generator(gateway, settings), judge, settings
    )
    report = aggregate(rows, settings, version=version, judge_alias=judge_alias)
    store.record(report)
    _print_report(report, rows)

    if args.json:
        Path(args.json).write_text(
            json.dumps(_report_dict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nчисла сохранены: {args.json}")

    if report.pass_rate is not None and report.pass_rate < args.min_pass_rate:
        print(f"\nдоля {report.pass_rate:.0%} ниже требуемой {args.min_pass_rate:.0%}")
        return 1
    return 0


def _report_dict(report: QualityReport) -> dict[str, Any]:
    return {
        "judge_alias": report.judge_alias,
        "judge_model": report.judge_model,
        "answering_alias": report.answering_alias,
        "golden_set_version": report.golden_set_version,
        "questions_total": report.questions_total,
        "questions_scored": report.questions_scored,
        "faithfulness_avg": report.faithfulness_avg,
        "answer_relevancy_avg": report.answer_relevancy_avg,
        "pass_rate": report.pass_rate,
        "skipped": report.skipped,
        "skip_reason": report.skip_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="взять первые N вопросов")
    parser.add_argument("--json", type=Path, help="куда сохранить числа прогона")
    parser.add_argument(
        "--no-store", action="store_true", help="не писать отчёт в Postgres"
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="выйти с ошибкой, если доля прошедших пороги ниже этого (для CI)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
