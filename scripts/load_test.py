"""Нагрузочный замер Backend: сколько запросов в секунду и какая задержка.

Клиент замкнутого цикла: N параллельных «абонентов», каждый шлёт следующий вопрос,
когда получил ответ на предыдущий. Для каждого уровня параллелизма (N) печатается
пропускная способность (успешных ответов в секунду), медианы и хвосты задержки, число
ошибок. Ответы читаются потоком SSE, как это делает виджет; фиксируются время до
первого куска ответа и время до конца.

СЦЕНАРИИ
* `direct`  — вопрос, на который отвечает прямой ответчик без модели (тарифы);
* `rag`     — вопрос через поиск по базе знаний и вызов модели (сервер
              `load_server.py` подменяет провайдера заглушкой, чтобы мерить свою
              задержку, а не чужую).

ЗАПУСК
    python scripts/load_server.py --port 8100          # в первом окне
    python scripts/load_test.py --url http://127.0.0.1:8100 --json golden_set/load_test.json

Клиент и сервер работают на одной машине и делят процессор: это учитывается в отчёте.
Если указан `--server-pid`, для каждого уровня дополнительно считается процессорное время
процесса сервера на один запрос и его память (нужен пакет `psutil`).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

SCENARIOS: dict[str, str] = {
    "direct": "Какие сейчас тарифы на холодную воду?",
    "rag": "Почему начисляются пени?",
}
_counter = itertools.count()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


async def one_request(client: httpx.AsyncClient, url: str, query: str) -> tuple[bool, float, float]:
    """Один вопрос потоком: (успех, секунд до первого куска, секунд до конца)."""
    n = next(_counter)
    body = {
        "system": "помощник абонента водоканала",
        "query": query,
        "parameters": {"stream": True},
        "metadata": {"subscriber_id": f"load-{n % 5000}", "session_id": f"load-{n}"},
    }
    started = time.perf_counter()
    first = 0.0
    try:
        async with client.stream("POST", url + "/v1/generate", json=body) as response:
            if response.status_code != 200:
                await response.aread()
                return False, 0.0, time.perf_counter() - started
            async for line in response.aiter_lines():
                if not first and line.startswith("event: token"):
                    first = time.perf_counter() - started
        return True, first, time.perf_counter() - started
    except httpx.HTTPError:
        return False, 0.0, time.perf_counter() - started


async def run_level(
    url: str, query: str, concurrency: int, duration: float, warmup: float, server: Any = None
) -> dict[str, Any]:
    limits = httpx.Limits(
        max_connections=concurrency + 2, max_keepalive_connections=concurrency + 2
    )
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        stop_warm = time.perf_counter() + warmup
        stop = stop_warm + duration
        results: list[tuple[bool, float, float]] = []
        cpu: list[float] = []

        async def sample_cpu() -> None:
            """Процессорное время сервера на границах окна замера."""
            for moment in (stop_warm, stop):
                await asyncio.sleep(max(0.0, moment - time.perf_counter()))
                times = server.cpu_times()
                cpu.append(times.user + times.system)

        async def worker() -> None:
            while time.perf_counter() < stop:
                started = time.perf_counter()
                result = await one_request(client, url, query)
                if started >= stop_warm:
                    results.append(result)

        began = time.perf_counter()
        sampler = [sample_cpu()] if server is not None else []
        await asyncio.gather(*(worker() for _ in range(concurrency)), *sampler)
        wall = time.perf_counter() - max(began, stop_warm)

    ok = [r for r in results if r[0]]
    firsts = [r[1] * 1000 for r in ok if r[1]]
    totals = [r[2] * 1000 for r in ok]
    extra: dict[str, Any] = {}
    if server is not None and len(cpu) == 2 and results:
        extra = {
            "cpu_сервера_с_на_запрос": round((cpu[1] - cpu[0]) / len(results), 3),
            "cpu_сервера_ядер": round((cpu[1] - cpu[0]) / duration, 2),
            "память_сервера_мб": round(server.memory_info().rss / 1e6),
        }
    return {
        **extra,
        "параллельно": concurrency,
        "запросов": len(results),
        "ошибок": len(results) - len(ok),
        "rps": round(len(ok) / duration, 1),
        "до_первого_куска_мс": {
            "p50": round(percentile(firsts, 50)),
            "p95": round(percentile(firsts, 95)),
            "p99": round(percentile(firsts, 99)),
        },
        "до_конца_мс": {
            "p50": round(percentile(totals, 50)),
            "p95": round(percentile(totals, 95)),
            "p99": round(percentile(totals, 99)),
            "среднее": round(statistics.mean(totals)) if totals else 0,
        },
        "окно_с": round(wall, 1),
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    server = None
    if args.server_pid:
        import psutil

        server = psutil.Process(args.server_pid)
    result: dict[str, Any] = {
        "url": args.url,
        "длительность_уровня_с": args.duration,
        "прогрев_с": args.warmup,
        "сценарии": {},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        for query in SCENARIOS.values():  # холодный старт: прогреть ленивые загрузки
            for _ in range(3):
                await one_request(client, args.url, query)
    for name in args.scenarios:
        query = SCENARIOS[name]
        rows = []
        print(f"\n=== сценарий {name}: «{query}»")
        head = (
            "параллельно",
            "RPS",
            "до 1-го куска p50/p95/p99, мс",
            "до конца p50/p95/p99, мс",
            "ошибок",
        )
        print(f"{head[0]:>11} {head[1]:>7} {head[2]:>32} {head[3]:>28} {head[4]:>7}")
        for level in args.levels:
            row = await run_level(args.url, query, level, args.duration, args.warmup, server)
            rows.append(row)
            f, t = row["до_первого_куска_мс"], row["до_конца_мс"]
            line = (
                f"{level:>11} {row['rps']:>7} {f['p50']:>10}/{f['p95']}/{f['p99']:<8}"
                f" {t['p50']:>12}/{t['p95']}/{t['p99']:<8} {row['ошибок']:>7}"
            )
            if "cpu_сервера_с_на_запрос" in row:
                cpu_s, cores = row["cpu_сервера_с_на_запрос"], row["cpu_сервера_ядер"]
                line += f"   CPU {cpu_s} с/запрос, {cores} ядра"
            print(line)
        result["сценарии"][name] = {"вопрос": query, "уровни": rows}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--levels", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument(
        "--duration", type=float, default=20.0, help="секунд на уровень (после прогрева)"
    )
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument(
        "--server-pid", type=int, help="pid процесса сервера: считать его CPU и память"
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    if args.json:
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nчисла сохранены: {args.json}")


if __name__ == "__main__":
    main()
