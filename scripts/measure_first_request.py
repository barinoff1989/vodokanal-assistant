"""Замер первого запроса после перезапуска по слагаемым (пункт 35 TODO).

ЗАЧЕМ

Живая проверка дала 21,6 с до первого куска против 1,9 с в прежнем замере при
том же прогреве. Причина не была установлена, и подставлять объяснение без
замера нельзя — правило, выведенное разделом 50.3, где ровно такой разбор по
слагаемым и превратил «восемь секунд» в три понятных числа.

ЧТО ИЗМЕРЯЕТСЯ, А ЧТО НАМЕРЕННО НЕТ

Провайдер из замера исключён: вместо него подставляется мгновенный ответ.
Иначе в числе смешались бы наша задержка и чужая, а по решению плана
тестирования гейт ставится на «Backend + Gateway без провайдера» — ухудшение у
провайдера иначе выглядело бы как регрессия нашего кода.

Отсюда же следует, что абсолютные числа здесь **меньше** тех, что видит абонент
на стенде: у него сверху лежит время провайдера (0,22–0,28 с на прогретом пути).
Сравнивать эти числа со стендовыми напрямую нельзя, а сравнивать между собой —
можно, и ради этого замер и написан.

**[ОГОВОРКА, БЕЗ КОТОРОЙ ЧИСЛА ВРУТ.]** Подставляя провайдера, замер заодно
**отключает загрузку LiteLLM**: шлюз строит маршрутизатор только когда вызов
модели не подменён (`if self._completion is None`). А импорт LiteLLM — самое
тяжёлое из ленивого, 7,0 с по разделу 50.3. Поэтому столбец «старт сервиса
(прогрев)» здесь показывает 0,02 с и **не измеряет главную работу прогрева**;
он годится, чтобы поймать *отсутствие* прогрева (тогда столбцы совпадают
в принципе), но не чтобы измерить его пользу. Ту меряет стенд.

ЗАПУСК

    python scripts/measure_first_request.py

Каждое слагаемое меряется в **отдельном процессе**: ленивые загрузки происходят
однажды, и второй замер в том же процессе показал бы ноль там, где на самом деле
секунды. Это та же поправка, из-за которой замер моделей поиска снимается со
второго запуска.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Замер выполняется в порождённом процессе: см. оговорку в описании модуля.
_PROBE = textwrap.dedent(
    """
    import json, os, sys, time
    sys.path.insert(0, {root!r})
    os.environ.setdefault("YANDEX_API_KEY", "замер-без-провайдера")

    marks = {{}}
    started = time.perf_counter()

    from app.api import create_app
    from app.backend.orchestrator import Orchestrator
    from app.config import get_settings
    from app.gateway.llm_gateway import LlmGateway
    from app.main import _build_knowledge_base, _build_outages, _build_regulated
    from app.models import GenerateRequest, GenerationParameters, RequestMetadata
    marks["импорт модулей"] = time.perf_counter() - started

    class _Chunk:
        def __init__(self, text):
            self.choices = [type("C", (), {{"delta": type("D", (), {{"content": text}})()}})()]
            self.usage = None

    async def instant(**kwargs):
        \"\"\"Мгновенный ответ вместо провайдера: меряем свою задержку, не чужую.\"\"\"
        if kwargs.get("stream"):
            async def gen():
                yield _Chunk("готово")
            return gen()
        return type("R", (), {{
            "choices": [type("C", (), {{"message": type("M", (), {{"content": "готово"}})()}})()],
            "usage": None,
        }})()

    started = time.perf_counter()
    settings = get_settings()
    orchestrator = Orchestrator(
        LlmGateway(completion=instant, settings=settings),
        knowledge_base=_build_knowledge_base(),
        direct=tuple(r for r in (_build_regulated(), _build_outages()) if r is not None),
        settings=settings,
    )
    marks["сборка приложения"] = time.perf_counter() - started

    from fastapi.testclient import TestClient
    app = create_app(orchestrator)

    started = time.perf_counter()
    client = TestClient(app)
    client.__enter__()
    marks["старт сервиса (прогрев)"] = time.perf_counter() - started

    body = {{
        "system": "помощник абонента водоканала",
        "query": "Как получить справку об отсутствии задолженности?",
        "parameters": {{"stream": False}},
        "metadata": {{"subscriber_id": "2100367945", "session_id": "измерение"}},
    }}
    for name in ("первый запрос", "второй запрос", "третий запрос"):
        started = time.perf_counter()
        response = client.post("/v1/generate", json=body)
        marks[name] = time.perf_counter() - started
        if response.status_code != 200:
            marks[name + " (код)"] = response.status_code

    client.__exit__(None, None, None)
    print("МЕТКИ" + json.dumps(marks, ensure_ascii=False))
    """
)


def probe(*, warm: bool) -> dict[str, float]:
    """Прогнать замер в отдельном процессе.

    :param warm: выполнять ли прогрев при старте сервиса.

    Прогрев выключается подменой метода, а не правкой кода: замер должен
    показывать разницу между «есть прогрев» и «нет прогрева» на **одном и том
    же** коде, иначе он мерил бы две разные программы.
    """
    source = _PROBE.format(root=str(ROOT))
    if not warm:
        source = source.replace(
            "    client = TestClient(app)",
            "    orchestrator.warmup = lambda: None  # прогрев выключен для сравнения\n"
            "    client = TestClient(app)",
        )
    # Дочерний процесс под Windows пишет в кодировке консоли, а не в UTF-8, и
    # русские метки приходили нечитаемыми. Кодировка задаётся ему явно.
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        env=environment,
    )
    for line in result.stdout.splitlines():
        if line.startswith("МЕТКИ"):
            return dict(json.loads(line[len("МЕТКИ") :]))
    sys.exit(
        "замер не дал результата\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="куда сохранить числа")
    args = parser.parse_args()

    print("Провайдер исключён: вместо него мгновенный ответ.")
    print("Числа меньше стендовых на время провайдера (0,22–0,28 с).\n")

    runs = {"с прогревом": probe(warm=True), "без прогрева": probe(warm=False)}

    names = list(runs["с прогревом"])
    print(f"{'слагаемое':<26} {'с прогревом':>14} {'без прогрева':>14}")
    for name in names:
        warm = runs["с прогревом"].get(name)
        cold = runs["без прогрева"].get(name)
        if not isinstance(warm, float) or not isinstance(cold, float):
            continue
        print(f"{name:<26} {warm:>13.2f}с {cold:>13.2f}с")

    if args.json:
        args.json.write_text(
            json.dumps(runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nчисла сохранены: {args.json}")


if __name__ == "__main__":
    main()
