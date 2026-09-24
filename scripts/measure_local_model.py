"""Замер четырёх характеристик локальной тестовой модели.

Закрывает открытый пункт: в ADR-300 стоят
ориентиры с пометкой «требует подтверждения при первой загрузке». Этот скрипт
даёт фактические числа, которыми их нужно заменить.

Замеряется:

* размер весов на диске — берётся у самой Ollama, не с сайта;
* потребление памяти при загруженной модели — разница до и после;
* скорость генерации в токенах в секунду;
* время до первого токена.

Запуск (Ollama должна быть поднята, модель загружена):

    python scripts/measure_local_model.py

Числа с одного прогона — ориентир, поэтому генерация повторяется трижды и
берётся медиана: первый прогон всегда дороже из-за загрузки весов в память.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("LOCAL_MODEL", "qwen2.5:7b-instruct-q4_K_M")
RUNS = 3
PROMPT = (
    "Объясни абоненту водоканала в трёх-четырёх предложениях, "
    "как подготовиться к поверке счётчика воды."
)


def _post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as response:
        return json.loads(response.read())


def weights_size() -> tuple[str, float]:
    """Размер весов на диске — по данным самой Ollama."""
    for model in _get("/api/tags").get("models", []):
        if model.get("name", "").startswith(MODEL.split(":")[0]):
            gigabytes = model.get("size", 0) / 1024**3
            return model["name"], gigabytes
    raise SystemExit(f"модель {MODEL} не загружена; выполните: ollama pull {MODEL}")


def memory_used() -> float:
    """Сколько памяти занимает модель, когда она поднята.

    Берётся у Ollama, а не у операционной системы: та показала бы и всё
    остальное, что работает на машине, а нас интересует именно модель.
    """
    running = _get("/api/ps").get("models", [])
    for model in running:
        if model.get("name", "").startswith(MODEL.split(":")[0]):
            return float(model.get("size_vram", 0) or model.get("size", 0)) / 1024**3
    return 0.0


def generation_run() -> tuple[float, float, int]:
    """Один прогон генерации: время до первого токена, скорость, число токенов."""
    started = time.perf_counter()
    first_at: float | None = None
    tokens = 0

    request = urllib.request.Request(
        f"{BASE}/api/generate",
        data=json.dumps(
            {"model": MODEL, "prompt": PROMPT, "stream": True, "options": {"num_predict": 200}}
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.strip():
                continue
            piece = json.loads(line)
            if piece.get("response"):
                tokens += 1
                if first_at is None:
                    first_at = time.perf_counter() - started
            if piece.get("done"):
                break

    total = time.perf_counter() - started
    speed = tokens / total if total else 0.0
    return (first_at or 0.0), speed, tokens


def main() -> int:
    print(f"адрес Ollama : {BASE}")
    print(f"модель       : {MODEL}")
    print("-" * 62)

    try:
        name, size_gb = weights_size()
    except urllib.error.URLError as exc:
        print(f"Ollama недоступна: {exc}")
        print("Запустите приложение Ollama и повторите.")
        return 1

    print(f"{'Размер весов на диске':38} {size_gb:5.2f} ГБ   ({name})")

    # Первый прогон поднимает модель в память — он же и прогрев.
    print(f"{'Прогрев (первый прогон)':38} ", end="", flush=True)
    warm_ttft, warm_speed, _ = generation_run()
    print(f"{warm_ttft:5.2f} с до первого токена")

    print(f"{'Память под моделью':38} {memory_used():5.2f} ГБ")

    ttfts: list[float] = []
    speeds: list[float] = []
    for index in range(RUNS):
        ttft, speed, tokens = generation_run()
        ttfts.append(ttft)
        speeds.append(speed)
        print(f"  прогон {index + 1}: {ttft:5.2f} с, {speed:5.1f} ток/с, {tokens} токенов")

    print("-" * 62)
    print(f"{'Время до первого токена (медиана)':38} {statistics.median(ttfts):5.2f} с")
    print(f"{'Скорость генерации (медиана)':38} {statistics.median(speeds):5.1f} ток/с")
    print()
    print("Назначение модели — офлайн-проверки промптов и слоя защиты без трат")
    print("на платный внешний API. Оценивать по этим числам задержку рабочего")
    print("контура нельзя: там отвечает управляемый API (ADR-300).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
