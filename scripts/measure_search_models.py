"""Замер моделей поиска на машине прототипа: эмбеддинги и переранжирование.

ADR-300 фиксирует **BGE-M3** для эмбеддингов и кросс-энкодер
**BGE-reranker-v2-m3** для переранжирования, обе с пометкой «своя, GPU». Машина
прототипа — без видеоускорителя, и на ней уже живёт локальная модель на 4,7 ГБ.

Вопрос, ради которого написан скрипт, один: **укладываются ли эти модели на
процессоре в бюджет задержки**. Отвечать на него рассуждением нельзя — проект
уже дважды получал числа, обратные ожиданиям: память локальной модели оказалась
меньше расчётной, а время до первого токена лучше на два порядка.

ЧТО ЛЕЖИТ НА СИНХРОННОМ ПУТИ АБОНЕНТА, А ЧТО НЕТ

| Операция | Путь | Бюджет |
|---|---|---|
| Эмбеддинг **запроса** | синхронный | входит в `P99 поиска < 100 мс` (DoD) |
| Переранжирование `top_k=20 → top_n=3` | синхронный | туда же |
| Эмбеддинг **корпуса** при индексации | офлайн | бюджета нет, важна пропускная способность |

Отсюда и состав замера: одиночный запрос меряется отдельно от пакета, а
переранжирование — на тех самых двадцати кандидатах, что задаёт ADR-200.

ПОЧЕМУ МЕДИАНА ИЗ ТРЁХ. Первый прогон дороже: модель прогревается, веса ложатся
в память. Одно измерение дало бы завышенную оценку — та же поправка, что в
замере локальной модели.

ЗАПУСК

    python -m pip install sentence-transformers psutil
    python scripts/measure_search_models.py

**Первый запуск скачивает веса, и в «загрузку модели» попадает скачивание.**
Первый прогон дал 494 с и 269 с, повторный на кэше — 7,8 с и 5,4 с. Числа
загрузки имеют смысл только со второго запуска; это проверено, а не
предположено.

Сравнить другую пару моделей:

    python scripts/measure_search_models.py --embedder intfloat/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EMBEDDER = "BAAI/bge-m3"
RERANKER = "BAAI/bge-reranker-v2-m3"
RUNS = 3

# Модели задаются ключами: замер нужен не один раз, а для сравнения. Когда
# выбранная ADR-300 пара не укладывается в бюджет, следующий вопрос — «а что
# укладывается», и линейка для ответа должна быть той же самой.

# Бюджеты, с которыми сравниваются числа (DoD).
SEARCH_BUDGET_MS = 100.0
TTFT_BUDGET_MS = 500.0

QUERY = "Почему выросла сумма в квитанции за холодную воду?"
"""Запрос абонента — короткий, как в жизни. Длину замер учитывает: эмбеддинг
длинного текста дороже, и мерить его запросом на страницу было бы нечестно."""

CHUNK = (
    "Перерасчёт размера платы за коммунальную услугу производится исполнителем "
    "при предоставлении потребителем показаний индивидуального прибора учёта. "
    "Основанием для перерасчёта является заявление потребителя, поданное в "
    "течение расчётного периода. Исполнитель обязан произвести перерасчёт не "
    "позднее месяца, следующего за месяцем подачи заявления, и отразить его "
    "результат в платёжном документе."
)
"""Фрагмент базы знаний правдоподобной длины — около 500 знаков.

Текст сочинён для замера и знанием проекта не является: измеряется скорость, а
не содержание. Настоящий корпус на прототипе собирается из открытых источников."""


@dataclass
class Timing:
    """Итог по одной операции: медиана и разброс."""

    label: str
    samples: list[float]
    budget_ms: float | None = None

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples) * 1000

    @property
    def first_ms(self) -> float:
        return self.samples[0] * 1000

    def line(self) -> str:
        verdict = ""
        if self.budget_ms is not None:
            fits = "укладывается" if self.median_ms <= self.budget_ms else "НЕ УКЛАДЫВАЕТСЯ"
            verdict = f"   бюджет {self.budget_ms:.0f} мс — {fits}"
        return (
            f"  {self.label:44} медиана {self.median_ms:8.1f} мс"
            f"   первый прогон {self.first_ms:8.1f} мс{verdict}"
        )


def format_memory(value: float) -> str:
    """Память с оговоркой, когда числу верить нельзя.

    Прирост RSS не ловит отображённые в память веса: замер BGE-M3 дал 0,33 ГБ
    при 4,25 ГБ на диске, а кросс-энкодер — ровно ноль. Печатать такое как факт
    нельзя, а молчать — значит потерять хоть какой-то ориентир.
    """
    if value < 0.05:
        return f"{value:8.2f} ГБ  (прирост RSS не показателен, см. пометку)"
    return f"{value:8.2f} ГБ  (нижняя оценка: RSS не ловит mmap весов)"


def memory_gb() -> float:
    """Прирост памяти процесса. Без psutil — ноль и пометка в отчёте."""
    try:
        import psutil
    except ImportError:
        return 0.0
    return psutil.Process().memory_info().rss / 1024**3


def cache_size_gb(model_name: str) -> float:
    """Сколько весов легло на диск. Берётся из кэша Hugging Face."""
    from huggingface_hub import constants

    folder = Path(constants.HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}"
    if not folder.exists():
        return 0.0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1024**3


def repeat(operation, runs: int = RUNS) -> list[float]:
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return samples


def measure_embedder(
    name: str, corpus_size: int
) -> tuple[list[Timing], float, float, float]:
    from sentence_transformers import SentenceTransformer

    before = memory_gb()
    load_started = time.perf_counter()
    model = SentenceTransformer(name, device="cpu")
    load_seconds = time.perf_counter() - load_started
    memory = max(0.0, memory_gb() - before)

    # Прогрев отдельно: он не должен попасть в медиану, но и не должен быть
    # спрятан — время первой загрузки уже трижды оказывалось для проекта
    # неожиданностью.
    model.encode([QUERY])

    timings = [
        Timing(
            "эмбеддинг одного запроса (синхронный путь)",
            repeat(lambda: model.encode([QUERY])),
            SEARCH_BUDGET_MS,
        ),
        Timing(
            f"эмбеддинг корпуса, {corpus_size} фрагментов (офлайн)",
            repeat(lambda: model.encode([CHUNK] * corpus_size, batch_size=8), runs=1),
        ),
    ]
    return timings, memory, load_seconds, cache_size_gb(name)


def measure_reranker(name: str, top_k: int) -> tuple[list[Timing], float, float, float]:
    from sentence_transformers import CrossEncoder

    before = memory_gb()
    load_started = time.perf_counter()
    model = CrossEncoder(name, device="cpu")
    load_seconds = time.perf_counter() - load_started
    memory = max(0.0, memory_gb() - before)

    pairs = [(QUERY, CHUNK)] * top_k
    model.predict(pairs[:1])

    timings = [
        Timing(
            f"переранжирование {top_k} кандидатов (синхронный путь)",
            repeat(lambda: model.predict(pairs)),
            SEARCH_BUDGET_MS,
        )
    ]
    return timings, memory, load_seconds, cache_size_gb(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--embedder", default=EMBEDDER, help="модель эмбеддингов")
    parser.add_argument("--reranker", default=RERANKER, help="модель переранжирования")
    parser.add_argument("--top-k", type=int, default=20, help="кандидатов на переранжирование")
    parser.add_argument("--corpus", type=int, default=64, help="фрагментов в пакете индексации")
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args(argv)

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print(
            "нужен sentence-transformers:\n"
            "    python -m pip install sentence-transformers psutil",
            file=sys.stderr,
        )
        return 1

    if memory_gb() == 0.0:
        print("psutil не установлен — память не измеряется, остальное меряется\n")

    print(f"Эмбеддинги — {args.embedder}")
    timings, memory, load, size = measure_embedder(args.embedder, args.corpus)
    print(f"  {'веса на диске':44} {size:8.2f} ГБ")
    print(f"  {'память под моделью':44} {format_memory(memory)}")
    print(f"  {'загрузка модели с диска':44} {load:8.1f} с")
    for timing in timings:
        print(timing.line())

    if not args.skip_reranker:
        print(f"\nПереранжирование — {args.reranker}")
        timings, memory, load, size = measure_reranker(args.reranker, args.top_k)
        print(f"  {'веса на диске':44} {size:8.2f} ГБ")
        print(f"  {'память под моделью':44} {format_memory(memory)}")
        print(f"  {'загрузка модели с диска':44} {load:8.1f} с")
        for timing in timings:
            print(timing.line())

    print(
        f"\nБюджет поиска — {SEARCH_BUDGET_MS:.0f} мс на весь синхронный путь "
        f"(DoD), время до первого токена — {TTFT_BUDGET_MS:.0f} мс.\n"
        "Эмбеддинг запроса и переранжирование складываются: их сумма и есть то,\n"
        "что абонент ждёт до начала ответа сверх работы самой модели."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
