"""Сервер для нагрузочного замера: тот же Backend, но вместо провайдера — заглушка.

ЗАЧЕМ. Нагрузку на сервис нужно мерить без внешнего провайдера: его задержка
(медиана 0,78 с до первого куска у YandexGPT) и его лимиты — чужие, и они бы
подменили собой результат. Здесь ответ провайдера заменён потоком заранее
заданного текста, всё остальное — настоящее: разбор вопроса, поиск в Qdrant,
лимиты в Redis, обезличивание, охранители на потоке, запись событий в Postgres.

ЧТО ОТЛИЧАЕТСЯ ОТ ОБЫЧНОГО ЗАПУСКА
* провайдер — заглушка (`--provider-delay-ms` добавляет к первому куску заданную
  паузу, по умолчанию 0);
* лимиты подняты, чтобы замер не упирался в 429: по умолчанию 20 запросов в минуту
  на абонента и 300 на сервис — это защита от злоупотреблений, а не пропускная
  способность;
* судья качества на живом пути выключен (`QUALITY_SAMPLE_RATE=0`): он вызывает
  модель на каждом ответе;
* Redis — база 1, телеметрия — база `telemetry_load`: рабочие данные разработки не
  затрагиваются;
* метрики на порту 9101, чтобы не мешать запущенному стеку.

ЗАПУСК
    docker compose up -d postgres redis qdrant
    docker exec vk-postgres psql -U vodokanal -d postgres -c "CREATE DATABASE telemetry_load"
    python scripts/load_server.py --port 8100

Нагрузку даёт `scripts/load_test.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ответ заглушки: типичный по длине ответ ассистента (616 знаков), отдаётся
# кусками по четыре слова (22 куска), как поток провайдера.
ANSWER = (
    "Справку об отсутствии задолженности можно получить в личном кабинете в разделе "
    "«Документы» или обратившись в клиентский офис с паспортом. Срок оформления — до "
    "трёх рабочих дней. Если задолженность есть, справка не выдаётся: сначала нужно "
    "погасить долг и передать показания счётчика, после этого запросите справку заново. "
    "Проверить текущее состояние лицевого счёта можно в том же разделе личного кабинета. "
    "Если у вас несколько лицевых счетов, справка оформляется отдельно на каждый из них. "
    "Для юридических лиц действует другой порядок: заявление подаётся в письменном виде, "
    "срок рассмотрения — до десяти рабочих дней."
)
CHUNK_WORDS = 4


class _Chunk:
    def __init__(self, text: str | None, usage: Any = None) -> None:
        delta = type("D", (), {"content": text})()
        self.choices = [type("C", (), {"delta": delta})()]
        self.usage = usage


def make_stub(delay_ms: float) -> Any:
    words = ANSWER.split()
    parts = [" ".join(words[i : i + CHUNK_WORDS]) + " " for i in range(0, len(words), CHUNK_WORDS)]
    usage = type("U", (), {"prompt_tokens": 320, "completion_tokens": 160, "total_tokens": 480})()

    async def completion(**kwargs: Any) -> Any:
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000)
        if kwargs.get("stream"):

            async def stream() -> Any:
                for part in parts:
                    yield _Chunk(part)
                yield _Chunk(None, usage)

            return stream()
        message = type("M", (), {"content": ANSWER})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice], "usage": usage})()

    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--provider-delay-ms", type=float, default=0.0)
    args = parser.parse_args()

    defaults = {
        "QUOTA_SUBSCRIBER_PER_MINUTE": "1000000",
        "QUOTA_SESSION_PER_MINUTE": "1000000",
        "QUOTA_SERVICE_PER_MINUTE": "1000000",
        "QUALITY_SAMPLE_RATE": "0",
        "REDIS_DB": "1",
        "TELEMETRY_DB": "telemetry_load",
        "METRICS_PORT": "9101",
        "VECTOR_STORE": "qdrant",
        "LLM_PROVIDER": "yandexgpt",
        "YANDEX_API_KEY": "нагрузочный-замер",
        "YANDEX_FOLDER_ID": "нагрузочный-замер",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    # Подмена до импорта `app.main`: приложение собирается при импорте модуля.
    from app.gateway import llm_gateway

    original = llm_gateway.LlmGateway.__init__
    stub = make_stub(args.provider_delay_ms)

    def patched(self: Any, *a: Any, **kw: Any) -> None:
        kw["completion"] = stub
        original(self, *a, **kw)

    llm_gateway.LlmGateway.__init__ = patched  # type: ignore[method-assign]

    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
