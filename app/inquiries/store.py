"""Хранилище обращений на прототипе — таблица вместо микросервиса владельца.

ЧЕМ ЭТО ЯВЛЯЕТСЯ

На MVP обращение регистрируется в **микросервисе обращений** владельца через его
REST API (ADR-009). На прототипе микросервиса нет: он внутри закрытой сети, и
описания интерфейса тоже нет (пункт 20 сведённого TODO). Его место занимает
таблица в Postgres.

**Шов проходит по адаптеру, а не здесь.** `InquiryServiceAdapter` зовёт у клиента
один метод — `create_inquiry(payload, key)`. На MVP клиентом станет обёртка над
REST владельца, и всё, что выше адаптера — гейты правила 4.7, привязка к сессии,
идемпотентность, автомат — не изменится. То же построение, что у приёма графика
отключений и у чтения Биллинга: заменяется чтение или запись, а не то, что за ними.

ПОЧЕМУ ЭТО НЕ НАРУШАЕТ ADR-003

Правило требует ходить в чужие системы только через их API. Здесь нарушать
нечего: чужой системы нет, а таблица — **наша**, заведённая нами на нашей же
базе-слепке. Это не прямой доступ к БД владельца, а замена отсутствующей системы,
и на MVP она исчезает вместе с прототипом.

ИДЕМПОТЕНТНОСТЬ ЖИВЁТ В БАЗЕ, А НЕ ТОЛЬКО В ПАМЯТИ

`WriteAdapter` держит виденные ключи в памяти процесса и честно оговаривает, что
перезапуск их теряет. Здесь ключ уникален **в таблице**: повтор после
перезапуска не создаст вторую запись, а вернёт номер первой. Это не дублирование
защиты, а её единственный надёжный уровень — память процесса остаётся быстрым
отсевом.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

__all__ = ["InquiryStore", "PostgresConnection", "SCHEMA_PATH"]

logger = logging.getLogger(__name__)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "docker" / "postgres" / "init" / "01_inquiries.sql"
)
"""Единственное место, где написана схема.

Postgres выполняет `docker/postgres/init` только при создании тома, поэтому на
уже поднятой базе файл читает и применяет это хранилище. Два места, где пишется
схема, разошлись бы при первой правке."""


class PostgresConnection(Protocol):
    """Минимум, который нужен от соединения. Ровно то, что даёт `psycopg`."""

    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


class InquiryStore:
    """Регистрация обращений в таблице `inquiries`.

    :param connect: как получить соединение. Функция, а не готовое соединение:
        сервис живёт дольше любого соединения, и держать одно на весь процесс
        значило бы падать после первого разрыва.
    """

    def __init__(self, connect: Any) -> None:
        self._connect = connect
        self._schema_applied = False

    def ensure_schema(self) -> None:
        """Применить схему, если её ещё нет.

        Идемпотентно по построению: в файле `CREATE TABLE IF NOT EXISTS`. Зовётся
        один раз за жизнь объекта — не при каждой записи, чтобы не платить
        обращением к базе на пути абонента.
        """
        if self._schema_applied:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            conn.commit()
        self._schema_applied = True

    def create_inquiry(self, payload: dict[str, Any], key: str) -> str:
        """Записать обращение и вернуть его номер.

        Подпись задана адаптером (`_call_system`), а не удобством хранилища: на
        MVP этот же метод будет обращаться к REST владельца.

        Повтор с тем же ключом **не создаёт вторую запись** — возвращается номер
        первой. Решение принимает база, а не проверка перед вставкой: между
        проверкой и вставкой помещается второй запрос.

        **Номера идут с пропусками, и это не дефект.** Отклонённая по конфликту
        вставка всё равно расходует значение последовательности: после повтора
        следующее обращение получит не соседний номер. Так устроен `IDENTITY` в
        Postgres, и настоящие системы регистрации ведут себя так же — важна
        уникальность номера, а не его непрерывность.
        """
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inquiries (
                        inquiry_type, subscriber_id, session_id,
                        subject, body, slots, idempotency_key
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING id
                    """,
                    (
                        payload["inquiry_type"],
                        payload["subscriber_id"],
                        payload.get("session_id", ""),
                        payload["subject"],
                        payload["body"],
                        json.dumps(payload.get("slots", {}), ensure_ascii=False),
                        key,
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    # Ключ уже был: запись не создана, номер берётся у первой.
                    cur.execute(
                        "SELECT id FROM inquiries WHERE idempotency_key = %s", (key,)
                    )
                    row = cur.fetchone()
                    if row is None:  # pragma: no cover — гонка с удалением строки
                        raise RuntimeError(
                            "обращение не создано и не найдено по ключу "
                            f"{key!r}: состояние таблицы противоречиво"
                        )
                    logger.info("повтор обращения по ключу %s: номер %s", key, row[0])
            conn.commit()
        return str(row[0])
