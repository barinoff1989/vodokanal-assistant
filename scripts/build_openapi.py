"""Сборка спецификации API из моделей данных.

ПОЧЕМУ ГЕНЕРАТОР, А НЕ ФАЙЛ, НАПИСАННЫЙ РУКАМИ.
В проекте трижды расходились два источника истины: устаревшая схема в `dz5`
породила ложное замечание ревьюера, счёт диаграмм считался от строки в матрице
вместо фактического состава, идентификатор каталога Яндекса жил в двух местах и
разошёлся при первой правке. Спецификация, набранная руками рядом с моделями, —
четвёртый такой случай, только отложенный.

Здесь схемы берутся **из тех же классов**, по которым работает сервис. Разойтись
им негде: расхождение означало бы, что модель изменилась, а спецификация — нет,
и это ловится проверкой `test_api_spec.py`, которая пересобирает файл и сравнивает.

ЧТО ЗАКРЫВАЕТ ЭТА СПЕЦИФИКАЦИЯ (семь проблем прежней редакции):

===== ===================================================================
   1   `servers: localhost` числился рабочим контуром — заменён адресами кластера
   2   `sources` принимались на входе, но не возвращались — теперь в ответе
   3   у события ошибки в потоке не было схемы — теперь есть
   4   `department` в метаданных — убран, а не дополнен (решение 2)
   5   опечатки и хвостовые пробелы — их здесь нет по построению
   6   перечень каналов — `lk_web`, `lk_mobile`
   7   `subscriber_id` вместо `user_id`, `inquiry_type` в метаданных
===== ===================================================================

Плюс пятое решение: код `503` при недоступности хранилища лимитов (ADR-400),
которого в прежней редакции не было.

Запуск::

    python scripts/build_openapi.py
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import yaml
from pydantic.json_schema import models_json_schema

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models import (  # noqa: E402
    DoneEvent,
    ErrorEvent,
    GenerateRequest,
    GenerateResponse,
    MetadataEvent,
    ProblemDetail,
    TokenEvent,
)

TARGET = ROOT / "AI_docs" / "API_Spec.yaml"

ERROR_CODES: dict[int, tuple[str, str]] = {
    400: ("Неверный запрос", "Тело не прошло проверку схемы."),
    401: ("Не авторизован", "Отсутствует или неверен внутренний токен."),
    429: (
        "Превышена квота",
        "Лимит на абонента, сессию или сервис исчерпан. "
        "Заголовок `Retry-After` обязателен.",
    ),
    502: ("Ошибка провайдера", "Провайдер модели ответил ошибкой."),
    503: (
        "Сервис временно недоступен",
        "Недоступно хранилище счётчиков лимитов (ADR-400). "
        "Код именно `503`, а не `429`: последний означал бы «вы превысили лимит» "
        "и возлагал причину на абонента, который ни при чём. "
        "Заголовок `Retry-After` обязателен.",
    ),
    504: ("Таймаут провайдера", "Провайдер модели не ответил вовремя."),
}


def _schemas() -> dict[str, Any]:
    """Схемы всех моделей контракта с общими определениями."""
    _, wrapper = models_json_schema(
        [
            (GenerateRequest, "validation"),
            (GenerateResponse, "serialization"),
            (ProblemDetail, "serialization"),
            (TokenEvent, "serialization"),
            (MetadataEvent, "serialization"),
            (DoneEvent, "serialization"),
            (ErrorEvent, "serialization"),
        ],
        ref_template="#/components/schemas/{model}",
        title="Схемы контракта",
    )
    return dict(wrapper.get("$defs", {}))


def _problem_response(status: int) -> dict[str, Any]:
    title, description = ERROR_CODES[status]
    response: dict[str, Any] = {
        "description": f"{title}. {description}",
        "content": {
            "application/problem+json": {
                "schema": {"$ref": "#/components/schemas/ProblemDetail"}
            }
        },
    }
    if status in (429, 503):
        response["headers"] = {
            "Retry-After": {
                "description": "Через сколько секунд повторять запрос.",
                "schema": {"type": "integer", "minimum": 1},
            }
        }
    return response


def _stream_example() -> dict[str, Any]:
    """Пример потока: порядок событий и разделение пустой строкой.

    Порядок не произволен — куски текста, затем метаданные, затем завершение.
    Потребитель, собравший разбор по этому примеру, не сломается на настоящем
    потоке.
    """
    events = [
        ("token", '{"event":"token","delta":"Поверку "}'),
        ("token", '{"event":"token","delta":"проводят раз в шесть лет."}'),
        ("metadata", '{"event":"metadata","sources":[],"confidence_score":0.0}'),
        ("done", '{"event":"done","finish_reason":"stop","trace_id":"trace-1"}'),
    ]
    body = "".join(
        "event: " + name + chr(10) + "data: " + data + chr(10) * 2
        for name, data in events
    )
    return {"summary": "Последовательность событий", "value": body}


def build() -> dict[str, Any]:
    """Собрать документ спецификации."""
    spec: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {
            "title": "AI-помощник абонента водоканала — шлюз к модели",
            "version": "1.0.0",
            "description": (
                "Контракт между Backend и шлюзом к модели.\n\n"
                "Разрешены **ровно два адреса**: генерация и проверка "
                "доступности. Адрес `/models` удалён из контракта сознательно, добавлять "
                "новые нельзя без обновления диаграммы контейнеров.\n\n"
                "Отдача метрик живёт на отдельном порту и в этот контракт не входит: "
                "её читает система сбора, а не виджет абонента.\n\n"
                "**Файл собирается из моделей данных** скриптом `scripts/build_openapi.py`. "
                "Править его руками бессмысленно — правку затрёт следующая сборка; менять "
                "нужно модели в `app/models.py`."
            ),
        },
        # Проблема 1: в прежней редакции рабочим контуром числился
        # localhost. Здесь адреса внутрикластерные, а машина разработчика помечена
        # как таковая.
        "servers": [
            {
                "url": "http://llm-gateway.ai-app.svc.cluster.local:8000",
                "description": "Рабочий контур: внутренний адрес кластера",
            },
            {
                "url": "http://localhost:8000",
                "description": "Машина разработчика. Рабочим контуром не является",
            },
        ],
        "security": [{"InternalToken": []}],
        "tags": [
            {"name": "generate", "description": "Генерация ответа абоненту"},
            {"name": "ops", "description": "Эксплуатация"},
        ],
        "paths": {
            "/v1/generate": {
                "post": {
                    "tags": ["generate"],
                    "summary": "Сгенерировать ответ абоненту",
                    "description": (
                        "По умолчанию ответ отдаётся **потоком**: абонент "
                        "должен видеть, что ответ печатается. Значение `stream: false` "
                        "допустимо для служебных вызовов вроде классификации, где "
                        "промежуточный вывод некому показывать.\n\n"
                        "Порядок проверок внутри шлюза зафиксирован и не меняется: "
                        "лимиты, обезличивание, вызов модели, охранители. Лимиты первыми — "
                        "отклонить по превышению дешевле, чем тратить процессор на "
                        "обезличивание."
                    ),
                    "parameters": [
                        {
                            "name": "X-Trace-Id",
                            "in": "header",
                            "required": False,
                            "description": (
                                "Опознаватель запроса. Если не передан, шлюз создаёт свой "
                                "и возвращает его в ответе: без него жалобу абонента "
                                "невозможно связать с записью в журнале."
                            ),
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GenerateRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": (
                                "Ответ. Формат зависит от `parameters.stream`: поток "
                                "событий либо готовый ответ целиком."
                            ),
                            "content": {
                                "text/event-stream": {
                                    "schema": {
                                        "$ref": "#/components/schemas/StreamEvent"
                                    },
                                    "examples": {"поток": _stream_example()},
                                },
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/GenerateResponse"
                                    }
                                },
                            },
                        },
                        **{str(code): _problem_response(code) for code in sorted(ERROR_CODES)},
                    },
                }
            },
            "/v1/healthz": {
                "get": {
                    "tags": ["ops"],
                    "summary": "Проверка доступности",
                    "description": (
                        "Без авторизации: проверку выполняет система оркестрации, у "
                        "которой внутреннего токена нет."
                    ),
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "Сервис отвечает",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "ok"}
                                        },
                                        "required": ["status"],
                                    }
                                }
                            },
                        }
                    },
                }
            },
        },
        "components": {
            "securitySchemes": {
                "InternalToken": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-Internal-Token",
                    "description": (
                        "Внутренний токен. Шлюз доступен только изнутри контура; "
                        "взаимная проверка сертификатов (mTLS) — дополнительная мера, "
                        "не замена."
                    ),
                }
            },
            "schemas": _schemas(),
        },
    }

    # Событие потока — объединение четырёх видов. Прежняя редакция схемы для
    # события ошибки не имела вовсе (проблема 3).
    spec["components"]["schemas"]["StreamEvent"] = {
        "description": (
            "Одно событие потока. Различаются по полю `event`. Ошибка внутри уже "
            "начатого потока приходит тем же телом RFC 7807, что и обычная ошибка: "
            "два формата на два пути пришлось бы поддерживать порознь."
        ),
        "oneOf": [
            {"$ref": "#/components/schemas/TokenEvent"},
            {"$ref": "#/components/schemas/MetadataEvent"},
            {"$ref": "#/components/schemas/DoneEvent"},
            {"$ref": "#/components/schemas/ErrorEvent"},
        ],
        "discriminator": {"propertyName": "event"},
    }
    return spec


def main() -> int:
    spec = build()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(
        "# Файл собран скриптом scripts/build_openapi.py из моделей app/models.py.\n"
        "# Править руками бессмысленно: правку затрёт следующая сборка.\n"
        + yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    schemas = spec["components"]["schemas"]
    print(f"собрано: {TARGET}")
    print(f"схем: {len(schemas)}, адресов: {len(spec['paths'])}")
    print(f"кодов ошибок: {', '.join(str(code) for code in sorted(ERROR_CODES))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
