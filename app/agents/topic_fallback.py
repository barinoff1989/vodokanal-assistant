"""Запасной классификатор темы: локальная модель, только когда правила молчат.

ПОЧЕМУ НЕ ПЕРВЫЙ ГЕЙТ, А ЗАПАСНОЙ. `Topic` решает, звать ли модель вообще: пять
тем (`outage`, `water_quality`, `tariff`, `account`, `template`) отвечаются без
неё и намеренно быстро (ADR-012, ADR-013). Поставить модель первым гейтом
значило бы звать её на каждый запрос — включая все пять путей, ради скорости
которых правила и написаны, — до того как узнать, что она не нужна.

Здесь модель зовётся **только когда `Triage._topic_by_rules` не нашла ничего**
— то есть ровно на том подмножестве, которое сегодня и так уходит в
`Topic.GENERAL` без вопросов. По корректности хуже, чем сейчас, не станет:
классификатор никогда не может увести на путь хуже отказа — только на верный
путь вместо него (в худшем случае возвращает `None`, и это то же самое
`GENERAL`, что и без него). **По задержке — честно, регрессия на этом
подмножестве есть:** таймаут `topic_fallback_timeout_seconds` (раздел 91
журнала — замер, а не ноль) добавляется к сегодняшнему быстрому провалу.
Плата принята сознательно: несколько секунд за шанс на верный ответ лучше,
чем мгновенный неверный, но только на подмножестве, где сегодня и так плохо —
не на путях, которые отвечают без модели намеренно быстро (см. выше). Живой
пример — раздел 90 журнала: «По моему адресу до скольки отключение?» не
подошла ни под один маркер `TOPIC_MARKERS['outage']`.

МОДЕЛЬ ЛОКАЛЬНАЯ. `local-test` (Qwen2.5-7B через Ollama) уже поднята для
судьи (ADR-007) и smoke-тестов — токенов не тратит, только процессор. На CPU
без GPU вызов не бесплатен и не быстрый (раздел 91), поэтому — таймаут и
вызов только на подмножестве, а не на каждом запросе (см. выше).

ЛЮБОЙ ОТКАЗ — ЭТО `None`, НЕ ИСКЛЮЧЕНИЕ. Модель не ответила, ответила не JSON'ом
или сама выбрала `general` (то есть «ничего конкретного не нашла») — вызывающий
код идёт по прежнему пути, как если бы классификатора не было вовсе. Отличать
эти причины друг от друга не нужно: результат для маршрутизации один и тот же.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from app.config import Settings, get_settings
from app.taxonomy import Topic, coerce_topic

__all__ = ["TopicFallbackClassifier"]

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Ты определяешь тему вопроса абонента водоканала. Отвечаешь строго одним "
    "объектом JSON, без пояснений вокруг."
)

_PROMPT = """\
Определи тему вопроса — ровно одно значение:

outage — жалоба на отсутствие воды либо вопрос про плановое/аварийное отключение
water_quality — вопрос о качестве, цвете, запахе, привкусе воды
tariff — вопрос о цене куба воды или водоотведения
account — запрос факта по лицевому счёту: задолженность, начисления, срок
          поверки, показания (не спор о них — «откуда долг» сюда не входит)
template — просьба дать бланк или образец заявления
general — всё остальное, включая вопросы не по теме

Вопрос: {query}

Верни только JSON: {{"topic": "<одно значение из списка выше>"}}
"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class TopicFallbackClassifier:
    """Определяет тему одним коротким вызовом локальной модели.

    :param settings: настройки приложения; из них берётся псевдоним модели и
        бюджет ожидания, если не заданы явно.
    :param model_alias: псевдоним модели из `litellm_config.yaml`.
    :param timeout_seconds: сколько ждать ответ, прежде чем считать модель
        недоступной.
    :param complete: функция обращения к модели. ``None`` — берётся
        `litellm.Router` лениво, как в шлюзе и у судьи.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_alias: str | None = None,
        timeout_seconds: float | None = None,
        complete: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._alias = model_alias or self._settings.topic_fallback_provider
        self._timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._settings.topic_fallback_timeout_seconds
        )
        self._complete = complete
        self._router_instance: Any = None

    async def classify(self, query: str) -> Topic | None:
        """Определить тему вопроса, который не узнали правила.

        ``None`` — модель недоступна, не ответила вовремя, вернула мусор либо
        сама выбрала `general`. Все эти случаи равнозначны для вызывающего
        кода: путь остаётся прежним.
        """
        try:
            raw = await asyncio.wait_for(self._call(query), timeout=self._timeout)
        except TimeoutError:
            logger.info("запасной классификатор темы не ответил за %.1f с", self._timeout)
            return None
        except Exception as exc:  # noqa: BLE001 — любой отказ модели равнозначен «не помогла»
            logger.info("запасной классификатор темы недоступен (%s)", exc)
            return None

        topic = _parse(raw)
        if topic is None or topic is Topic.GENERAL:
            return None
        return topic

    async def _call(self, query: str) -> Any:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _PROMPT.format(query=query.strip())},
        ]
        params: dict[str, Any] = {
            "model": self._alias,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 30,
            # На случай провайдера, который режим не понимает; маршрутизатор
            # litellm_settings его не читает — тот же приём, что у судьи.
            "response_format": {"type": "json_object"},
            "drop_params": True,
            # Таймаут отдан litellm, а не только внешнему asyncio.wait_for:
            # замер (раздел 91 журнала) показал, что снаружи он не успевает
            # прервать вызов — asyncio.wait_for не может отменить блокирующий
            # ввод-вывод внутри клиента Ollama, только не дождаться его.
            # Здесь — тот же приём, что уже стоит в самом шлюзе
            # (litellm_config.yaml, `router_settings.timeout`).
            "timeout": self._timeout,
        }
        if self._complete is not None:
            return await self._complete(**params)
        return await self._router().acompletion(**params)

    def _router(self) -> Any:
        if self._router_instance is None:
            from app.gateway.model_router import build_router

            self._router_instance = build_router(self._settings)
        return self._router_instance


def _parse(raw: Any) -> Topic | None:
    """Достать тему из ответа модели. Мусор — тоже `None`, не исключение."""
    text = _extract_text(raw)
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "topic" not in payload:
        return None
    # coerce_topic никогда не бросает — значение вне справочника становится
    # `general`, а он в classify() уже приравнен к «не помогла».
    return coerce_topic(payload["topic"])


def _extract_text(raw: Any) -> str:
    try:
        return str(raw.choices[0].message.content or "")
    except (AttributeError, IndexError, TypeError):
        return str(raw or "")
