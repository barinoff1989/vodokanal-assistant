"""Проверки запасного классификатора темы.

Модель подменена: сторожим разбор ответа, таймаут, отказ модели и то, что
`general`/мусор равнозначны отсутствию классификатора — во всех случаях
`classify()` отдаёт `None`, а не бросает исключение.
"""

from __future__ import annotations

import asyncio

from app.agents.topic_fallback import TopicFallbackClassifier
from app.taxonomy import Topic


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


def _classifier(complete, *, timeout: float = 0.5) -> TopicFallbackClassifier:
    return TopicFallbackClassifier(complete=complete, timeout_seconds=timeout)


async def test_распознанная_тема_возвращается():
    async def complete(**_: object) -> _Response:
        return _Response('{"topic": "outage"}')

    topic = await _classifier(complete).classify("до скольки отключение?")
    assert topic is Topic.OUTAGE


async def test_general_равнозначен_отсутствию_классификатора():
    async def complete(**_: object) -> _Response:
        return _Response('{"topic": "general"}')

    assert await _classifier(complete).classify("что угодно") is None


async def test_мусор_в_ответе_это_none():
    async def complete(**_: object) -> _Response:
        return _Response("не могу ответить")

    assert await _classifier(complete).classify("вопрос") is None


async def test_значение_вне_справочника_это_none():
    async def complete(**_: object) -> _Response:
        return _Response('{"topic": "неизвестное"}')

    assert await _classifier(complete).classify("вопрос") is None


async def test_ошибка_модели_не_пробрасывается():
    async def complete(**_: object) -> _Response:
        raise RuntimeError("провайдер недоступен")

    assert await _classifier(complete).classify("вопрос") is None


async def test_таймаут_не_пробрасывается():
    async def complete(**_: object) -> _Response:
        await asyncio.sleep(1)
        return _Response('{"topic": "outage"}')

    topic = await _classifier(complete, timeout=0.01).classify("вопрос")
    assert topic is None


async def test_json_внутри_пояснения_разбирается():
    async def complete(**_: object) -> _Response:
        return _Response('Вот ответ: {"topic": "tariff"} — готово')

    assert await _classifier(complete).classify("сколько стоит куб?") is Topic.TARIFF
