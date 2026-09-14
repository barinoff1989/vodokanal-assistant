"""Проверки кросс-энкодера переранжирования (app/kb/reranker.py).

Настоящая модель здесь не грузится, кроме одного `slow`-теста — тот же приём,
что у `SentenceTransformerEmbedder` в test_kb_search.py: `local_files_only=True`
падает без кэша быстро и без сети, этого достаточно, чтобы проверить сообщение
об ошибке, не платя за загрузку torch/модели в обычном прогоне.
"""

from __future__ import annotations

import pytest

from app.kb.reranker import CrossEncoderReranker
from app.models import ContextChunk


def _chunk(chunk_id: str) -> ContextChunk:
    return ContextChunk(
        chunk_id=chunk_id,
        text=f"текст {chunk_id}",
        source_title="s",
        source_url="u",
        relevance_score=0.0,
    )


def test_по_умолчанию_модель_берётся_только_из_кэша():
    reranker = CrossEncoderReranker("любая/модель")
    assert reranker._local_files_only is True


def test_ноль_или_один_кандидат_не_требует_модели():
    """Короткое замыкание до `_ensure()` — модель не должна грузиться ради
    входа, где переранжировать нечего."""
    reranker = CrossEncoderReranker("несуществующая/модель-которая-не-грузится")

    assert reranker.rerank("запрос", []) == []

    one = [(_chunk("c1"), 0.9)]
    assert reranker.rerank("запрос", one) == one


@pytest.mark.slow
def test_пустой_кэш_объясняет_себя_а_не_выглядит_отказом_сети():
    reranker = CrossEncoderReranker("несуществующая/модель-для-проверки")
    with pytest.raises(RuntimeError) as caught:
        reranker.warm_up()

    message = str(caught.value)
    assert "локальном кэше" in message
    assert "make install-search" in message
