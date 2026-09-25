"""Обход несовместимости `gte-multilingual-reranker-base` с transformers 5.x.

Модель приходит с собственным кодом (`trust_remote_code`, репозиторий
`Alibaba-NLP/new-impl`, просмотрен 25 сентября 2026: чистый PyTorch, без сети и
запуска процессов). Под transformers 5.16 не сохраняемые в файле буферы этого кода
(`position_ids`, кэш вращательных эмбеддингов) остаются неинициализированными: в
`position_ids[0]` лежит мусор вроде 6466274787328, и первый же `predict` падает с
`IndexError`. После пересоздания буферов оценки осмысленные (проверено на трёх
парах: уместный фрагмент 0,95, вода про оплату 0,35, нерелевантный 0,04).

Используется только замерами (`measure_reranking.py`, `measure_reranker_latency.py`).
Если модель будет выбрана для службы, обход надо либо перенести в загрузчик
реранкера, либо зафиксировать transformers версии 4.x.
"""

from __future__ import annotations

from typing import Any


def fix_gte_buffers(cross_encoder: Any) -> int:
    """Пересоздать буферы у модулей `NewEmbeddings`; вернуть, сколько таких модулей найдено."""
    import torch

    model = cross_encoder.model
    config = model.config
    fixed = 0
    for module in model.modules():
        if type(module).__name__ == "NewEmbeddings":
            module.position_ids = torch.arange(config.max_position_embeddings)
            module._init_rope(config)
            fixed += 1
    return fixed
