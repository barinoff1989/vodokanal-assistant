"""Переранжирование топ-k кандидатов кросс-энкодером (ADR-300), опционально.

**Выключено по умолчанию, и это не временная деталь.** Замер на машине
прототипа без видеоускорителя: `BGE-reranker-v2-m3` — 5 774 мс на 20
кандидатах при бюджете поиска 100 мс, превышение в 58 раз. Это не
«неоптимально» — это неработоспособно на синхронном пути абонента. Реализация
здесь существует для офлайн-сценариев и для тех, кто поднимет её на GPU
(`settings.reranker_device`), а не для прод-трафика прототипа.

**Что переранжирование меняет, а что нет.** Кросс-энкодер переставляет
кандидатов местами внутри уже отобранного окна `top_k` — который порядка
меняет `KnowledgeBase.search()`. Отсечение по `threshold` при этом
по-прежнему делается по **косинусной** оценке (0.872, ADR-300), не по оценке
кросс-энкодера: у неё другая шкала (логит, не косинус в [0,1]), и
калиброванного порога для неё в проекте нет — известна лишь цена такого
варианта («отсекать оценкой кросс-энкодера»), число не зафиксировано. Выдумывать
его здесь значило бы нарушить то же правило, из-за которого принято решение ADR-300.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.models import ContextChunk

__all__ = ["CrossEncoderReranker", "Reranker"]


class Reranker(Protocol):
    """Минимум, который нужен от кросс-энкодера переранжирования."""

    def rerank(
        self, query: str, candidates: Sequence[tuple[ContextChunk, float]]
    ) -> list[tuple[ContextChunk, float]]:
        """Та же оценка (`float`) при каждом фрагменте — только новый порядок.

        Оценка не переписывается кросс-энкодером: `KnowledgeBase.search()`
        по-прежнему сверяет её с `threshold` на калиброванной косинусной шкале.
        """
        ...


class CrossEncoderReranker:
    """Кросс-энкодер из `sentence-transformers` (`CrossEncoder`, не `SentenceTransformer`).

    Другой класс модели, не другая роль того же `SentenceTransformerEmbedder`:
    кросс-энкодер оценивает пару (запрос, текст) целиком через один проход
    модели, а не считает косинус между двумя независимо посчитанными
    векторами — отсюда и качество выше, и цена дороже (ADR-300).
    """

    def __init__(
        self, model_name: str, device: str = "cpu", *, local_files_only: bool = True
    ) -> None:
        """:param local_files_only: см. `SentenceTransformerEmbedder` — то же
        требование среды (закрытая сеть на MVP), не оптимизация ради оптимизации."""
        self._model_name = model_name
        self._device = device
        self._local_files_only = local_files_only
        self._model: object | None = None

    def warm_up(self) -> None:
        """Поднять модель заранее, а не на первом запросе абонента.

        Индексация базы знаний сама прогревает `Embedder` (`encode()` вызывается
        при сборке корпуса) — у реранкера такого естественного триггера нет,
        поэтому подъём нужно звать явно (`build_knowledge_base`), тем же
        приёмом, что уже применён к `SentenceTransformerEmbedder` четыре раза
        по следам одной и той же ошибки (ADR-300)."""
        self._ensure()

    def _ensure(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            try:
                self._model = CrossEncoder(
                    self._model_name,
                    device=self._device,
                    local_files_only=self._local_files_only,
                )
            except Exception as exc:
                if not self._local_files_only:
                    raise
                raise RuntimeError(
                    f"модель {self._model_name!r} не найдена в локальном кэше. "
                    "Скачать один раз: make install-search — либо поднять сервис "
                    "с EMBEDDING_LOCAL_FILES_ONLY=false, если сеть доступна"
                ) from exc
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[tuple[ContextChunk, float]]
    ) -> list[tuple[ContextChunk, float]]:
        if len(candidates) <= 1:
            return list(candidates)

        model = self._ensure()
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        # `predict` не типизирован под наш протокол, но по форме совпадает:
        # массив чисел, по одному на пару, тот же приём, что у Embedder.encode.
        cross_scores = model.predict(pairs)  # type: ignore[attr-defined]
        ranked = sorted(
            zip(candidates, cross_scores, strict=True), key=lambda pair: -pair[1]
        )
        # Оценка, уходящая наружу, — исходная косинусная (`pair[0][1]`), не
        # `cross_scores`: см. докстринг модуля.
        return [pair for pair, _cross_score in ranked]
