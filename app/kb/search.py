"""Поиск по базе знаний: эмбеддинг запроса, косинусная близость, порог.

Корпус прототипа — FAQ воронежского филиала (ADR-013), двадцать пар
вопрос-ответ из открытого источника. Одна пара ложится в один фрагмент: длина
ответа от 89 до 1019 знаков, резать нечего.

ЧТО ЭМБЕДДИТСЯ — ВОПРОС И ОТВЕТ **РАЗНЫМИ ВЕКТОРАМИ**, близость берётся по
лучшему из них.

Первая редакция эмбеддила вопрос вместе с ответом одним куском. Довод был верен:
вопрос из FAQ сформулирован так же, как его задаёт абонент, и выбрасывать его
нельзя. Неверным оказалось следствие — что их надо склеить.

**Вопрос тонет в ответе.** Медиана ответа 472 знака, максимум 1019; у `faq-11`
вопрос 24 знака против 611 — отношение один к двадцати пяти. Вектор склейки
определяется ответом, а не вопросом, и оценка падает тем сильнее, чем
многословнее ответ. Дословный вопрос корпуса «Почему начисляются пени?» набирал
0,844 — ниже порога 0,86, при том что ответ на него лежит в базе знаний
буквально.

Два вектора на один фрагмент это снимают, ничего не выбрасывая: совпадение с
вопросом больше не разбавляется, а совпадение по существу ответа по-прежнему
находится. **Единица поиска перестала совпадать с единицей контекста:** ищется
по частям, а в промпт уходит пара целиком — иначе модель получила бы ответ без
вопроса, к которому он относится.

ПРЕФИКСЫ `query:` И `passage:` ОБЯЗАТЕЛЬНЫ. Модель e5 обучена с ними, и без них
близость считается по другому распределению — это свойство модели, а не
украшение. Отсюда же следует, что менять модель эмбеддингов, не меняя префиксы,
нельзя.

ХРАНИЛИЩЕ — В ПАМЯТИ, А НЕ QDRANT. Двадцать фрагментов не нуждаются в контейнере
со своим жизненным циклом; на MVP хранилищем будет Qdrant по ADR-006. Форма
запроса от этого не меняется: `top_k` кандидатов по близости, отсев порогом,
`top_n` в промпт.

**Переранжирования между ними нет** (ADR-014): кросс-энкодер ADR-007 стоит 5,8 с
на двадцати кандидатах при бюджете поиска 100 мс — замерено, а не предположено.

МОДЕЛЬ ВНЕДРЯЕТСЯ СНАРУЖИ. Импорт `sentence-transformers` тянет torch и занимает
секунды; проверки не должны за это платить, а подмена модели — единственный
способ проверить сам поиск, а не качество эмбеддингов.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from app.models import ContextChunk

__all__ = ["Embedder", "KnowledgeBase", "SentenceTransformerEmbedder", "cosine"]

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder(Protocol):
    """Минимум, который нужен от модели эмбеддингов."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Векторы для набора текстов, по одному на текст."""
        ...


def cosine(first: Sequence[float], second: Sequence[float]) -> float:
    """Косинусная близость (ADR-006, метрика зафиксирована там).

    Результат подрезается к отрезку от нуля до единицы: `relevance_score` в
    контракте объявлен именно так, а вычислительная погрешность даёт единицу с
    небольшим хвостом на совпадающих векторах.
    """
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norm = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    if norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / norm))


@dataclass(frozen=True, slots=True)
class _Entry:
    """Фрагмент корпуса вместе с векторами его частей.

    Векторов несколько: вопрос и ответ индексируются порознь, чтобы короткий
    вопрос не растворялся в длинном ответе. Близость фрагмента — лучшая из
    близостей его частей.
    """

    chunk: ContextChunk
    vectors: tuple[Sequence[float], ...]

    def similarity(self, query: Sequence[float]) -> float:
        """Лучшая близость среди частей.

        Максимум, а не среднее: среднее вернуло бы то самое разбавление, ради
        устранения которого части и разведены.
        """
        return max(cosine(query, vector) for vector in self.vectors)


class KnowledgeBase:
    """Проиндексированный корпус и поиск по нему."""

    def __init__(self, entries: Sequence[_Entry], embedder: Embedder) -> None:
        self._entries = tuple(entries)
        self._embedder = embedder

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_file(cls, path: Path, embedder: Embedder) -> KnowledgeBase:
        """Прочитать корпус и проиндексировать его целиком.

        Индексация двадцати фрагментов занимает доли секунды (замер ADR-014: 48
        фрагментов в секунду), поэтому делается при запуске, а не заранее. Когда
        корпус вырастет, это станет отдельным шагом приёма — как у графика
        отключений.
        """
        items = json.loads(path.read_text(encoding="utf-8"))
        texts = [f"{item['question']}\n{item['answer']}" for item in items]
        # Части индексируются порознь, но одним вызовом: модель дорого поднимать,
        # а не звать, и разбивать вызов незачем.
        # Склейка «вопрос + ответ» третьим видом **не берётся**: проверено
        # замером — hit@1, hit@3 и MRR не сдвинулись ни на единицу, а при
        # действующем пороге стало на один ответ хуже. Её близость всегда лежит
        # между близостями частей, то есть максимум её никогда не выбирает.
        views = [(item["question"], item["answer"]) for item in items]
        encoded = embedder.encode(
            [PASSAGE_PREFIX + part for view in views for part in view]
        )
        per_item = len(views[0]) if views else 0

        entries = [
            _Entry(
                chunk=ContextChunk(
                    chunk_id=item["chunk_id"],
                    text=text,
                    source_title=item["source_title"],
                    source_url=item["source_url"],
                    relevance_score=0.0,
                ),
                vectors=tuple(
                    encoded[index * per_item : (index + 1) * per_item]
                ),
            )
            for index, (item, text) in enumerate(zip(items, texts, strict=True))
        ]
        return cls(entries, embedder)

    def search(
        self, query: str, *, top_n: int, threshold: float, top_k: int | None = None
    ) -> list[ContextChunk]:
        """Найти фрагменты для промпта, от самого близкого.

        Порядок ровно тот, что задан ADR-006: `top_k` кандидатов по близости,
        затем отсев порогом, затем `top_n` в промпт. Переранжирования между
        ними нет (ADR-014).

        **Пустой результат — законный ответ, а не сбой.** Ниже порога контекст
        считается не найденным, и включается запасной путь без модели
        (раздел 8.2): лучше сказать «не знаю», чем дать модели чужой фрагмент и
        получить уверенный вымысел.
        """
        if not query.strip() or not self._entries:
            return []

        (vector,) = self._embedder.encode([QUERY_PREFIX + query])
        scored = sorted(
            ((entry.similarity(vector), entry) for entry in self._entries),
            key=lambda pair: -pair[0],
        )
        candidates = scored[: top_k] if top_k is not None else scored
        return [
            entry.chunk.model_copy(update={"relevance_score": score})
            for score, entry in candidates[:top_n]
            if score >= threshold
        ]


class SentenceTransformerEmbedder:
    """Модель эмбеддингов из `sentence-transformers`.

    Импорт лениво: он тянет torch и занимает секунды, а модулю, который просто
    объявляет тип, платить за это незачем — тот же приём, что с LiteLLM в шлюзе.
    """

    def __init__(
        self, model_name: str, device: str = "cpu", *, local_files_only: bool = True
    ) -> None:
        """
        :param local_files_only: брать веса только из локального кэша.

        **По умолчанию — только из кэша, и это не оптимизация, а требование
        среды.** На MVP сервис живёт в закрытой сети, где `huggingface.co`
        недоступен вовсе; обращение к нему при старте там обернулось бы
        ожиданием сетевых таймаутов на пустом месте.

        На прототипе это заодно экономит время. Замер подъёма сервиса:

        ===================================== ======
          с обращением к `huggingface.co`      19,1 с
          только из кэша                        5,3 с
        ===================================== ======

        То есть **четырнадцать секунд из сорока шести** уходило на проверку,
        не обновилась ли модель, — при том что менять её молча нельзя: смена
        модели эмбеддингов означает переиндексацию корпуса (ADR-006).
        """
        self._model_name = model_name
        self._device = device
        self._local_files_only = local_files_only
        self._model: object | None = None

    def warm_up(self) -> None:
        """Поднять модель заранее.

        Проект четырежды находил тяжёлую загрузку, спрятанную в первый запрос
        абонента (разделы 50.3, 51.1, 54.5 и замер ADR-014). Здесь она поднята
        явно.
        """
        self._ensure()

    def _ensure(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    local_files_only=self._local_files_only,
                )
            except Exception as exc:
                if not self._local_files_only:
                    raise
                # Отказ из-за пустого кэша выглядит как отказ сети, и без
                # подсказки следующий человек будет искать не там.
                raise RuntimeError(
                    f"модель {self._model_name!r} не найдена в локальном кэше. "
                    "Скачать один раз: make install-search — либо поднять сервис "
                    "с EMBEDDING_LOCAL_FILES_ONLY=false, если сеть доступна"
                ) from exc
        return self._model

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        model = self._ensure()
        # `sentence-transformers` не типизирован под наш протокол, но по форме
        # совпадает: `encode` возвращает массив векторов.
        vectors = model.encode(list(texts), normalize_embeddings=True)  # type: ignore[attr-defined]
        return cast("list[list[float]]", vectors.tolist())
