"""Поиск по базе знаний: эмбеддинг запроса, косинусная близость, порог.

Корпус прототипа — FAQ воронежского филиала (ADR-100), двадцать пар
вопрос-ответ из открытого источника.

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

ХРАНИЛИЩЕ — ЗА ПРОТОКОЛОМ `VectorStore`. По умолчанию — `InMemoryVectorStore`
(двадцать фрагментов не нуждались в контейнере со своим жизненным циклом), но
`KnowledgeBase` не знает, память под капотом или Qdrant (`app/kb/qdrant_store.py`,
`settings.vector_store`) — обе реализации отдают одинаковый контракт:
по одной паре «фрагмент, лучшая близость среди его частей» на фрагмент. Форма
запроса от этого не меняется: `top_k` кандидатов по близости, отсев порогом,
`top_n` в промпт.

**Переранжирования между ними нет** (ADR-300): кросс-энкодер стоит 5,8 с
на двадцати кандидатах при бюджете поиска 100 мс — замерено, а не предположено.

МОДЕЛЬ ВНЕДРЯЕТСЯ СНАРУЖИ. Импорт `sentence-transformers` тянет torch и занимает
секунды; проверки не должны за это платить, а подмена модели — единственный
способ проверить сам поиск, а не качество эмбеддингов.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from app.models import ContextChunk

if TYPE_CHECKING:
    from app.kb.reranker import Reranker

__all__ = [
    "PASSAGE_PREFIX",
    "QUERY_PREFIX",
    "Embedder",
    "InMemoryVectorStore",
    "KbItem",
    "KnowledgeBase",
    "SentenceTransformerEmbedder",
    "VectorStore",
    "cosine",
    "faq_items",
    "split_answer",
]

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder(Protocol):
    """Минимум, который нужен от модели эмбеддингов."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Векторы для набора текстов, по одному на текст."""
        ...


def cosine(first: Sequence[float], second: Sequence[float]) -> float:
    """Косинусная близость (ADR-200, метрика зафиксирована там).

    Результат подрезается к отрезку от нуля до единицы: `relevance_score` в
    контракте объявлен именно так, а вычислительная погрешность даёт единицу с
    небольшим хвостом на совпадающих векторах.
    """
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norm = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    if norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / norm))


_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")


def split_answer(answer: str, max_chars: int) -> list[str]:
    """Разрезать ответ на части, годные для сравнения с вопросом абонента.

    :param answer: текст ответа.
    :param max_chars: предел длины части.

    **Режется по готовым границам, а не по счёту знаков.** Сначала по строкам:
    одиннадцать ответов из двадцати — списки способов («по телефону…», «в личном
    кабинете…», «в офисе…»), и каждый пункт отвечает на свой вопрос абонента.
    Длинная строка режется дальше по предложениям, и предложения набираются в
    часть жадно, пока помещаются, — иначе вопрос сравнивался бы с обрывком.

    Предложение длиннее предела **не режется**: обрубок на середине мысли даёт
    вектор, не значащий ничего, и лучше оставить одну длинную часть, чем две
    бессмысленные.
    """
    parts: list[str] = []
    for line in (piece.strip() for piece in answer.split("\n")):
        if not line:
            continue
        if len(line) <= max_chars:
            parts.append(line)
            continue

        current = ""
        for sentence in _SENTENCE_END.split(line):
            if not sentence:
                continue
            if current and len(current) + 1 + len(sentence) > max_chars:
                parts.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            parts.append(current)
    return parts or [answer.strip()]


@dataclass(frozen=True, slots=True)
class KbItem:
    """Запись корпуса до индексации — общая форма для всех источников.

    :param title: то, что говорит, о чём текст: вопрос FAQ либо путь заголовков.
    :param body: сам текст.
    :param synthetic: собрана ли запись генератором, а не пришла от владельца.
    :param inquiry_type: значение `InquiryType`, к которому относится запись;
        ``None`` — общая запись, подходит под любой тип (ADR-200).
    """

    chunk_id: str
    title: str
    body: str
    source_title: str
    source_url: str | None = None
    synthetic: bool = False
    inquiry_type: str | None = None


def faq_items(path: Path) -> list[KbItem]:
    """Пары «вопрос-ответ» из корпуса FAQ."""
    return [
        KbItem(
            chunk_id=item["chunk_id"],
            title=item["question"],
            body=item["answer"],
            source_title=item["source_title"],
            source_url=item["source_url"],
        )
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]


@dataclass(frozen=True, slots=True)
class _Entry:
    """Фрагмент корпуса вместе с векторами его частей.

    Векторов несколько: вопрос и ответ индексируются порознь, чтобы короткий
    вопрос не растворялся в длинном ответе. Близость фрагмента — лучшая из
    близостей его частей.
    """

    chunk: ContextChunk
    vectors: tuple[Sequence[float], ...]
    inquiry_type: str | None = None

    def similarity(self, query: Sequence[float]) -> float:
        """Лучшая близость среди частей.

        Максимум, а не среднее: среднее вернуло бы то самое разбавление, ради
        устранения которого части и разведены.
        """
        return max(cosine(query, vector) for vector in self.vectors)


class VectorStore(Protocol):
    """Куда `KnowledgeBase` кладёт и где ищет векторы фрагментов.

    Единственный контракт: `search_parts` отдаёт **по одной паре на фрагмент**
    (чанк, лучшая близость среди его частей) — группировка внутри фрагмента
    (title-вектор vs части ответа) уже сделана реализацией, `KnowledgeBase` её
    не повторяет. Обе реализации проекта (`InMemoryVectorStore`,
    `app.kb.qdrant_store.QdrantVectorStore`) держат этот контракт одинаково,
    поэтому смена хранилища не меняет `KnowledgeBase.search()` ни на строку.
    """

    def upsert(self, entries: Sequence[_Entry]) -> None:
        """Добавить фрагменты в хранилище."""
        ...

    def search_parts(
        self, vector: Sequence[float], inquiry_type: str | None = None
    ) -> list[tuple[ContextChunk, float]]:
        """Все фрагменты против вектора запроса, по одной паре на фрагмент.

        `inquiry_type` — фильтр, который участвует **в самом поиске**, а не
        применяется к готовой выдаче (ADR-200): из кандидатов исключаются
        фрагменты чужого типа. Фрагменты без типа (общие регламенты) подходят
        под любой. ``None`` — без фильтра.
        """
        ...

    def __len__(self) -> int:
        """Сколько различных фрагментов в хранилище (не векторов-частей)."""
        ...


class InMemoryVectorStore:
    """Хранилище прототипа — список в памяти процесса (ADR-100, таблица упрощённых замен).

    Оправдано только объёмом: двадцать фрагментов не нуждаются в контейнере со
    своим жизненным циклом. Живёт, пока жив процесс — ничего не пишется на диск.
    """

    def __init__(self) -> None:
        self._entries: list[_Entry] = []

    def upsert(self, entries: Sequence[_Entry]) -> None:
        self._entries.extend(entries)

    def search_parts(
        self, vector: Sequence[float], inquiry_type: str | None = None
    ) -> list[tuple[ContextChunk, float]]:
        return [
            (entry.chunk, entry.similarity(vector))
            for entry in self._entries
            if inquiry_type is None
            or entry.inquiry_type is None
            or entry.inquiry_type == inquiry_type
        ]

    def __len__(self) -> int:
        return len(self._entries)


class KnowledgeBase:
    """Проиндексированный корпус и поиск по нему."""

    def __init__(
        self, store: VectorStore, embedder: Embedder, reranker: Reranker | None = None
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker

    def __len__(self) -> int:
        return len(self._store)

    @property
    def embedder(self) -> Embedder:
        """Модель эмбеддингов, которой проиндексирован корпус.

        Открыта наружу, чтобы другие потребители того же процесса (запасной
        классификатор темы, `app/agents/topic_fallback.py`) не грузили вторую
        копию весов ради того же самого — модель эмбеддингов на прототипе не
        бесплатна по времени подъёма."""
        return self._embedder

    @classmethod
    def from_items(
        cls,
        items: Sequence[KbItem],
        embedder: Embedder,
        *,
        part_max_chars: int = 300,
        store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> KnowledgeBase:
        """Проиндексировать готовые записи, откуда бы они ни пришли.

        :param items: записи корпуса — пара FAQ либо раздел документа.
        :param part_max_chars: предел длины части тела при разрезании.
        :param store: куда положить векторы. По умолчанию — `InMemoryVectorStore`;
            для Qdrant передать `app.kb.qdrant_store.QdrantVectorStore(...)`.
        :param reranker: кросс-энкодер поверх `top_k` (ADR-300). По умолчанию
            выключен — `app.kb.reranker.CrossEncoderReranker` неработоспособен
            на CPU синхронного пути (ADR-300), опция для офлайн/GPU-сценариев.

        **Один код на два источника, потому что устройство у них одно.** У пары
        FAQ вопрос и ответ; у раздела документа — путь заголовков и текст
        раздела. Заголовок несёт ровно то же, что вопрос: он говорит, о чём
        текст, словами, близкими к вопросу абонента. Значит и индексируются они
        одинаково: заголовок отдельным вектором, тело — частями.
        """
        views = [
            (item.title, *split_answer(item.body, part_max_chars)) for item in items
        ]
        # Части индексируются порознь, но одним вызовом: модель дорого поднимать,
        # а не звать, и разбивать вызов незачем.
        # Склейка «вопрос + ответ» отдельным видом **не берётся**: проверено
        # замером — hit@1, hit@3 и MRR не сдвинулись ни на единицу, а при
        # действующем пороге стало на один ответ хуже. Её близость всегда лежит
        # между близостями частей, то есть максимум её никогда не выбирает.
        encoded = embedder.encode(
            [PASSAGE_PREFIX + part for view in views for part in view]
        )

        # Векторы вернулись одним списком — разложить обратно по фрагментам.
        # Частей у фрагментов разное число, поэтому по срезам, а не по шагу.
        grouped: list[list[Sequence[float]]] = []
        cursor = 0
        for view in views:
            grouped.append(list(encoded[cursor : cursor + len(view)]))
            cursor += len(view)

        entries = [
            _Entry(
                chunk=ContextChunk(
                    chunk_id=item.chunk_id,
                    text="\n".join((item.title, item.body)),
                    source_title=item.source_title,
                    source_url=item.source_url,
                    synthetic=item.synthetic,
                    relevance_score=0.0,
                ),
                vectors=tuple(vectors),
                inquiry_type=item.inquiry_type,
            )
            for item, vectors in zip(items, grouped, strict=True)
        ]
        store = store if store is not None else InMemoryVectorStore()
        store.upsert(entries)
        return cls(store, embedder, reranker)

    @classmethod
    def from_file(
        cls,
        path: Path,
        embedder: Embedder,
        *,
        part_max_chars: int = 300,
        store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> KnowledgeBase:
        """Прочитать корпус FAQ и проиндексировать его целиком.

        :param part_max_chars: предел длины части ответа при разрезании.
        :param store: см. `from_items`.
        :param reranker: см. `from_items`.

        Индексация двадцати фрагментов занимает доли секунды (замер ADR-300: 48
        фрагментов в секунду), поэтому делается при запуске, а не заранее. Когда
        корпус вырастет, это станет отдельным шагом приёма — как у графика
        отключений.
        """
        return cls.from_items(
            faq_items(path),
            embedder,
            part_max_chars=part_max_chars,
            store=store,
            reranker=reranker,
        )

    def search(
        self,
        query: str,
        *,
        top_n: int,
        threshold: float,
        top_k: int | None = None,
        inquiry_type: str | None = None,
    ) -> list[ContextChunk]:
        """Найти фрагменты для промпта, от самого близкого.

        Порядок ровно тот, что задан ADR-200: `top_k` кандидатов по близости,
        затем переранжирование (если включено, ADR-300), затем отсев порогом,
        затем `top_n` в промпт. Переранжирование меняет **порядок** внутри
        `top_k` — оценка, с которой сверяется `threshold`, остаётся косинусной
        (см. `app/kb/reranker.py`: у кросс-энкодера другая шкала, калиброванного
        порога для неё нет). По умолчанию реранкера нет — переранжирования между
        ними нет (ADR-300).

        **Фильтр `inquiry_type` — внутри поиска** (ADR-200): чужой тип
        исключается до отбора `top_k`, а не после — иначе нужный фрагмент мог
        бы не попасть в двадцать кандидатов из-за соседей другого типа. Общие
        фрагменты (без типа) проходят любой фильтр.

        **Пустой результат — законный ответ, а не сбой.** Ниже порога контекст
        считается не найденным, и включается запасной путь без модели:
лучше сказать «не знаю», чем дать модели чужой фрагмент и
        получить уверенный вымысел.
        """
        if not query.strip() or len(self._store) == 0:
            return []

        (vector,) = self._embedder.encode([QUERY_PREFIX + query])
        scored = sorted(
            self._store.search_parts(vector, inquiry_type), key=lambda pair: -pair[1]
        )
        candidates = scored[:top_k] if top_k is not None else scored
        if self._reranker is not None:
            candidates = self._reranker.rerank(query, candidates)
        return [
            chunk.model_copy(update={"relevance_score": score})
            for chunk, score in candidates[:top_n]
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
        модели эмбеддингов означает переиндексацию корпуса (ADR-200).
        """
        self._model_name = model_name
        self._device = device
        self._local_files_only = local_files_only
        self._model: object | None = None

    def warm_up(self) -> None:
        """Поднять модель заранее.

        Проект четырежды находил тяжёлую загрузку, спрятанную в первый запрос
        абонента (замер ADR-300). Здесь она поднята
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
