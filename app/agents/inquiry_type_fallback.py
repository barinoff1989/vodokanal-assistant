"""Запасной классификатор типа обращения: эмбеддинги, тем же приёмом, что у темы.

ПОЧЕМУ ТОЛЬКО СЕМЬ ТИПОВ ИЗ ШЕСТНАДЦАТИ. `InquiryType` — регистрационная ось
(ADR-011), но регистрация на прототипе предлагается только по семи закрытым
типам (`app/backend/registration.py`, `REGISTRABLE`): поверка, опломбировка,
установка/снятие прибора, справка, обследование, приём документов. Ошибка
классификации по остальным девяти (`refund`, `debt`, `payment`, `penalty`,
`accrual_recalculation`, `inquiry_cancellation`) сегодня ни на что
функционально не влияет — они не запускают предложение регистрации, их
единственный потребитель — разбивка метрики (раздел 56.11).

**Решающий довод — качество данных, не только объём работы.** Разбор
`fixtures/inquiries_voronezh.csv` по этим девяти типам показал: реальные
тексты — по существу споры («откуда задолженность, если я плачу», «почему
сумма выросла втрое»), а не запросы на регистрацию. Строить на них примеры
для эмбеддингов значило бы учить классификатор смешивать спор с заявкой —
ту же ошибку, которую `AccountResponder._DISPUTE_MARKERS` отдельно отсекает
на соседней оси. Расширение на эти девять типов — отдельная работа, когда
появится размеченный корпус вопросов-заявок, а не жалоб.

ПОЧЕМУ НЕ ГЕНЕРАТИВНАЯ МОДЕЛЬ. Тот же довод, что у `topic_fallback.py`:
классификация — не генерация, k-NN по размеченным примерам той же моделью
эмбеддингов, что уже поднята и тёплая для поиска по базе знаний
(`KnowledgeBase.embedder`) и для запасного классификатора темы. Второй копии
весов нет, вызов — не вызов внешнего провайдера.

ПРИМЕРЫ — СМЕСЬ НАСТОЯЩИХ ТЕКСТОВ И ПЕРЕФОРМУЛИРОВОК, И ЭТО РАЗВЕДЕНО ЯВНО.
`fixtures/inquiries_voronezh.csv` даёт девять настоящих текстов на тип, но их
регистр разный: где-то это фразы, близкие к тому, как спросил бы абонент в
чате («Необходима поверка счётчика горячей воды»), где-то — операторская
пометка без формы вопроса («снятие кп», «хв», «пу хвс»). Для типов со вторым
случаем (`INSPECTION`, `METER_INSTALLATION`, `CERTIFICATE`) примеры
переформулированы под чат и помечены явно — как когда-то `TOPIC_MARKERS` для
тем без наблюдённого словаря.

ПОРОГ И РАЗРЫВ — СВОИ, НЕ ПЕРЕНЕСЕНЫ С ТЕМЫ. У `Topic` шесть значений, здесь
семь плюс более узкая предметная область (все примеры — про счётчики,
документы, справки), значит соседние типы в среднем ближе друг к другу.
Числа `TYPE_MATCH_THRESHOLD`/`TYPE_MATCH_MARGIN` — стартовое приближение,
требуют того же прогона на отложенной части `fixtures/inquiries_voronezh.csv`,
что описан в `scripts/measure_classifier.py`, прежде чем считаться проверенными.

ЛЮБОЙ ОТКАЗ — ЭТО `None`, НЕ ИСКЛЮЧЕНИЕ. Эмбеддер недоступен, тип не набрал
порог или разрыв с соседним типом мал — вызывающий код идёт по прежнему пути
(`classify_by_keywords` → `other`), как если бы классификатора не было вовсе.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.kb.search import PASSAGE_PREFIX, QUERY_PREFIX, Embedder, cosine
from app.taxonomy import InquiryType

__all__ = ["EXAMPLES", "InquiryTypeFallbackClassifier"]

logger = logging.getLogger(__name__)

TYPE_MATCH_THRESHOLD = 0.90
"""Ниже — тип не найден. [ПРЕДПОЛОЖЕНИЕ] — стартовое значение по образцу
`TOPIC_MATCH_THRESHOLD` (`app/agents/topic_fallback.py`), не проверено прогоном
на отложенной части `fixtures/inquiries_voronezh.csv`. Пересчитать первым
(пункт TODO), прежде чем доверять числу."""

TYPE_MATCH_MARGIN = 0.05
"""Обязательный отрыв лучшего типа от второго. [ПРЕДПОЛОЖЕНИЕ], см. выше —
перенесено с `TOPIC_MATCH_MARGIN` как стартовая точка, не как измеренная
величина. Семь типов из одной предметной области (счётчики, документы) дадут
в среднем более тесные совпадения, чем шесть разнородных тем, — разрыв,
вероятно, придётся расширить по итогам замера."""


@dataclass(frozen=True, slots=True)
class _Example:
    text: str
    inquiry_type: InquiryType
    real: bool
    """Взят дословно (с лёгкой правкой опечаток) из `fixtures/inquiries_voronezh.csv`
    (``True``) или переформулирован под форму вопроса в чате (``False``,
    предположение)."""


EXAMPLES: tuple[_Example, ...] = (
    # --- meter_verification — настоящие тексты дают форму вопроса --------- #
    _Example(
        "Необходима поверка счётчика горячей воды",
        InquiryType.METER_VERIFICATION, real=True,
    ),
    _Example("Заказ поверки счётчика", InquiryType.METER_VERIFICATION, real=True),
    _Example(
        "Заказать поверку счётчика горячей воды",
        InquiryType.METER_VERIFICATION, real=True,
    ),
    _Example(
        "Обращаюсь за поверкой приборов учёта, прошу не считать по среднему",
        InquiryType.METER_VERIFICATION, real=True,
    ),
    _Example(
        "Хочу заказать поверку счётчика холодной воды",
        InquiryType.METER_VERIFICATION, real=False,
    ),
    # --- meter_sealing — настоящие тексты дают форму вопроса --------------- #
    _Example(
        "Прошу направить специалиста для ввода в эксплуатацию прибора учёта холодной воды",
        InquiryType.METER_SEALING, real=True,
    ),
    _Example(
        "Прошу направить специалиста для опломбировки счётчика",
        InquiryType.METER_SEALING, real=True,
    ),
    _Example("Требуется опломбировать счётчики", InquiryType.METER_SEALING, real=True),
    _Example(
        "Нужна опломбировка нового прибора учёта",
        InquiryType.METER_SEALING, real=False,
    ),
    # --- meter_installation — настоящие тексты почти все короткие коды ----- #
    _Example(
        "Хочу заменить прибор учёта холодной воды",
        InquiryType.METER_INSTALLATION, real=False,
    ),
    _Example(
        "Нужна замена счётчика горячей воды",
        InquiryType.METER_INSTALLATION, real=False,
    ),
    _Example(
        "Прошу установить новый прибор учёта воды",
        InquiryType.METER_INSTALLATION, real=False,
    ),
    _Example(
        "Требуется установка счётчика на кухне",
        InquiryType.METER_INSTALLATION, real=False,
    ),
    # --- meter_unsealing — настоящие тексты дают форму вопроса ------------- #
    _Example(
        "Снятие пломбы прибора учёта холодной воды",
        InquiryType.METER_UNSEALING, real=True,
    ),
    _Example(
        "Прошу снять пломбу со счётчика для установки обратного клапана",
        InquiryType.METER_UNSEALING, real=True,
    ),
    _Example(
        "Нужно снять прибор учёта с эксплуатации досрочно",
        InquiryType.METER_UNSEALING, real=True,
    ),
    _Example("Как снять пломбу со счётчика воды?", InquiryType.METER_UNSEALING, real=False),
    # --- certificate — «справка» в этой системе про сделку с недвижимостью #
    _Example("Нужна справка для продажи квартиры", InquiryType.CERTIFICATE, real=False),
    _Example(
        "Прошу справку об отсутствии задолженности для сделки",
        InquiryType.CERTIFICATE, real=False,
    ),
    _Example(
        "Нужно снять контрольные показания для продажи квартиры",
        InquiryType.CERTIFICATE, real=True,
    ),
    _Example(
        "Обследовать на степень благоустройства перед сделкой",
        InquiryType.CERTIFICATE, real=True,
    ),
    # --- inspection — все настоящие тексты буквально «снятие кп» ----------- #
    _Example("Прошу прислать специалиста для обследования", InquiryType.INSPECTION, real=False),
    _Example("Нужно снять контрольные показания счётчика", InquiryType.INSPECTION, real=False),
    _Example("Хочу заказать обследование прибора учёта", InquiryType.INSPECTION, real=False),
    # --- document_submission — настоящие тексты дают форму вопроса --------- #
    _Example(
        "Направляю акт метрологической поверки",
        InquiryType.DOCUMENT_SUBMISSION, real=True,
    ),
    _Example(
        "Прикладываю документы о смене собственника",
        InquiryType.DOCUMENT_SUBMISSION, real=True,
    ),
    _Example(
        "Направляю паспорт прибора учёта с отметкой прохождения поверки",
        InquiryType.DOCUMENT_SUBMISSION, real=True,
    ),
    _Example("Прошу принять акт поверки счётчика", InquiryType.DOCUMENT_SUBMISSION, real=False),
)
"""Примеры ограничены семью регистрируемыми типами (`REGISTRABLE`) — остальные
девять на прототипе не запускают действие, расширение на них требует другого
корпуса (см. заголовок модуля). ``real`` — не для классификатора, а чтобы при
следующей правке было видно, где переформулировка, а где цитата."""


class InquiryTypeFallbackClassifier:
    """k-NN по эмбеддингам размеченных примеров — тип ближайшего соседа.

    :param embedder: модель эмбеддингов. Берётся у уже поднятой базы знаний
        (`KnowledgeBase.embedder`) — вторая копия весов не грузится.
    :param examples: размеченные примеры. По умолчанию — :data:`EXAMPLES`.
    :param threshold: порог близости. По умолчанию — :data:`TYPE_MATCH_THRESHOLD`.
    :param margin: обязательный отрыв от второго (другого) типа. По умолчанию —
        :data:`TYPE_MATCH_MARGIN`.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        examples: Sequence[_Example] = EXAMPLES,
        threshold: float = TYPE_MATCH_THRESHOLD,
        margin: float = TYPE_MATCH_MARGIN,
    ) -> None:
        self._embedder = embedder
        self._examples = tuple(examples)
        self._threshold = threshold
        self._margin = margin
        self._vectors: tuple[Sequence[float], ...] | None = None

    def _ensure_vectors(self) -> tuple[Sequence[float], ...]:
        """Проэмбеддить примеры один раз, лениво — при первом обращении.

        Не в конструкторе: сборка приложения не должна ждать модель, если
        запасной классификатор в итоге не понадобится ни разу за прогон
        проверок."""
        if self._vectors is None:
            self._vectors = tuple(
                self._embedder.encode(
                    [PASSAGE_PREFIX + example.text for example in self._examples]
                )
            )
        return self._vectors

    async def classify(self, query: str) -> InquiryType | None:
        """Определить тип обращения, который не узнали ключевые слова.

        ``None`` — ближайший пример не набрал порог, либо лучший тип не
        оторвался от второго на нужный разрыв. Вызывающий код в обоих случаях
        идёт по прежнему пути — `classify_by_keywords`, затем `other`.

        Асинхронный метод ради единого интерфейса с
        `TopicFallbackClassifier.classify` и `Triage.classify_async`: сам
        подсчёт синхронный и быстрый, `await` не ждёт внешнего провайдера.
        """
        if not query.strip() or not self._examples:
            return None
        try:
            (vector,) = self._embedder.encode([QUERY_PREFIX + query])
            vectors = self._ensure_vectors()
        except Exception as exc:  # noqa: BLE001 — отказ эмбеддера не роняет путь
            logger.warning("запасной классификатор типа обращения недоступен (%s)", exc)
            return None

        scored = sorted(
            (
                (cosine(vector, example_vector), example.inquiry_type)
                for example, example_vector in zip(self._examples, vectors, strict=True)
            ),
            key=lambda pair: -pair[0],
        )
        best_score, best_type = scored[0]
        if best_score < self._threshold:
            return None

        runner_up = next(
            (score for score, itype in scored[1:] if itype is not best_type), 0.0
        )
        if best_score - runner_up < self._margin:
            return None
        return best_type
