"""Схемы данных: контракт шлюза и состояние обращения.

Источники: контракт API (контекст, раздел 7), состояние обращения и слоты
(раздел 10), таксономия (раздел 9). ER-диаграмма в проекте пропущена
осознанно (раздел 41.8), поэтому сущности выводятся напрямую отсюда.

ЧЕТЫРЕ РЕШЕНИЯ ПО ОТКРЫТЫМ ВОПРОСАМ СПЕЦИФИКАЦИИ (раздел 7.6). Модели их
кодируют, но сам файл спецификации ещё не обновлён — пункт 5 сведённого TODO
остаётся открытым, и до шага 8 расхождение нужно устранить:

1. `subscriber_id` вместо `user_id` (пункт 7 раздела 7.6). Пользователь —
   абонент, а не сотрудник; идентификатор назван по тому, кто он есть.
2. `department` из метаданных запроса УБРАН, хотя раздел 7.1 его содержал.
   Причина: департаменты не пользователи чат-интерфейса (раздел 1), и по этой
   же причине из Quota Manager убрали квоты по департаментам (раздел 35.1).
   Подразделение — свойство документа базы знаний (раздел 9.5), а не запроса.
   Это решение в обратную сторону от того, что предлагал раздел 7.6, пункт 4.
3. `sources` ДОБАВЛЕНЫ в ответ (пункт 2 раздела 7.6). Раньше поля источника
   принимались на входе, но не возвращались, — при том что DoD прямо требует
   «ответ содержит sources» (раздел 18). Выбран вариант «добавить в ответ», а
   не «удалить с входа».
4. У события ошибки в потоке ПОЯВИЛАСЬ схема (пункт 3 раздела 7.6): то же тело
   RFC 7807, что и у обычной ошибки. Один формат ошибки на оба пути дешевле в
   поддержке, чем два.

Отдельно про персональные данные: поля, куда они попадают, помечены
`repr=False`. Правило 4.2 запрещает логировать сырые ПДн, а самый частый способ
нарушить его — не отправка наружу, а случайный `print` или запись объекта в
журнал при отладке.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.taxonomy import InquiryType

__all__ = [
    "AuditEntry",
    "Channel",
    "ContextChunk",
    "DoneEvent",
    "ErrorEvent",
    "FinishReason",
    "GenerateRequest",
    "GenerateResponse",
    "GenerationParameters",
    "InquiryState",
    "MetadataEvent",
    "PiiReport",
    "ProblemDetail",
    "RequestMetadata",
    "SessionState",
    "Slots",
    "SourceRef",
    "SuggestedAction",
    "TokenEvent",
    "TriageResult",
    "Usage",
]

MAX_QUERY_LENGTH = 2000
"""Ограничение раздела 7.2. Проверяется здесь, а не в обработчике: иначе оно
повторится в трёх местах и разойдётся."""


class Channel(StrEnum):
    """Откуда пришёл запрос (раздел 7.6, пункт 6)."""

    LK_WEB = "lk_web"
    LK_MOBILE = "lk_mobile"


class FinishReason(StrEnum):
    """Почему генерация завершилась.

    `guardrail` — не из словаря провайдера: так помечается обрыв потока
    синхронной проверкой (раздел 35.1). Без отдельного значения этот случай
    был бы неотличим от нормального завершения в аналитике.
    """

    STOP = "stop"
    LENGTH = "length"
    GUARDRAIL = "guardrail"
    ERROR = "error"


class InquiryState(StrEnum):
    """Состояния обращения (раздел 10, `InquiryFSM`).

    `AWAITING_CONFIRMATION` — обязательный привал перед любой операцией записи
    (правило 4.7). Переходы задаются на шаге 2, здесь только сам перечень.
    """

    INIT = "init"
    CLASSIFIED = "classified"
    FILLING = "filling"
    DRAFT_READY = "draft_ready"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SUBMITTED = "submitted"
    COMPLETED = "completed"


# --- контракт шлюза: запрос -------------------------------------------------- #


class ContextChunk(BaseModel):
    """Фрагмент базы знаний, найденный поиском и переданный модели."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    source_title: str
    source_url: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)


class GenerationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "vodokanal-assistant"
    max_tokens: int = Field(default=200, ge=1, le=4096)
    """По умолчанию 200, а не «побольше»: замер на YandexGPT показал 4,73 с на
    длинном ответе при цели DoD «полный ответ < 3 с» (раздел 46.7). Бюджет в три
    секунды — это примерно 150–200 токенов. Пункт 31 сведённого TODO."""

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    stream: bool = True
    """Поток по умолчанию: правило 4.3 требует, чтобы ответ доходил до абонента
    постепенно. Значение `False` допустимо для служебных вызовов вроде
    классификации, где промежуточный вывод некому показывать."""


class RequestMetadata(BaseModel):
    """Метаданные запроса: аудит, квоты, аналитика (раздел 9.7)."""

    model_config = ConfigDict(extra="forbid")

    subscriber_id: str
    session_id: str
    channel: Channel = Channel.LK_WEB
    inquiry_type: InquiryType | None = None
    """Может отсутствовать до классификации — на самом первом обращении
    абонента типа ещё нет."""

    @field_validator("inquiry_type", mode="before")
    @classmethod
    def _normalize_inquiry_type(cls, value: Any) -> Any:
        """Нормализация на входе, а не в трёх местах дальше.

        Через этот класс проходит всё, что приходит извне, — здесь и снимаются
        пробелы с регистром, из-за которых прототип не находил документы
        (раздел 11.1).
        """
        if value is None or isinstance(value, InquiryType):
            return value
        if isinstance(value, str):
            from app.taxonomy import parse

            return parse(value)
        return value


class GenerateRequest(BaseModel):
    """Запрос к шлюзу (раздел 7.1)."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(min_length=1)
    """Обязателен (раздел 7.2): без системной части модель отвечает вне роли."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH, repr=False)
    """`repr=False` — текст абонента может содержать ФИО, адрес, лицевой счёт."""

    context: list[ContextChunk] = Field(default_factory=list)
    """Может быть пустым, но тогда обязателен запасной путь (раздел 7.2)."""

    parameters: GenerationParameters = Field(default_factory=GenerationParameters)
    metadata: RequestMetadata

    @property
    def has_context(self) -> bool:
        """Пустой контекст — признак того, что нужен ответ без модели."""
        return bool(self.context)


# --- контракт шлюза: ответ ---------------------------------------------------- #


class SourceRef(BaseModel):
    """Источник, на котором основан ответ. Показывается абоненту."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_title: str
    source_url: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)


class SuggestedAction(BaseModel):
    """Кнопка следующего шага в виджете (раздел 8.4)."""

    model_config = ConfigDict(extra="forbid")

    action: str
    label: str
    inquiry_type: InquiryType | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class PiiReport(BaseModel):
    """Отчёт об обезличивании — единственное, что разрешено писать в журнал.

    Правило 4.2: сами значения персональных данных сюда не попадают ни при
    каких условиях, только перечень типов найденных сущностей.
    """

    model_config = ConfigDict(extra="forbid")

    pii_detected: bool = False
    entities: list[str] = Field(default_factory=list)

    @field_validator("entities")
    @classmethod
    def _entities_are_type_names(cls, value: list[str]) -> list[str]:
        """Защита от того, что в отчёт положат значение, а не тип сущности.

        Проверка грубая — по длине и наличию цифр, — но ловит самый вероятный
        промах: запись `"1234567890"` вместо `"ACCOUNT_NUMBER"`.
        """
        for entity in value:
            if any(char.isdigit() for char in entity):
                raise ValueError(
                    f"в pii_report попало значение, а не тип сущности: {entity!r}"
                )
        return value


class GenerateResponse(BaseModel):
    """Ответ шлюза (раздел 7.3) с полями, которых требует DoD раздела 18."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(repr=False)
    model: str
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Field(default_factory=Usage)
    sources: list[SourceRef] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    disclaimer: str | None = None
    pii_report: PiiReport = Field(default_factory=PiiReport)
    routing: dict[str, Any] = Field(default_factory=dict)
    trace_id: str


# --- ошибки ------------------------------------------------------------------- #


class ProblemDetail(BaseModel):
    """Тело ошибки по RFC 7807 (правило 4.5).

    `trace_id` — не из стандарта, но добавлен намеренно: без него жалобу
    абонента невозможно связать с записью в журнале.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    title: str
    status: int = Field(ge=100, le=599)
    detail: str | None = None
    instance: str | None = None
    trace_id: str | None = None


# --- события потока (раздел 7.3) ---------------------------------------------- #


class TokenEvent(BaseModel):
    """Очередной кусочек ответа."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["token"] = "token"
    delta: str = Field(repr=False)


class MetadataEvent(BaseModel):
    """Источники и оценки — приходят после текста, перед завершением."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["metadata"] = "metadata"
    sources: list[SourceRef] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    disclaimer: str | None = None


class DoneEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: Literal["done"] = "done"
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Field(default_factory=Usage)
    trace_id: str


class ErrorEvent(BaseModel):
    """Ошибка внутри уже начатого потока.

    Схемы у этого события не было (раздел 7.6, пункт 3). Взято то же тело
    RFC 7807, что и у обычной ошибки: два разных формата ошибки на два пути
    пришлось бы поддерживать порознь.
    """

    model_config = ConfigDict(extra="forbid")

    event: Literal["error"] = "error"
    problem: ProblemDetail


# --- состояние обращения (раздел 10) ------------------------------------------ #


class Slots(BaseModel):
    """Данные, которые собираются у абонента.

    Три поля обязательны для черновика (раздел 10); проверяет их полноту не эта
    модель, а Draft Validator оркестратора (раздел 41.2) — здесь они могут быть
    пустыми, потому что заполняются постепенно, в диалоге.
    """

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(default=None, repr=False)
    account_number: str | None = Field(default=None, repr=False)
    phone: str | None = Field(default=None, repr=False)
    email: str | None = Field(default=None, repr=False)
    address: str | None = Field(default=None, repr=False)
    reason: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict, repr=False)

    REQUIRED: ClassVar[tuple[str, ...]] = ("full_name", "account_number", "reason")

    @field_validator(
        "full_name", "account_number", "phone", "email", "address", "reason", mode="before"
    )
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """Пробельное значение — это отсутствие значения, а не значение.

        Без этого поле из одних пробелов считалось бы заполненным, и черновик
        ушёл бы на подтверждение абонента с пустым по сути именем. Тот же класс
        ошибки, что хвостовые пробелы в таксономии (раздел 11.1): пробел
        выглядит как данные и ведёт себя как данные, ничего не ломая явно.
        """
        if isinstance(value, str):
            return value.strip() or None
        return value

    def missing_required(self) -> tuple[str, ...]:
        """Каких обязательных полей не хватает для черновика."""
        return tuple(name for name in self.REQUIRED if not getattr(self, name))

    @property
    def is_complete(self) -> bool:
        return not self.missing_required()


class TriageResult(BaseModel):
    """Результат классификации обращения (раздел 9.6)."""

    model_config = ConfigDict(extra="forbid")

    inquiry_type: InquiryType
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_slots: Slots = Field(default_factory=Slots)
    fallback_used: bool = False
    """Классификация по ключевым словам вместо модели. Отдельным полем, а не
    догадкой по низкой уверенности: долю таких ответов нужно видеть в метриках
    (раздел 37.2, Fallback Rate)."""

    @field_validator("inquiry_type", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, str) and not isinstance(value, InquiryType):
            from app.taxonomy import parse

            return parse(value)
        return value


class AuditEntry(BaseModel):
    """Запись аудита. Обязательна на каждом шаге записи (правило 4.7)."""

    model_config = ConfigDict(extra="forbid")

    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    """Сюда не кладутся значения персональных данных — только идентификаторы и
    названия полей, по тому же правилу, что и в `PiiReport`."""


class SessionState(BaseModel):
    """Состояние диалога с абонентом (раздел 10, `SessionState`)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    subscriber_id: str
    current_state: InquiryState = InquiryState.INIT
    inquiry_type: InquiryType | None = None
    inquiry_type_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    slots: Slots = Field(default_factory=Slots)
    draft_text: str | None = Field(default=None, repr=False)
    submission_result: dict[str, Any] | None = None
    audit_log: list[AuditEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def record(self, actor: str, action: str, **details: Any) -> AuditEntry:
        """Добавить запись аудита. Отдельный метод, чтобы её нельзя было забыть."""
        entry = AuditEntry(actor=actor, action=action, details=details)
        self.audit_log.append(entry)
        return entry

    @property
    def awaits_confirmation(self) -> bool:
        """Ждём ли подтверждения абонента (правило 4.7)."""
        return self.current_state is InquiryState.AWAITING_CONFIRMATION
