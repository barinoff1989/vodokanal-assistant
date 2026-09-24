"""Схемы данных: контракт шлюза и состояние обращения.

Источники: контракт API, состояние обращения и слоты,
таксономия. ER-диаграмма в проекте пропущена
осознанно, поэтому сущности выводятся напрямую отсюда.

ЧЕТЫРЕ РЕШЕНИЯ ПО ОТКРЫТЫМ ВОПРОСАМ СПЕЦИФИКАЦИИ. Модели их
кодируют, но сам файл спецификации ещё не обновлён —
расхождение нужно устранить:

1. `subscriber_id` вместо `user_id`. Пользователь —
   абонент, а не сотрудник; идентификатор назван по тому, кто он есть.
2. `department` из метаданных запроса УБРАН, хотя прежняя редакция его содержала.
   Причина: департаменты не пользователи чат-интерфейса, и по этой
   же причине из Quota Manager убрали квоты по департаментам.
   Подразделение — свойство документа базы знаний, а не запроса.
   Это решение в обратную сторону от того, что предлагала прежняя редакция.
3. `sources` ДОБАВЛЕНЫ в ответ. Раньше поля источника
   принимались на входе, но не возвращались, — при том что DoD прямо требует
   «ответ содержит sources». Выбран вариант «добавить в ответ», а
   не «удалить с входа».
4. У события ошибки в потоке ПОЯВИЛАСЬ схема: то же тело
   RFC 7807, что и у обычной ошибки. Один формат ошибки на оба пути дешевле в
   поддержке, чем два.

Отдельно про персональные данные: поля, куда они попадают, помечены
`repr=False`. Правило защиты ПДн запрещает логировать сырые ПДн, а самый частый способ
нарушить его — не отправка наружу, а случайный `print` или запись объекта в
журнал при отладке.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.taxonomy import InquiryType, Topic

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
    "Intent",
    "MetadataEvent",
    "PiiReport",
    "ProblemDetail",
    "RequestMetadata",
    "SessionState",
    "Slots",
    "SourceRef",
    "SuggestedAction",
    "TemplateFill",
    "TokenEvent",
    "TriageResult",
    "Usage",
]

MAX_QUERY_LENGTH = 2000
"""Ограничение длины запроса. Проверяется здесь, а не в обработчике: иначе оно
повторится в трёх местах и разойдётся."""


class Channel(StrEnum):
    """Откуда пришёл запрос."""

    LK_WEB = "lk_web"
    LK_MOBILE = "lk_mobile"


class FinishReason(StrEnum):
    """Почему генерация завершилась.

    `guardrail` — не из словаря провайдера: так помечается обрыв потока
    синхронной проверкой. Без отдельного значения этот случай
    был бы неотличим от нормального завершения в аналитике.
    """

    STOP = "stop"
    LENGTH = "length"
    GUARDRAIL = "guardrail"
    ERROR = "error"


class Intent(StrEnum):
    """Явное действие абонента над черновиком обращения.

    **Намерение приходит признаком, а не выводится из текста реплики.** Слова
    «да», «хорошо», «ага» абонент говорит и в ответ на что-то другое; прочитать
    их как подтверждение значило бы выполнить запись, которой он не просил.
    Запись требует **явного** подтверждения, а вывод из свободного текста
    явным не бывает.

    Признак ставит виджет по нажатию кнопки — там намерение однозначно по
    построению. Тот же довод, по которому тема определяется правилами, а не
    моделью: звать модель, чтобы узнать, разрешена ли
    запись, — значит поставить решение о записи в зависимость от угадывания.
    """

    CONFIRM = "confirm"
    REJECT = "reject"


class InquiryState(StrEnum):
    """Состояния обращения (`InquiryFSM`).

    `AWAITING_CONFIRMATION` — обязательный привал перед любой операцией записи.
Переходы задаются в `app/fsm.py`, здесь только сам перечень.
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

    synthetic: bool = False
    """Собран ли фрагмент генератором, а не получен от владельца.

    **Часть контракта, а не служебная пометка.** На прототипе настоящих
    регламентов нет (закрытая сеть), и вместо них индексируются синтетические
    документы. Ответ, построенный на выдуманной процедуре, обязан быть отличим
    от ответа по настоящему регламенту — иначе на демонстрации его примут за
    второе.

    По умолчанию `False`: корпус FAQ собран с публичного сайта водоканала и
    синтетическим не является."""


class GenerationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "vodokanal-assistant"
    max_tokens: int = Field(default=200, ge=1, le=4096)
    """По умолчанию 200, а не «побольше»: замер на YandexGPT показал 4,73 с на
    длинном ответе при цели DoD «полный ответ < 3 с». Бюджет в три
    секунды — это примерно 150–200 токенов."""

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    stream: bool = True
    """Поток по умолчанию: правило потоковой выдачи требует, чтобы ответ доходил до абонента
    постепенно. Значение `False` допустимо для служебных вызовов вроде
    классификации, где промежуточный вывод некому показывать."""


class RequestMetadata(BaseModel):
    """Метаданные запроса: аудит, квоты, аналитика."""

    model_config = ConfigDict(extra="forbid")

    subscriber_id: str
    session_id: str
    channel: Channel = Channel.LK_WEB
    inquiry_type: InquiryType | None = None
    """Может отсутствовать до классификации — на самом первом обращении
    абонента типа ещё нет."""

    topic: Topic | None = None
    """Тематическая ось (ADR-300): о чём вопрос, а не во что регистрировать.

    Определяет путь ответа. `outage` уводит на точный поиск по графику
    отключений, `water_quality` — на утверждённую формулировку; оба минуют
    модель. Отсутствует до классификации, как и `inquiry_type`."""

    intent: Intent | None = None
    """Что абонент делает с показанным ему черновиком.

    Пусто у обычного вопроса. Заполнено — значит абонент нажал кнопку под
    черновиком, и это единственный способ дойти до записи: правило подтверждения записи не
    признаёт подтверждением ни текст реплики, ни молчание.

    Намерение без ожидающего черновика — не ошибка абонента, а рассогласование:
    состояние истекло, сессия чужая или виджет отстал. Обрабатывается ответом
    «черновик не найден», а не записью.
    """

    address: str | None = None
    """Адрес абонента. Нужен там, где ответ зависит от места, — сейчас это
    график отключений.

    **Ведёт адрес Биллинг** (файл `ЛСФЛ`, источник 1), не запрос: `Orchestrator`
    проставляет его сам по `subscriber_id` (`_with_address`), а присланное
    клиентом значение не используется — иначе можно было бы прислать чужой адрес
    и получить график по нему. До оркестратора поле пустое; служебные вызовы и
    проверки, которым Биллинг не нужен, задают его напрямую.

    Строка в том виде, в каком её ведёт источник
    («г. Тестовый, ул. Набережная, д. 65, кв. 156»)."""

    @field_validator("inquiry_type", mode="before")
    @classmethod
    def _normalize_inquiry_type(cls, value: Any) -> Any:
        """Нормализация на входе, а не в трёх местах дальше.

        Через этот класс проходит всё, что приходит извне, — здесь и снимаются
        пробелы с регистром, из-за которых прототип не находил документы.
        """
        if value is None or isinstance(value, InquiryType):
            return value
        if isinstance(value, str):
            from app.taxonomy import parse

            return parse(value)
        return value


class GenerateRequest(BaseModel):
    """Запрос к шлюзу."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(min_length=1)
    """Обязателен: без системной части модель отвечает вне роли."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH, repr=False)
    """`repr=False` — текст абонента может содержать ФИО, адрес, лицевой счёт."""

    context: list[ContextChunk] = Field(default_factory=list)
    """Может быть пустым, но тогда обязателен запасной путь."""

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

    synthetic: bool = False
    """Составлен ли источник генератором, а не получен от владельца.

    Показывается абоненту вместе с самим источником: ответ по выдуманной
    процедуре обязан быть отличим от ответа по настоящему регламенту. Признак
    приходит из :class:`ContextChunk` и дальше не выводится — иначе появилось бы
    второе место, где решается, настоящий документ или нет."""


class SuggestedAction(BaseModel):
    """Кнопка следующего шага в виджете."""

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

    Правило защиты ПДн: сами значения персональных данных сюда не попадают ни при
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
    """Ответ шлюза с полями, которых требует DoD."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(repr=False)
    model: str
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Field(default_factory=Usage)
    sources: list[SourceRef] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    disclaimer: str | None = None
    document_url: str | None = None
    """Ссылка на готовый бланк заявления — то же, что в :class:`MetadataEvent`,
    для ответа целиком (служебные вызовы)."""
    pii_report: PiiReport = Field(default_factory=PiiReport)
    routing: dict[str, Any] = Field(default_factory=dict)
    trace_id: str


# --- ошибки ------------------------------------------------------------------- #


class ProblemDetail(BaseModel):
    """Тело ошибки по RFC 7807.

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
    retry_after: int | None = None
    """Через сколько секунд повторять — для `429` и `503` (ADR-400).

    Не из RFC 7807, как и `trace_id`, и добавлено по той же причине: величину
    надо передать, а места для неё в стандарте нет. Заголовок `Retry-After`
    собирается отсюда.

    Раньше поля не было, и HTTP-слой выскабливал число из русского текста
    `detail`. На `429` это работало случайно — там в пояснении стоит остаток
    окна; на `503` цифр в тексте нет вовсе, и заголовок всегда получал
    зашитую тридцатку, а настройка `quota_retry_after_seconds` не влияла ни на
    что. Величина, которую передают прозой, однажды перестаёт передаваться."""


# --- события потока ---------------------------------------------- #


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

    document_url: str | None = None
    """Ссылка на готовый бланк заявления для печати (тема `template`).

    Заполнена, когда ответчик образцов собрал бланк и положил его в хранилище
    (`app/documents/artifact.py`). Виджет показывает её отдельной панелью —
    состояние Document Panel на C3 виджета. На прототипе ведёт на маршрут стенда,
    не входящий в контракт `/v1`."""


class DoneEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: Literal["done"] = "done"
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Field(default_factory=Usage)
    pii_report: PiiReport = Field(default_factory=PiiReport)
    """Тот же отчёт, что и в ответе целиком: значений он не
    содержит по построению, только перечень типов найденного. Без него два пути
    — поток и ответ целиком — отдавали бы разное."""

    routing: dict[str, Any] = Field(default_factory=dict)
    """Псевдоним провайдера и фактически ответившая модель.

    Их расхождение и есть срабатывание запасного провайдера. Поле уже есть в
    ответе целиком; здесь оно появляется, чтобы поток отдавал то же
    самое — иначе переключение видно только в одном из двух режимов.

    Внутренние длительности сюда не попадают: обезличивание и проверка
    охранителей — величины сервера, виджету в кабинете они не нужны, и их место
    на порту метрик."""

    trace_id: str


class ErrorEvent(BaseModel):
    """Ошибка внутри уже начатого потока.

    Схемы у этого события не было. Взято то же тело
    RFC 7807, что и у обычной ошибки: два разных формата ошибки на два пути
    пришлось бы поддерживать порознь.
    """

    model_config = ConfigDict(extra="forbid")

    event: Literal["error"] = "error"
    problem: ProblemDetail


# --- состояние обращения ------------------------------------------ #


class Slots(BaseModel):
    """Данные, которые собираются у абонента.

    Три поля обязательны для черновика; проверяет их полноту не эта
    модель, а Draft Validator оркестратора — здесь они могут быть
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
        ошибки, что хвостовые пробелы в таксономии: пробел
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
    """Результат классификации обращения."""

    model_config = ConfigDict(extra="forbid")

    inquiry_type: InquiryType
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_slots: Slots = Field(default_factory=Slots)
    fallback_used: bool = False
    """Классификация по ключевым словам вместо модели. Отдельным полем, а не
    догадкой по низкой уверенности: долю таких ответов нужно видеть в метриках
    (метрика Fallback Rate)."""

    @field_validator("inquiry_type", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, str) and not isinstance(value, InquiryType):
            from app.taxonomy import parse

            return parse(value)
        return value


class AuditEntry(BaseModel):
    """Запись аудита. Обязательна на каждом шаге записи."""

    model_config = ConfigDict(extra="forbid")

    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    """Сюда не кладутся значения персональных данных — только идентификаторы и
    названия полей, по тому же правилу, что и в `PiiReport`."""


class TemplateFill(BaseModel):
    """Заполнение бланка заявления по репликам.

    Отдельно от `Slots`: слоты — это данные обращения, которое ассистент
    регистрирует сам; здесь — поля бумажного бланка, который абонент подаёт сам.
    Ответы — свободный текст абонента, поэтому `repr=False`.
    """

    model_config = ConfigDict(extra="forbid")

    template_id: str
    pending: str | None = None
    """Имя поля, которое сейчас спрашивают. `None` — все поля собраны."""

    answers: dict[str, str] = Field(default_factory=dict, repr=False)
    """Имя поля → ответ абонента. Пустая строка — поле пропущено, останется
    прочерком в бланке."""


class SessionState(BaseModel):
    """Состояние диалога с абонентом (`SessionState`)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    subscriber_id: str
    current_state: InquiryState = InquiryState.INIT
    inquiry_type: InquiryType | None = None
    inquiry_type_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    slots: Slots = Field(default_factory=Slots)
    draft_text: str | None = Field(default=None, repr=False)
    submission_result: dict[str, Any] | None = None
    template_fill: TemplateFill | None = None
    """Идёт ли сбор полей бланка заявления. `None` — не идёт."""

    audit_log: list[AuditEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def record(self, actor: str, action: str, **details: Any) -> AuditEntry:
        """Добавить запись аудита. Отдельный метод, чтобы её нельзя было забыть."""
        entry = AuditEntry(actor=actor, action=action, details=details)
        self.audit_log.append(entry)
        return entry

    @property
    def awaits_confirmation(self) -> bool:
        """Ждём ли подтверждения абонента."""
        return self.current_state is InquiryState.AWAITING_CONFIRMATION
