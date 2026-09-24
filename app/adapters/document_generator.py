"""Формирование документа и доставка ссылки на него абоненту.

Путь документа целиком::

    Renderer -> Artifact Store Client -> S3/MinIO
        -> ссылка -> Render API -> Оркестратор -> Backend -> виджет -> Абонент

Последнее звено было последним пробелом: в диаграмме виджета не встречалось ни
слова «файл», ни «ссылка». Здесь путь заканчивается ссылкой с ограниченным
сроком жизни, которую виджет показывает вложением в ленте диалога.

ЭТО НЕ ЗАПИСЬ ВО ВНЕШНЮЮ СИСТЕМУ.
Документ кладётся в наше собственное хранилище, а не в чужую систему, поэтому
гейты записи здесь другие: подтверждение абонента нужно (черновик показывается
до него, готовый документ — после), а идемпотентность обеспечивается именем
объекта, а не ключом операции.

> **[ПРЕДПОЛОЖЕНИЕ]** Хранилище готовых документов и срок жизни ссылки в проекте
> не заданы ни одним документом. Взято S3/MinIO как
> уже принятая технология и сутки на срок жизни; требует подтверждения.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from app.models import InquiryState, SessionState

__all__ = ["DEFAULT_LINK_TTL_SECONDS", "Document", "DocumentGenerator"]

DEFAULT_LINK_TTL_SECONDS = 24 * 3600
"""Срок жизни ссылки на документ.

Ссылка ведёт на документ с персональными данными абонента, поэтому вечной быть
не может. Сутки — предположение: достаточно, чтобы абонент успел скачать, и
достаточно мало, чтобы утёкшая ссылка быстро перестала работать.
"""


class _StoreLike(Protocol):
    """Минимум, который нужен от хранилища объектов."""

    def put_object(self, key: str, body: bytes, content_type: str) -> None: ...
    def presigned_url(self, key: str, ttl_seconds: int) -> str: ...


@dataclass(frozen=True, slots=True)
class Document:
    """Готовый документ и ссылка на него."""

    object_key: str
    url: str
    template: str
    size_bytes: int
    ttl_seconds: int = DEFAULT_LINK_TTL_SECONDS


class DocumentGenerator:
    """Собирает документ по шаблону и кладёт его в хранилище.

    :param store: хранилище объектов. Внедряется снаружи: на прототипе вместо
        MinIO работает дубль, чтобы проверки не требовали поднятого контейнера.
    """

    def __init__(self, store: _StoreLike | None = None) -> None:
        self._store = store

    def render(self, template: str, values: dict[str, Any]) -> bytes:
        """Подставить значения в шаблон.

        Заглушка: настоящие шаблоны — типовые формы заявлений, которые придут
        образцами от владельца базы знаний (часть A8 запроса данных). До этого
        рендер собирает простой текст, а не пытается воспроизвести форму, которой
        никто не видел.
        """
        lines = [f"Шаблон: {template}", ""]
        lines.extend(f"{name}: {value}" for name, value in values.items())
        return "\n".join(lines).encode("utf-8")

    def build(
        self,
        session: SessionState,
        template: str,
        values: dict[str, Any],
        *,
        ttl_seconds: int = DEFAULT_LINK_TTL_SECONDS,
    ) -> Document:
        """Сформировать документ и вернуть ссылку на него.

        Готовый документ выдаётся **после** подтверждения абонента: до него
        абонент видит черновик в диалоге подтверждения, и это разные состояния
        одного документа.

        :raises RuntimeError: хранилище не задано либо абонент не подтвердил.
        """
        if session.current_state is not InquiryState.AWAITING_CONFIRMATION:
            raise RuntimeError(
                f"готовый документ выдаётся после подтверждения; "
                f"текущее состояние: {session.current_state.value}"
            )
        if self._store is None:
            raise RuntimeError("хранилище документов не задано")

        body = self.render(template, values)

        # Имя объекта детерминировано: тот же документ той же сессии не создаёт
        # второй копии. Идемпотентность здесь обеспечивается именем, а не ключом
        # операции — хранилище просто перезапишет одинаковое содержимое.
        digest = hashlib.sha256(body).hexdigest()[:16]
        object_key = f"documents/{session.session_id}/{template}-{digest}.txt"

        self._store.put_object(object_key, body, "text/plain; charset=utf-8")
        url = self._store.presigned_url(object_key, ttl_seconds)

        session.record(
            "document_generator",
            "document:ready",
            template=template,
            object_key=object_key,
            ttl_seconds=ttl_seconds,
        )
        return Document(
            object_key=object_key,
            url=url,
            template=template,
            size_bytes=len(body),
            ttl_seconds=ttl_seconds,
        )
