"""Учёт лимитов: сколько запросов разрешено абоненту, сессии и сервису.

Раздел 35.1 задаёт три предела и требует при превышении отвечать `429` с
заголовком `Retry-After` — в исходной версии диаграммы заголовок был пропущен,
и это считалось дефектом.

Квот по департаментам здесь нет намеренно: департаменты не пользователи
чат-интерфейса (раздел 1), и эта ошибка уже была исправлена в разделе 35.1.

ПОВЕДЕНИЕ ПРИ ОТКАЗЕ ХРАНИЛИЩА — ADR-008.
Redis недоступен — запросы отклоняются. Не пропускаются: отказ хранилища вполне
может быть *следствием* всплеска нагрузки, и тогда пропуск снял бы защиту ровно
в тот момент, когда она нужна. Расход токенов при этом — единственная статья,
растущая болезненно при масштабировании (раздел 42.5).

ОКНО СЧЁТА — ФИКСИРОВАННАЯ МИНУТА.
Счётчик живёт минуту и обнуляется на её границе. Скользящее окно точнее, но
требует хранить отметки времени каждого запроса; на прототипе это лишняя
сложность. Плата известна: на стыке двух минут абонент может отправить двойной
лимит. При лимитах порядка десятков запросов в минуту это несущественно.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

__all__ = [
    "QuotaDecision",
    "QuotaManager",
    "QuotaScope",
    "QuotaStoreUnavailableError",
]

WINDOW_SECONDS = 60


class QuotaScope(StrEnum):
    """Чей предел исчерпан.

    Значение попадает в метку метрики и в текст ошибки: без него по счётчику
    отказов нельзя понять, упёрся ли один абонент в свой лимит или сервис целиком
    в общий, а это разные инциденты с разными действиями.
    """

    SUBSCRIBER = "subscriber"
    SESSION = "session"
    SERVICE = "service"


class QuotaStoreUnavailableError(RuntimeError):
    """Хранилище счётчиков недоступно.

    Отдельный тип, а не обычная ошибка подключения: по ADR-008 это состояние
    отображается в `503`, тогда как превышение лимита — в `429`. Смешение кодов
    превратило бы отказ инфраструктуры в «активность абонентов, упирающихся в
    лимит», невидимую в метриках.
    """


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    """Решение по запросу."""

    allowed: bool
    scope: QuotaScope | None = None
    retry_after: int = 0
    used: int = 0
    limit: int = 0

    @property
    def detail(self) -> str:
        """Пояснение для тела ошибки. Значений счётчиков абонента не раскрывает."""
        if self.allowed:
            return ""
        return f"превышен лимит запросов ({self.scope}); повторите через {self.retry_after} с"


class _RedisLike(Protocol):
    """Минимум, который нужен от клиента Redis. Ради подмены в тестах."""

    def incr(self, name: str) -> int: ...
    def expire(self, name: str, time: int) -> bool: ...


class QuotaManager:
    """Считает три предела на счётчиках с временем жизни в одну минуту.

    :param client: клиент хранилища. Передаётся снаружи, чтобы шлюз не создавал
        подключение сам и чтобы проверки не требовали поднятого Redis.
    :param subscriber_limit: запросов в минуту от одного абонента.
    :param session_limit: запросов в минуту в рамках одной сессии.
    :param service_limit: запросов в минуту по сервису целиком.
    :param store_retry_after: что вернуть в `Retry-After` при отказе хранилища.
    """

    def __init__(
        self,
        client: _RedisLike,
        *,
        subscriber_limit: int,
        session_limit: int,
        service_limit: int,
        store_retry_after: int = 30,
    ) -> None:
        self._client = client
        self._subscriber_limit = subscriber_limit
        self._session_limit = session_limit
        self._service_limit = service_limit
        self._store_retry_after = store_retry_after

    def check(self, subscriber_id: str, session_id: str) -> QuotaDecision:
        """Учесть запрос и решить, обслуживать ли его.

        Счётчики увеличиваются **до** проверки, включая отклонённые запросы. Это
        не побочный эффект, а защита: иначе абонент, упёршийся в лимит, получал бы
        неограниченное число бесплатных попыток и продолжал нагружать шлюз.

        :raises QuotaStoreUnavailableError: хранилище недоступно (ADR-008).
        """
        window = int(time.time()) // WINDOW_SECONDS
        retry_after = self._seconds_until_window_ends()

        checks = (
            (QuotaScope.SUBSCRIBER, f"quota:sub:{subscriber_id}:{window}", self._subscriber_limit),
            (QuotaScope.SESSION, f"quota:sess:{session_id}:{window}", self._session_limit),
            (QuotaScope.SERVICE, f"quota:svc:{window}", self._service_limit),
        )

        exceeded: QuotaDecision | None = None
        for scope, key, limit in checks:
            used = self._bump(key)
            # Проверяются все три, а не до первого превышения: счётчики должны
            # увеличиться одинаково независимо от того, какой предел сработал,
            # иначе учёт разъедется между пределами.
            if used > limit and exceeded is None:
                exceeded = QuotaDecision(
                    allowed=False,
                    scope=scope,
                    retry_after=retry_after,
                    used=used,
                    limit=limit,
                )

        return exceeded if exceeded is not None else QuotaDecision(allowed=True)

    def _bump(self, key: str) -> int:
        """Увеличить счётчик и продлить ему жизнь до конца окна."""
        try:
            used = self._client.incr(key)
            if used == 1:
                # Время жизни ставится только при создании: иначе счётчик,
                # обновляемый каждым запросом, никогда бы не истёк.
                self._client.expire(key, WINDOW_SECONDS)
            return used
        except QuotaStoreUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 — важен факт недоступности, не причина
            raise QuotaStoreUnavailableError(str(exc)) from exc

    @staticmethod
    def _seconds_until_window_ends() -> int:
        """Сколько ждать до обнуления счётчика.

        Возвращается фактическое время до границы окна, а не постоянное значение
        из настроек: сказать «повторите через 30 секунд», когда счётчик обнулится
        через две, — значит без нужды задержать абонента, а когда через
        пятьдесят — обречь его на второй отказ.
        """
        return WINDOW_SECONDS - int(time.time()) % WINDOW_SECONDS

    @property
    def store_retry_after(self) -> int:
        """`Retry-After` для случая недоступности хранилища (ADR-008)."""
        return self._store_retry_after
