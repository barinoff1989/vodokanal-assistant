"""Готовый бланк как файл: HTML для печати и хранилище ссылок.

Путь `template` (`app/documents/responder.py`) собирает текст бланка и до сих пор
отдавал его **строкой в ленте диалога**. Абоненту нужен не текст для копирования,
а лист, который можно распечатать и отнести в водоканал. Этот модуль закрывает
последнее звено находки F4 (раздел 41.4) на прототипе.

ЧТО ЗДЕСЬ ЕСТЬ И ЧЕГО НЕТ.

* :func:`render_html` — оборачивает готовый текст бланка в самостоятельную
  страницу с версткой под A4 и кнопкой «Печать». Не PDF: настоящий рендер в PDF
  тянет системные зависимости (cairo/pango) и на прототипе ничего не доказывает —
  браузер печатает страницу в PDF сам. PDF — область MVP.
* :class:`ArtifactStore` — **дубль объектного хранилища**. На MVP готовые
  документы лежат в S3/MinIO (`app/adapters/document_generator.py`, ADR раздела
  41.4); здесь — каталог процесса, а «ссылка с ограниченным сроком жизни» —
  непрозрачный токен в пути и проверка возраста файла при чтении. Тот же приём,
  что CSV вместо API Биллинга и хранилище в памяти вместо Qdrant: шов проходит по
  хранилищу, реализация на прототипе другая.

ПОЧЕМУ ХРАНИЛИЩЕ, А НЕ ОТДАЧА ТЕКСТА НАПРЯМУЮ. В бланке персональные данные
абонента — ФИО, лицевой счёт, адрес. Ссылка на них не может быть вечной: срок
жизни (`document_link_ttl_seconds`) — единственный механизм удаления, тот же
довод, что у срока жизни состояния сессии (раздел 77.5). Просроченный файл
удаляется при первом же обращении к хранилищу.

ЭТО НЕ ЗАПИСЬ И НЕ ТРЕБУЕТ HITL. Путь `template` — Этап 1: ассистент отдаёт
образец, ничего не оформляет и никуда не пишет (`app/documents/responder.py`).
Гейт `AWAITING_CONFIRMATION` из адаптера Этапа 2 сюда не применяется.
"""

from __future__ import annotations

import html
import logging
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ArtifactStore", "TOKEN_RE", "render_html"]

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\A[A-Za-z0-9_-]{16,64}\Z")
"""Разрешённый вид токена. Проверяется до обращения к файловой системе: токен
приходит из URL, и `../` в нём открыл бы чтение чужих файлов."""

_PAGE = """\
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: #eceff1; color: #10202a;
    font-family: "Segoe UI", system-ui, sans-serif;
  }}
  .sheet {{
    max-width: 210mm; margin: 16px auto; padding: 20mm 22mm;
    background: #fff; box-shadow: 0 1px 6px rgba(0,0,0,.18);
  }}
  pre {{
    white-space: pre-wrap; word-wrap: break-word; margin: 0;
    font-family: "Consolas", "DejaVu Sans Mono", ui-monospace, monospace;
    font-size: 13px; line-height: 1.5;
  }}
  .bar {{
    position: sticky; top: 0; display: flex; gap: 8px; align-items: center;
    padding: 8px 12px; background: #063a40; color: #fff; font-size: 13px;
  }}
  .bar button {{
    font: inherit; padding: 4px 12px; border: 0; border-radius: 4px;
    background: #12b5c4; color: #04222b; cursor: pointer;
  }}
  .note {{
    margin: 0 auto 4px; max-width: 210mm; padding: 0 22mm;
    font-size: 12px; color: #4c6169;
  }}
  @media print {{
    body {{ background: #fff; }}
    .bar, .note {{ display: none; }}
    .sheet {{ box-shadow: none; margin: 0; padding: 0; max-width: none; }}
  }}
</style>
</head>
<body>
<div class="bar">
  <button type="button" onclick="window.print()">Печать</button>
  <span>Образец заявления — заполните недостающие поля и подайте привычным способом</span>
</div>
{note}
<div class="sheet"><pre>{body}</pre></div>
</body>
</html>
"""


def render_html(title: str, body_text: str, *, disclaimer: str | None = None) -> str:
    """Обернуть готовый текст бланка в страницу для печати.

    :param title: заголовок вкладки — обычно название образца.
    :param body_text: собранный текст бланка (`render_template` + справочный блок).
    :param disclaimer: оговорка под панелью; на печать не выводится.

    Текст экранируется целиком: в него подставлены ФИО и адрес абонента, а
    `<` в фамилии сломал бы разметку.
    """
    note = f'<p class="note">{html.escape(disclaimer)}</p>' if disclaimer else ""
    return _PAGE.format(
        title=html.escape(title),
        body=html.escape(body_text),
        note=note,
    )


@dataclass(slots=True)
class ArtifactStore:
    """Каталог готовых бланков — дубль S3/MinIO на прототипе.

    :param root: каталог для файлов. Создаётся при первой записи.
    :param ttl_seconds: сколько ссылка живёт. Файл старше этого срока считается
        отсутствующим и удаляется при обращении.
    :param url_prefix: начало ссылки, которую увидит абонент. Маршрут отдаёт
        стенд (`app/api.py`), в контракт `/v1` он не входит.
    """

    root: Path
    ttl_seconds: int
    url_prefix: str = "/documents"

    def put(self, html_text: str) -> str | None:
        """Сохранить страницу, вернуть ссылку. ``None`` — записать не удалось.

        Токен непрозрачный (`secrets.token_urlsafe`): по нему нельзя перебрать
        чужие документы, как по последовательному номеру.
        """
        token = secrets.token_urlsafe(18)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._sweep()
            (self.root / f"{token}.html").write_text(html_text, encoding="utf-8")
        except OSError as exc:
            # Не роняем ответ абоненту: текст бланка он уже получил в ленте,
            # файл — дополнение. Тот же принцип best-effort, что у телеметрии.
            logger.warning("бланк не сохранён в хранилище: %s", exc)
            return None
        return f"{self.url_prefix}/{token}"

    def get(self, token: str) -> str | None:
        """Прочитать страницу по токену. ``None`` — нет такой или просрочена."""
        if not TOKEN_RE.match(token):
            return None
        path = self.root / f"{token}.html"
        try:
            if not path.is_file() or self._expired(path):
                return None
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("бланк не прочитан из хранилища: %s", exc)
            return None

    def _expired(self, path: Path) -> bool:
        return (time.time() - path.stat().st_mtime) > self.ttl_seconds

    def _sweep(self) -> None:
        """Удалить просроченные файлы. Зовётся при записи — отдельного планировщика
        на прототипе нет, а ссылок за сессию демонстрации немного."""
        try:
            for path in self.root.glob("*.html"):
                if self._expired(path):
                    path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("уборка хранилища бланков не удалась: %s", exc)
