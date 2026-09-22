"""Диагностика подключения к YandexGPT — по шагам, до первой поломки.

Отвечает на вопрос «будут ли проблемы с подключением», фактами, а не
предположениями. Каждая проверка независима и печатает вердикт; выполнять
до начала код-шага 4.

Запуск без ключа — проверятся только сеть и доверие сертификату:

    python scripts/check_yandex.py

Запуск с ключом — дополнительно авторизация, формат ответа и поток. Ключ и
каталог задаются переменными окружения оболочки (см. `env.example.sh`),
файлы `.env` этот скрипт не читает:

    export YANDEX_API_KEY=...
    export YANDEX_FOLDER_ID=...
    python scripts/check_yandex.py

Ключ нигде не печатается: в выводе только его длина и первые символы.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request

HOST = "llm.api.cloud.yandex.net"
BASE = os.getenv("YANDEX_API_BASE", f"https://{HOST}/v1")
API_KEY = os.getenv("YANDEX_API_KEY", "")
FOLDER = os.getenv("YANDEX_FOLDER_ID", "")
MODEL = os.getenv("YANDEX_MODEL", f"gpt://{FOLDER}/yandexgpt/latest")
TIMEOUT = 15

OK, FAIL, SKIP = "ДА", "НЕТ", "ПРОПУЩЕНО"
results: list[tuple[str, str, str]] = []


def report(name: str, verdict: str, detail: str = "") -> None:
    results.append((name, verdict, detail))
    mark = {OK: "[ OK ]", FAIL: "[ !! ]", SKIP: "[ -- ]"}[verdict]
    print(f"{mark} {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"       {line}")


# --- 1. DNS ----------------------------------------------------------------- #

def check_dns() -> bool:
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(HOST, 443)}
        report("Имя разрешается в адрес", OK, ", ".join(sorted(addrs)))
        return True
    except Exception as exc:
        report("Имя разрешается в адрес", FAIL, f"{type(exc).__name__}: {exc}")
        return False


# --- 2. TCP ----------------------------------------------------------------- #

def check_tcp() -> bool:
    try:
        with socket.create_connection((HOST, 443), timeout=TIMEOUT):
            report("Соединение на порт 443 устанавливается", OK)
            return True
    except Exception as exc:
        report(
            "Соединение на порт 443 устанавливается",
            FAIL,
            f"{type(exc).__name__}: {exc}\n"
            "Похоже на блокировку сети или корпоративный прокси.",
        )
        return False


# --- 3. Доверие сертификату — ключевой вопрос ------------------------------- #

def check_tls() -> bool:
    """Главная проверка: доверяет ли Python сертификату без сертификатов НУЦ.

    Если нет — решение переводить прототип на Яндекс теряет смысл, потому что
    ровно из-за этого от GigaChat и отказались.
    """
    ctx = ssl.create_default_context()
    try:
        with (
            socket.create_connection((HOST, 443), timeout=TIMEOUT) as sock,
            ctx.wrap_socket(sock, server_hostname=HOST) as tls,
        ):
            cert = tls.getpeercert() or {}
            issuer = dict(x[0] for x in cert.get("issuer", ()))
            name = issuer.get("organizationName") or issuer.get("commonName") or "?"
            report(
                "Сертификат доверен стандартным набором",
                OK,
                f"издатель: {name}\n"
                f"протокол: {tls.version()}\n"
                "Корневые сертификаты НУЦ Минцифры НЕ нужны.",
            )
            return True
    except ssl.SSLCertVerificationError as exc:
        report(
            "Сертификат доверен стандартным набором",
            FAIL,
            f"{exc.verify_message}\n"
            "Нужны те же корневые сертификаты, что и для GigaChat.\n"
            "Решение переводить прототип на Яндекс требует пересмотра.",
        )
        return False
    except Exception as exc:
        report("Сертификат доверен стандартным набором", FAIL, f"{type(exc).__name__}: {exc}")
        return False


# --- 4. Авторизация и формат ответа ----------------------------------------- #

def _post(url: str, payload: dict, auth: str) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": auth},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def check_auth() -> str | None:
    """Какой заголовок авторизации принимает эндпоинт — Api-Key или Bearer.

    Однозначного ответа в проекте нет, поэтому пробуются оба и печатается тот,
    что сработал: это значение и пойдёт в конфигурацию шлюза.
    """
    if not API_KEY:
        report("Авторизация принимается", SKIP, "YANDEX_API_KEY не задан")
        return None
    if not FOLDER and "gpt://" in MODEL and "//" in MODEL.split("gpt://")[1][:1]:
        report("Авторизация принимается", SKIP, "YANDEX_FOLDER_ID не задан")
        return None

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Ответь одним словом: работает"}],
        "max_tokens": 10,
        "temperature": 0,
    }
    url = f"{BASE.rstrip('/')}/chat/completions"

    for scheme in (f"Api-Key {API_KEY}", f"Bearer {API_KEY}"):
        label = scheme.split()[0]
        status, body = _post(url, payload, scheme)
        if status == 200:
            try:
                text = json.loads(body)["choices"][0]["message"]["content"]
            except Exception:
                text = "(ответ пришёл, но структура отличается от OpenAI)"
            report(
                "Авторизация принимается",
                OK,
                f"рабочий заголовок: Authorization: {label} <ключ>\n"
                f"ответ модели: {text[:120]}",
            )
            return scheme
        report(f"  попытка с заголовком {label}", FAIL, f"HTTP {status}: {body[:220]}")

    report("Авторизация принимается", FAIL, "Ни Api-Key, ни Bearer не подошли — см. ответы выше")
    return None


# --- 5. Потоковая отдача ---------------------------------------------------- #

def check_stream(auth: str) -> None:
    """Правило 4.3 требует потока. Проверяем, что он действительно поток."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Подробно, не менее чем в десяти предложениях, объясни, "
                "как абоненту подготовиться к поверке счётчика воды.",
            }
        ],
        "max_tokens": 500,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{BASE.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": auth},
        method="POST",
    )
    started = time.perf_counter()
    first_at: float | None = None
    chunks = 0
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT * 2) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                chunks += 1
                if first_at is None:
                    first_at = time.perf_counter() - started
    except Exception as exc:
        report("Ответ приходит потоком", FAIL, f"{type(exc).__name__}: {exc}")
        return

    total = time.perf_counter() - started
    # Критерий строгий: мало получить больше одного фрагмента — первый обязан
    # прийти заметно раньше последнего. Иначе провайдер собрал ответ целиком
    # и лишь потом отдал его частями, а это не поток в смысле правила 4.3.
    incremental = chunks > 2 and first_at is not None and first_at < total * 0.8
    if incremental and first_at is not None:
        report(
            "Ответ приходит потоком",
            OK,
            f"фрагментов: {chunks}\n"
            f"первый фрагмент через {first_at:.2f} с, весь ответ за {total:.2f} с\n"
            f"ориентир DoD: время до первого токена < 0.5 с, весь ответ < 3 с",
        )
    else:
        report(
            "Ответ приходит потоком",
            FAIL,
            f"фрагментов получено: {chunks} — это не поток, а один кусок.\n"
            "Нарушает правило 4.3 (ответ доходит до абонента потоком).",
        )


# --- главный сценарий ------------------------------------------------------- #

def main() -> int:
    print(f"Проверка подключения к {HOST}")
    print(f"адрес API : {BASE}")
    print(f"модель    : {MODEL if FOLDER or 'gpt://' not in MODEL else '(не задан каталог)'}")
    print(f"ключ      : {'задан, длина ' + str(len(API_KEY)) if API_KEY else 'НЕ ЗАДАН'}")
    print("-" * 70)

    if not check_dns():
        return finish()
    if not check_tcp():
        return finish()
    if not check_tls():
        return finish()

    auth = check_auth()
    if auth:
        check_stream(auth)
    else:
        report("Ответ приходит потоком", SKIP, "нет рабочей авторизации")

    return finish()


def finish() -> int:
    print("-" * 70)
    failed = [name for name, verdict, _ in results if verdict == FAIL and not name.startswith("  ")]
    skipped = [name for name, verdict, _ in results if verdict == SKIP]
    if failed:
        print("ВЕРДИКТ: проблемы есть —", "; ".join(failed))
        return 1
    if skipped:
        print("ВЕРДИКТ: пройденные проверки в порядке; не проверено —", "; ".join(skipped))
        return 0
    print("ВЕРДИКТ: препятствий не найдено, подключение возможно")
    return 0


if __name__ == "__main__":
    sys.exit(main())
