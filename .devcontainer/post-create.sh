#!/usr/bin/env bash
# Разовая настройка Codespace/dev container после создания.
#
# Что НЕ делается здесь намеренно:
# - make install-pii / install-search — тяжёлые extras (torch, веса моделей),
#   нужны только на шагах 3 и 5; ставятся вручную, когда до них дошла работа.
# - ollama pull — 4.5 ГБ на каждый пересоздание контейнера без prebuild;
#   разработчик решает сам, когда тянуть модель (`make model-pull`).
set -euo pipefail

# В этом образе на PATH одновременно несколько Python (свой у Codespaces —
# /home/codespace/.python/current, свой у devcontainer-образа, ещё и conda),
# и голый `python`/`python3` резолвится в версию НЕ из диапазона pyproject.toml
# (`>=3.12,<3.13`) — например, в 3.14. `python3.12` при этом на месте
# (/usr/bin/python3.12). Строим venv явно на нём, а не полагаемся на PATH.
echo "=== Виртуальное окружение (python3.12) ==="
if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Чтобы venv был активен и в новых терминалах, открытых после этого скрипта
# (postCreateCommand выполняется один раз, а не при каждом открытии терминала).
BASHRC_LINE="[ -f \"$(pwd)/.venv/bin/activate\" ] && source \"$(pwd)/.venv/bin/activate\""
grep -qxF "$BASHRC_LINE" ~/.bashrc 2>/dev/null || echo "$BASHRC_LINE" >>~/.bashrc

echo "=== Зависимости Python (make install) ==="
make install

echo "=== Локальные слепки систем (docker compose: postgres, redis) ==="
docker compose up -d

echo "=== Ollama ==="
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
# systemd недоступен внутри dev-контейнера — поднимаем сервер вручную в фоне.
nohup ollama serve >/tmp/ollama.log 2>&1 &

cat <<'EOF'

Готово. Дальше вручную:
  make model-pull   — скачать qwen2.5:7b-instruct-q4_K_M (~4.5 ГБ) для LLM_PROVIDER=local-test
  make preflight    — проверить, что Ollama и модель на месте
  make check        — lint + typecheck + тесты (не требует Ollama/модель)

Ключи внешних провайдеров (YANDEX_API_KEY и т.п.) задаются через
Settings → Secrets and variables → Codespaces — .env создавать не нужно.
EOF
