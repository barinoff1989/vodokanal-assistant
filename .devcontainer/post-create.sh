#!/usr/bin/env bash
# Разовая настройка Codespace/dev container после создания.
#
# Что НЕ делается здесь намеренно:
# - make install-pii / install-search — тяжёлые extras (torch, веса моделей),
#   нужны только на шагах 3 и 5; ставятся вручную, когда до них дошла работа.
# - ollama pull — 4.5 ГБ на каждый пересоздание контейнера без prebuild;
#   разработчик решает сам, когда тянуть модель (`make model-pull`).
set -euo pipefail

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
