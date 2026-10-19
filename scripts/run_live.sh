#!/usr/bin/env bash
# 每小時自動執行即時訊號檢查的包裝腳本(供 crontab 掛載)。
# 用法(每小時第 1 分執行):
#   crontab -e 然後加入一行(路徑換成你的實際路徑):
#   1 * * * * /Volumes/workplace/EnglishTraining/trading/scripts/run_live.sh >> /Volumes/workplace/EnglishTraining/trading/output/run_live.log 2>&1
# Telegram(可選):在下面兩行填入後取消註解,或改在系統環境設定。
#   export TELEGRAM_BOT_TOKEN="..."
#   export TELEGRAM_CHAT_ID="..."

set -euo pipefail
# 腳本所在目錄的上一層 = 專案根
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "找不到 venv python: $PY" >&2
  exit 1
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
"$PY" scripts/run_live.py
