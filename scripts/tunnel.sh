#!/usr/bin/env bash
# tunnel.sh — Start an ngrok HTTPS tunnel to the local FastAPI server on port 8000.
#
# Prerequisites:
#   brew install ngrok   (or download from https://ngrok.com/download)
#   ngrok config add-authtoken <YOUR_TOKEN>
#
# Usage:
#   bash scripts/tunnel.sh
#   bash scripts/tunnel.sh 9000   # optional: override port

set -euo pipefail

PORT="${1:-8000}"

# Verify ngrok is installed
if ! command -v ngrok &>/dev/null; then
  echo "[tunnel] ERROR: ngrok not found. Install it with:"
  echo "         brew install ngrok"
  echo "         or visit https://ngrok.com/download"
  exit 1
fi

echo "[tunnel] Starting ngrok tunnel → http://localhost:${PORT}"
echo "[tunnel] Press Ctrl-C to stop."
echo ""

# --log=stdout pipes ngrok logs to the terminal so the public URL is visible.
ngrok http "${PORT}" --log=stdout
