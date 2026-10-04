#!/usr/bin/env bash
# Start StreamClearance locally. Mock mode unless GEMINI_API_KEY is set (SC_MOCK=1 forces mock).
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${SC_PORT:-8430}" "$@"
