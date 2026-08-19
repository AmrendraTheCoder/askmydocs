#!/usr/bin/env bash
# Start AskMyDocs (backend + UI, one process, one port).
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/uvicorn backend.main:app --reload
