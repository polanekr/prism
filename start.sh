#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    echo 'PRISM environment missing. Run: uv sync --locked' >&2
    exit 1
fi
exec .venv/bin/python -m prism_app "$@"
