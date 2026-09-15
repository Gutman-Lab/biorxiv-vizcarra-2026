"""Log embedding run timing/stats to a local JSONL file."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = _PROJECT_ROOT / '.embedding_runs.jsonl'


def append_run(
    record: dict[str, Any],
    *,
    log_path: Path | None = None,
) -> None:
    path = log_path or DEFAULT_LOG_PATH
    row = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **record,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')


def load_runs(*, log_path: Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    path = log_path or DEFAULT_LOG_PATH
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit is not None and limit >= 0:
        rows = rows[-limit:]
    return rows


def latest_run_per_model(*, log_path: Path | None = None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_runs(log_path=log_path):
        model_id = row.get('model_id')
        if model_id:
            latest[model_id] = row
    return latest
