"""Server-side API handlers for crash logs and rank output retrieval."""

import json
from pathlib import Path


def read_crash_logs(session_dir: str) -> list[dict]:
    """Read all crash report JSON files from a session directory."""
    logs = []
    session_path = Path(session_dir)
    if not session_path.exists():
        return logs
    for f in sorted(session_path.glob("*.crash.json")):
        try:
            logs.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return logs


def read_rank_logs(session_dir: str, rank: int, tail: int = 100) -> str:
    """Read the last N lines of a rank's log file."""
    log_path = Path(session_dir) / f"rank-{rank}.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text().splitlines()
    return "\n".join(lines[-tail:])
