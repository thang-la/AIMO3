from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...<truncated>"


def _json_safe(value: Any, max_chars: int) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, max_chars) for v in value]
    return _truncate_text(repr(value), max_chars)


class DebugTracer:
    def __init__(
        self,
        *,
        enabled: bool,
        max_chars: int = 1200,
        file_path: Path | None = None,
    ):
        self.enabled = enabled
        self.max_chars = max(200, max_chars)
        self.file_path = file_path
        self._file = None
        if self.enabled and self.file_path is not None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.file_path.open("a", encoding="utf-8")

    def _write_line(self, line: str) -> None:
        target = self._file
        if target is not None:
            target.write(line + "\n")
            target.flush()
        else:
            print(line, file=sys.stderr, flush=True)

    def log(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": _json_safe(payload, self.max_chars),
        }
        self._write_line(json.dumps(data, ensure_ascii=False))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:  # pragma: no cover
        self.close()
