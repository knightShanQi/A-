from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TaskRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}

    def _write(self, records: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.path)

    def record_submitted(self, task_id: str, *, task_type: str, params: dict[str, Any]) -> None:
        records = self._read()
        now = time.time()
        existing = records.get(task_id, {})
        records[task_id] = {
            **existing,
            "task_id": task_id,
            "task_type": task_type,
            "params": params,
            "status": "running",
            "submitted_at": existing.get("submitted_at", now),
            "updated_at": now,
            "error": "",
        }
        self._write(records)

    def record_status(self, task_id: str, *, status: str, error: str = "") -> None:
        records = self._read()
        now = time.time()
        existing = records.get(task_id, {"task_id": task_id})
        records[task_id] = {
            **existing,
            "status": status,
            "updated_at": now,
            "error": str(error or ""),
        }
        self._write(records)

    def get(self, task_id: str) -> dict[str, Any]:
        return dict(self._read().get(task_id, {}))
