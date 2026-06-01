from __future__ import annotations

from a_share_predictor.task_registry import TaskRegistry


def test_task_registry_persists_submitted_and_completed_status(tmp_path):
    path = tmp_path / "tasks.json"
    registry = TaskRegistry(path)

    registry.record_submitted("task-1", task_type="market_backtest", params={"top_k": 5})
    registry.record_status("task-1", status="completed")

    reloaded = TaskRegistry(path).get("task-1")
    assert reloaded["task_type"] == "market_backtest"
    assert reloaded["params"] == {"top_k": 5}
    assert reloaded["status"] == "completed"
    assert reloaded["error"] == ""
