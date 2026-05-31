from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from a_share_predictor.dashboard import (
    GLOBAL_MODEL_TEST_END,
    GLOBAL_MODEL_TEST_START,
    GLOBAL_MODEL_TRAIN_END,
    GLOBAL_MODEL_TRAIN_START,
)
from a_share_predictor.modeling import (
    get_market_wide_model_status,
    load_cached_market_wide_model,
    load_market_proxy_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "model_source_availability_2026-05-28.md"


CONFIGS = [
    {"horizon_days": 3, "positive_return": 0.10, "label": "current_focus_board_h3_r1000"},
    {"horizon_days": 5, "positive_return": 0.03, "label": "legacy_default_h5_r300"},
]


def _audit_config(horizon_days: int, positive_return: float, label: str) -> dict[str, object]:
    status = get_market_wide_model_status(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=GLOBAL_MODEL_TRAIN_START,
        train_end=GLOBAL_MODEL_TRAIN_END,
        test_start=GLOBAL_MODEL_TEST_START,
        test_end=GLOBAL_MODEL_TEST_END,
    )
    market_model = load_cached_market_wide_model(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=GLOBAL_MODEL_TRAIN_START,
        train_end=GLOBAL_MODEL_TRAIN_END,
        test_start=GLOBAL_MODEL_TEST_START,
        test_end=GLOBAL_MODEL_TEST_END,
    )
    proxy_model = load_market_proxy_model(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=GLOBAL_MODEL_TRAIN_START,
        train_end=GLOBAL_MODEL_TRAIN_END,
        test_start=GLOBAL_MODEL_TEST_START,
        test_end=GLOBAL_MODEL_TEST_END,
    )
    return {
        "label": label,
        "horizon_days": horizon_days,
        "positive_return": positive_return,
        "market_model_ready": bool(status.get("model_ready")),
        "partial_dataset_ready": bool(status.get("partial_ready")),
        "market_model_loaded": market_model is not None,
        "proxy_model_loaded": proxy_model is not None,
        "model_path": str(status.get("model_path") or ""),
        "partial_path": str(status.get("partial_path") or ""),
        "completed_symbol_count": int(status.get("completed_symbol_count", 0) or 0),
        "partial_symbol_count": int(status.get("partial_symbol_count", 0) or 0),
        "partial_row_count": int(status.get("partial_row_count", 0) or 0),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_audit_config(**config) for config in CONFIGS]
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "model_source_availability_summary.csv"
    json_path = OUTPUT_DIR / "model_source_availability.json"
    df.to_csv(csv_path, index=False)

    current_row = df.loc[df["label"].eq("current_focus_board_h3_r1000")].iloc[0]
    legacy_row = df.loc[df["label"].eq("legacy_default_h5_r300")].iloc[0]
    findings = [
        f"The current focus-board audit configuration (`h3 / 10%`) has no market-wide model and no proxy model available: market model ready = {bool(current_row['market_model_ready'])}, proxy loaded = {bool(current_row['proxy_model_loaded'])}.",
        f"The older default-style configuration (`h5 / 3%`) still has no full market-wide model ready, but it does load a proxy model = {bool(legacy_row['proxy_model_loaded'])}.",
        "That means the current short-horizon audit path is structurally forced into `local_fast_fallback`, while the older longer-horizon path can at least retain a proxy-model-based ranking layer.",
        "This is not just a modeling-quality issue. It is a model-availability issue: different horizon/target settings are exercising different ranking architectures in production.",
        "Practical implication: part of the low annualized return and weak explanation quality on the current short-horizon path may come from running the weakest fallback architecture, not only from the score formulas layered on top.",
    ]

    payload = {
        "summary": rows,
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# Model Source Availability 2026-05-28",
        "",
        "## Purpose",
        "",
        "Audit whether the audited trading paths are actually running on the same model-source architecture, or whether some configurations are forced into fallback mode before any ranking/scoring logic is applied.",
        "",
        "## Key Findings",
        "",
    ]
    for finding in findings:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- If the short-horizon focus-board path has neither a market-wide model nor a proxy model, then score-layer audits on that path are partly auditing a degraded architecture, not the intended full stack.",
            "- Comparing `h3 / 10%` and `h5 / 3%` directly also becomes an architecture comparison, not just a target-window comparison, because they are using different upstream model availability states.",
            "- Any optimization plan for annualized return therefore has to separate two questions: whether the ranking formulas are weak, and whether the strongest intended model source is even present for the audited configuration.",
            "",
            "## Next Actions",
            "",
            "1. Treat `h3 / 10%` model-source unavailability as a first-class bottleneck in the current trading stack.",
            "2. Before deleting more score layers on that path, decide whether the intended fix is to train/restore a compatible market model or proxy for `h3 / 10%`, or to explicitly retire that configuration.",
            "",
            f"Summary CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
            "",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
