from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from a_share_predictor.dashboard import DEFAULT_VIEW_PARAMS
from a_share_predictor.modeling import (
    GLOBAL_MODEL_TEST_END,
    GLOBAL_MODEL_TEST_START,
    GLOBAL_MODEL_TRAIN_END,
    GLOBAL_MODEL_TRAIN_START,
    get_market_wide_model_status,
    load_cached_market_wide_model,
    load_market_proxy_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "h3_restore_or_retire_2026-05-28.md"


def _config_status(horizon_days: int, positive_return: float, label: str) -> dict[str, object]:
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
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    default_h = int(DEFAULT_VIEW_PARAMS["horizon_days"])
    default_r = float(DEFAULT_VIEW_PARAMS["positive_return"])
    default_row = _config_status(default_h, default_r, "current_default")
    legacy_row = _config_status(5, 0.03, "available_proxy_reference")

    rows = [default_row, legacy_row]
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "h3_restore_or_retire_summary.csv"
    json_path = OUTPUT_DIR / "h3_restore_or_retire.json"
    df.to_csv(csv_path, index=False)

    recommendation = "retire_default_until_model_restored"
    rationale = [
        "The current default path (`h3 / 10%`) has neither a market-wide model nor a proxy model available, so the UI default is structurally forced into local fallback mode.",
        "The only matching maintained training entrypoint in the repo (`scripts/train_market_wide_model.py`) hardcodes `h5 / 3%`, which means the workspace currently optimizes and refreshes a different configuration from the default UI path.",
        "An optimization program that continues pruning score layers on `h3 / 10%` before restoring model-source support is mostly auditing a degraded fallback architecture.",
    ]
    if default_row["proxy_model_loaded"] or default_row["market_model_loaded"]:
        recommendation = "restore_and_keep"

    findings = [
        f"Current UI default = `h{default_h} / {default_r * 100:.0f}%`, from `DEFAULT_VIEW_PARAMS`.",
        f"Default path availability: market model loaded = {bool(default_row['market_model_loaded'])}, proxy loaded = {bool(default_row['proxy_model_loaded'])}.",
        f"Reference `h5 / 3%` availability: market model loaded = {bool(legacy_row['market_model_loaded'])}, proxy loaded = {bool(legacy_row['proxy_model_loaded'])}.",
        f"Recommendation: `{recommendation}`.",
        *rationale,
    ]

    payload = {
        "summary": rows,
        "recommendation": recommendation,
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# h3 Restore Or Retire 2026-05-28",
        "",
        "## Purpose",
        "",
        "Turn the model-source availability evidence into a concrete configuration decision for the current default short-horizon trading path.",
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
            "## Recommendation",
            "",
        ]
    )
    if recommendation == "retire_default_until_model_restored":
        lines.extend(
            [
                "- Retire `h3 / 10%` as the default production trading configuration for now.",
                "- If you want to preserve it as a research track, do so only after restoring a compatible market-wide or proxy model for that exact horizon/target setting.",
                "- Until then, route default trading analysis to a configuration that actually has model support, such as `h5 / 3%`, or explicitly label the current path as fallback-only.",
            ]
        )
    else:
        lines.extend(
            [
                "- Keep the current default configuration because matching model support is already present.",
                "- Continue score-level pruning only after verifying that the restored model-source path is actually active end-to-end.",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "1. Either train/restore a compatible `h3 / 10%` market model or proxy, or change the default configuration away from that unsupported path.",
            "2. Only after that decision should further score-layer simplification on the default path be treated as high-priority optimization work.",
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
