from __future__ import annotations

from datetime import datetime
from pathlib import Path

from a_share_predictor.database_source import load_env_file

load_env_file()

from a_share_predictor.modeling import MODEL_SCHEMA_VERSION, train_market_wide_model


LOG_PATH = Path(__file__).resolve().parents[1] / ".cache" / "train_market_wide_model.run.log"


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {message}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def main() -> None:
    log(f"Start market-wide model training for schema v{MODEL_SCHEMA_VERSION}.")
    result = train_market_wide_model(
        horizon_days=5,
        positive_return=0.03,
        refresh=False,
    )
    log(f"Train window: {result.train_start} -> {result.train_end}")
    log(f"Test window: {result.test_start} -> {result.test_end}")
    log(f"Train samples: {result.train_sample_size:,}")
    log(f"Test samples: {result.test_sample_size:,}")
    log(f"Eligible symbols: {result.eligible_symbols:,} / Universe: {result.universe_size:,}")
    log(f"ROC AUC: {result.metrics.get('roc_auc')}")
    log(f"Precision: {result.metrics.get('precision')}")
    log(f"Recall: {result.metrics.get('recall')}")
    log(f"Top bucket return: {result.metrics.get('top_bucket_return')}")
    log(f"Quality label: {result.quality_label}")
    log(f"Regime calibrators: {len(getattr(result, 'regime_calibrators', {}))}")
    log(f"Regime distribution: {getattr(result, 'regime_distribution', {})}")
    log(result.summary)


if __name__ == "__main__":
    main()
