from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = PROJECT_ROOT / "docs" / "strategy_hard_filter_optimization_plan_2026-05-31.md"
DEFAULT_INPUT_PATH = PROJECT_ROOT / ".cache" / "hard_filter_plan_comparison_full" / "plan_scored_candidates.csv"
DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / ".cache" / "hard_filter_plan_comparison" / "prepared_fast_history_v1_20160527_20260526.pkl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "hard_filter_plan_strategy_models"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "hard_filter_plan_strategy_model_calibration_2026-06-01.md"

NUMERIC_FEATURE_COLUMNS = [
    "candidate_priority",
    "strategy_rank",
    "latest_price",
    "change_pct",
    "amount",
    "turnover",
    "industry_ret_2d_pct",
    "industry_up_count",
    "model_probability",
    "model_score",
    "priority_score",
    "model_priority_80_20",
    "market_ret",
    "up_ratio",
    "above_ma20_ratio",
    "limit_up_count",
    "limit_down_count",
    "amount_ma5_ma20",
    "amount_ma20_ma60",
    "up_amount_ratio",
    "strong_amount_ratio",
    "trend_score",
    "flow_score",
    "sse_close",
    "bull_score",
]
BOOLEAN_FEATURE_COLUMNS = [
    "trend_green",
    "flow_green",
    "internal_green",
    "market_green",
    "v3_full_green",
    "v3_yellow",
    "is_bull_strict",
    "is_bull_loose",
]
CATEGORICAL_FEATURE_COLUMNS = [
    "plan_market_bucket",
    "market_state",
    "bull_bear_state",
]
REQUIRED_METRIC_COLUMNS = [
    "strategy_family",
    "horizon_days",
    "target_return",
    "sample_count",
    "test_sample_count",
    "positive_rate",
    "test_positive_rate",
    "auc",
    "brier",
    "calibration_gap",
    "avg_return",
    "top_bucket_return",
    "top_bucket_win_rate",
    "top_bucket_max_drawdown",
    "probability_p05",
    "probability_p50",
    "probability_p95",
]


class StrategySpec:
    def __init__(self, strategy_family: str, horizons: tuple[int, ...], target_returns: dict[int, float]) -> None:
        self.strategy_family = strategy_family
        self.horizons = horizons
        self.target_returns = target_returns


STRATEGY_SPECS = {
    "strategy1": StrategySpec("strategy1", (3, 5), {3: 0.02, 5: 0.03}),
    "strategy2": StrategySpec("strategy2", (1, 3), {1: 0.00, 3: 0.03}),
    "strategy3": StrategySpec("strategy3", (3, 5), {3: 0.02, 5: 0.03}),
}


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return numeric
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_strategy_family(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "strategy1" in text or "策略1" in text:
        return "strategy1"
    if "strategy2" in text or "策略2" in text:
        return "strategy2"
    if "strategy3" in text or "策略3" in text:
        return "strategy3"
    if "plan_p1" in text:
        if text.startswith("1") or "_1" in text:
            return "strategy1"
        if text.startswith("2") or "_2" in text:
            return "strategy2"
        if text.startswith("3") or "_3" in text:
            return "strategy3"
    return "unknown"


def _safe_auc(y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(y_prob, dtype=float)))


def _safe_brier(y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(brier_score_loss(labels, np.asarray(y_prob, dtype=float)))


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def load_plan_candidates(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return frame
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["market_date", "symbol"]).copy()
    if "strategy_family" not in frame.columns:
        frame["strategy_family"] = frame.get("candidate_strategy", pd.Series("", index=frame.index)).map(normalize_strategy_family)
    else:
        frame["strategy_family"] = frame["strategy_family"].map(normalize_strategy_family)
    for column in NUMERIC_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOLEAN_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("boolean").fillna(False).astype(float)
    return frame.reset_index(drop=True)


def _load_history(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    history_path = Path(path)
    if not history_path.exists():
        return pd.DataFrame()
    if history_path.suffix.lower() == ".pkl":
        frame = pd.read_pickle(history_path)
    else:
        frame = pd.read_csv(history_path, encoding="utf-8-sig")
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    date_column = "trade_date" if "trade_date" in frame.columns else "market_date"
    frame["market_date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    for column in ["close", "high", "low"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["symbol", "market_date", "close"]).sort_values(["symbol", "market_date"]).reset_index(drop=True)


def build_forward_label_frame(history: pd.DataFrame, horizons: Iterable[int] = (1, 3, 5)) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["symbol", "market_date"])
    local = history[["symbol", "market_date", "close", "high", "low"]].copy()
    grouped = local.groupby("symbol", sort=False, group_keys=False)
    close = pd.to_numeric(local["close"], errors="coerce")
    for horizon in sorted({int(value) for value in horizons}):
        future_close = grouped["close"].shift(-horizon)
        shifted_highs = [grouped["high"].shift(-offset) for offset in range(1, horizon + 1)]
        shifted_lows = [grouped["low"].shift(-offset) for offset in range(1, horizon + 1)]
        future_high = pd.concat(shifted_highs, axis=1).max(axis=1)
        future_low = pd.concat(shifted_lows, axis=1).min(axis=1)
        local[f"forward_return_{horizon}d"] = future_close / close.replace(0.0, np.nan) - 1.0
        local[f"max_high_return_{horizon}d"] = future_high / close.replace(0.0, np.nan) - 1.0
        local[f"max_drawdown_{horizon}d"] = future_low / close.replace(0.0, np.nan) - 1.0
    return local.drop(columns=["close", "high", "low"])


def attach_forward_labels(candidates: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    frame = candidates.copy()
    if isinstance(history, pd.DataFrame) and not history.empty:
        labels = build_forward_label_frame(history, horizons=(1, 3, 5))
        frame = frame.merge(labels, on=["symbol", "market_date"], how="left")
    if "hold_3d_return" in frame.columns:
        frame["forward_return_3d"] = pd.to_numeric(frame.get("forward_return_3d"), errors="coerce").fillna(
            pd.to_numeric(frame["hold_3d_return"], errors="coerce")
        )
    if "max_high_return" in frame.columns:
        frame["max_high_return_3d"] = pd.to_numeric(frame.get("max_high_return_3d"), errors="coerce").fillna(
            pd.to_numeric(frame["max_high_return"], errors="coerce")
        )
    if "max_drawdown" in frame.columns:
        frame["max_drawdown_3d"] = pd.to_numeric(frame.get("max_drawdown_3d"), errors="coerce").fillna(
            pd.to_numeric(frame["max_drawdown"], errors="coerce")
        )
    for horizon in (1, 3, 5):
        for prefix in ("forward_return", "max_high_return", "max_drawdown"):
            column = f"{prefix}_{horizon}d"
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["hit_3pct_3d"] = frame["max_high_return_3d"].ge(0.03).astype(float)
    frame["hit_5pct_5d"] = frame["max_high_return_5d"].ge(0.05).astype(float)
    frame["drawdown_risk_3d"] = frame["max_drawdown_3d"].le(-0.05).astype(float)
    return frame


def build_design_matrix(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty:
        columns = feature_columns or []
        return pd.DataFrame(columns=columns), columns
    parts: list[pd.DataFrame] = []
    numeric = pd.DataFrame(index=frame.index)
    for column in NUMERIC_FEATURE_COLUMNS:
        if column in frame.columns:
            numeric[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOLEAN_FEATURE_COLUMNS:
        if column in frame.columns:
            numeric[column] = frame[column].astype("boolean").fillna(False).astype(float)
    parts.append(numeric)
    categorical = [column for column in CATEGORICAL_FEATURE_COLUMNS if column in frame.columns]
    if categorical:
        cat_frame = frame[categorical].fillna("missing").astype(str)
        parts.append(pd.get_dummies(cat_frame, columns=categorical, prefix=categorical, dtype=float))
    matrix = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    if feature_columns is None:
        feature_columns = sorted(matrix.columns.tolist())
    matrix = matrix.reindex(columns=feature_columns, fill_value=0.0)
    return matrix, feature_columns


def _assign_time_split(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["market_date"], errors="coerce").dropna().sort_values().unique()
    result = pd.Series("train", index=frame.index, dtype=object)
    if len(dates) < 5:
        order = frame.sort_values("market_date").index.to_list()
        train_end = max(int(len(order) * 0.60), 1)
        calibrate_end = max(int(len(order) * 0.80), train_end + 1)
        result.loc[order[train_end:calibrate_end]] = "calibrate"
        result.loc[order[calibrate_end:]] = "test"
        return result
    train_end = max(int(len(dates) * 0.65), 1)
    calibrate_end = max(int(len(dates) * 0.82), train_end + 1)
    train_dates = set(pd.to_datetime(dates[:train_end]).normalize())
    calibrate_dates = set(pd.to_datetime(dates[train_end:calibrate_end]).normalize())
    normalized = pd.to_datetime(frame["market_date"], errors="coerce").dt.normalize()
    result.loc[normalized.isin(calibrate_dates)] = "calibrate"
    result.loc[~normalized.isin(train_dates) & ~normalized.isin(calibrate_dates)] = "test"
    return result


def _fit_base_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=17,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train.astype(int))
    return model


def _fit_probability_calibrator(raw_prob: np.ndarray, y_true: pd.Series) -> LogisticRegression | None:
    labels = y_true.astype(int).to_numpy()
    if len(labels) < 30 or len(np.unique(labels)) < 2:
        return None
    calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
    calibrator.fit(np.asarray(raw_prob, dtype=float).reshape(-1, 1), labels)
    return calibrator


def _apply_calibrator(calibrator: LogisticRegression | None, raw_prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_prob, dtype=float), 1e-4, 1.0 - 1e-4)
    if calibrator is None:
        return clipped
    return np.clip(calibrator.predict_proba(clipped.reshape(-1, 1))[:, 1], 1e-4, 1.0 - 1e-4)


def _probability_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    bucketed_parts: list[pd.DataFrame] = []
    for _, group in predictions.groupby(["strategy_family", "horizon_days"], dropna=False):
        local = group.copy()
        local["rank"] = local["calibrated_probability"].rank(method="first")
        bucket_count = min(10, max(1, int(len(local) // 20)))
        if bucket_count <= 1:
            local["probability_bucket"] = 0
        else:
            local["probability_bucket"] = pd.qcut(local["rank"], q=bucket_count, labels=False, duplicates="drop")
        bucketed_parts.append(local)
    local = pd.concat(bucketed_parts, ignore_index=True, sort=False)
    return (
        local.groupby(["strategy_family", "horizon_days", "probability_bucket"], as_index=False)
        .agg(
            sample_count=("target", "size"),
            avg_probability=("calibrated_probability", "mean"),
            realized_positive_rate=("target", "mean"),
            avg_return=("forward_return", "mean"),
            avg_max_high_return=("max_high_return", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
        )
        .sort_values(["strategy_family", "horizon_days", "probability_bucket"])
    )


def train_strategy_horizon_model(
    frame: pd.DataFrame,
    *,
    strategy_family: str,
    horizon_days: int,
    target_return: float,
    min_samples: int = 500,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object] | None]:
    local = frame.loc[frame["strategy_family"].eq(strategy_family)].copy()
    return_column = f"forward_return_{int(horizon_days)}d"
    high_column = f"max_high_return_{int(horizon_days)}d"
    drawdown_column = f"max_drawdown_{int(horizon_days)}d"
    local = local.dropna(subset=[return_column]).copy()
    local["target"] = local[return_column].ge(float(target_return)).astype(int)
    metric_base = {
        "strategy_family": strategy_family,
        "horizon_days": int(horizon_days),
        "target_return": float(target_return),
        "sample_count": int(len(local)),
        "status": "trained",
    }
    if len(local) < int(min_samples) or local["target"].nunique() < 2:
        metric_base.update(
            {
                "status": "insufficient_sample",
                "test_sample_count": 0,
                "positive_rate": float(local["target"].mean()) if not local.empty else float("nan"),
                "test_positive_rate": float("nan"),
                "auc": float("nan"),
                "brier": float("nan"),
                "calibration_gap": float("nan"),
                "avg_return": _safe_mean(local[return_column]) if return_column in local.columns else float("nan"),
                "top_bucket_return": float("nan"),
                "top_bucket_win_rate": float("nan"),
                "top_bucket_max_drawdown": float("nan"),
                "probability_p05": float("nan"),
                "probability_p50": float("nan"),
                "probability_p95": float("nan"),
            }
        )
        return metric_base, pd.DataFrame(), None

    local["split"] = _assign_time_split(local)
    if not local["split"].eq("test").any():
        local.loc[local.tail(max(int(len(local) * 0.18), 1)).index, "split"] = "test"
    if not local["split"].eq("calibrate").any():
        non_test = local.loc[~local["split"].eq("test")]
        local.loc[non_test.tail(max(int(len(non_test) * 0.20), 1)).index, "split"] = "calibrate"
    train = local.loc[local["split"].eq("train")].copy()
    calibrate = local.loc[local["split"].eq("calibrate")].copy()
    test = local.loc[local["split"].eq("test")].copy()
    if len(train) < int(min_samples * 0.40) or train["target"].nunique() < 2 or test.empty:
        metric_base["status"] = "insufficient_time_split"
        metric_base["test_sample_count"] = int(len(test))
        return metric_base, pd.DataFrame(), None

    x_train, feature_columns = build_design_matrix(train)
    x_calibrate, _ = build_design_matrix(calibrate, feature_columns)
    x_test, _ = build_design_matrix(test, feature_columns)
    model = _fit_base_model(x_train, train["target"])
    raw_calibrate = model.predict_proba(x_calibrate)[:, 1] if not calibrate.empty else np.array([], dtype=float)
    calibrator = _fit_probability_calibrator(raw_calibrate, calibrate["target"]) if not calibrate.empty else None
    raw_test = model.predict_proba(x_test)[:, 1]
    calibrated_test = _apply_calibrator(calibrator, raw_test)
    predictions = test[
        [
            "market_date",
            "symbol",
            "name",
            "candidate_strategy",
            "strategy_family",
            return_column,
            high_column,
            drawdown_column,
            "target",
            "split",
        ]
    ].copy()
    predictions = predictions.rename(
        columns={
            return_column: "forward_return",
            high_column: "max_high_return",
            drawdown_column: "max_drawdown",
        }
    )
    predictions["horizon_days"] = int(horizon_days)
    predictions["target_return"] = float(target_return)
    predictions["raw_probability"] = raw_test
    predictions["calibrated_probability"] = calibrated_test
    predictions["model_score"] = np.clip(calibrated_test * 100.0, 0.0, 100.0)

    top_cutoff = max(int(np.ceil(len(predictions) * 0.10)), 1)
    top_bucket = predictions.sort_values("calibrated_probability", ascending=False).head(top_cutoff)
    probability = pd.Series(calibrated_test)
    metric_base.update(
        {
            "test_sample_count": int(len(predictions)),
            "train_sample_count": int(len(train)),
            "calibration_sample_count": int(len(calibrate)),
            "positive_rate": float(local["target"].mean()),
            "test_positive_rate": float(predictions["target"].mean()),
            "auc": _safe_auc(predictions["target"], predictions["calibrated_probability"]),
            "brier": _safe_brier(predictions["target"], predictions["calibrated_probability"]),
            "calibration_gap": abs(float(predictions["calibrated_probability"].mean()) - float(predictions["target"].mean())),
            "avg_return": _safe_mean(predictions["forward_return"]),
            "top_bucket_return": _safe_mean(top_bucket["forward_return"]),
            "top_bucket_win_rate": float(top_bucket["target"].mean()) if not top_bucket.empty else float("nan"),
            "top_bucket_max_drawdown": _safe_mean(top_bucket["max_drawdown"]),
            "probability_p05": float(probability.quantile(0.05)),
            "probability_p50": float(probability.quantile(0.50)),
            "probability_p95": float(probability.quantile(0.95)),
            "feature_count": int(len(feature_columns)),
            "calibrator": "platt_logistic" if calibrator is not None else "none",
            "split_train_start": train["market_date"].min(),
            "split_train_end": train["market_date"].max(),
            "split_calibration_start": calibrate["market_date"].min() if not calibrate.empty else None,
            "split_calibration_end": calibrate["market_date"].max() if not calibrate.empty else None,
            "split_test_start": test["market_date"].min(),
            "split_test_end": test["market_date"].max(),
        }
    )
    model_payload = {
        "strategy_family": strategy_family,
        "horizon_days": int(horizon_days),
        "target_return": float(target_return),
        "feature_columns": feature_columns,
        "model": model,
        "calibrator": calibrator,
        "metrics": metric_base,
    }
    return metric_base, predictions, model_payload


def write_report(
    *,
    report_path: Path,
    output_dir: Path,
    metadata: dict[str, object],
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Hard Filter Plan Strategy Model Retraining And Calibration",
        "",
        "## Scope",
        "",
        f"- Strategy plan: `{metadata['strategy_plan']}`",
        f"- Candidate source: `{metadata['candidate_source']}`",
        f"- History source: `{metadata.get('history_source') or ''}`",
        f"- Candidate rows after labels: `{metadata['candidate_rows']}`",
        "- Training mode: per-strategy logistic meta-model plus Platt probability calibration.",
        "- Time split: train, calibration, and test are separated by market date.",
        "",
        "## Per-Strategy Metrics",
        "",
        "| strategy | horizon | samples | test | pos_rate | AUC | Brier | calibration_gap | avg_return | top_bucket_return | top_bucket_win_rate | p05/p50/p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics.sort_values(["strategy_family", "horizon_days"]).to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.get("strategy_family", "")),
                    str(int(row.get("horizon_days", 0) or 0)),
                    str(int(row.get("sample_count", 0) or 0)),
                    str(int(row.get("test_sample_count", 0) or 0)),
                    f"{float(row.get('positive_rate', 0.0) or 0.0):.3f}",
                    f"{float(row.get('auc', float('nan'))):.3f}" if not pd.isna(row.get("auc")) else "",
                    f"{float(row.get('brier', float('nan'))):.4f}" if not pd.isna(row.get("brier")) else "",
                    f"{float(row.get('calibration_gap', float('nan'))):.4f}" if not pd.isna(row.get("calibration_gap")) else "",
                    f"{float(row.get('avg_return', float('nan'))) * 100:.2f}%" if not pd.isna(row.get("avg_return")) else "",
                    f"{float(row.get('top_bucket_return', float('nan'))) * 100:.2f}%"
                    if not pd.isna(row.get("top_bucket_return"))
                    else "",
                    f"{float(row.get('top_bucket_win_rate', float('nan'))) * 100:.2f}%"
                    if not pd.isna(row.get("top_bucket_win_rate"))
                    else "",
                    (
                        f"{float(row.get('probability_p05', 0.0)):.3f}/"
                        f"{float(row.get('probability_p50', 0.0)):.3f}/"
                        f"{float(row.get('probability_p95', 0.0)):.3f}"
                    )
                    if not pd.isna(row.get("probability_p05"))
                    else "",
                ]
            )
            + " |"
        )
    rows.extend(
        [
            "",
            "## Calibration Buckets",
            "",
            "The full decile table is written to `strategy_calibration_bins.csv`.",
        ]
    )
    if not bins.empty:
        top_bins = bins.sort_values(["strategy_family", "horizon_days", "probability_bucket"]).groupby(
            ["strategy_family", "horizon_days"], as_index=False
        ).tail(1)
        rows.append("")
        rows.append("| strategy | horizon | top_bucket_samples | avg_probability | realized_rate | avg_return |")
        rows.append("|---|---:|---:|---:|---:|---:|")
        for row in top_bins.to_dict("records"):
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("strategy_family", "")),
                        str(int(row.get("horizon_days", 0) or 0)),
                        str(int(row.get("sample_count", 0) or 0)),
                        f"{float(row.get('avg_probability', 0.0)):.3f}",
                        f"{float(row.get('realized_positive_rate', 0.0)):.3f}",
                        f"{float(row.get('avg_return', 0.0)) * 100:.2f}%",
                    ]
                )
                + " |"
            )
    rows.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{output_dir / 'summary.json'}`",
            f"- Metrics: `{output_dir / 'strategy_model_metrics.csv'}`",
            f"- Calibration bins: `{output_dir / 'strategy_calibration_bins.csv'}`",
            f"- Test predictions: `{output_dir / 'strategy_model_predictions.csv'}`",
            f"- Model bundle: `{output_dir / 'strategy_model_bundle.pkl'}`",
        ]
    )
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_strategy_model_retraining(
    *,
    candidates_path: Path,
    output_dir: Path,
    strategy_plan_path: Path = DEFAULT_PLAN_PATH,
    history_path: Path | None = DEFAULT_HISTORY_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    min_samples: int = 500,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] candidates={candidates_path}", flush=True)
    candidates = load_plan_candidates(candidates_path)
    print(f"[load] candidate rows={len(candidates)}", flush=True)
    history = _load_history(history_path)
    print(f"[load] history rows={len(history)}", flush=True)
    frame = attach_forward_labels(candidates, history)
    labeled_path = output_dir / "strategy_training_frame.csv"
    frame.to_csv(labeled_path, index=False, encoding="utf-8-sig")

    metric_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    model_bundle: dict[str, object] = {
        "strategy_plan": str(strategy_plan_path),
        "candidate_source": str(candidates_path),
        "history_source": str(history_path) if history_path is not None else "",
        "models": {},
    }
    for family, spec in STRATEGY_SPECS.items():
        for horizon in spec.horizons:
            target_return = spec.target_returns[int(horizon)]
            print(f"[train] {family} horizon={horizon} target={target_return:g}", flush=True)
            metrics, predictions, model_payload = train_strategy_horizon_model(
                frame,
                strategy_family=family,
                horizon_days=int(horizon),
                target_return=float(target_return),
                min_samples=int(min_samples),
            )
            metric_rows.append(metrics)
            if not predictions.empty:
                prediction_parts.append(predictions)
            if model_payload is not None:
                model_bundle["models"][f"{family}_{int(horizon)}d"] = model_payload

    metrics_frame = pd.DataFrame(metric_rows)
    for column in REQUIRED_METRIC_COLUMNS:
        if column not in metrics_frame.columns:
            metrics_frame[column] = np.nan
    predictions_frame = (
        pd.concat(prediction_parts, ignore_index=True, sort=False)
        if prediction_parts
        else pd.DataFrame(columns=["strategy_family", "horizon_days"])
    )
    bins = _probability_bins(predictions_frame)

    metrics_path = output_dir / "strategy_model_metrics.csv"
    predictions_path = output_dir / "strategy_model_predictions.csv"
    bins_path = output_dir / "strategy_calibration_bins.csv"
    bundle_path = output_dir / "strategy_model_bundle.pkl"
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    bins.to_csv(bins_path, index=False, encoding="utf-8-sig")
    with bundle_path.open("wb") as handle:
        pickle.dump(model_bundle, handle)

    metadata = {
        "strategy_plan": str(strategy_plan_path),
        "candidate_source": str(candidates_path),
        "history_source": str(history_path) if history_path is not None else "",
        "candidate_rows": int(len(frame)),
        "strategy_counts": frame["strategy_family"].value_counts().to_dict(),
        "trained_model_count": int(len(model_bundle["models"])),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "calibration_bins_path": str(bins_path),
        "model_bundle_path": str(bundle_path),
        "training_frame_path": str(labeled_path),
        "report_path": str(report_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    write_report(report_path=report_path, output_dir=output_dir, metadata=metadata, metrics=metrics_frame, bins=bins)
    return metadata


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain and calibrate per-strategy models for the 2026-05-31 hard-filter plan.")
    parser.add_argument("--candidates-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--strategy-plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--min-samples", type=int, default=500)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run_strategy_model_retraining(
        candidates_path=Path(args.candidates_path),
        history_path=Path(args.history_path) if str(args.history_path).strip() else None,
        strategy_plan_path=Path(args.strategy_plan_path),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report_path),
        min_samples=int(args.min_samples),
    )
    print(
        json.dumps(
            {
                "candidate_rows": payload["candidate_rows"],
                "trained_model_count": payload["trained_model_count"],
                "metrics_path": payload["metrics_path"],
                "report_path": payload["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
