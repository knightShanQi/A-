from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from a_share_predictor.modeling import (
    EXTERNAL_SNAPSHOT_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    _apply_incremental_probability_upgrade,
    _append_market_regime_features,
    _append_market_resonance_features,
    _augment_model_features,
    _build_backtest_metrics,
    _ensemble_probability,
    load_cached_market_wide_model,
    load_market_proxy_model,
)
from a_share_predictor.features import build_training_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
OUTPUT_DIR = CACHE_DIR / "backtests"


def _snapshot_paths() -> list[Path]:
    return sorted(CACHE_DIR.glob("market_snapshot_history_store_v1_*.pkl"))


def _load_snapshot(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    return data.copy()


def _load_market_history() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _snapshot_paths():
        frame = _load_snapshot(path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    history = pd.concat(frames, ignore_index=True)
    history["symbol"] = history["symbol"].astype(str).str.zfill(6)
    history["trade_date"] = pd.to_datetime(history.get("trade_date", history.get("date")), errors="coerce")
    history = history.dropna(subset=["symbol", "trade_date"])
    history = history.sort_values(["symbol", "trade_date"])
    history = history.drop_duplicates(["symbol", "trade_date"], keep="last").reset_index(drop=True)
    return history


def _legacy_proxy_paths() -> list[Path]:
    return sorted(CACHE_DIR.glob("global_market_proxy*_h*_r*_*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)


def _load_latest_available_proxy_model():
    for path in _legacy_proxy_paths():
        try:
            with path.open("rb") as handle:
                model = pickle.load(handle)
        except Exception:
            continue
        if hasattr(model, "fitted_model"):
            return model, path
    return None, None


def _month_window(trade_dates: Iterable[pd.Timestamp], latest_date: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(trade_dates), errors="coerce").dropna().unique()))
    if dates.empty:
        raise RuntimeError("No valid trade dates found in local market snapshots.")
    end_date = pd.Timestamp(latest_date) if latest_date else dates.max()
    end_date = dates[dates <= end_date].max()
    start_anchor = end_date - pd.DateOffset(months=1)
    eligible = dates[dates >= start_anchor]
    start_date = eligible.min() if len(eligible) else dates[max(0, len(dates) - 20)]
    return pd.Timestamp(start_date), pd.Timestamp(end_date)


def _prepare_symbol_daily(group: pd.DataFrame) -> pd.DataFrame:
    daily = group.copy()
    daily["date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "vol", "amount", "turnover"):
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
    if "volume" not in daily.columns and "vol" in daily.columns:
        daily["volume"] = daily["vol"]
    if "volume" in daily.columns:
        daily["volume"] = daily["volume"].fillna(0.0)
    if "turnover" in daily.columns:
        daily["turnover"] = daily["turnover"].fillna(1.0)
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily.set_index("date", drop=False)


def _build_recent_dataset(
    history: pd.DataFrame,
    *,
    horizon_days: int,
    positive_return: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total_symbols = int(history["symbol"].nunique())
    for index, (symbol, group) in enumerate(history.groupby("symbol", sort=True), start=1):
        if index == 1 or index % 250 == 0:
            print(f"prepared symbols {index}/{total_symbols}", flush=True)
        daily = _prepare_symbol_daily(group)
        if len(daily) < max(80, horizon_days + 30):
            continue
        try:
            dataset = build_training_frame(
                daily,
                horizon_days=horizon_days,
                positive_return=positive_return,
            )
            dataset = _augment_model_features(dataset)
        except Exception:
            continue
        if dataset.empty:
            continue
        dataset = dataset.copy()
        if "signal_date" not in dataset.columns:
            dataset["signal_date"] = pd.to_datetime(dataset.index, errors="coerce")
        dataset["symbol"] = str(symbol).zfill(6)
        if "name" in group.columns:
            dataset["name"] = str(group["name"].dropna().iloc[-1]) if group["name"].notna().any() else ""
        signal_dates = pd.to_datetime(dataset["signal_date"], errors="coerce")
        mask = signal_dates.between(start_date, end_date, inclusive="both")
        selected = dataset.loc[mask].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True)
    return _attach_fast_context_features(dataset)


def _recent_dataset_cache_path(
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon_days: int,
    positive_return: float,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / (
        f"recent_market_model_dataset_{start_date:%Y%m%d}_{end_date:%Y%m%d}_"
        f"h{horizon_days}_r{int(positive_return * 10000)}.pkl"
    )


def _build_or_load_recent_dataset(
    history: pd.DataFrame,
    *,
    horizon_days: int,
    positive_return: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    cache_path = _recent_dataset_cache_path(
        start_date=start_date,
        end_date=end_date,
        horizon_days=horizon_days,
        positive_return=positive_return,
    )
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                print(f"loaded cached recent dataset: {cache_path}", flush=True)
                return cached.copy()
        except Exception:
            pass
    dataset = _build_recent_dataset(
        history,
        horizon_days=horizon_days,
        positive_return=positive_return,
        start_date=start_date,
        end_date=end_date,
    )
    if not dataset.empty:
        with cache_path.open("wb") as handle:
            pickle.dump(dataset, handle)
    return dataset


def _attach_fast_context_features(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return dataset
    enriched = dataset.copy()
    enriched["signal_date"] = pd.to_datetime(enriched["signal_date"], errors="coerce")
    date_stats = (
        enriched.groupby("signal_date", dropna=True)
        .agg(
            market_ret_5=("ret_5", "mean"),
            market_ret_20=("ret_20", "mean"),
            market_close_vs_ma20=("close_vs_ma20", "mean"),
            market_volatility_10=("volatility_10", "mean"),
            market_range_position_20=("range_position_20", "mean"),
        )
        .reset_index()
    )
    enriched = enriched.merge(date_stats, on="signal_date", how="left")
    enriched["relative_strength_5"] = pd.to_numeric(enriched["ret_5"], errors="coerce").fillna(0.0) - pd.to_numeric(
        enriched["market_ret_5"], errors="coerce"
    ).fillna(0.0)
    enriched["relative_strength_20"] = pd.to_numeric(enriched["ret_20"], errors="coerce").fillna(0.0) - pd.to_numeric(
        enriched["market_ret_20"], errors="coerce"
    ).fillna(0.0)
    for column in MARKET_FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    for column in EXTERNAL_SNAPSHOT_COLUMNS:
        if column in {"news_positive_ratio_7d", "fund_positive_ratio_5d"}:
            enriched[column] = 0.5
        else:
            enriched[column] = 0.0
    enriched = _append_market_regime_features(enriched)
    enriched = _append_market_resonance_features(enriched)
    return enriched.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _daily_top_panel(dataset: pd.DataFrame, probability_column: str, *, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date, group in dataset.groupby("signal_date", sort=True):
        ranked = group.sort_values(probability_column, ascending=False).head(top_n)
        if ranked.empty:
            continue
        universe_avg = float(group["future_return"].mean())
        rows.append(
            {
                "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "sample_size": int(len(group)),
                "selected": int(len(ranked)),
                "avg_probability_pct": round(float(ranked[probability_column].mean()) * 100, 2),
                "avg_return_pct": round(float(ranked["future_return"].mean()) * 100, 2),
                "win_rate_pct": round(float((ranked["future_return"] > 0).mean()) * 100, 2),
                "target_hit_rate_pct": round(float(ranked["target"].mean()) * 100, 2),
                "universe_avg_return_pct": round(universe_avg * 100, 2),
                "excess_return_pct": round((float(ranked["future_return"].mean()) - universe_avg) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def _candidate_gate_mask(dataset: pd.DataFrame) -> pd.Series:
    frame = dataset.copy()
    ret_20 = pd.to_numeric(frame.get("ret_20"), errors="coerce").fillna(0.0)
    ret_10 = pd.to_numeric(frame.get("ret_10"), errors="coerce").fillna(0.0)
    close_vs_ma20 = pd.to_numeric(frame.get("close_vs_ma20"), errors="coerce").fillna(0.0)
    ma_alignment = pd.to_numeric(frame.get("ma_alignment_score"), errors="coerce").fillna(0.5)
    range_position_20 = pd.to_numeric(frame.get("range_position_20"), errors="coerce").fillna(0.5)
    breakout_distance_20 = pd.to_numeric(frame.get("breakout_distance_20"), errors="coerce").fillna(0.0)
    volume_ratio_5 = pd.to_numeric(frame.get("volume_ratio_5"), errors="coerce").fillna(1.0)
    upper_shadow_ratio = pd.to_numeric(frame.get("upper_shadow_ratio"), errors="coerce").fillna(0.0)
    volatility_20 = pd.to_numeric(frame.get("volatility_20"), errors="coerce").fillna(0.0)
    risk_pressure = pd.to_numeric(frame.get("risk_pressure"), errors="coerce").fillna(50.0)
    stretch_risk = pd.to_numeric(frame.get("stretch_risk"), errors="coerce").fillna(0.0)
    launch_readiness = pd.to_numeric(frame.get("launch_readiness"), errors="coerce").fillna(50.0)
    market_resonance = pd.to_numeric(frame.get("market_resonance"), errors="coerce").fillna(50.0)
    relative_strength_20 = pd.to_numeric(frame.get("relative_strength_20"), errors="coerce").fillna(0.0)

    base_gate = (
        ret_20.gt(-0.08)
        & ret_10.gt(-0.06)
        & close_vs_ma20.gt(-0.04)
        & ma_alignment.ge(0.33)
        & range_position_20.ge(0.25)
        & breakout_distance_20.ge(-0.14)
        & volume_ratio_5.between(0.60, 3.50, inclusive="both")
        & upper_shadow_ratio.le(0.06)
        & volatility_20.le(0.08)
        & risk_pressure.le(240.0)
        & stretch_risk.le(70.0)
        & relative_strength_20.ge(-0.14)
    )
    score = (
        ret_20.rank(pct=True) * 0.16
        + close_vs_ma20.rank(pct=True) * 0.14
        + ma_alignment.rank(pct=True) * 0.10
        + range_position_20.rank(pct=True) * 0.10
        + breakout_distance_20.rank(pct=True) * 0.10
        + volume_ratio_5.clip(upper=2.5).rank(pct=True) * 0.08
        + launch_readiness.rank(pct=True) * 0.12
        + market_resonance.rank(pct=True) * 0.12
        + relative_strength_20.rank(pct=True) * 0.12
        - risk_pressure.rank(pct=True) * 0.08
        - stretch_risk.rank(pct=True) * 0.06
    )
    score = score.fillna(0.0)
    if "signal_date" in frame.columns:
        threshold = score.groupby(pd.to_datetime(frame["signal_date"], errors="coerce")).transform(
            lambda values: values.quantile(0.75)
        )
    else:
        threshold = pd.Series(score.quantile(0.75), index=score.index)
    return (base_gate & score.ge(threshold.fillna(score.quantile(0.75)))).fillna(False)


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return numeric


def _align_features_for_estimator(model, features: pd.DataFrame) -> pd.DataFrame:
    estimator = getattr(model, "fitted_model", model)
    feature_names = None
    try:
        feature_names = getattr(estimator.named_steps.get("imputer"), "feature_names_in_", None)
    except Exception:
        feature_names = None
    if feature_names is None:
        feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is None:
        return features
    return features.reindex(columns=list(feature_names), fill_value=0.0)


def run_backtest(
    *,
    horizon_days: int,
    positive_return: float,
    latest_date: str | None,
    top_n_values: tuple[int, ...],
) -> dict[str, object]:
    history = _load_market_history()
    if history.empty:
        raise RuntimeError("No local market snapshot history found.")

    start_date, end_date = _month_window(history["trade_date"], latest_date)
    dataset = _build_or_load_recent_dataset(
        history,
        horizon_days=horizon_days,
        positive_return=positive_return,
        start_date=start_date,
        end_date=end_date,
    )
    if dataset.empty:
        raise RuntimeError("No evaluable model samples were produced for the requested window.")

    features = dataset.reindex(columns=MODEL_FEATURE_COLUMNS)
    wide_model = load_cached_market_wide_model(horizon_days=horizon_days, positive_return=positive_return)
    proxy_model = None if wide_model is not None else load_market_proxy_model(
        horizon_days=horizon_days,
        positive_return=positive_return,
    )
    legacy_proxy_path = None
    if wide_model is None and proxy_model is None:
        proxy_model, legacy_proxy_path = _load_latest_available_proxy_model()
    if wide_model is not None:
        probabilities, _ = _ensemble_probability(
            wide_model.fitted_models,
            features,
            weights=wide_model.ensemble_weights,
            calibrator=wide_model.calibrator,
            calibration_feature_frame=dataset,
            regime_calibrators=wide_model.regime_calibrators,
        )
        model_train_window = f"{wide_model.train_start} -> {wide_model.train_end}"
        model_test_window = f"{wide_model.test_start} -> {wide_model.test_end}"
        model_quality_label = wide_model.quality_label
        model_kind = "market_wide"
    elif proxy_model is not None:
        proxy_features = _align_features_for_estimator(proxy_model, features)
        probabilities = proxy_model.fitted_model.predict_proba(proxy_features)[:, 1]
        model_train_window = "partial-market proxy cache"
        model_test_window = "proxy validation split"
        model_quality_label = str(
            getattr(proxy_model, "validation_summary", "")
            or getattr(proxy_model, "source_label", "")
            or "legacy proxy model"
        )
        candidate_name = str(getattr(proxy_model, "candidate_name", "legacy"))
        model_kind = f"market_proxy:{candidate_name}"
        if legacy_proxy_path is not None:
            model_kind = f"legacy_{model_kind}"
    else:
        raise RuntimeError("No cached market-wide model or trainable proxy model is available.")
    probabilities, _ = _apply_incremental_probability_upgrade(probabilities, features)
    dataset = dataset.copy()
    dataset["model_probability"] = probabilities
    y_true = dataset["target"].to_numpy(dtype=int)
    metrics = _build_backtest_metrics(y_true, probabilities, dataset["future_return"].astype(float))

    panels: dict[str, dict[str, object]] = {}
    daily_frames: dict[int, pd.DataFrame] = {}
    for top_n in top_n_values:
        panel = _daily_top_panel(dataset, "model_probability", top_n=top_n)
        daily_frames[top_n] = panel
        panels[f"top_{top_n}"] = {
            "days": int(len(panel)),
            "selected_total": int(panel["selected"].sum()) if not panel.empty else 0,
            "avg_probability_pct": round(float(panel["avg_probability_pct"].mean()), 2) if not panel.empty else 0.0,
            "avg_return_pct": round(float(panel["avg_return_pct"].mean()), 2) if not panel.empty else 0.0,
            "win_rate_pct": round(float(panel["win_rate_pct"].mean()), 2) if not panel.empty else 0.0,
            "target_hit_rate_pct": round(float(panel["target_hit_rate_pct"].mean()), 2) if not panel.empty else 0.0,
            "excess_return_pct": round(float(panel["excess_return_pct"].mean()), 2) if not panel.empty else 0.0,
        }
    candidate_mask = _candidate_gate_mask(dataset)
    candidate_dataset = dataset.loc[candidate_mask].copy()
    candidate_panels: dict[str, dict[str, object]] = {}
    candidate_daily_paths = {}
    for top_n in top_n_values:
        panel = _daily_top_panel(candidate_dataset, "model_probability", top_n=top_n)
        candidate_panels[f"candidate_top_{top_n}"] = {
            "days": int(len(panel)),
            "selected_total": int(panel["selected"].sum()) if not panel.empty else 0,
            "avg_probability_pct": round(float(panel["avg_probability_pct"].mean()), 2) if not panel.empty else 0.0,
            "avg_return_pct": round(float(panel["avg_return_pct"].mean()), 2) if not panel.empty else 0.0,
            "win_rate_pct": round(float(panel["win_rate_pct"].mean()), 2) if not panel.empty else 0.0,
            "target_hit_rate_pct": round(float(panel["target_hit_rate_pct"].mean()), 2) if not panel.empty else 0.0,
            "excess_return_pct": round(float(panel["excess_return_pct"].mean()), 2) if not panel.empty else 0.0,
        }
        daily_frames[f"candidate_top_{top_n}"] = panel

    probability_thresholds = [0.5, 0.6, 0.7, 0.8, float(metrics.get("precision_gate_threshold", 1.0) or 1.0)]
    threshold_rows = []
    for threshold in sorted(set(round(value, 4) for value in probability_thresholds if value >= 0.0 and value <= 1.0)):
        selected = dataset[dataset["model_probability"] >= threshold]
        threshold_rows.append(
            {
                "threshold": threshold,
                "support": int(len(selected)),
                "precision_pct": round(float(selected["target"].mean()) * 100, 2) if not selected.empty else 0.0,
                "direction_win_rate_pct": round(float((selected["future_return"] > 0).mean()) * 100, 2) if not selected.empty else 0.0,
                "avg_return_pct": round(float(selected["future_return"].mean()) * 100, 2) if not selected.empty else 0.0,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_h{horizon_days}_r{int(positive_return * 10000)}"
    sample_path = OUTPUT_DIR / f"recent_market_model_samples_{stamp}.csv"
    summary_path = OUTPUT_DIR / f"recent_market_model_summary_{stamp}.json"
    daily_paths = {}
    export_cols = [
        "signal_date",
        "symbol",
        "name",
        "model_probability",
        "target",
        "future_return",
    ]
    dataset[[column for column in export_cols if column in dataset.columns]].to_csv(sample_path, index=False, encoding="utf-8-sig")
    for top_n, panel in daily_frames.items():
        label = str(top_n)
        path = OUTPUT_DIR / f"recent_market_model_daily_{label}_{stamp}.csv"
        panel.to_csv(path, index=False, encoding="utf-8-sig")
        daily_paths[label if label.startswith("candidate_") else f"top_{label}"] = str(path)

    signal_dates = pd.to_datetime(dataset["signal_date"], errors="coerce")
    summary = {
        "horizon_days": int(horizon_days),
        "positive_return": float(positive_return),
        "positive_return_pct": round(float(positive_return) * 100, 2),
        "requested_signal_start": start_date.strftime("%Y-%m-%d"),
        "requested_signal_end": end_date.strftime("%Y-%m-%d"),
        "actual_signal_start": signal_dates.min().strftime("%Y-%m-%d") if signal_dates.notna().any() else "",
        "actual_signal_end": signal_dates.max().strftime("%Y-%m-%d") if signal_dates.notna().any() else "",
        "history_rows": int(len(history)),
        "history_symbols": int(history["symbol"].nunique()),
        "sample_rows": int(len(dataset)),
        "sample_symbols": int(dataset["symbol"].nunique()),
        "sample_days": int(pd.to_datetime(dataset["signal_date"], errors="coerce").nunique()),
        "metrics": {key: _safe_float(value) for key, value in metrics.items()},
        "top_selection": panels,
        "candidate_gate": {
            "sample_rows": int(len(candidate_dataset)),
            "sample_symbols": int(candidate_dataset["symbol"].nunique()) if not candidate_dataset.empty else 0,
            "sample_days": int(pd.to_datetime(candidate_dataset["signal_date"], errors="coerce").nunique())
            if not candidate_dataset.empty
            else 0,
            "coverage_pct": round(float(len(candidate_dataset) / max(len(dataset), 1)) * 100, 2),
        },
        "candidate_top_selection": candidate_panels,
        "thresholds": threshold_rows,
        "model_kind": model_kind,
        "model_train_window": model_train_window,
        "model_test_window": model_test_window,
        "model_quality_label": model_quality_label,
        "sample_path": str(sample_path),
        "daily_paths": daily_paths,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the current market-wide model on the most recent month of A-share data.")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--positive-return", type=float, default=0.03)
    parser.add_argument("--latest-date", type=str, default="")
    parser.add_argument("--top-n", type=str, default="50,100")
    args = parser.parse_args()
    top_n_values = tuple(int(item.strip()) for item in args.top_n.split(",") if item.strip())
    summary = run_backtest(
        horizon_days=int(args.horizon_days),
        positive_return=float(args.positive_return),
        latest_date=args.latest_date or None,
        top_n_values=top_n_values or (50,),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
