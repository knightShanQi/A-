from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from a_share_predictor.data import _call_tushare_api
from a_share_predictor.database_source import load_env_file


load_env_file()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / ".cache" / "ten_year_model_strategy_validation"
DEFAULT_V3_DIR = PROJECT_ROOT / ".cache" / "ten_year_market_regime_v3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "ten_year_sse3500_filter"


def _fetch_sse_index(date_from: str, date_to: str, output_dir: Path, *, force: bool = False) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sse_000001_daily.csv"
    if path.exists() and not force:
        return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["market_date"])
    start = pd.to_datetime(date_from).strftime("%Y%m%d")
    end = pd.to_datetime(date_to).strftime("%Y%m%d")
    frame = _call_tushare_api(
        "index_daily",
        params={"ts_code": "000001.SH", "start_date": start, "end_date": end},
        fields="ts_code,trade_date,close,open,high,low,pct_chg,vol,amount",
    )
    if frame.empty:
        raise RuntimeError("Failed to fetch SSE Composite index_daily from Tushare.")
    frame = frame.rename(columns={"trade_date": "market_date", "pct_chg": "sse_pct_chg"})
    frame["market_date"] = pd.to_datetime(frame["market_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame.rename(
        columns={
            "close": "sse_close",
            "open": "sse_open",
            "high": "sse_high",
            "low": "sse_low",
            "vol": "sse_vol",
            "amount": "sse_amount",
        }
    )
    for column in ["sse_close", "sse_open", "sse_high", "sse_low", "sse_pct_chg", "sse_vol", "sse_amount"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["market_date", "sse_close"]).sort_values("market_date").reset_index(drop=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def _load_model_calendar(path: Path, date_from: str, date_to: str) -> pd.DataFrame:
    dates: set[pd.Timestamp] = set()
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=["market_date"], chunksize=500_000):
        parsed = pd.to_datetime(chunk["market_date"], errors="coerce").dropna().dt.normalize()
        dates.update(pd.Timestamp(value) for value in parsed.unique())
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    calendar = pd.DataFrame({"market_date": sorted(dates)})
    return calendar.loc[calendar["market_date"].between(start, end, inclusive="both")].copy()


def _summarize(selected: pd.DataFrame, calendar: pd.DataFrame, *, rule: str, positive_return: float = 0.03) -> dict[str, object]:
    frame = selected.copy()
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
    frame["hold_3d_return"] = pd.to_numeric(frame["hold_3d_return"], errors="coerce")
    frame = frame.dropna(subset=["market_date", "hold_3d_return"])
    daily = (
        frame.groupby("market_date", as_index=False)
        .agg(selected=("symbol", "count"), avg_return=("hold_3d_return", "mean"))
        .sort_values("market_date")
    )
    curve = calendar[["market_date"]].merge(daily, on="market_date", how="left")
    curve["selected"] = curve["selected"].fillna(0).astype(int)
    curve["avg_return"] = curve["avg_return"].fillna(0.0)
    curve["equity"] = (1.0 + curve["avg_return"]).cumprod()
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["running_max"].replace(0.0, np.nan) - 1.0
    ending = float(curve["equity"].iloc[-1]) if not curve.empty else 1.0
    annualized = ending ** (252.0 / len(curve)) - 1.0 if len(curve) and ending > 0 else -1.0
    max_drawdown = float(curve["drawdown"].min()) if not curve.empty else 0.0
    trade_returns = frame["hold_3d_return"].dropna()
    active_returns = daily["avg_return"].dropna()
    return {
        "rule": rule,
        "calendar_days": int(len(calendar)),
        "active_days": int((curve["selected"] > 0).sum()),
        "coverage_pct": round(float((curve["selected"] > 0).mean()) * 100.0, 2) if not curve.empty else 0.0,
        "selected_rows": int(len(frame)),
        "avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else 0.0,
        "trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else 0.0,
        "target_hit_rate": round(float((trade_returns >= float(positive_return)).mean()), 4) if not trade_returns.empty else 0.0,
        "active_daily_return": round(float(active_returns.mean()), 6) if not active_returns.empty else 0.0,
        "active_daily_win_rate": round(float((active_returns > 0).mean()), 4) if not active_returns.empty else 0.0,
        "annualized_return": round(float(annualized), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "ending_equity": round(float(ending), 6),
        "return_drawdown_ratio": round(float(annualized / abs(max_drawdown)), 6) if max_drawdown else None,
    }


def run_sse3500_filter(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    v3_dir: str | Path = DEFAULT_V3_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_from: str = "2016-05-27",
    date_to: str = "2026-05-26",
    threshold: float = 3500.0,
    force: bool = False,
) -> dict[str, object]:
    source_path = Path(source_dir)
    v3_path = Path(v3_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sse = _fetch_sse_index(date_from, date_to, output_path, force=force)
    calendar = _load_model_calendar(source_path / "model_scores.csv", date_from, date_to)
    calendar = calendar.merge(sse[["market_date", "sse_close"]], on="market_date", how="left")
    eligible_calendar = calendar.loc[calendar["sse_close"].ge(float(threshold))].copy()

    selected = pd.read_csv(v3_path / "v3_existing_overlay_selected_trades.csv", encoding="utf-8-sig", parse_dates=["market_date"])
    selected["hold_3d_return"] = pd.to_numeric(selected["hold_3d_return"], errors="coerce")
    selected = selected.merge(sse[["market_date", "sse_close"]], on="market_date", how="left")
    eligible_selected = selected.loc[selected["sse_close"].ge(float(threshold))].copy()

    rules = [
        "existing_v2_top3_score68_full",
        "existing_v3_trend_flow_top3",
        "existing_v3_full_green_top3",
        "existing_v3_trend_flow_cand40_100_top3",
    ]
    rows: list[dict[str, object]] = []
    for rule in rules:
        original = selected.loc[selected["rule"].eq(rule)].copy()
        filtered = eligible_selected.loc[eligible_selected["rule"].eq(rule)].copy()
        rows.append({"filter_mode": "baseline_full_calendar", **_summarize(original, calendar, rule=rule)})
        rows.append({"filter_mode": "sse3500_as_cash_full_calendar", **_summarize(filtered, calendar, rule=rule)})
        rows.append({"filter_mode": "sse3500_conditional_calendar", **_summarize(filtered, eligible_calendar, rule=rule)})

    result = pd.DataFrame(rows)
    result = result.sort_values(["filter_mode", "return_drawdown_ratio", "annualized_return"], ascending=[True, False, False])
    result.to_csv(output_path / "sse3500_filter_rules.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_path / "sse3500_selected_trades_with_index.csv", index=False, encoding="utf-8-sig")
    summary = {
        "date_from": date_from,
        "date_to": date_to,
        "threshold": float(threshold),
        "calendar_days": int(len(calendar)),
        "eligible_days": int(len(eligible_calendar)),
        "below_threshold_days": int(calendar["sse_close"].lt(float(threshold)).sum()),
        "missing_sse_days": int(calendar["sse_close"].isna().sum()),
        "rules": result.replace({np.nan: None}).to_dict("records"),
        "summary_path": str(output_path / "sse3500_summary.json"),
        "rules_path": str(output_path / "sse3500_filter_rules.csv"),
        "index_path": str(output_path / "sse_000001_daily.csv"),
        "selected_trades_path": str(output_path / "sse3500_selected_trades_with_index.csv"),
    }
    (output_path / "sse3500_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay SSE Composite 3500 filter on V2/V3 backtest selections.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--v3-dir", default=str(DEFAULT_V3_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--date-from", default="2016-05-27")
    parser.add_argument("--date-to", default="2026-05-26")
    parser.add_argument("--threshold", type=float, default=3500.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_sse3500_filter(
        source_dir=args.source_dir,
        v3_dir=args.v3_dir,
        output_dir=args.output_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        threshold=float(args.threshold),
        force=bool(args.force),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
