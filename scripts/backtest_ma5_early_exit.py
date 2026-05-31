from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from search_strategy_model_combinations import (
    DEFAULT_BULL_PATH,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SSE_PATH,
    DEFAULT_V3_DIR,
    SearchRule,
    _enrich_candidates,
    _load_candidates,
    _load_model_calendar,
    _select_top,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "openclaw_market_data.duckdb"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "ten_year_ma5_early_exit"


def _load_price_window(db_path: Path, symbols: list[str], date_from: str, date_to: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "trade_date", "close", "ma5"])
    con = duckdb.connect(str(db_path), read_only=True)
    symbols_df = pd.DataFrame({"symbol": symbols})
    con.register("symbols_df", symbols_df)
    query = """
        select p.symbol, p.trade_date, p.close
        from a_share_daily_prices p
        inner join symbols_df s on p.symbol = s.symbol
        where p.trade_date between (cast(? as date) - interval 30 day)
                              and (cast(? as date) + interval 15 day)
          and p.close is not null
        order by p.symbol, p.trade_date
    """
    prices = con.execute(query, [date_from, date_to]).fetchdf()
    con.close()
    if prices.empty:
        return prices
    prices["symbol"] = prices["symbol"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["symbol", "trade_date", "close"]).sort_values(["symbol", "trade_date"]).copy()
    prices["ma5"] = prices.groupby("symbol")["close"].transform(lambda series: series.rolling(5, min_periods=5).mean())
    prices["forward_index"] = prices.groupby("symbol").cumcount()
    return prices


def _attach_ma5_exit(selected: pd.DataFrame, prices: pd.DataFrame, hold_days: int = 3) -> pd.DataFrame:
    frame = selected.copy()
    if frame.empty:
        return frame
    frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")
    frame["hold_3d_return"] = pd.to_numeric(frame["hold_3d_return"], errors="coerce")
    frame = frame.dropna(subset=["market_date", "symbol", "entry_price", "hold_3d_return"]).copy()
    if frame.empty or prices.empty:
        frame["ma5_exit_return"] = frame["hold_3d_return"]
        frame["ma5_exit_triggered"] = False
        frame["ma5_exit_day"] = pd.NA
        frame["ma5_exit_date"] = pd.NaT
        frame["ma5_exit_close"] = np.nan
        frame["ma5_exit_ma5"] = np.nan
        return frame

    prices = prices.sort_values(["symbol", "trade_date"]).copy()
    by_symbol = {symbol: group.reset_index(drop=True) for symbol, group in prices.groupby("symbol", sort=False)}
    returns: list[float] = []
    triggered_flags: list[bool] = []
    exit_days: list[int | None] = []
    exit_dates: list[pd.Timestamp | pd.NaT] = []
    exit_closes: list[float | None] = []
    exit_ma5s: list[float | None] = []

    for row in frame.itertuples(index=False):
        symbol = str(row.symbol).zfill(6)
        market_date = pd.Timestamp(row.market_date).normalize()
        entry_price = float(row.entry_price)
        fallback_return = float(row.hold_3d_return)
        history = by_symbol.get(symbol)
        chosen_return = fallback_return
        triggered = False
        chosen_day: int | None = None
        chosen_date: pd.Timestamp | pd.NaT = pd.NaT
        chosen_close: float | None = None
        chosen_ma5: float | None = None
        if history is not None and entry_price > 0:
            future = history.loc[history["trade_date"].gt(market_date)].head(int(hold_days)).copy()
            for day_index, price_row in enumerate(future.itertuples(index=False), start=1):
                close_value = float(price_row.close)
                ma5_value = float(price_row.ma5) if pd.notna(price_row.ma5) else np.nan
                if pd.notna(ma5_value) and close_value < ma5_value:
                    chosen_return = close_value / entry_price - 1.0
                    triggered = True
                    chosen_day = day_index
                    chosen_date = pd.Timestamp(price_row.trade_date)
                    chosen_close = close_value
                    chosen_ma5 = ma5_value
                    break
        returns.append(chosen_return)
        triggered_flags.append(triggered)
        exit_days.append(chosen_day)
        exit_dates.append(chosen_date)
        exit_closes.append(chosen_close)
        exit_ma5s.append(chosen_ma5)

    frame["ma5_exit_return"] = returns
    frame["ma5_exit_triggered"] = triggered_flags
    frame["ma5_exit_day"] = exit_days
    frame["ma5_exit_date"] = exit_dates
    frame["ma5_exit_close"] = exit_closes
    frame["ma5_exit_ma5"] = exit_ma5s
    return frame


def _summarize(selected: pd.DataFrame, calendar: pd.DataFrame, return_column: str, rule_name: str) -> tuple[dict[str, object], pd.DataFrame]:
    frame = selected.copy()
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    frame = frame.dropna(subset=["market_date", return_column]).copy()
    if frame.empty:
        daily = pd.DataFrame(columns=["market_date", "selected", "avg_return"])
    else:
        daily = (
            frame.groupby("market_date", as_index=False)
            .agg(
                selected=("symbol", "count"),
                avg_return=(return_column, "mean"),
                avg_model_score=("model_score", "mean"),
                triggered=("ma5_exit_triggered", "sum") if "ma5_exit_triggered" in frame.columns else ("symbol", "count"),
            )
            .sort_values("market_date")
        )
    curve = calendar[["market_date"]].merge(daily, on="market_date", how="left")
    curve["selected"] = curve["selected"].fillna(0).astype(int)
    curve["avg_return"] = curve["avg_return"].fillna(0.0)
    curve["triggered"] = curve.get("triggered", pd.Series(0, index=curve.index)).fillna(0).astype(int)
    curve["equity"] = (1.0 + curve["avg_return"]).cumprod()
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["running_max"].replace(0.0, np.nan) - 1.0
    trade_returns = frame[return_column].dropna()
    active_daily = curve.loc[curve["selected"].gt(0), "avg_return"]
    ending = float(curve["equity"].iloc[-1]) if not curve.empty else 1.0
    annualized = ending ** (252.0 / len(curve)) - 1.0 if len(curve) and ending > 0 else 0.0
    max_drawdown = float(curve["drawdown"].min()) if not curve.empty else 0.0
    triggered_count = int(frame["ma5_exit_triggered"].sum()) if "ma5_exit_triggered" in frame.columns else 0
    summary = {
        "rule": rule_name,
        "return_column": return_column,
        "calendar_days": int(len(curve)),
        "active_days": int((curve["selected"] > 0).sum()),
        "coverage_pct": round(float((curve["selected"] > 0).mean()) * 100.0, 2) if not curve.empty else 0.0,
        "selected_rows": int(len(frame)),
        "avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else None,
        "trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else None,
        "target_hit_rate": round(float((trade_returns >= 0.03).mean()), 4) if not trade_returns.empty else None,
        "active_daily_return": round(float(active_daily.mean()), 6) if not active_daily.empty else None,
        "active_daily_win_rate": round(float((active_daily > 0).mean()), 4) if not active_daily.empty else None,
        "annualized_return": round(float(annualized), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "ending_equity": round(float(ending), 6),
        "return_drawdown_ratio": round(float(annualized / abs(max_drawdown)), 6) if max_drawdown < 0 else None,
        "ma5_exit_triggered_count": triggered_count,
        "ma5_exit_triggered_pct": round(float(triggered_count / len(frame)) * 100.0, 2) if len(frame) else 0.0,
    }
    return summary, curve


def run_backtest(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    v3_dir: str | Path = DEFAULT_V3_DIR,
    bull_path: str | Path = DEFAULT_BULL_PATH,
    sse_path: str | Path = DEFAULT_SSE_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_from: str = "2016-05-27",
    date_to: str = "2026-05-26",
) -> dict[str, object]:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calendar = _load_model_calendar(source_path / "model_scores.csv", date_from, date_to)
    candidates = _load_candidates(source_path, date_from, date_to)
    candidates = _enrich_candidates(candidates, Path(v3_dir), Path(bull_path), Path(sse_path))
    candidates["bull_score"] = pd.to_numeric(candidates.get("bull_score"), errors="coerce")

    rule = SearchRule("all", "v3_full_green", 68.0, None, 3, "model_score", None, 0)
    filtered = candidates.loc[candidates["bull_score"].ge(6.0)].copy()
    selected = _select_top(filtered, rule)
    prices = _load_price_window(Path(db_path), sorted(selected["symbol"].dropna().astype(str).str.zfill(6).unique()), date_from, date_to)
    selected = _attach_ma5_exit(selected, prices, hold_days=3)

    base_summary, base_curve = _summarize(selected, calendar, "hold_3d_return", "all_strategy_score68_top3_v3_full_green_bull6_hold3d")
    ma5_summary, ma5_curve = _summarize(selected, calendar, "ma5_exit_return", "all_strategy_score68_top3_v3_full_green_bull6_ma5_early_exit")

    selected.to_csv(output_path / "ma5_early_exit_selected.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [
            base_curve.assign(rule=base_summary["rule"]),
            ma5_curve.assign(rule=ma5_summary["rule"]),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(output_path / "ma5_early_exit_daily_curve.csv", index=False, encoding="utf-8-sig")
    summary = {
        "date_from": date_from,
        "date_to": date_to,
        "rule": {
            "candidate_pool": "strategy1/2/3",
            "model_score_threshold": 68,
            "top_n": 3,
            "market_filter": "V3 full_green",
            "bull_filter": "bull_score >= 6",
            "entry": "next trading day open, inherited from existing validation entry_price",
            "exit": "sell at close when close < MA5 within next 3 trading days; otherwise original 3-day hold return",
        },
        "base_hold3d": base_summary,
        "ma5_early_exit": ma5_summary,
        "paths": {
            "selected": str(output_path / "ma5_early_exit_selected.csv"),
            "daily_curve": str(output_path / "ma5_early_exit_daily_curve.csv"),
            "summary": str(output_path / "ma5_early_exit_summary.json"),
        },
    }
    (output_path / "ma5_early_exit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest V3 full green + bull6 top3 with MA5 early exit.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--v3-dir", default=str(DEFAULT_V3_DIR))
    parser.add_argument("--bull-path", default=str(DEFAULT_BULL_PATH))
    parser.add_argument("--sse-path", default=str(DEFAULT_SSE_PATH))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--date-from", default="2016-05-27")
    parser.add_argument("--date-to", default="2026-05-26")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_backtest(
        source_dir=args.source_dir,
        v3_dir=args.v3_dir,
        bull_path=args.bull_path,
        sse_path=args.sse_path,
        db_path=args.db_path,
        output_dir=args.output_dir,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
