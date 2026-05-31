from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from a_share_predictor.data import fetch_a_share_universe, normalize_symbol
from a_share_predictor.news_impact import analyze_symbol_news_impact, summarize_category_impact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "news_impact_research"
DEFAULT_PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")


def _safe_date_token(value: str | None) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "", str(value or "")) or "none"


def _symbol_cache_path(
    cache_dir: Path,
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
    news_limit: int,
    include_disclosures: bool,
    horizons: Sequence[int],
) -> Path:
    horizon_token = "-".join(str(int(value)) for value in horizons)
    disclosure_token = "with_disclosures" if include_disclosures else "news_only"
    return (
        cache_dir
        / f"{symbol}_{_safe_date_token(start_date)}_{_safe_date_token(end_date)}_{news_limit}_{horizon_token}_{disclosure_token}.pkl"
    )


def _read_symbols_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        for piece in re.split(r"[,，\s]+", line.strip()):
            if not piece:
                continue
            try:
                values.append(normalize_symbol(piece))
            except ValueError:
                continue
    return values


def _dedupe_symbols(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in symbols:
        try:
            symbol = normalize_symbol(raw)
        except ValueError:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def _round_robin_by_prefix(universe: pd.DataFrame, prefixes: Sequence[str], limit: int) -> list[str]:
    if universe.empty:
        return []
    frame = universe.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    frame = frame.loc[frame["symbol"].str.len().eq(6)].drop_duplicates("symbol").sort_values("symbol")
    buckets: dict[str, list[str]] = {
        prefix: frame.loc[frame["symbol"].str.startswith(prefix), "symbol"].tolist()
        for prefix in prefixes
    }
    selected: list[str] = []
    seen: set[str] = set()
    while len(selected) < limit:
        added = False
        for prefix in prefixes:
            bucket = buckets.get(prefix, [])
            while bucket and bucket[0] in seen:
                bucket.pop(0)
            if not bucket:
                continue
            symbol = bucket.pop(0)
            selected.append(symbol)
            seen.add(symbol)
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def resolve_symbols(args: argparse.Namespace) -> pd.DataFrame:
    requested: list[str] = []
    if args.symbols:
        requested.extend(piece.strip() for piece in re.split(r"[,，\s]+", args.symbols) if piece.strip())
    if args.symbols_file:
        requested.extend(_read_symbols_file(Path(args.symbols_file)))
    if requested:
        symbols = _dedupe_symbols(requested)
        return pd.DataFrame({"symbol": symbols, "name": symbols})

    universe = fetch_a_share_universe()
    if universe.empty:
        return pd.DataFrame(columns=["symbol", "name"])
    view = universe.copy()
    view["symbol"] = view["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    view["name"] = view.get("name", view["symbol"]).fillna("").astype(str)
    if args.exclude_st:
        view = view.loc[~view["name"].str.upper().str.contains("ST", regex=False)].copy()
    prefixes = tuple(piece.strip() for piece in str(args.prefixes or "").split(",") if piece.strip()) or DEFAULT_PREFIXES
    symbols = _round_robin_by_prefix(view, prefixes, int(args.max_symbols))
    selected = pd.DataFrame({"symbol": symbols}).merge(view[["symbol", "name"]].drop_duplicates("symbol"), on="symbol", how="left")
    selected["name"] = selected["name"].fillna(selected["symbol"])
    return selected


def _empty_result(symbol: str, name: str, error: str | None = None) -> dict[str, object]:
    return {
        "symbol": symbol,
        "name": name,
        "event_count": 0,
        "impact_sample_count": 0,
        "events": pd.DataFrame(),
        "event_impacts": pd.DataFrame(),
        "category_summary": pd.DataFrame(),
        "latest_signal": {},
        "error": error,
        "cached": False,
    }


def analyze_one_symbol(
    symbol: str,
    name: str,
    *,
    start_date: str | None,
    end_date: str | None,
    news_limit: int,
    horizons: Sequence[int],
    include_disclosures: bool,
    cache_dir: Path,
    refresh: bool,
    pause_seconds: float,
) -> dict[str, object]:
    clean_symbol = normalize_symbol(symbol)
    cache_path = _symbol_cache_path(
        cache_dir,
        clean_symbol,
        start_date=start_date,
        end_date=end_date,
        news_limit=news_limit,
        include_disclosures=include_disclosures,
        horizons=horizons,
    )
    if cache_path.exists() and not refresh:
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict):
                payload["cached"] = True
                return payload
        except Exception:
            pass

    if pause_seconds > 0:
        time.sleep(float(pause_seconds))
    try:
        result = analyze_symbol_news_impact(
            clean_symbol,
            start_date=start_date,
            end_date=end_date,
            news_limit=int(news_limit),
            horizons=tuple(int(value) for value in horizons),
            include_disclosures=bool(include_disclosures),
        )
        for key in ("events", "event_impacts", "category_summary"):
            frame = result.get(key)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame = frame.copy()
                frame["sample_symbol"] = clean_symbol
                frame["sample_name"] = name
                result[key] = frame
        payload = {
            "symbol": clean_symbol,
            "name": name,
            "event_count": int(result.get("event_count", 0)),
            "impact_sample_count": int(result.get("impact_sample_count", 0)),
            "events": result.get("events", pd.DataFrame()),
            "event_impacts": result.get("event_impacts", pd.DataFrame()),
            "category_summary": result.get("category_summary", pd.DataFrame()),
            "latest_signal": result.get("latest_signal", {}),
            "error": None,
            "cached": False,
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return payload
    except Exception as exc:
        return _empty_result(clean_symbol, name, str(exc))


def _concat_frames(results: Sequence[dict[str, object]], key: str) -> pd.DataFrame:
    frames = [item.get(key) for item in results if isinstance(item.get(key), pd.DataFrame) and not item.get(key).empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_category_direction(impact_df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    if impact_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (category, direction), group in impact_df.groupby(["event_category", "event_direction"], dropna=False):
        row: dict[str, object] = {
            "event_category": category,
            "event_direction": direction,
            "event_count": int(len(group)),
            "avg_expected_impact_score": round(float(pd.to_numeric(group.get("expected_impact_score"), errors="coerce").mean()), 2),
            "avg_open_gap_pct": round(float(pd.to_numeric(group.get("open_gap_pct"), errors="coerce").mean()), 4),
        }
        for horizon in horizons:
            return_column = f"return_{int(horizon)}d_pct"
            hit_column = f"direction_hit_{int(horizon)}d"
            returns = pd.to_numeric(group.get(return_column), errors="coerce")
            hits = pd.to_numeric(group.get(hit_column), errors="coerce") if hit_column in group.columns else pd.Series(dtype=float)
            row[f"avg_return_{int(horizon)}d_pct"] = round(float(returns.mean()), 4) if returns.notna().any() else None
            row[f"median_return_{int(horizon)}d_pct"] = round(float(returns.median()), 4) if returns.notna().any() else None
            row[f"positive_return_rate_{int(horizon)}d"] = round(float((returns > 0).mean()), 4) if returns.notna().any() else None
            row[f"direction_hit_rate_{int(horizon)}d"] = round(float(hits.mean()), 4) if hits.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["event_count", "event_category"], ascending=[False, True]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row.get(column)
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    meta: pd.DataFrame,
    category_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# News Impact Research Run",
        "",
        f"- Generated at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Symbols requested: {len(meta)}",
        f"- Symbols with impact samples: {int((meta['impact_sample_count'] > 0).sum()) if not meta.empty else 0}",
        f"- Total events: {int(meta['event_count'].sum()) if not meta.empty else 0}",
        f"- Total impact samples: {int(meta['impact_sample_count'].sum()) if not meta.empty else 0}",
        f"- Date window: {args.start_date or 'auto'} to {args.end_date or 'today'}",
        f"- Include disclosures: {not args.no_disclosures}",
        "",
        "## Category Summary",
        "",
    ]
    if category_summary.empty:
        lines.append("No category summary was produced.")
    else:
        columns = [
            column
            for column in (
                "event_category",
                "event_count",
                "bullish_count",
                "bearish_count",
                "neutral_count",
                "avg_open_gap_pct",
                "avg_return_1d_pct",
                "direction_hit_rate_1d",
                "avg_return_3d_pct",
                "direction_hit_rate_3d",
                "avg_return_5d_pct",
                "direction_hit_rate_5d",
            )
            if column in category_summary.columns
        ]
        lines.append(_markdown_table(category_summary[columns]))
    lines.extend(["", "## Category And Direction Summary", ""])
    if direction_summary.empty:
        lines.append("No direction summary was produced.")
    else:
        columns = [
            column
            for column in (
                "event_category",
                "event_direction",
                "event_count",
                "avg_open_gap_pct",
                "avg_return_1d_pct",
                "direction_hit_rate_1d",
                "avg_return_3d_pct",
                "direction_hit_rate_3d",
                "avg_return_5d_pct",
                "direction_hit_rate_5d",
            )
            if column in direction_summary.columns
        ]
        lines.append(_markdown_table(direction_summary[columns].head(30)))
    lines.append("")
    (output_dir / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a larger A-share news impact research dataset.")
    parser.add_argument("--symbols", default="", help="Comma/space separated symbols. Overrides universe sampling.")
    parser.add_argument("--symbols-file", default="", help="Text file containing symbols. Overrides universe sampling.")
    parser.add_argument("--max-symbols", type=int, default=80, help="Maximum symbols sampled from the A-share universe.")
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES), help="Comma-separated symbol prefixes for sampling.")
    parser.add_argument("--exclude-st", action="store_true", default=True, help="Exclude ST/*ST names when universe names are available.")
    parser.add_argument("--include-st", dest="exclude_st", action="store_false", help="Do not exclude ST/*ST names.")
    parser.add_argument("--start-date", default="20240101", help="News/disclosure and price start date in YYYYMMDD.")
    parser.add_argument("--end-date", default=None, help="End date in YYYYMMDD. Defaults to today.")
    parser.add_argument("--news-limit", type=int, default=300, help="Maximum normalized events retained per symbol.")
    parser.add_argument("--horizons", default="1,3,5", help="Comma-separated trading-day horizons.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel symbol workers.")
    parser.add_argument("--pause-seconds", type=float, default=0.15, help="Per-symbol pause before live fetch.")
    parser.add_argument("--no-disclosures", action="store_true", help="Skip CNInfo disclosure fetches.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached per-symbol payloads.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for CSV/JSON/Markdown artifacts.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "symbol_cache"
    horizons = tuple(int(piece.strip()) for piece in str(args.horizons).split(",") if piece.strip())
    symbols = resolve_symbols(args)
    if symbols.empty:
        raise RuntimeError("No symbols resolved for news impact research.")

    results: list[dict[str, object]] = []
    max_workers = max(1, min(int(args.workers), 8))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                analyze_one_symbol,
                row["symbol"],
                row.get("name", row["symbol"]),
                start_date=args.start_date,
                end_date=args.end_date,
                news_limit=int(args.news_limit),
                horizons=horizons,
                include_disclosures=not args.no_disclosures,
                cache_dir=cache_dir,
                refresh=bool(args.refresh),
                pause_seconds=float(args.pause_seconds),
            ): row["symbol"]
            for _, row in symbols.iterrows()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            results.append(payload)
            print(
                f"[{index}/{len(futures)}] {payload['symbol']} events={payload['event_count']} "
                f"impacts={payload['impact_sample_count']} cached={payload['cached']} "
                f"error={payload['error'] or ''}"
            )

    events = _concat_frames(results, "events")
    impacts = _concat_frames(results, "event_impacts")
    category_summary = summarize_category_impact(impacts, horizons=horizons, min_events=2) if not impacts.empty else pd.DataFrame()
    direction_summary = summarize_category_direction(impacts, horizons=horizons) if not impacts.empty else pd.DataFrame()
    meta = pd.DataFrame(
        [
            {
                "symbol": item["symbol"],
                "name": item["name"],
                "event_count": item["event_count"],
                "impact_sample_count": item["impact_sample_count"],
                "cached": item["cached"],
                "error": item["error"],
            }
            for item in results
        ]
    ).sort_values(["impact_sample_count", "event_count"], ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "news_events.csv", index=False, encoding="utf-8-sig")
    impacts.to_csv(output_dir / "news_event_impacts.csv", index=False, encoding="utf-8-sig")
    category_summary.to_csv(output_dir / "category_summary.csv", index=False, encoding="utf-8-sig")
    direction_summary.to_csv(output_dir / "category_direction_summary.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(output_dir / "symbol_meta.csv", index=False, encoding="utf-8-sig")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "symbols": int(len(symbols)),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "news_limit": args.news_limit,
                "horizons": list(horizons),
                "include_disclosures": not args.no_disclosures,
                "workers": max_workers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(output_dir, args=args, meta=meta, category_summary=category_summary, direction_summary=direction_summary)

    print("\nCompleted news impact research run")
    print(f"symbols={len(meta)} events={int(meta['event_count'].sum())} impacts={int(meta['impact_sample_count'].sum())}")
    print(f"output_dir={output_dir}")
    if not category_summary.empty:
        columns = [column for column in ("event_category", "event_count", "avg_return_1d_pct", "avg_return_3d_pct", "avg_return_5d_pct") if column in category_summary.columns]
        print(category_summary[columns].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
