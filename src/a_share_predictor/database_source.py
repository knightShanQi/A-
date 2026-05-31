from __future__ import annotations

import datetime as dt
import os
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.local"
DEFAULT_DAILY_TABLE = "a_share_daily_prices"
DEFAULT_DAILY_SERIES_TABLE = "a_share_daily_price_series"
DEFAULT_CALENDAR_TABLE = "a_share_trade_calendar"
ENABLED_SOURCE_VALUES = {"database", "postgres", "postgresql", "supabase"}
STORAGE_MODES = {"row", "series"}
SERIES_VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]


def load_env_file(path: Path | None = None) -> dict[str, str]:
    env_path = path or DEFAULT_ENV_FILE
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def is_enabled() -> bool:
    load_env_file()
    source = (
        os.getenv("OPENCLAW_MARKET_DATA_SOURCE")
        or os.getenv("A_SHARE_MARKET_DATA_SOURCE")
        or os.getenv("MARKET_DATA_SOURCE")
        or ""
    ).strip().lower()
    return source in ENABLED_SOURCE_VALUES


def _table_name() -> str:
    default_table = DEFAULT_DAILY_SERIES_TABLE if _storage_mode() == "series" else DEFAULT_DAILY_TABLE
    table = os.getenv("OPENCLAW_DAILY_PRICE_TABLE", default_table).strip() or default_table
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table):
        raise ValueError(f"unsafe table name: {table}")
    return ".".join(f'"{part}"' for part in table.split("."))


def _calendar_table_name() -> str:
    table = os.getenv("OPENCLAW_TRADE_CALENDAR_TABLE", DEFAULT_CALENDAR_TABLE).strip() or DEFAULT_CALENDAR_TABLE
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table):
        raise ValueError(f"unsafe table name: {table}")
    return ".".join(f'"{part}"' for part in table.split("."))


def _storage_mode() -> str:
    load_env_file()
    mode = (os.getenv("OPENCLAW_DAILY_PRICE_STORAGE") or "row").strip().lower()
    if mode not in STORAGE_MODES:
        return "row"
    return mode


def _database_url() -> str:
    load_env_file()
    explicit = (os.getenv("DATABASE_URL") or os.getenv("SUPABASE_POSTGRES_URL") or "").strip()
    if explicit:
        return explicit
    project_ref = os.getenv("SUPABASE_PROJECT_REF", "").strip()
    password = os.getenv("SUPABASE_DB_PASSWORD", "").strip()
    pooler_host = os.getenv("SUPABASE_POOLER_HOST", "").strip()
    if project_ref and password and pooler_host:
        return (
            f"postgresql://postgres.{project_ref}:{quote(password, safe='')}"
            f"@{pooler_host}:6543/postgres?sslmode=require"
        )
    raise RuntimeError("DATABASE_URL or SUPABASE_POSTGRES_URL is required for database market data")


def _connect(attempts: int = 3):
    import psycopg

    last_exc: Exception | None = None
    for attempt in range(max(int(attempts), 1)):
        try:
            connection = psycopg.connect(_database_url(), connect_timeout=30)
            _ensure_read_write_session(connection)
            return connection
        except psycopg.OperationalError as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(2.0 * float(attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("PostgreSQL connection failed without an exception")


def _ensure_read_write_session(connection) -> None:
    was_autocommit = connection.autocommit
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("set default_transaction_read_only = off")
    finally:
        connection.autocommit = was_autocommit


def _query_frame(connection, query: str, params: Iterable[object] = ()) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        columns = [description.name for description in cursor.description or []]
    return pd.DataFrame(rows, columns=columns)


def _parse_date_param(value: str | None, default: str | None = None) -> str | None:
    raw = str(value or default or "").strip()
    if not raw:
        return None
    parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _format_tushare_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def _normalize_symbol(symbol: str) -> str:
    digits = re.sub(r"\D", "", str(symbol))
    if len(digits) != 6:
        raise ValueError(f"cannot normalize stock symbol: {symbol}")
    return digits


def _date_value(value: str | None, default: str | None = None) -> dt.date | None:
    parsed = _parse_date_param(value, default)
    if not parsed:
        return None
    return dt.date.fromisoformat(parsed)


def _array_at(values: object, index: int) -> object:
    if isinstance(values, (list, tuple)) and 0 <= index < len(values):
        return values[index]
    return None


def _series_rows_to_daily_frame(rows: pd.DataFrame, *, start: dt.date | None, end: dt.date | None) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for row in rows.to_dict("records"):
        dates = row.get("dates")
        if not isinstance(dates, (list, tuple)):
            continue
        for index, raw_date in enumerate(dates):
            parsed = pd.to_datetime(raw_date, errors="coerce")
            if pd.isna(parsed):
                continue
            trade_date = pd.Timestamp(parsed).date()
            if start is not None and trade_date < start:
                continue
            if end is not None and trade_date > end:
                continue
            record = {
                "symbol": row.get("symbol"),
                "name": row.get("name") or "",
                "trade_date": trade_date,
                "pre_close": None,
                "change": None,
                "pct_chg": None,
            }
            for column in SERIES_VALUE_COLUMNS:
                record[column] = _array_at(row.get(column), index)
            records.append(record)
    return pd.DataFrame(records)


def _market_suffix(symbol: str) -> str:
    if str(symbol).startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    if str(symbol).startswith(("000", "001", "002", "003", "300", "301")):
        return "SZ"
    if str(symbol).startswith(("430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "874", "875", "876", "877", "878", "879", "920")):
        return "BJ"
    return "SH"


def fetch_daily_history(symbol: str, start_date: str = "20220101", end_date: str | None = None) -> pd.DataFrame:
    clean_symbol = _normalize_symbol(symbol)
    start = _parse_date_param(start_date)
    end = _parse_date_param(end_date, dt.date.today().strftime("%Y%m%d"))
    table = _table_name()
    if _storage_mode() == "series":
        start_value = _date_value(start_date)
        end_value = _date_value(end_date, dt.date.today().strftime("%Y%m%d"))
        start_year = start_value.year if start_value is not None else 1990
        end_year = end_value.year if end_value is not None else dt.date.today().year
        with _connect() as connection:
            frame = _query_frame(
                connection,
                f"""
                select symbol, name, year, dates, open, high, low, close,
                       volume, amount, turnover_rate
                from {table}
                where symbol = %s
                  and year between %s and %s
                order by year
                """,
                (clean_symbol, int(start_year), int(end_year)),
            )
        return normalize_daily_history_frame(
            _series_rows_to_daily_frame(frame, start=start_value, end=end_value),
            source="supabase",
        )
    with _connect() as connection:
        query = f"""
            select symbol, name, trade_date, open, high, low, close, pre_close,
                   change, pct_chg, volume, amount, turnover_rate
            from {table}
            where symbol = %s
              and (%s::date is null or trade_date >= %s::date)
              and (%s::date is null or trade_date <= %s::date)
            order by trade_date
        """
        frame = _query_frame(connection, query, (clean_symbol, start, start, end, end))
    return normalize_daily_history_frame(frame, source="supabase")


def normalize_daily_history_frame(frame: pd.DataFrame, *, source: str = "database") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
    normalized["date"] = pd.to_datetime(normalized["trade_date"], errors="coerce")
    normalized = normalized.dropna(subset=["symbol", "date", "close"]).sort_values("date").copy()
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount", "turnover_rate"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["change_amount"] = normalized.get("change")
    if "pre_close" in normalized.columns:
        base = normalized["pre_close"].where(normalized["pre_close"].ne(0))
        normalized["change_amount"] = normalized["change_amount"].where(normalized["change_amount"].notna(), normalized["close"] - normalized["pre_close"])
        normalized["change_pct"] = normalized.get("pct_chg").where(normalized.get("pct_chg").notna(), (normalized["close"] / base - 1.0) * 100)
    else:
        normalized["change_pct"] = normalized.get("pct_chg")
    normalized["change_amount"] = normalized["change_amount"].where(normalized["change_amount"].notna(), normalized["close"].diff())
    normalized["change_pct"] = normalized["change_pct"].where(normalized["change_pct"].notna(), normalized["close"].pct_change() * 100)
    normalized["turnover"] = normalized.get("turnover_rate")
    keep_columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "change_pct",
        "change_amount",
        "turnover",
    ]
    result = normalized[[column for column in keep_columns if column in normalized.columns]].copy()
    result = result.set_index("date", drop=False)
    result.attrs["data_source"] = source
    return result


def fetch_recent_trade_dates(end_date: str | None = None, limit: int = 30) -> list[str]:
    end_value = _date_value(end_date, dt.date.today().strftime("%Y%m%d"))
    end = end_value.isoformat() if end_value is not None else None
    table = _table_name()
    if _storage_mode() == "series":
        try:
            calendar_table = _calendar_table_name()
            with _connect() as connection:
                frame = _query_frame(
                    connection,
                    f"""
                    select trade_date
                    from {calendar_table}
                    where (%s::date is null or trade_date <= %s::date)
                    order by trade_date desc
                    limit %s
                    """,
                    (end, end, int(limit)),
                )
        except Exception:
            max_year = end_value.year if end_value is not None else dt.date.today().year
            min_year = max(1990, max_year - 2)
            with _connect() as connection:
                frame = _query_frame(
                    connection,
                    f"""
                    select distinct trade_date
                    from (
                        select unnest(dates) as trade_date
                        from {table}
                        where year between %s and %s
                    ) as expanded
                    where (%s::date is null or trade_date <= %s::date)
                    order by trade_date desc
                    limit %s
                    """,
                    (min_year, max_year, end, end, int(limit)),
                )
        if frame.empty:
            return []
        return sorted(_format_tushare_date(value) for value in frame["trade_date"] if _format_tushare_date(value))
    with _connect() as connection:
        frame = _query_frame(
            connection,
            f"""
            select distinct trade_date
            from {table}
            where (%s::date is null or trade_date <= %s::date)
            order by trade_date desc
            limit %s
            """,
            (end, end, int(limit)),
        )
    if frame.empty:
        return []
    return sorted(_format_tushare_date(value) for value in frame["trade_date"] if _format_tushare_date(value))


def fetch_daily_snapshot(trade_date: str) -> pd.DataFrame:
    target_value = _date_value(trade_date)
    if target_value is None:
        return pd.DataFrame()
    target = target_value.isoformat()
    table = _table_name()
    if _storage_mode() == "series":
        with _connect() as connection:
            frame = _query_frame(
                connection,
                f"""
                select symbol, name, %s::date as trade_date,
                       open[idx] as open,
                       high[idx] as high,
                       low[idx] as low,
                       close[idx] as close,
                       null::numeric as pre_close,
                       null::numeric as change,
                       null::numeric as pct_chg,
                       volume[idx] as volume,
                       amount[idx] as amount,
                       turnover_rate[idx] as turnover_rate
                from (
                    select symbol, name, open, high, low, close, volume, amount, turnover_rate,
                           array_position(dates, %s::date) as idx
                    from {table}
                    where year = %s
                      and %s::date = any(dates)
                ) as matched
                order by symbol
                """,
                (target, target, target_value.year, target),
            )
    else:
        with _connect() as connection:
            frame = _query_frame(
                connection,
                f"""
                select symbol, name, trade_date, open, high, low, close, pre_close,
                       change, pct_chg, volume, amount, turnover_rate
                from {table}
                where trade_date = %s::date
                order by symbol
                """,
                (target,),
            )
    if frame.empty:
        return frame
    frame = frame.copy()
    for column in ["pre_close", "change", "pct_chg"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["name"] = frame["name"].where(frame["name"].str.len().gt(0), frame["symbol"])
    frame["industry"] = ""
    frame["market"] = frame["symbol"].map(_market_suffix)
    frame["ts_code"] = frame["symbol"] + "." + frame["market"]
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount", "turnover_rate"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["vol"] = pd.to_numeric(frame.get("volume"), errors="coerce")
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    pre_close = pd.to_numeric(frame.get("pre_close"), errors="coerce")
    base = pre_close.where(pre_close.ne(0))
    frame["change"] = frame["change"].where(frame["change"].notna(), close - pre_close)
    frame["pct_chg"] = frame["pct_chg"].where(frame["pct_chg"].notna(), (close / base - 1.0) * 100)
    return frame.reset_index(drop=True)


def fetch_daily_window(end_date: str | None = None, window: int = 20) -> pd.DataFrame:
    dates = fetch_recent_trade_dates(end_date=end_date, limit=max(int(window), 2))
    frames = [fetch_daily_snapshot(trade_date) for trade_date in dates]
    valid = [frame for frame in frames if not frame.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True).sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def fetch_universe(limit: int | None = None) -> pd.DataFrame:
    table = _table_name()
    limit_clause = "" if limit is None else "limit %s"
    params: Iterable[object] = () if limit is None else (int(limit),)
    if _storage_mode() == "series":
        with _connect() as connection:
            frame = _query_frame(
                connection,
                f"""
                select symbol, max(nullif(name, '')) as name, max(year) as latest_year
                from {table}
                group by symbol
                order by symbol
                {limit_clause}
                """,
                params,
            )
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "name"])
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["name"] = frame["name"].fillna("").astype(str)
        frame["name"] = frame["name"].where(frame["name"].str.len().gt(0), frame["symbol"])
        return frame[["symbol", "name"]].reset_index(drop=True)
    with _connect() as connection:
        frame = _query_frame(
            connection,
            f"""
            select symbol, max(nullif(name, '')) as name, max(trade_date) as latest_trade_date
            from {table}
            group by symbol
            order by symbol
            {limit_clause}
            """,
            params,
        )
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "name"])
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["name"] = frame["name"].where(frame["name"].str.len().gt(0), frame["symbol"])
    return frame[["symbol", "name"]].reset_index(drop=True)
