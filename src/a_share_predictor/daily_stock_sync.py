from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import getpass
import json
import math
import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote, unquote, urlparse

import pandas as pd
import requests


DEFAULT_BAIDU_SHARE_URL = "https://pan.baidu.com/s/17i-xHfAtZizB15vBTMdBFw?login_type=weixin&_at_=1779716418067#list/path=%2F"
DEFAULT_TABLE_NAME = "a_share_daily_prices"
DEFAULT_SERIES_TABLE_NAME = "a_share_daily_price_series"
DEFAULT_CALENDAR_TABLE_NAME = "a_share_trade_calendar"
DEFAULT_DOWNLOAD_DIR = Path(".cache") / "baidu_daily_stock"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.local"
SUPPORTED_DATA_SUFFIXES = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}
SUPPORTED_ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar"}
DOWNLOAD_ONLY_ARCHIVE_SUFFIXES: set[str] = set()
STORAGE_MODES = {"row", "series"}

COLUMN_ALIASES = {
    "symbol": {
        "symbol",
        "code",
        "stock_code",
        "ticker",
        "ts_code",
        "证券代码",
        "股票代码",
        "代码",
        "品种代码",
    },
    "name": {"name", "stock_name", "security_name", "证券简称", "股票简称", "名称", "股票名称"},
    "trade_date": {"trade_date", "date", "交易日期", "日期", "datetime"},
    "time": {"time", "时间"},
    "open": {"open", "开盘", "开盘价"},
    "high": {"high", "最高", "最高价"},
    "low": {"low", "最低", "最低价"},
    "close": {"close", "收盘", "收盘价", "最新价"},
    "pre_close": {"pre_close", "prev_close", "昨收", "昨收价", "前收盘"},
    "change": {"change", "涨跌额", "涨跌"},
    "pct_chg": {"pct_chg", "change_pct", "涨跌幅", "涨跌幅%", "涨幅"},
    "volume": {"volume", "vol", "成交量", "成交量(手)", "成交量(股)"},
    "amount": {"amount", "成交额", "成交额(元)", "成交额(千元)", "成交金额"},
    "turnover_rate": {"turnover_rate", "turnover", "换手率", "换手率%"},
}

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "volume",
    "amount",
    "turnover_rate",
]

SERIES_VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]


@dataclass(frozen=True)
class SyncResult:
    table: str
    discovered_files: int
    imported_files: int
    rows_read: int
    rows_written: int


def _normalize_column_name(column: object) -> str:
    text = str(column or "").strip()
    text = re.sub(r"\s+", "_", text)
    return text.strip("_")


def _canonical_column_map(columns: Iterable[object]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            lookup[alias.lower()] = canonical
    result: dict[str, str] = {}
    for column in columns:
        normalized = _normalize_column_name(column)
        canonical = lookup.get(normalized.lower(), normalized)
        result[str(column)] = canonical
    return result


def _normalize_symbol(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "." in text and re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    match = re.search(r"(\d{1,6})", text)
    if not match:
        return None
    return match.group(1).zfill(6)


def _symbol_from_filename(path: str | Path) -> str | None:
    stem = Path(path).stem.strip()
    match = re.fullmatch(r"(?:sh|sz|bj)?(\d{6})", stem, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _coerce_trade_date(value: object) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{8}", text):
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"": None, "--": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_daily_stock_frame(frame: pd.DataFrame, *, source_file: str = "") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    normalized = frame.copy()
    normalized = normalized.rename(columns=_canonical_column_map(normalized.columns))
    if "trade_date" not in normalized.columns and "datetime" in normalized.columns:
        normalized["trade_date"] = normalized["datetime"]

    inferred_symbol = _symbol_from_filename(source_file) if source_file else None
    if "symbol" not in normalized.columns and inferred_symbol:
        normalized["symbol"] = inferred_symbol
        if inferred_symbol:
            normalized["symbol"] = inferred_symbol

    required = {"symbol", "trade_date", "close"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"{source_file or 'input'} missing required columns: {', '.join(sorted(missing))}")

    normalized["symbol"] = normalized["symbol"].map(_normalize_symbol)
    if inferred_symbol:
        normalized["symbol"] = inferred_symbol
    normalized["trade_date"] = normalized["trade_date"].map(_coerce_trade_date)
    for column in NUMERIC_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _coerce_numeric_series(normalized[column])
        else:
            normalized[column] = pd.NA

    if "name" not in normalized.columns:
        normalized["name"] = ""
    normalized["name"] = normalized["name"].fillna("").astype(str).str.strip()

    normalized = normalized.dropna(subset=["symbol", "trade_date", "close"]).copy()
    if normalized.empty:
        return pd.DataFrame()

    normalized = _aggregate_duplicate_daily_rows(normalized)
    normalized["source_file"] = Path(source_file).name if source_file else ""
    normalized["source_updated_at"] = pd.Timestamp.utcnow()
    normalized = normalized[
        [
            "symbol",
            "name",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "volume",
            "amount",
            "turnover_rate",
            "source_file",
            "source_updated_at",
        ]
    ]
    return normalized.sort_values(["symbol", "trade_date"]).drop_duplicates(["symbol", "trade_date"], keep="last")


def _aggregate_duplicate_daily_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not frame.duplicated(["symbol", "trade_date"]).any():
        return frame
    sort_columns = [column for column in ["trade_date", "time", "datetime"] if column in frame.columns]
    view = frame.sort_values(sort_columns).copy() if sort_columns else frame.copy()
    aggregations = {
        "name": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "pre_close": "last",
        "change": "last",
        "pct_chg": "last",
        "volume": "sum",
        "amount": "sum",
        "turnover_rate": "last",
    }
    return (
        view.groupby(["symbol", "trade_date"], as_index=False)
        .agg({column: func for column, func in aggregations.items() if column in view.columns})
        .reset_index(drop=True)
    )


def _detect_csv_dialect(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()[:8192]
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            sample = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        sample = raw.decode("utf-8", errors="ignore")
        encoding = "utf-8"

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        sep = dialect.delimiter
    except csv.Error:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return encoding, sep


def read_daily_stock_file(path: Path) -> pd.DataFrame:
    if any(part.lower() in {"1min", "5min", "15min", "30min", "60min"} for part in path.parts):
        minute_frame = _read_minute_daily_file(path)
        if minute_frame is not None:
            return minute_frame
    if _symbol_from_filename(path):
        yearly_frame = _read_stock_yearly_file(path)
        if yearly_frame is not None:
            return yearly_frame
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        encoding, sep = _detect_csv_dialect(path)
        return pd.read_csv(path, encoding=encoding, sep=sep)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported daily stock file: {path}")


def _read_minute_daily_file(path: Path) -> pd.DataFrame | None:
    symbol = _symbol_from_filename(path)
    if not symbol:
        return None
    encoding, sep = _detect_csv_dialect(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sep)
        first_row: dict[str, str] | None = None
        last_row: dict[str, str] | None = None
        high = float("nan")
        low = float("nan")
        volume = 0.0
        amount = 0.0
        for row in reader:
            if first_row is None:
                first_row = row
            last_row = row
            row_high = pd.to_numeric(str(row.get("最高", "")).replace(",", ""), errors="coerce")
            row_low = pd.to_numeric(str(row.get("最低", "")).replace(",", ""), errors="coerce")
            row_volume = pd.to_numeric(str(row.get("成交量", "")).replace(",", ""), errors="coerce")
            row_amount = pd.to_numeric(str(row.get("成交额", "")).replace(",", ""), errors="coerce")
            if pd.notna(row_high):
                high = float(row_high) if pd.isna(high) else max(high, float(row_high))
            if pd.notna(row_low):
                low = float(row_low) if pd.isna(low) else min(low, float(row_low))
            if pd.notna(row_volume):
                volume += float(row_volume)
            if pd.notna(row_amount):
                amount += float(row_amount)
    if first_row is None or last_row is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": first_row.get("日期"),
                "open": first_row.get("开盘"),
                "high": high,
                "low": low,
                "close": last_row.get("收盘"),
                "volume": volume,
                "amount": amount,
            }
        ]
    )


def _read_stock_yearly_file(path: Path) -> pd.DataFrame | None:
    symbol = _symbol_from_filename(path)
    if not symbol or path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
        return None
    encoding, sep = _detect_csv_dialect(path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sep)
        if not reader.fieldnames or "日期" not in reader.fieldnames or "收盘价" not in reader.fieldnames:
            return None
        for row in reader:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": row.get("日期"),
                    "close": row.get("收盘价"),
                    "turnover_rate": row.get("换手率"),
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def iter_data_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_DATA_SUFFIXES else []
    if not root.exists():
        return []
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_SUFFIXES)
    one_minute_files = [path for path in files if any(part.lower() == "1min" for part in path.parts)]
    return one_minute_files or files


def _extract_rar_archive(archive_path: Path, target_dir: Path) -> None:
    tar_exe = shutil.which("tar")
    if not tar_exe:
        raise RuntimeError("RAR extraction requires a tar/libarchive executable on PATH.")
    completed = subprocess.run(
        [tar_exe, "-xf", str(archive_path), "-C", str(target_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"RAR extraction failed for {archive_path}: {detail}")


def extract_supported_archives(root: Path, *, strict: bool = False) -> list[Path]:
    extracted_roots: list[Path] = []
    candidates = [root] if root.is_file() else list(root.rglob("*")) if root.exists() else []
    for archive_path in candidates:
        if not archive_path.is_file() or archive_path.suffix.lower() not in SUPPORTED_ARCHIVE_SUFFIXES:
            continue
        target_dir = archive_path.with_suffix("")
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            if archive_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(target_dir)
            elif archive_path.suffix.lower() == ".7z":
                import py7zr

                with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                    archive.extractall(target_dir)
            elif archive_path.suffix.lower() == ".rar":
                _extract_rar_archive(archive_path, target_dir)
        except Exception:
            if strict:
                raise
            continue
        extracted_roots.append(target_dir)
    return extracted_roots


class BaiduPanShareClient:
    """Best-effort public-share downloader for Baidu Pan web shares.

    Baidu frequently changes the web API and usually requires a valid browser
    Cookie for public share downloads. The sync command therefore accepts a
    local input directory as the deterministic path, while this client handles
    the common logged-in share page flow when credentials are available.
    """

    def __init__(self, share_url: str, *, cookie: str = "", password: str = "", timeout: float = 20.0):
        self.share_url = share_url
        self.password = password
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Referer": share_url,
            }
        )
        if cookie:
            self.session.headers["Cookie"] = cookie
        self._state: dict | None = None

    def _logid(self) -> str:
        return base64.b64encode(str(int(time.time() * 1000)).encode("utf-8")).decode("ascii")

    def _shorturl(self) -> str:
        path = urlparse(self.share_url).path.rstrip("/")
        match = re.search(r"/s/([^/?#]+)", path)
        if not match:
            raise ValueError(f"cannot parse Baidu share shorturl from {self.share_url}")
        return match.group(1)

    def _initial_page_state(self) -> dict:
        if self._state is not None:
            return self._state
        response = self.session.get(self.share_url, timeout=self.timeout)
        response.raise_for_status()
        html = response.text

        state: dict = {}
        match = re.search(r"locals\.mset\((\{.*?\})\);", html, re.S)
        if match:
            try:
                state.update(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass
        match = re.search(r"yunData\.setData\((\{.*?\})\);", html, re.S)
        if match:
            try:
                state.update(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass
        self._state = state
        return state

    def _list_dir(self, dir_path: str = "/") -> list[dict]:
        state = self._initial_page_state()
        params = {
            "app_id": "250528",
            "web": "1",
            "channel": "chunlei",
            "clienttype": "0",
            "page": "1",
            "num": "100",
            "order": "time",
            "desc": "1",
            "root": "1" if dir_path == "/" else "0",
            "dir": dir_path,
            "logid": self._logid(),
        }
        if state.get("share_uk") and state.get("shareid"):
            params["uk"] = str(state["share_uk"])
            params["shareid"] = str(state["shareid"])
        else:
            params["shorturl"] = self._shorturl()
        response = self.session.get("https://pan.baidu.com/share/list", params=params, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        if int(body.get("errno", 0)) != 0:
            raise RuntimeError(f"Baidu share list failed: {body.get('errmsg') or body.get('errno')}")
        return list(body.get("list") or [])

    def list_files(self) -> list[dict]:
        files: list[dict] = []
        state = self._initial_page_state()
        pending = list(state.get("file_list") or []) or [{"path": "/", "isdir": 1}]
        while pending:
            current_item = pending.pop(0)
            if int(current_item.get("isdir", 0)) == 0:
                files.append(current_item)
                continue
            current = str(current_item.get("path") or "/")
            try:
                entries = self._list_dir(current)
            except (RuntimeError, requests.RequestException):
                if current == "/":
                    raise
                continue
            for item in entries:
                is_dir = int(item.get("isdir", 0)) == 1
                server_filename = str(item.get("server_filename") or item.get("path") or "")
                item_path = str(item.get("path") or f"{current.rstrip('/')}/{server_filename}")
                if is_dir:
                    item["path"] = item_path
                    pending.append(item)
                else:
                    files.append(item)
        return files

    def _download_file_items(self, items: list[dict], destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        state = self._initial_page_state()
        for item in items:
            filename = str(item.get("server_filename") or Path(str(item.get("path", "download"))).name)
            suffix = Path(filename).suffix.lower()
            if (
                suffix not in SUPPORTED_DATA_SUFFIXES
                and suffix not in SUPPORTED_ARCHIVE_SUFFIXES
                and suffix not in DOWNLOAD_ONLY_ARCHIVE_SUFFIXES
            ):
                continue
            target = destination / filename
            tmp_target = target.with_suffix(target.suffix + ".tmp")
            expected_size = int(item.get("size") or 0)
            if expected_size > 0 and target.exists() and target.stat().st_size == expected_size:
                downloaded.append(target)
                continue
            last_exc: Exception | None = None
            for attempt in range(8):
                try:
                    resume_at = tmp_target.stat().st_size if tmp_target.exists() else 0
                    headers = {"Range": f"bytes={resume_at}-"} if resume_at > 0 else None
                    dlink = str(item.get("dlink") or "")
                    if not dlink:
                        dlink = self._resolve_dlink(item, state)
                    if dlink:
                        response = self.session.get(dlink, stream=True, timeout=self.timeout, headers=headers)
                    else:
                        response = self._transfer_and_download(item, state, headers=headers)
                    response.raise_for_status()
                    mode = "ab" if resume_at > 0 and response.status_code == 206 else "wb"
                    with tmp_target.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=256 * 1024):
                            if chunk:
                                handle.write(chunk)
                    if expected_size > 0 and tmp_target.stat().st_size != expected_size:
                        raise requests.RequestException(
                            f"incomplete Baidu download for {filename}: {tmp_target.stat().st_size}/{expected_size}"
                        )
                    tmp_target.replace(target)
                    last_exc = None
                    break
                except requests.RequestException as exc:
                    last_exc = exc
                    if attempt < 7:
                        time.sleep(1.5 * (attempt + 1))
            if last_exc is not None:
                raise last_exc
            downloaded.append(target)
        return downloaded

    def download_files(self, destination: Path, *, backfill_years: set[int] | None = None) -> list[Path]:
        files = _select_preferred_share_files(self.list_files(), backfill_years=backfill_years)
        return self._download_file_items(files, destination)

    def download_intraday_files(
        self,
        destination: Path,
        *,
        years: set[int] | None = None,
        intervals: set[int] | None = None,
        include_legacy_rar: bool = False,
    ) -> list[Path]:
        files = select_intraday_share_files(
            self.list_files(),
            years=years,
            intervals=intervals,
            include_legacy_rar=include_legacy_rar,
        )
        return self._download_file_items(files, destination)

    def _ensure_own_dir(self, path: str, state: dict) -> None:
        response = self.session.post(
            "https://pan.baidu.com/api/create",
            params={
                "a": "commit",
                "bdstoken": str(state.get("bdstoken", "")),
                "channel": "chunlei",
                "web": "1",
                "app_id": "250528",
            },
            data={"path": path, "isdir": "1", "block_list": "[]"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        if int(body.get("errno", 0)) not in {0, -8}:
            raise RuntimeError(f"Baidu mkdir failed: {body}")

    def _transfer_and_download(self, item: dict, state: dict, headers: dict | None = None):
        fs_id = item.get("fs_id")
        filename = str(item.get("server_filename") or Path(str(item.get("path", "download"))).name)
        if not fs_id:
            raise RuntimeError(f"Baidu file has no fs_id: {filename}")
        own_dir = f"/openclaw_sync/{fs_id}"
        self._ensure_own_dir(own_dir, state)
        response = self.session.post(
            "https://pan.baidu.com/share/transfer",
            params={
                "app_id": "250528",
                "channel": "chunlei",
                "clienttype": "0",
                "web": "1",
                "bdstoken": str(state.get("bdstoken", "")),
                "shareid": str(state.get("shareid", "")),
                "from": str(state.get("share_uk", "")),
            },
            data={"fsidlist": json.dumps([int(fs_id)]), "path": own_dir},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        if int(body.get("errno", 0)) != 0 and not body.get("duplicated"):
            raise RuntimeError(f"Baidu transfer failed for {filename}: {body}")
        own_path = f"{own_dir}/{filename}"
        pcs_response = self.session.get(
            "https://d.pcs.baidu.com/rest/2.0/pcs/file",
            params={"method": "download", "app_id": "250528", "path": own_path},
            stream=True,
            timeout=self.timeout,
            headers=headers,
        )
        return pcs_response

    def _resolve_dlink(self, item: dict, state: dict) -> str:
        fs_id = item.get("fs_id")
        if not fs_id:
            return ""
        uk = state.get("share_uk") or state.get("uk")
        shareid = state.get("shareid") or state.get("share_id")
        sign = state.get("sign")
        timestamp = state.get("timestamp")
        bdstoken = state.get("bdstoken", "")
        if not (uk and shareid and sign and timestamp):
            return ""
        response = self.session.post(
            "https://pan.baidu.com/api/sharedownload",
            params={
                "app_id": "250528",
                "channel": "chunlei",
                "clienttype": "0",
                "web": "1",
                "sign": sign,
                "timestamp": timestamp,
                "bdstoken": bdstoken,
                "logid": self._logid(),
            },
            data={
                "encrypt": "0",
                "product": "share",
                "uk": str(uk),
                "primaryid": str(shareid),
                "fid_list": json.dumps([int(fs_id)]),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        entries = body.get("list") or []
        return str(entries[0].get("dlink") if entries else "")


def discover_or_download_inputs(
    *,
    input_dir: Path | None = None,
    share_url: str = DEFAULT_BAIDU_SHARE_URL,
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
    baidu_cookie: str = "",
    baidu_password: str = "",
    skip_download: bool = False,
    backfill_years: set[int] | None = None,
) -> list[Path]:
    roots: list[Path] = []
    if input_dir is not None:
        roots.append(input_dir)

    if not skip_download and share_url:
        client = BaiduPanShareClient(share_url, cookie=baidu_cookie, password=baidu_password)
        downloaded = client.download_files(download_dir, backfill_years=backfill_years)
        roots.extend(downloaded or [download_dir])

    for root in list(roots):
        roots.extend(extract_supported_archives(root))

    files: list[Path] = []
    for root in roots:
        files.extend(iter_data_files(root))
    return sorted(dict.fromkeys(path.resolve() for path in files))


def _select_preferred_share_files(files: list[dict], *, backfill_years: set[int] | None = None) -> list[dict]:
    today = dt.date.today()
    today_name = today.strftime("%Y%m%d")
    today_int = int(today_name)
    dated_names: list[int] = []
    for item in files:
        text = f"{item.get('server_filename') or ''}/{item.get('path') or ''}"
        for match in re.findall(r"20\d{6}", text):
            value = int(match)
            if value <= today_int:
                dated_names.append(value)
    latest_available_name = str(max(dated_names)) if dated_names else today_name
    year_name = f"{today.year}.7z"
    scored: list[tuple[int, dict]] = []
    requested_year_names = {f"{int(year)}.7z" for year in (backfill_years or set())}
    for item in files:
        filename = str(item.get("server_filename") or "")
        path = str(item.get("path") or "")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_DATA_SUFFIXES and suffix not in SUPPORTED_ARCHIVE_SUFFIXES:
            continue
        score = 100
        if requested_year_names:
            if "每日指标" in path and filename in requested_year_names:
                score = int(filename[:4])
            else:
                continue
        elif "每日指标" in path and filename == year_name:
            score = 0
        elif latest_available_name in filename and ("每日数据" in path or "每日指标" in path):
            score = 1
        elif "stock_basic" in filename.lower() or "股票列表" in filename:
            score = 20
        else:
            continue
        scored.append((score, item))
    return [item for _, item in sorted(scored, key=lambda pair: (pair[0], str(pair[1].get("path") or "")))]


def _path_covers_year(path: str, year: int) -> bool:
    if re.search(rf"(?:^|/){int(year)}(?:/|$)", path):
        return True
    for start, end in re.findall(r"(?<!\d)(20\d{2})\s*[-－~～]\s*(20\d{2})(?!\d)", path):
        if int(start) <= int(year) <= int(end):
            return True
    return False


def select_intraday_share_files(
    files: list[dict],
    *,
    years: set[int] | None = None,
    intervals: set[int] | None = None,
    include_legacy_rar: bool = False,
) -> list[dict]:
    requested_intervals = intervals or {1, 5, 15, 30, 60}
    requested_names = {f"{int(interval)}min.7z" for interval in requested_intervals}
    if include_legacy_rar:
        requested_names.update({f"{int(interval)}min.rar" for interval in requested_intervals})

    selected: list[tuple[int, str, dict]] = []
    for item in files:
        filename = str(item.get("server_filename") or "").strip()
        lower_name = filename.lower()
        if lower_name not in requested_names:
            continue
        path = str(item.get("path") or "").replace("\\", "/")
        if years:
            if not any(_path_covers_year(path, int(year)) for year in years):
                continue
        elif lower_name.endswith(".rar") and not include_legacy_rar:
            continue
        if lower_name.endswith(".rar") and not include_legacy_rar:
            continue
        interval = int(re.match(r"(\d+)min", lower_name).group(1))
        year_score = 0
        if years:
            matched_years = [int(year) for year in years if _path_covers_year(path, int(year))]
            year_score = min(matched_years) if matched_years else 9999
        selected.append((year_score, f"{interval:02d}", item))
    return [item for _, _, item in sorted(selected, key=lambda row: (row[0], row[1], str(row[2].get("path") or "")))]


def load_daily_stock_files(paths: Iterable[Path]) -> tuple[pd.DataFrame, int, int]:
    frames: list[pd.DataFrame] = []
    imported_files = 0
    rows_read = 0
    for path in paths:
        raw = read_daily_stock_file(path)
        rows_read += int(len(raw))
        try:
            normalized = normalize_daily_stock_frame(raw, source_file=str(path))
        except ValueError as exc:
            if "missing required columns" in str(exc):
                continue
            raise
        if normalized.empty:
            continue
        imported_files += 1
        frames.append(normalized)
    if not frames:
        return pd.DataFrame(), imported_files, rows_read
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["symbol", "trade_date", "source_updated_at"])
    merged = merged.drop_duplicates(["symbol", "trade_date"], keep="last").reset_index(drop=True)
    return merged, imported_files, rows_read


def _safe_table_name(table_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table_name or ""):
        raise ValueError(f"unsafe table name: {table_name}")
    return table_name


def _quote_table_name(table_name: str) -> str:
    return ".".join(f'"{part}"' for part in _safe_table_name(table_name).split("."))


def _safe_index_name(table_name: str, suffix: str) -> str:
    base = _safe_table_name(table_name).replace(".", "_")
    return re.sub(r"[^A-Za-z0-9_]", "_", f"{base}_{suffix}")[:63]


def _resolve_storage_mode(storage_mode: str | None = None) -> str:
    value = (storage_mode or os.getenv("OPENCLAW_DAILY_PRICE_STORAGE") or "row").strip().lower()
    if value not in STORAGE_MODES:
        raise ValueError(f"unsupported daily price storage mode: {value}")
    return value


def _resolve_table_name(table_name: str | None, storage_mode: str) -> str:
    explicit = (table_name or "").strip()
    if explicit:
        return explicit
    env_table = os.getenv("OPENCLAW_DAILY_PRICE_TABLE", "").strip()
    if env_table:
        return env_table
    return DEFAULT_SERIES_TABLE_NAME if storage_mode == "series" else DEFAULT_TABLE_NAME


def _resolve_calendar_table_name() -> str:
    return os.getenv("OPENCLAW_TRADE_CALENDAR_TABLE", DEFAULT_CALENDAR_TABLE_NAME).strip() or DEFAULT_CALENDAR_TABLE_NAME


def _database_url_from_env(database_url: str | None = None) -> str:
    value = (database_url or os.getenv("DATABASE_URL") or os.getenv("SUPABASE_POSTGRES_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL or SUPABASE_POSTGRES_URL is required")
    if "[YOUR-PASSWORD]" in value:
        raise RuntimeError("replace [YOUR-PASSWORD] in the PostgreSQL URL before running the sync")
    return value


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


def _build_supabase_database_url(project_ref: str, password: str) -> str:
    project_ref = str(project_ref or "").strip()
    if not re.fullmatch(r"[a-z0-9]{20}", project_ref):
        raise ValueError(f"invalid Supabase project ref: {project_ref}")
    return f"postgresql://postgres:{quote(password, safe='')}@db.{project_ref}.supabase.co:5432/postgres?sslmode=require"


def _to_db_value(value: object) -> object:
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return value
    return value


def ensure_daily_stock_table(connection, table_name: str = DEFAULT_TABLE_NAME) -> None:
    quoted_table = _quote_table_name(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            create table if not exists {quoted_table} (
                symbol text not null,
                name text,
                trade_date date not null,
                open numeric,
                high numeric,
                low numeric,
                close numeric not null,
                pre_close numeric,
                change numeric,
                pct_chg numeric,
                volume numeric,
                amount numeric,
                turnover_rate numeric,
                source_file text,
                source_updated_at timestamptz not null default now(),
                ingested_at timestamptz not null default now(),
                primary key (symbol, trade_date)
            )
            """
        )
    connection.commit()


def ensure_daily_stock_series_table(connection, table_name: str = DEFAULT_SERIES_TABLE_NAME) -> None:
    quoted_table = _quote_table_name(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            create table if not exists {quoted_table} (
                symbol text not null,
                name text,
                year smallint not null,
                dates date[] not null,
                open real[],
                high real[],
                low real[],
                close real[] not null,
                volume real[],
                amount real[],
                turnover_rate real[],
                source_files text[],
                source_updated_at timestamptz not null default now(),
                ingested_at timestamptz not null default now(),
                primary key (symbol, year)
            )
            """
        )
        cursor.execute(f'create index if not exists "{_safe_index_name(table_name, "year_idx")}" on {quoted_table} (year)')
    connection.commit()


def ensure_trade_calendar_table(connection, table_name: str = DEFAULT_CALENDAR_TABLE_NAME) -> None:
    quoted_table = _quote_table_name(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            create table if not exists {quoted_table} (
                trade_date date primary key,
                ingested_at timestamptz not null default now()
            )
            """
        )
    connection.commit()


def upsert_daily_stock_frame(connection, frame: pd.DataFrame, table_name: str = DEFAULT_TABLE_NAME) -> int:
    if frame.empty:
        return 0

    quoted_table = _quote_table_name(table_name)
    columns = [
        "symbol",
        "name",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "volume",
        "amount",
        "turnover_rate",
        "source_file",
        "source_updated_at",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)
    update_assignments: list[str] = []
    for column in columns:
        if column in {"symbol", "trade_date"}:
            continue
        if column in {"source_file", "source_updated_at"}:
            update_assignments.append(f"{column}=excluded.{column}")
        else:
            update_assignments.append(f"{column}=coalesce(excluded.{column}, target.{column})")
    update_sql = ", ".join(update_assignments)
    records = [
        tuple(_to_db_value(row[column]) for column in columns)
        for row in frame[columns].to_dict("records")
    ]
    if len(records) > 5000:
        return _copy_upsert_daily_stock_records(connection, records, columns, quoted_table, update_sql)
    with connection.cursor() as cursor:
        cursor.executemany(
            f"""
            insert into {quoted_table} as target ({column_sql})
            values ({placeholders})
            on conflict (symbol, trade_date) do update set
                {update_sql},
                ingested_at = now()
            """,
            records,
        )
    connection.commit()
    return len(records)


def upsert_trade_calendar(connection, frame: pd.DataFrame, table_name: str = DEFAULT_CALENDAR_TABLE_NAME) -> int:
    if frame.empty or "trade_date" not in frame.columns:
        return 0
    dates = (
        pd.to_datetime(frame["trade_date"], errors="coerce")
        .dropna()
        .dt.date
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not dates:
        return 0
    quoted_table = _quote_table_name(table_name)
    with connection.cursor() as cursor:
        cursor.executemany(
            f"""
            insert into {quoted_table} (trade_date)
            values (%s)
            on conflict (trade_date) do update set ingested_at = now()
            """,
            [(value,) for value in dates],
        )
    connection.commit()
    return len(dates)


def upsert_daily_stock_series_frame(connection, frame: pd.DataFrame, table_name: str = DEFAULT_SERIES_TABLE_NAME) -> int:
    if frame.empty:
        return 0

    quoted_table = _quote_table_name(table_name)
    records = daily_stock_frame_to_series_records(frame)
    if not records:
        return 0

    columns = [
        "symbol",
        "name",
        "year",
        "dates",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "source_files",
        "source_updated_at",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)
    update_sql = ", ".join(
        [
            "name=coalesce(nullif(excluded.name, ''), target.name)",
            "dates=excluded.dates",
            "open=excluded.open",
            "high=excluded.high",
            "low=excluded.low",
            "close=excluded.close",
            "volume=excluded.volume",
            "amount=excluded.amount",
            "turnover_rate=excluded.turnover_rate",
            "source_files=excluded.source_files",
            "source_updated_at=excluded.source_updated_at",
            "ingested_at=now()",
        ]
    )
    with connection.cursor() as cursor:
        cursor.executemany(
            f"""
            insert into {quoted_table} as target ({column_sql})
            values ({placeholders})
            on conflict (symbol, year) do update set {update_sql}
            """,
            [tuple(record[column] for column in columns) for record in records],
        )
    connection.commit()
    return len(frame)


def daily_stock_frame_to_series_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []

    normalized = frame.copy()
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], errors="coerce")
    normalized = normalized.dropna(subset=["symbol", "trade_date", "close"]).copy()
    if normalized.empty:
        return []
    normalized["year"] = normalized["trade_date"].dt.year.astype(int)
    normalized = normalized.sort_values(["symbol", "year", "trade_date", "source_updated_at"])
    normalized = normalized.drop_duplicates(["symbol", "trade_date"], keep="last")

    records: list[dict[str, object]] = []
    for (symbol, year), group in normalized.groupby(["symbol", "year"], sort=True):
        group = group.sort_values("trade_date")
        dates = [
            value.date() if isinstance(value, pd.Timestamp) else value
            for value in group["trade_date"].tolist()
        ]
        names = [str(value).strip() for value in group.get("name", pd.Series(dtype=object)).tolist() if str(value).strip()]
        source_files = sorted({str(value).strip() for value in group.get("source_file", pd.Series(dtype=object)).tolist() if str(value).strip()})
        source_updated_at = pd.to_datetime(group.get("source_updated_at", pd.Series([pd.Timestamp.utcnow()])).max(), errors="coerce")
        record: dict[str, object] = {
            "symbol": str(symbol).zfill(6),
            "name": names[-1] if names else "",
            "year": int(year),
            "dates": dates,
            "source_files": source_files or None,
            "source_updated_at": source_updated_at.to_pydatetime() if pd.notna(source_updated_at) else dt.datetime.now(dt.timezone.utc),
        }
        for column in SERIES_VALUE_COLUMNS:
            record[column] = _series_numeric_values(group[column]) if column in group.columns else None
        records.append(record)
    return records


def _series_numeric_values(series: pd.Series) -> list[float | None] | None:
    values: list[float | None] = []
    has_value = False
    for value in series.tolist():
        db_value = _to_db_value(value)
        if db_value is None:
            values.append(None)
            continue
        numeric = float(db_value)
        values.append(numeric)
        has_value = True
    return values if has_value else None


def _copy_upsert_daily_stock_records(connection, records: list[tuple], columns: list[str], quoted_table: str, update_sql: str) -> int:
    temp_table = "tmp_openclaw_daily_prices"
    column_sql = ", ".join(columns)
    chunk_size = 200_000
    written = 0
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        with connection.cursor() as cursor:
            cursor.execute("set statement_timeout = 0")
            cursor.execute(f"create temp table {temp_table} (like {quoted_table} including defaults) on commit drop")
            with cursor.copy(f"copy {temp_table} ({column_sql}) from stdin") as copy:
                for record in chunk:
                    copy.write_row(record)
            cursor.execute(
                f"""
                insert into {quoted_table} as target ({column_sql})
                select {column_sql}
                from {temp_table}
                on conflict (symbol, trade_date) do update set
                    {update_sql},
                    ingested_at = now()
                """
            )
        connection.commit()
        written += len(chunk)
    return len(records)


def _connect_postgres_with_retry(database_url: str, *, attempts: int = 3):
    import psycopg

    last_exc: Exception | None = None
    for attempt in range(max(int(attempts), 1)):
        try:
            connection = psycopg.connect(database_url, connect_timeout=30)
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


def sync_daily_stock_data(
    *,
    database_url: str | None = None,
    table_name: str | None = None,
    input_dir: Path | None = None,
    share_url: str = DEFAULT_BAIDU_SHARE_URL,
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
    baidu_cookie: str = "",
    baidu_password: str = "",
    skip_download: bool = False,
    backfill_years: set[int] | None = None,
    storage_mode: str | None = None,
) -> SyncResult:
    resolved_storage_mode = _resolve_storage_mode(storage_mode)
    resolved_table_name = _resolve_table_name(table_name, resolved_storage_mode)
    files = discover_or_download_inputs(
        input_dir=input_dir,
        share_url=share_url,
        download_dir=download_dir,
        baidu_cookie=baidu_cookie,
        baidu_password=baidu_password,
        skip_download=skip_download,
        backfill_years=backfill_years,
    )
    frame, imported_files, rows_read = load_daily_stock_files(files)
    if frame.empty:
        return SyncResult(table=resolved_table_name, discovered_files=len(files), imported_files=imported_files, rows_read=rows_read, rows_written=0)

    db_url = _database_url_from_env(database_url)
    with _connect_postgres_with_retry(db_url) as connection:
        calendar_table_name = _resolve_calendar_table_name()
        ensure_trade_calendar_table(connection, calendar_table_name)
        upsert_trade_calendar(connection, frame, calendar_table_name)
        if resolved_storage_mode == "series":
            ensure_daily_stock_series_table(connection, resolved_table_name)
            rows_written = upsert_daily_stock_series_frame(connection, frame, resolved_table_name)
        else:
            ensure_daily_stock_table(connection, resolved_table_name)
            rows_written = upsert_daily_stock_frame(connection, frame, resolved_table_name)

    return SyncResult(
        table=resolved_table_name,
        discovered_files=len(files),
        imported_files=imported_files,
        rows_read=rows_read,
        rows_written=rows_written,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync daily A-share stock files from Baidu Pan/local files into PostgreSQL.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="Local env file loaded before syncing.")
    parser.add_argument("--database-url", default=None, help="PostgreSQL URL. Defaults to DATABASE_URL or SUPABASE_POSTGRES_URL.")
    parser.add_argument("--supabase-project-ref", default="vrddzvdrbzffynmacbua")
    parser.add_argument("--prompt-db-password", action="store_true", help="Prompt for the Supabase database password without echoing it.")
    parser.add_argument("--prompt-baidu-cookie", action="store_true", help="Prompt for the Baidu Pan cookie without storing it.")
    parser.add_argument("--table-name", default=None)
    parser.add_argument("--storage-mode", choices=sorted(STORAGE_MODES), default=None, help="Use row for one row per day, or series for compact symbol-year arrays.")
    parser.add_argument("--share-url", default=DEFAULT_BAIDU_SHARE_URL)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--input-dir", type=Path, default=None, help="Use already downloaded files from this path.")
    parser.add_argument("--baidu-cookie", default=None)
    parser.add_argument("--baidu-password", default=None)
    parser.add_argument("--backfill-years", default="", help="Comma-separated years or ranges, e.g. 2020-2026 or 2018,2019,2020.")
    parser.add_argument("--skip-download", action="store_true", help="Only import --input-dir or existing --download-dir files.")
    return parser


def parse_years(value: str) -> set[int]:
    years: set[int] = set()
    for part in str(value or "").split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start_year = int(start_text)
            end_year = int(end_text)
            if end_year < start_year:
                start_year, end_year = end_year, start_year
            years.update(range(start_year, end_year + 1))
        else:
            years.add(int(token))
    current_year = dt.date.today().year
    return {year for year in years if 1990 <= year <= current_year}


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    load_env_file(args.env_file)
    input_dir = args.input_dir
    if input_dir is None and args.skip_download:
        input_dir = args.download_dir
    database_url = args.database_url
    if args.prompt_db_password:
        database_url = _build_supabase_database_url(
            args.supabase_project_ref,
            getpass.getpass("Supabase database password: "),
        )
    baidu_cookie = args.baidu_cookie if args.baidu_cookie is not None else os.getenv("BAIDU_PAN_COOKIE", "")
    if args.prompt_baidu_cookie:
        baidu_cookie = getpass.getpass("Baidu Pan cookie: ")
    baidu_password = args.baidu_password if args.baidu_password is not None else os.getenv("BAIDU_PAN_PASSWORD", "")
    backfill_years = parse_years(args.backfill_years)
    result = sync_daily_stock_data(
        database_url=database_url,
        table_name=args.table_name,
        input_dir=input_dir,
        share_url=args.share_url,
        download_dir=args.download_dir,
        baidu_cookie=baidu_cookie,
        baidu_password=baidu_password,
        skip_download=bool(args.skip_download),
        backfill_years=backfill_years or None,
        storage_mode=args.storage_mode,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
