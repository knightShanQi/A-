import pandas as pd
import sys
import datetime as dt

from a_share_predictor import daily_stock_sync as sync


def test_normalize_daily_stock_frame_accepts_chinese_columns():
    raw = pd.DataFrame(
        [
            {
                "股票代码": "600001.SH",
                "股票简称": "测试股",
                "交易日期": "20260522",
                "开盘价": "10.10",
                "最高价": "10.80",
                "最低价": "9.90",
                "收盘价": "10.50",
                "涨跌幅": "3.20%",
                "成交量(手)": "120,000",
                "成交额(元)": "250,000,000",
                "换手率": "6.8%",
            }
        ]
    )

    result = sync.normalize_daily_stock_frame(raw, source_file="sample.csv")

    assert result.loc[0, "symbol"] == "600001"
    assert result.loc[0, "name"] == "测试股"
    assert result.loc[0, "trade_date"].strftime("%Y-%m-%d") == "2026-05-22"
    assert result.loc[0, "close"] == 10.5
    assert result.loc[0, "pct_chg"] == 3.2
    assert result.loc[0, "turnover_rate"] == 6.8


def test_normalize_daily_stock_frame_zero_pads_numeric_symbol_from_yearly_file():
    raw = pd.DataFrame([{"股票代码": 1, "日期": "2026-01-05", "收盘价": 11.5, "换手率": 0.45}])

    result = sync.normalize_daily_stock_frame(raw, source_file="sz000001.csv")

    assert result.loc[0, "symbol"] == "000001"
    assert result.loc[0, "close"] == 11.5
    assert result.loc[0, "turnover_rate"] == 0.45


def test_normalize_daily_stock_frame_prefers_stock_filename_over_bad_inner_symbol():
    raw = pd.DataFrame([{"股票代码": 1, "日期": "2026-01-06", "收盘价": 54.9}])

    result = sync.normalize_daily_stock_frame(raw, source_file=r"E:\openclaw\.cache\baidu_daily_stock\2026\2026\sz002311.csv")

    assert result.loc[0, "symbol"] == "002311"


def test_parse_years_accepts_ranges_and_lists():
    assert sync.parse_years("2024-2026,2020") == {2020, 2024, 2025, 2026}


def test_select_preferred_share_files_uses_latest_available_trade_file(monkeypatch):
    class FixedDate(dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 5, 27)

    monkeypatch.setattr(sync.dt, "date", FixedDate)

    selected = sync._select_preferred_share_files(
        [
            {"server_filename": "2026.7z", "path": "/apps/全部股票/股票/每日指标/2026/2026.7z"},
            {"server_filename": "20260526.csv", "path": "/apps/全部股票/股票/每日指标/2026/每日数据/20260526.csv"},
            {"server_filename": "20260526.7z", "path": "/share/有竞价/竞价不复权/2026/每日数据/20260526.7z"},
            {"server_filename": "stock_basic.csv", "path": "/apps/全部股票/股票/每日指标/stock_basic.csv"},
        ]
    )

    assert [item["server_filename"] for item in selected] == ["2026.7z", "20260526.csv", "20260526.7z", "stock_basic.csv"]


def test_select_intraday_share_files_filters_year_and_interval():
    selected = sync.select_intraday_share_files(
        [
            {"server_filename": "1min.7z", "path": "/share/auction/2026/1min.7z", "size": 100},
            {"server_filename": "5min.7z", "path": "/share/auction/2026/5min.7z", "size": 50},
            {"server_filename": "1min.7z", "path": "/share/auction/2025/1min.7z", "size": 90},
            {"server_filename": "1min.rar", "path": "/share/auction/2000-2025/1min.rar", "size": 900},
        ],
        years={2026},
        intervals={1},
    )

    assert [item["path"] for item in selected] == ["/share/auction/2026/1min.7z"]


def test_select_intraday_share_files_can_include_legacy_rar():
    selected = sync.select_intraday_share_files(
        [
            {"server_filename": "1min.rar", "path": "/share/auction/2000-2025/1min.rar", "size": 900},
            {"server_filename": "5min.rar", "path": "/share/auction/2000-2025/5min.rar", "size": 500},
        ],
        intervals={5},
        include_legacy_rar=True,
    )

    assert [item["server_filename"] for item in selected] == ["5min.rar"]


def test_select_intraday_share_files_matches_year_ranges():
    selected = sync.select_intraday_share_files(
        [
            {"server_filename": "15min.rar", "path": "/share/auction/2000-2025（分k）/15min.rar", "size": 500},
            {"server_filename": "15min.7z", "path": "/share/auction/2026/15min.7z", "size": 50},
        ],
        years={2025},
        intervals={15},
        include_legacy_rar=True,
    )

    assert [item["server_filename"] for item in selected] == ["15min.rar"]


def test_iter_data_files_prefers_coarser_minute_files(tmp_path):
    root = tmp_path / "20260528"
    one_minute = root / "1min"
    sixty_minute = root / "60min"
    one_minute.mkdir(parents=True)
    sixty_minute.mkdir(parents=True)
    one_file = one_minute / "sz000001.csv"
    sixty_file = sixty_minute / "sz000001.csv"
    daily_file = root / "daily.csv"
    one_file.write_text("symbol,trade_date,close\n000001,2026-05-28,10.1\n", encoding="utf-8")
    sixty_file.write_text("symbol,trade_date,close\n000001,2026-05-28,10.1\n", encoding="utf-8")
    daily_file.write_text("symbol,trade_date,close\n000001,2026-05-28,10.1\n", encoding="utf-8")

    assert sync.iter_data_files(root) == [sixty_file]


def test_baidu_download_falls_back_to_transfer_when_direct_link_forbidden(tmp_path):
    class FakeResponse:
        def __init__(self, status_code, chunks=()):
            self.status_code = status_code
            self._chunks = list(chunks)
            self.closed = False

        def raise_for_status(self):
            if self.status_code >= 400:
                raise sync.requests.HTTPError(f"{self.status_code} error", response=self)

        def iter_content(self, chunk_size):
            return iter(self._chunks)

        def close(self):
            self.closed = True

    class FakeSession:
        def __init__(self):
            self.direct_response = FakeResponse(403)

        def get(self, *args, **kwargs):
            return self.direct_response

    client = sync.BaiduPanShareClient.__new__(sync.BaiduPanShareClient)
    client.session = FakeSession()
    client.timeout = 1.0
    client._initial_page_state = lambda: {"ok": True}
    client._resolve_dlink = lambda item, state: "https://example.invalid/file"
    transfers = []

    def fake_transfer(item, state, headers=None):
        transfers.append((item, state, headers))
        return FakeResponse(200, [b"archive"])

    client._transfer_and_download = fake_transfer

    downloaded = client._download_file_items(
        [{"server_filename": "1min.7z", "size": len(b"archive"), "fs_id": 123}],
        tmp_path,
    )

    assert downloaded == [tmp_path / "1min.7z"]
    assert (tmp_path / "1min.7z").read_bytes() == b"archive"
    assert client.session.direct_response.closed is True
    assert len(transfers) == 1


def test_extract_supported_archives_uses_tar_for_rar(tmp_path, monkeypatch):
    archive = tmp_path / "15min.rar"
    archive.write_bytes(b"placeholder")
    calls = []

    monkeypatch.setattr(sync.shutil, "which", lambda name: "tar.exe" if name == "tar" else None)

    def fake_run(args, check, capture_output, text):
        calls.append(args)
        return sync.subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sync.subprocess, "run", fake_run)

    extracted = sync.extract_supported_archives(archive, strict=True)

    assert extracted == [tmp_path / "15min"]
    assert calls == [["tar.exe", "-xf", str(archive), "-C", str(tmp_path / "15min")]]


def test_load_daily_stock_files_deduplicates_symbol_and_date(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("symbol,trade_date,close\n600001,2026-05-22,10.1\n", encoding="utf-8")
    second.write_text("symbol,trade_date,close\n600001,2026-05-22,10.2\n", encoding="utf-8")

    frame, imported_files, rows_read = sync.load_daily_stock_files([first, second])

    assert imported_files == 2
    assert rows_read == 2
    assert len(frame) == 1
    assert frame.loc[0, "close"] == 10.2


def test_load_daily_stock_files_skips_non_price_files(tmp_path):
    basic = tmp_path / "stock_basic.csv"
    price = tmp_path / "price.csv"
    basic.write_text("symbol,name\n600001,Test\n", encoding="utf-8")
    price.write_text("symbol,trade_date,close\n600001,2026-05-22,10.1\n", encoding="utf-8")

    frame, imported_files, rows_read = sync.load_daily_stock_files([basic, price])

    assert imported_files == 1
    assert rows_read == 2
    assert len(frame) == 1
    assert frame.loc[0, "symbol"] == "600001"


def test_upsert_daily_stock_frame_uses_expected_conflict_key():
    frame = sync.normalize_daily_stock_frame(
        pd.DataFrame([{"symbol": "600001", "trade_date": "20260522", "close": 10.5}])
    )
    connection = FakeConnection()

    written = sync.upsert_daily_stock_frame(connection, frame, table_name="public.daily_prices")

    assert written == 1
    assert connection.commits == 1
    assert "on conflict (symbol, trade_date)" in connection.cursor_obj.sql.lower()
    assert "coalesce(excluded.open, target.open)" in connection.cursor_obj.sql.lower()
    assert len(connection.cursor_obj.records) == 1
    assert connection.cursor_obj.records[0][0] == "600001"


def test_daily_stock_frame_to_series_records_groups_symbol_year():
    frame = sync.normalize_daily_stock_frame(
        pd.DataFrame(
            [
                {"symbol": "600001", "trade_date": "20220104", "close": 10.5, "turnover_rate": 1.2},
                {"symbol": "600001", "trade_date": "20220105", "close": 10.8, "turnover_rate": 1.5},
            ]
        ),
        source_file="sh600001.csv",
    )

    records = sync.daily_stock_frame_to_series_records(frame)

    assert len(records) == 1
    assert records[0]["symbol"] == "600001"
    assert records[0]["year"] == 2022
    assert records[0]["dates"] == [dt.date(2022, 1, 4), dt.date(2022, 1, 5)]
    assert records[0]["close"] == [10.5, 10.8]
    assert records[0]["turnover_rate"] == [1.2, 1.5]


def test_build_supabase_database_url_escapes_password():
    url = sync._build_supabase_database_url("vrddzvdrbzffynmacbua", "pa:ss/word")

    assert url == "postgresql://postgres:pa%3Ass%2Fword@db.vrddzvdrbzffynmacbua.supabase.co:5432/postgres?sslmode=require"


def test_load_env_file_sets_missing_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("DATABASE_URL=postgresql://example\nBAIDU_PAN_COOKIE=\"cookie-value\"\n", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("BAIDU_PAN_COOKIE", raising=False)

    loaded = sync.load_env_file(env_file)

    assert loaded["DATABASE_URL"] == "postgresql://example"
    assert loaded["BAIDU_PAN_COOKIE"] == "cookie-value"


def test_load_env_file_strips_utf8_bom(monkeypatch, tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("\ufeffDATABASE_URL=postgresql://example\n", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    loaded = sync.load_env_file(env_file)

    assert loaded["DATABASE_URL"] == "postgresql://example"


def test_database_url_from_env_rejects_placeholder_password():
    try:
        sync._database_url_from_env("postgresql://postgres:[YOUR-PASSWORD]@db.example.supabase.co:5432/postgres")
    except RuntimeError as exc:
        assert "replace [YOUR-PASSWORD]" in str(exc)
    else:
        raise AssertionError("expected placeholder database URL to be rejected")


def test_sync_daily_stock_data_skips_database_when_no_rows(monkeypatch, tmp_path):
    class UnexpectedConnectModule:
        @staticmethod
        def connect(*args, **kwargs):
            raise AssertionError("database connection should not be attempted when no rows are loaded")

    monkeypatch.setattr(sync, "discover_or_download_inputs", lambda **kwargs: [tmp_path / "empty.csv"])
    monkeypatch.setattr(sync, "load_daily_stock_files", lambda paths: (pd.DataFrame(), 0, 0))
    monkeypatch.setitem(sys.modules, "psycopg", UnexpectedConnectModule)

    result = sync.sync_daily_stock_data(skip_download=True, input_dir=tmp_path)

    assert result == sync.SyncResult(
        table=sync.DEFAULT_TABLE_NAME,
        discovered_files=1,
        imported_files=0,
        rows_read=0,
        rows_written=0,
    )


def test_ensure_read_write_session_turns_off_default_read_only():
    connection = FakeSessionConnection()

    sync._ensure_read_write_session(connection)

    assert connection.autocommit is False
    assert connection.autocommit_values == [True, False]
    assert connection.cursor_obj.sql == ["set default_transaction_read_only = off"]


def test_main_uses_download_dir_when_skip_download_has_no_input_dir(monkeypatch, tmp_path, capsys):
    calls = {}
    download_dir = tmp_path / "downloads"
    env_file = tmp_path / ".env.local"
    monkeypatch.setenv("BAIDU_PAN_COOKIE", "cookie-from-env")
    monkeypatch.setenv("BAIDU_PAN_PASSWORD", "password-from-env")
    monkeypatch.setattr(sync, "load_env_file", lambda path: {"loaded": str(path)})

    def fake_sync_daily_stock_data(**kwargs):
        calls.update(kwargs)
        return sync.SyncResult(
            table=kwargs["table_name"],
            discovered_files=0,
            imported_files=0,
            rows_read=0,
            rows_written=0,
        )

    monkeypatch.setattr(sync, "sync_daily_stock_data", fake_sync_daily_stock_data)

    sync.main(
        [
            "--env-file",
            str(env_file),
            "--table-name",
            "public.daily_prices",
            "--download-dir",
            str(download_dir),
            "--skip-download",
        ]
    )

    assert calls["input_dir"] == download_dir
    assert calls["skip_download"] is True
    assert calls["table_name"] == "public.daily_prices"
    assert calls["baidu_cookie"] == "cookie-from-env"
    assert calls["baidu_password"] == "password-from-env"
    assert '"table": "public.daily_prices"' in capsys.readouterr().out


class FakeSessionConnection:
    def __init__(self):
        self._autocommit = False
        self.autocommit_values = []
        self.cursor_obj = FakeSessionCursor()

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        self.autocommit_values.append(value)
        self._autocommit = value

    def cursor(self):
        return self.cursor_obj


class FakeSessionCursor:
    def __init__(self):
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql):
        self.sql.append(sql)


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1


class FakeCursor:
    def __init__(self):
        self.sql = ""
        self.records = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def executemany(self, sql, records):
        self.sql = sql
        self.records = list(records)
