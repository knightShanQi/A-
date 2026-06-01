from __future__ import annotations

import mimetypes
from types import SimpleNamespace

import pandas as pd

from a_share_predictor import api_service
from a_share_predictor import api as api_module
from a_share_predictor import daily_review
from a_share_predictor.api import app


def test_api_routes_are_registered():
    paths = {route.path for route in app.routes}

    assert "/api/health" in paths
    assert "/api/board/quick" in paths
    assert "/api/board/enhanced" in paths
    assert "/api/symbol/{symbol}" in paths
    assert "/api/tasks/rebuild-ranking" in paths
    assert "/api/tasks/market-backtest" in paths
    assert "/api/backtests/market/latest" in paths
    assert "/api/tasks/{task_id}" in paths


def test_frontend_asset_mime_types_are_browser_safe():
    assert mimetypes.guess_type("index.js")[0] == "application/javascript"
    assert mimetypes.guess_type("chunk.mjs")[0] == "application/javascript"
    assert mimetypes.guess_type("index.css")[0] == "text/css"


def test_symbol_detail_can_be_forced_to_safe_cached_payload(monkeypatch):
    calls = []
    cached = {
        "freshness": {"data_date": "2026-05-14", "latest_market_data_date": "2026-05-14"},
        "board": {
            "rows": [
                {
                    "symbol": "600000",
                    "name": "浦发银行",
                    "rank_score": 72.0,
                    "p_hit": 61.0,
                    "expected_return_pct": 7.5,
                    "action_label": "观察",
                }
            ],
            "freshness": {"data_date": "2026-05-14", "latest_market_data_date": "2026-05-14"},
        },
    }

    def fail_live_detail(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("safe detail must not call live detail by default")

    monkeypatch.delenv("A_SHARE_ENABLE_LIVE_DETAIL", raising=False)
    monkeypatch.setenv("A_SHARE_DISABLE_LIVE_DETAIL", "1")
    monkeypatch.setattr(api_service, "load_quick_board_payload", lambda params: cached)
    monkeypatch.setattr(api_service, "_build_symbol_detail", fail_live_detail)
    monkeypatch.setattr(
        api_service,
        "load_a_share_universe",
        lambda: pd.DataFrame([{"symbol": "600000", "name": "浦发银行"}]),
    )

    payload = api_service.load_symbol_detail_payload("600000", api_service.normalize_api_params())

    assert calls == []
    assert payload["symbol"] == "600000"
    assert payload["hero"]["name"] == "浦发银行"
    assert payload["display_context"]["detail_mode"] == "safe_cached"
    assert payload["display_context"]["p_hit"] == 61.0
    assert payload["display_context"]["rank_score"] == 72.0
    assert payload["charts"]["daily"] is None


def test_symbol_detail_defaults_to_live_single_symbol_model(monkeypatch):
    calls = []

    class FakeChart:
        def to_json(self):
            return "{}"

    def fake_build_symbol_detail(symbol, horizon_days, positive_return):
        calls.append((symbol, horizon_days, positive_return))
        return {
            "profile": {"股票简称": "浦发银行", "行业": "银行"},
            "rule_context": SimpleNamespace(board_label="主板", price_limit_label="10%"),
            "quant_signal": SimpleNamespace(total_score=67.0, primary_signal="trend", summary="量化偏强"),
            "model": SimpleNamespace(
                signal_label="观察",
                backtest_summary="单票模型",
                strategy_score=71.0,
                agreement_score=66.0,
                quality_label="single_symbol",
                risk_label="中性",
            ),
            "backtest": SimpleNamespace(
                status_label="ok",
                summary="回测完成",
                target_precision=0.9,
                selection_summary="",
                achieved_precision=0.62,
                trade_count=12,
                latest_signal_active=True,
            ),
            "tomorrow_plan": SimpleNamespace(
                setup_label="等待回踩",
                bias="观察",
                buy_point="低吸",
                sell_point="破位",
                confidence=63.0,
            ),
            "stage": SimpleNamespace(label="趋势中继", summary="结构正常", intraday_expectation="看承接"),
            "daily": pd.DataFrame({"date": ["2026-05-14"], "close": [10.0]}),
            "minute": pd.DataFrame(),
            "snapshot": {},
            "intraday": {"score": 0.58, "label": "偏强"},
            "sector_signal": {"sector_score": 56.0},
            "fund_signal": {"fund_score": 55.0},
            "news_signal": {"sentiment_score": 54.0},
            "fund_flow_df": pd.DataFrame(),
            "news_df": pd.DataFrame(),
            "analysis_date": "2026-05-14",
            "latest_market_data_date": "2026-05-14",
        }

    monkeypatch.delenv("A_SHARE_ENABLE_LIVE_DETAIL", raising=False)
    monkeypatch.delenv("A_SHARE_DISABLE_LIVE_DETAIL", raising=False)
    monkeypatch.setattr(api_service, "_build_symbol_detail", fake_build_symbol_detail)
    monkeypatch.setattr(
        api_service,
        "_detail_display_context",
        lambda detail: {
            "probability_up": 63.0,
            "raw_probability_up": 59.0,
            "enhanced_probability_up": 61.0,
            "rank_score": 72.0,
            "enhanced_attention_score": 74.0,
            "expected_return_pct": 8.0,
            "market_state_label": "trend",
            "computed_at": "2026-05-14 18:00:00",
        },
    )
    monkeypatch.setattr(api_service, "make_daily_chart", lambda daily: FakeChart())
    monkeypatch.setattr(api_service, "make_minute_chart", lambda minute: FakeChart())

    payload = api_service.load_symbol_detail_payload("600000", api_service.normalize_api_params())

    assert calls and calls[0][0] == "600000"
    assert payload["hero"]["name"] == "浦发银行"
    assert payload["display_context"]["p_hit"] > 0.0
    assert payload["display_context"]["rank_score"] == 72.0
    assert payload["charts"]["daily"] == {}


def test_symbol_detail_default_live_failure_falls_back_to_safe_cached_payload(monkeypatch):
    calls = []
    cached = {
        "freshness": {"data_date": "2026-05-14", "latest_market_data_date": "2026-05-14"},
        "board": {"rows": [], "freshness": {"data_date": "2026-05-14", "latest_market_data_date": "2026-05-14"}},
    }

    def fail_live_detail(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("live fetch failed")

    monkeypatch.delenv("A_SHARE_ENABLE_LIVE_DETAIL", raising=False)
    monkeypatch.delenv("A_SHARE_DISABLE_LIVE_DETAIL", raising=False)
    monkeypatch.setattr(api_service, "_build_symbol_detail", fail_live_detail)
    monkeypatch.setattr(api_service, "load_quick_board_payload", lambda params: cached)
    monkeypatch.setattr(
        api_service,
        "load_a_share_universe",
        lambda: pd.DataFrame([{"symbol": "600519", "name": "贵州茅台"}]),
    )

    payload = api_service.load_symbol_detail_payload("600519", api_service.normalize_api_params())

    assert calls
    assert payload["display_context"]["detail_mode"] == "safe_cached"
    assert payload["display_context"]["detail_fallback_reason"] == "live fetch failed"
    assert payload["display_context"]["p_hit"] > 0.0


def test_symbol_detail_safe_payload_scores_uncached_search_result(monkeypatch):
    monkeypatch.delenv("A_SHARE_ENABLE_LIVE_DETAIL", raising=False)
    monkeypatch.setenv("A_SHARE_DISABLE_LIVE_DETAIL", "1")
    monkeypatch.setattr(
        api_service,
        "load_quick_board_payload",
        lambda params: {
            "freshness": {"data_date": "2026-05-14", "latest_market_data_date": "2026-05-14"},
            "board": {"rows": [], "freshness": {"data_date": "2026-05-14", "latest_market_data_date": "2026-05-14"}},
        },
    )
    monkeypatch.setattr(
        api_service,
        "load_a_share_universe",
        lambda: pd.DataFrame([{"symbol": "600519", "name": "贵州茅台"}]),
    )

    payload = api_service.load_symbol_detail_payload("600519", api_service.normalize_api_params())
    context = payload["display_context"]

    assert payload["symbol"] == "600519"
    assert payload["hero"]["name"] == "贵州茅台"
    assert context["detail_mode"] == "safe_cached"
    assert context["p_hit"] > 0.0
    assert context["rank_score"] > 0.0
    assert context["final_rank_score"] > 0.0
    assert context["calibration_method"] == "explicit"
    assert context["launch_signal_label"] in {"breakout", "ready", "watch", "wait"}


def test_quick_board_uses_cached_ranking_without_full_rebuild(monkeypatch):
    cached = pd.DataFrame(
        [
            {
                "symbol": "600000",
                "name": "浦发银行",
                "rank": 1,
                "attention_score": 70.0,
                "ranking_score": 70.0,
                "probability_up": 60.0,
            }
        ]
    )

    def fail_full_rebuild(*args, **kwargs):
        raise AssertionError("quick board must not run full focus-board rebuild")

    monkeypatch.setattr(api_service, "load_latest_close_quick_board", lambda **kwargs: (pd.DataFrame(), {}))
    monkeypatch.setattr(api_service, "_build_focus_board", fail_full_rebuild)
    monkeypatch.setattr(
        api_service,
        "_read_market_rankings_cache",
        lambda *args, **kwargs: (
            cached,
            {
                "data_mode": "history",
                "market_data_date": "2026-05-08",
                "latest_market_data_date": "2026-05-08",
                "cache_stale": False,
            },
        ),
    )
    monkeypatch.setattr(api_service, "_latest_market_close_date", lambda: "2026-05-08")
    monkeypatch.setattr(api_service, "load_latest_review_summary", lambda **kwargs: None)
    monkeypatch.setattr(api_service, "load_a_share_universe", lambda: pd.DataFrame([{"symbol": "600000"}]))
    monkeypatch.setattr(
        api_service,
        "_build_display_board",
        lambda board, board_size, ranking_by, data_mode, loading: board.head(board_size).copy(),
    )

    params = api_service.normalize_api_params(board_size=50)
    payload = api_service.load_quick_board_payload(params)

    assert payload["board"]["rows"][0]["symbol"] == "600000"
    assert payload["board"]["meta"]["quick_source"] == "cached_market_ranking"
    assert payload["board"]["rows"][0]["raw_probability_up"] == 60.0
    assert payload["board"]["rows"][0]["enhanced_probability_up"] == 60.0
    assert round(payload["board"]["rows"][0]["calibrated_probability_up"], 2) == 56.58
    assert round(payload["board"]["rows"][0]["p_hit"], 2) == 56.58
    assert payload["board"]["rows"][0]["calibration_method"] == "market_state_shrinkage"
    assert "expected_return_pct" in payload["board"]["rows"][0]
    assert "drawdown_risk_pct" in payload["board"]["rows"][0]
    assert payload["board"]["rows"][0]["rank_score"] == 70.0
    assert payload["board"]["rows"][0]["final_rank_score"] == 70.0
    assert "launch_signal_label" in payload["board"]["rows"][0]
    assert "launch_phase_label" in payload["board"]["rows"][0]
    assert "launch_reason_text" in payload["board"]["rows"][0]
    assert "intraday_sector_sync_score" in payload["board"]["rows"][0]
    assert "intraday_sector_state" in payload["board"]["rows"][0]
    assert "risk_level_label" in payload["board"]["rows"][0]
    assert "suggested_position_pct" in payload["board"]["rows"][0]
    assert payload["freshness"]["data_date"] == "2026-05-08"
    assert payload["freshness"]["is_latest_model_result"] is True
    assert payload["board"]["freshness"]["consistency_key"]
    assert "review_health" in payload
    assert payload["model_snapshot"]["model_schema_version"] == api_service.API_MODEL_CONTRACT_VERSION
    assert payload["model_snapshot"]["is_comparable"] is True


def test_probability_contract_prefers_explicit_ranking_fields():
    frame = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "probability_up": 55.0,
                "raw_probability_up": 48.0,
                "enhanced_probability_up": 53.0,
                "display_probability_up": 61.0,
                "ranking_score": 82.0,
                "enhanced_attention_score": 70.0,
            }
        ]
    )

    normalized = api_service.apply_probability_contract(frame)

    assert normalized.loc[0, "raw_probability_up"] == 48.0
    assert normalized.loc[0, "enhanced_probability_up"] == 53.0
    assert normalized.loc[0, "calibrated_probability_up"] == 61.0
    assert normalized.loc[0, "probability_up"] == 61.0
    assert normalized.loc[0, "calibration_method"] == "explicit"
    assert normalized.loc[0, "probability_confidence"] > 0.0
    assert normalized.loc[0, "p_hit"] == 61.0
    assert normalized.loc[0, "expected_return_pct"] == 0.0
    assert normalized.loc[0, "drawdown_risk_pct"] >= 0.0
    assert normalized.loc[0, "rank_score"] == 82.0
    assert normalized.loc[0, "final_rank_score"] == 82.0
    assert normalized.loc[0, "launch_signal_label"] in {"breakout", "ready", "watch", "wait"}
    assert normalized.loc[0, "risk_level_label"] in {"low", "medium", "high", "extreme"}
    assert normalized.loc[0, "suggested_position_pct"] >= 0.0


def test_daily_lightweight_model_is_serialized_for_api(monkeypatch):
    params = api_service.normalize_api_params(horizon_days=3, positive_return_pct=10, board_size=50)

    monkeypatch.setattr(
        api_service,
        "load_daily_lightweight_backtest_model",
        lambda **kwargs: {
            "status": "ready",
            "sample_count": 12,
            "review_days": 3,
            "base_win_rate": 0.625,
            "model_parameter_update_allowed": False,
            "panels": {
                "strategy": pd.DataFrame(
                    [
                        {
                            "candidate_strategy_label": "strategy1",
                            "sample_count": 7,
                            "win_rate_pct": 71.4,
                        }
                    ]
                )
            },
        },
    )

    payload = api_service.serialize_daily_lightweight_model(params)
    health = api_service.build_review_health({}, {}, params, payload)

    assert payload["status"] == "ready"
    assert payload["panels"]["strategy"][0]["candidate_strategy_label"] == "strategy1"
    assert health["daily_lightweight_status"] == "ready"
    assert health["daily_lightweight_model_independent"] is True


def test_market_backtest_payload_serializes_latest_result(monkeypatch):
    monkeypatch.setattr(
        api_service,
        "load_latest_market_backtest",
        lambda result_limit=50: {
            "summary": {"trade_count": 2, "win_rate": 0.5, "annualized_return": 0.12},
            "summary_path": "summary.json",
            "results_path": "trade_like_results.csv",
            "portfolio_nav_path": "portfolio_daily_nav.csv",
            "portfolio_trades_path": "portfolio_trades.csv",
            "results": pd.DataFrame([{"symbol": "600001", "forward_return": 0.03}]),
            "portfolio_daily_nav": pd.DataFrame([{"trade_date": "2026-04-22", "equity": 1010000.0}]),
            "portfolio_trades": pd.DataFrame([{"symbol": "600001", "net_return": 0.02}]),
        },
    )

    payload = api_service.load_market_backtest_payload(result_limit=10)

    assert payload["status"] == "ready"
    assert payload["summary"]["trade_count"] == 2
    assert payload["summary"]["annualized_return"] == 0.12
    assert payload["results"][0]["symbol"] == "600001"
    assert payload["portfolio_nav_path"] == "portfolio_daily_nav.csv"
    assert payload["portfolio_trades_path"] == "portfolio_trades.csv"
    assert payload["portfolio_daily_nav"][0]["equity"] == 1010000.0
    assert payload["portfolio_trades"][0]["symbol"] == "600001"


def test_market_backtest_payload_returns_missing_state(monkeypatch):
    monkeypatch.setattr(api_service, "load_latest_market_backtest", lambda result_limit=50: {})

    payload = api_service.load_market_backtest_payload(result_limit=0)

    assert payload == {
        "status": "missing",
        "summary": {},
        "results": [],
        "summary_path": "",
        "results_path": "",
        "portfolio_nav_path": "",
        "portfolio_trades_path": "",
        "portfolio_daily_nav": [],
        "portfolio_trades": [],
    }


def test_load_enhanced_board_payload_returns_review_linked_snapshot(monkeypatch):
    params = api_service.normalize_api_params(horizon_days=3, positive_return_pct=10, board_size=50)

    monkeypatch.setattr(
        api_service,
        "load_quick_board_payload",
        lambda params: {
            "board": api_service.serialize_board(pd.DataFrame(), params=params, source="enhanced_board"),
            "freshness": {"status": "stale"},
        },
    )
    monkeypatch.setattr(
        api_service,
        "load_latest_review_bundle",
        lambda **kwargs: {
            "summary": {"board_date": "2026-05-26", "review_date": "2026-05-27", "review_count": 2},
            "details": pd.DataFrame([{"symbol": "600001", "selection_score": 88.0}]),
            "review_meta": {"board_date": "2026-05-26", "review_date": "2026-05-27"},
            "snapshot_board": pd.DataFrame([{"symbol": "600001", "selection_score": 88.0}]),
            "snapshot_meta": {"board_date": "2026-05-26", "cache_version": 10},
        },
    )
    monkeypatch.setattr(api_service, "load_cached_review_panels", lambda params: {})
    monkeypatch.setattr(api_service, "serialize_daily_lightweight_model", lambda params: {"status": "missing"})
    monkeypatch.setattr(api_service, "build_review_health", lambda summary, panels, params, model: {"status": "ok"})
    monkeypatch.setattr(api_service, "build_model_snapshot", lambda params, freshness, review_health, source: {"source": source})
    monkeypatch.setattr(api_service, "_latest_market_close_date", lambda: "2026-05-28")
    monkeypatch.setattr(api_service, "serialize_market_context", lambda payload: payload)

    payload = api_service.load_enhanced_board_payload(params)

    assert payload["review_summary"]["review_count"] == 2
    assert payload["review_details"][0]["symbol"] == "600001"
    assert payload["review_snapshot_meta"]["cache_version"] == 10
    assert payload["review_snapshot_board"][0]["selection_score"] == 88.0


def test_serialize_board_excludes_growth_and_star_board_symbols():
    frame = pd.DataFrame(
        [
            {"symbol": "600001", "name": "MainA", "rank": 1, "probability_up": 60.0, "attention_score": 70.0},
            {"symbol": "300001", "name": "GrowthA", "rank": 2, "probability_up": 99.0, "attention_score": 99.0},
            {"symbol": "688001", "name": "StarA", "rank": 3, "probability_up": 98.0, "attention_score": 98.0},
            {"symbol": "000001", "name": "MainB", "rank": 4, "probability_up": 58.0, "attention_score": 68.0},
        ]
    )

    payload = api_service.serialize_board(frame, params=api_service.normalize_api_params(), source="unit_test")
    symbols = [row["symbol"] for row in payload["rows"]]

    assert symbols == ["600001", "000001"]
    assert payload["meta"]["security_scope"] == "main_board_non_st_ex_growth_star"
    assert payload["meta"]["raw_row_count"] == 4
    assert payload["meta"]["filtered_row_count"] == 2
    assert payload["meta"]["excluded_row_count"] == 2
    assert payload["rows"][0]["rank"] == 1
    assert payload["rows"][1]["rank"] == 2


def test_serialize_board_warns_when_strategy1_and_strategy2_are_empty():
    frame = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "name": "MainA",
                "rank": 1,
                "probability_up": 60.0,
                "attention_score": 70.0,
                "candidate_strategy": "strategy3",
            },
            {
                "symbol": "600002",
                "name": "MainB",
                "rank": 2,
                "probability_up": 58.0,
                "attention_score": 68.0,
                "candidate_strategy": "策略3·多因子主升预备",
            },
        ]
    )

    payload = api_service.serialize_board(frame, params=api_service.normalize_api_params(), source="unit_test")

    assert payload["meta"]["strategy_counts"] == {"strategy1": 0, "strategy2": 0, "strategy3": 2}
    assert "策略1、策略2未筛选出股票" in payload["meta"]["strategy_notice"]
    assert "策略3" in payload["meta"]["strategy_notice"]


def test_serialize_board_can_return_raw_all_market_scope():
    frame = pd.DataFrame(
        [
            {"symbol": "600001", "name": "MainA", "rank": 1, "probability_up": 60.0, "attention_score": 70.0},
            {"symbol": "300001", "name": "GrowthA", "rank": 2, "probability_up": 99.0, "attention_score": 99.0},
            {"symbol": "688001", "name": "StarA", "rank": 3, "probability_up": 98.0, "attention_score": 98.0},
        ]
    )
    params = api_service.normalize_api_params(security_scope="all")

    payload = api_service.serialize_board(frame, params=params, source="unit_test")

    assert [row["symbol"] for row in payload["rows"]] == ["600001", "300001", "688001"]
    assert payload["meta"]["security_scope"] == "all"
    assert payload["meta"]["raw_row_count"] == 3
    assert payload["meta"]["filtered_row_count"] == 3
    assert payload["meta"]["excluded_row_count"] == 0


def test_freshness_contract_marks_stale_cache():
    freshness = api_service.build_freshness_contract(
        {
            "market_data_date": "2026-05-07",
            "latest_market_data_date": "2026-05-08",
            "cache_stale": True,
            "model_schema_version": "model-v1",
        },
        api_service.normalize_api_params(horizon_days=3, positive_return_pct=10),
        source="unit_test",
    )

    assert freshness["source"] == "unit_test"
    assert freshness["data_date"] == "2026-05-07"
    assert freshness["is_latest_data"] is False
    assert freshness["is_latest_model_result"] is False
    assert freshness["freshness_label"] == "stale"
    assert "model-v1" in freshness["consistency_key"]


def test_review_health_contract_tracks_target_gap():
    health = api_service.build_review_health(
        {
            "review_count": 30,
            "win_rate_pct": 64.0,
            "target_hit_rate_pct": 42.0,
            "avg_return_pct": 1.8,
            "calibration_gap_pct": 6.0,
        },
        {"strategy_panel": [{"sample_count": 12}]},
        api_service.normalize_api_params(horizon_days=3, positive_return_pct=10),
    )

    assert health["status_label"] == "usable"
    assert health["target_precision_pct"] == 90.0
    assert health["precision_gap_to_target_pct"] == 26.0
    assert health["sample_count"] >= 30


def test_model_snapshot_is_stable_for_same_contract():
    params = api_service.normalize_api_params(horizon_days=3, positive_return_pct=10)
    freshness = api_service.build_freshness_contract(
        {"market_data_date": "2026-05-08", "latest_market_data_date": "2026-05-08"},
        params,
        source="unit_test",
    )
    health = api_service.build_review_health(
        {"review_count": 30, "win_rate_pct": 64.0, "target_hit_rate_pct": 42.0, "avg_return_pct": 1.8},
        {},
        params,
    )

    first = api_service.build_model_snapshot(params, freshness, health, source="unit_test")
    second = api_service.build_model_snapshot(params, freshness, health, source="unit_test")

    assert first["model_version_id"] == second["model_version_id"]
    assert first["signature"] == second["signature"]
    assert first["target_precision_pct"] == 90.0
    assert first["is_comparable"] is True


def test_review_panels_are_short_cached(monkeypatch):
    calls = []

    def fake_load_review_battle_panels(**kwargs):
        calls.append(kwargs)
        return {
            "strategy_panel": pd.DataFrame([{"candidate_strategy_label": "strategy", "sample_count": 3}]),
            "short_market_state_panel": pd.DataFrame(),
            "long_market_state_panel": pd.DataFrame(),
            "combo_panel": pd.DataFrame(),
            "meta": {"generated_at": "now"},
        }

    monkeypatch.setattr(api_service, "load_review_battle_panels", fake_load_review_battle_panels)
    api_service.API_REVIEW_PANEL_CACHE.clear()
    params = api_service.normalize_api_params()

    first = api_service.load_cached_review_panels(params)
    second = api_service.load_cached_review_panels(params)

    assert len(calls) == 1
    assert first["meta"]["cache_hit"] is False
    assert second["meta"]["cache_hit"] is True
    assert second["strategy_panel"][0]["sample_count"] == 3


def test_intraday_sector_sync_enters_adaptive_profile():
    details = pd.DataFrame(
        [
            {
                "rank": 1,
                "next_day_return": 0.06,
                "win": 1.0,
                "hit_target": 1.0,
                "intraday_high_return": 0.08,
                "attention_score": 60.0,
                "probability_up": 55.0,
                "enhanced_attention_score": 62.0,
                "quant_score": 50.0,
                "launch_score": 58.0,
                "market_resonance_score": 55.0,
                "intraday_sector_sync_score": 80.0,
                "launch_specialist_score": 60.0,
                "launch_regime_fit_score": 55.0,
                "launch_window_score": 60.0,
                "stage_label": "强势",
                "precision_gate_label": "strong",
            },
            {
                "rank": 2,
                "next_day_return": -0.02,
                "win": 0.0,
                "hit_target": 0.0,
                "intraday_high_return": 0.01,
                "attention_score": 58.0,
                "probability_up": 54.0,
                "enhanced_attention_score": 60.0,
                "quant_score": 50.0,
                "launch_score": 57.0,
                "market_resonance_score": 54.0,
                "intraday_sector_sync_score": 35.0,
                "launch_specialist_score": 59.0,
                "launch_regime_fit_score": 54.0,
                "launch_window_score": 58.0,
                "stage_label": "弱势",
                "precision_gate_label": "weak",
            },
        ]
    )

    profile = daily_review._derive_adaptive_profile([{"details": details}])

    assert "intraday_sector_sync_score" in profile["weights"]
    assert "intraday_sector_sync_score" in profile["factor_edges"]
    assert profile["factor_edges"]["intraday_sector_sync_score"] > 0


def test_adaptive_rank_score_uses_intraday_sector_sync_score():
    board = pd.DataFrame(
        [
            {
                "attention_score": 50.0,
                "probability_up": 50.0,
                "enhanced_attention_score": 50.0,
                "quant_score": 50.0,
                "launch_score": 50.0,
                "market_resonance_score": 50.0,
                "intraday_sector_sync_score": 80.0,
                "launch_specialist_score": 50.0,
                "launch_regime_fit_score": 50.0,
                "launch_window_score": 50.0,
            },
            {
                "attention_score": 50.0,
                "probability_up": 50.0,
                "enhanced_attention_score": 50.0,
                "quant_score": 50.0,
                "launch_score": 50.0,
                "market_resonance_score": 50.0,
                "intraday_sector_sync_score": 20.0,
                "launch_specialist_score": 50.0,
                "launch_regime_fit_score": 50.0,
                "launch_window_score": 50.0,
            },
        ]
    )

    score = daily_review.compute_adaptive_rank_score(board, None)

    assert score.iloc[0] > score.iloc[1]


def test_prediction_contract_derives_rank_score_when_missing():
    frame = pd.DataFrame(
        [
            {
                "symbol": "600002",
                "probability_up": 70.0,
                "predicted_upside_pct": 12.0,
                "enhanced_attention_score": 75.0,
                "market_resonance_score": 65.0,
                "intraday_execution_score": 72.0,
                "sector_strength_score": 66.0,
                "market_state_label": "trend",
            }
        ]
    )

    normalized = api_service.apply_probability_contract(frame)

    assert round(normalized.loc[0, "p_hit"], 2) == 65.06
    assert normalized.loc[0, "calibration_method"] == "market_state_shrinkage"
    assert normalized.loc[0, "probability_band_low"] < normalized.loc[0, "p_hit"]
    assert normalized.loc[0, "probability_band_high"] > normalized.loc[0, "p_hit"]
    assert normalized.loc[0, "expected_return_pct"] == 12.0
    assert normalized.loc[0, "drawdown_risk_pct"] > 0.0
    assert normalized.loc[0, "market_state_display"] == "趋势扩散"
    assert normalized.loc[0, "intraday_sector_sync_score"] > 60.0
    assert normalized.loc[0, "relative_intraday_alpha"] == 6.0
    assert normalized.loc[0, "intraday_sector_state"] in {
        "confirmed_sync",
        "sector_lead_wait",
        "stock_lead_watch",
        "divergence_risk",
        "neutral_sync",
    }
    assert normalized.loc[0, "rank_score"] > 0.0
    assert normalized.loc[0, "board_resonance_strength"] > 0.0
    assert normalized.loc[0, "long_setup_quality"] > 0.0
    assert 0.0 <= normalized.loc[0, "crowding_risk"] <= 100.0
    assert normalized.loc[0, "crowding_risk_label"] in {"拥挤低", "正常", "偏拥挤", "量化拥挤"}
    assert normalized.loc[0, "launch_window_score"] > 0.0
    assert normalized.loc[0, "launch_specialist_score"] > 0.0
    assert normalized.loc[0, "launch_signal_label"] in {"breakout", "ready", "watch", "wait"}
    assert normalized.loc[0, "launch_phase_label"] in {
        "crowded",
        "breakout_confirmed",
        "pre_launch",
        "pullback_setup",
        "wait_confirm",
    }
    assert isinstance(normalized.loc[0, "launch_reason_text"], str)
    assert normalized.loc[0, "stop_loss_pct"] >= 0.0
    assert normalized.loc[0, "take_profit_pct"] >= 5.0
    assert isinstance(normalized.loc[0, "risk_control_note"], str)


def test_probability_contract_allows_zero_execution_weight_for_launch_window():
    frame = pd.DataFrame(
        [
            {
                "symbol": "600003",
                "display_probability_up": 60.0,
                "probability_up": 60.0,
                "predicted_upside_pct": 12.0,
                "drawdown_risk_pct": 5.0,
                "execution_score": 100.0,
                "market_state_label": "trend",
            }
        ]
    )

    baseline = api_service.apply_probability_contract(frame)
    execution_off = api_service.apply_probability_contract(frame, launch_window_execution_weight=0.0)

    assert round(baseline.loc[0, "launch_window_score"] - execution_off.loc[0, "launch_window_score"], 2) == 22.0
    assert baseline.loc[0, "rank_score"] == execution_off.loc[0, "rank_score"]
    assert baseline.loc[0, "final_rank_score"] == execution_off.loc[0, "final_rank_score"]


def test_rebuild_task_reuses_running_future(monkeypatch):
    submitted = []

    class FakeFuture:
        def done(self):
            return False

    def fake_submit(*args):
        submitted.append(args)
        return FakeFuture()

    monkeypatch.setattr(api_service.API_TASK_EXECUTOR, "submit", fake_submit)
    monkeypatch.setattr(api_service, "_get_async_task_progress", lambda task_id: {"completed": 0, "total": 1})
    api_service.API_TASK_FUTURES.clear()

    params = api_service.normalize_api_params()
    first = api_service.start_rebuild_ranking_task(params)
    second = api_service.start_rebuild_ranking_task(params)

    assert first["status"] == "running"
    assert second["status"] == "running"
    assert len(submitted) == 1


def test_market_backtest_task_reuses_running_future_for_strategy_aliases(monkeypatch):
    submitted = []
    recorded = []

    class FakeFuture:
        def done(self):
            return False

    def fake_submit(*args, **kwargs):
        submitted.append((args, kwargs))
        return FakeFuture()

    monkeypatch.setattr(api_service.API_TASK_EXECUTOR, "submit", fake_submit)
    monkeypatch.setattr(api_service, "_get_async_task_progress", lambda task_id: {"completed": 0, "total": 1})
    monkeypatch.setattr(
        api_service.DEFAULT_TASK_REGISTRY,
        "record_submitted",
        lambda task_id, *, task_type, params: recorded.append((task_id, task_type, params)),
    )
    monkeypatch.setattr(api_service.DEFAULT_TASK_REGISTRY, "record_status", lambda *args, **kwargs: None)
    api_service.API_TASK_FUTURES.clear()

    first = api_service.start_market_backtest_task(
        date_from="2026-04-21",
        date_to="2026-04-30",
        strategy_mode="strategy1",
    )
    second = api_service.start_market_backtest_task(
        date_from="2026-04-21",
        date_to="2026-04-30",
        strategy_mode="1",
    )

    assert first["status"] == "running"
    assert second["status"] == "running"
    assert len(submitted) == 1
    assert len(recorded) == 1
    assert recorded[0][1] == "market_backtest"
    assert submitted[0][1]["strategy_mode"] == "strategy1"


def test_create_app_serves_frontend_missing_message_without_dist(monkeypatch):
    monkeypatch.setattr(api_module, "frontend_dist_available", lambda: False)
    created_app = api_module.create_app()
    root_route = next(route for route in created_app.routes if route.path == "/" and "GET" in getattr(route, "methods", set()))

    response = root_route.endpoint()

    assert response.status_code == 200
    assert b"Frontend assets are not built yet" in response.body
