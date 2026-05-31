import pickle

import pandas as pd

import a_share_predictor.daily_review as daily_review


def test_load_latest_review_summary_uses_metadata_not_filename_token(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
            "board_date": "2026-04-10",
            "review_date": "2026-04-13",
            "horizon_days": 5,
            "positive_return": 0.03,
            "ranking_by": "关注分数",
            "board_size": 50,
        },
        "summary": {"board_date": "2026-04-10", "review_date": "2026-04-13", "review_count": 1},
        "details": pd.DataFrame([{"symbol": "000001", "direction_hit": 1.0}]),
    }
    path = review_dir / f"review_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_ulegacytoken_20260410_20260413.pkl"
    path.write_bytes(pickle.dumps(payload))

    summary = daily_review.load_latest_review_summary(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )
    details = daily_review.load_latest_review_details(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    assert summary["review_date"] == "2026-04-13"
    assert len(details) == 1


def test_load_latest_snapshot_board_uses_metadata_not_filename_token(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    board = pd.DataFrame([{"symbol": "000001", "attention_score": 88.0}])
    payload = {
        "meta": {
            "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
            "board_date": "2026-04-11",
            "latest_market_data_date": "2026-04-13",
            "ranking_by": "关注分数",
            "board_size": 50,
            "horizon_days": 5,
            "positive_return": 0.03,
            "captured_at": "2026-04-13 09:30:00",
        },
        "board": board,
    }
    path = review_dir / f"snapshot_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_ulegacytoken_20260411.pkl"
    path.write_bytes(pickle.dumps(payload))

    loaded_board, meta = daily_review.load_latest_snapshot_board(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    assert loaded_board["symbol"].tolist() == ["000001"]
    assert meta["board_date"] == "2026-04-11"


def test_load_latest_snapshot_and_review_prefer_latest_meta_dates(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    older_snapshot = {
        "meta": {
            "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
            "board_date": "2026-04-11",
            "latest_market_data_date": "2026-04-11",
            "ranking_by": "关注分数",
            "board_size": 50,
            "horizon_days": 5,
            "positive_return": 0.03,
            "captured_at": "2026-04-11 15:00:00",
        },
        "board": pd.DataFrame([{"symbol": "000001"}]),
    }
    newer_snapshot = {
        "meta": {
            "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
            "board_date": "2026-04-14",
            "latest_market_data_date": "2026-04-14",
            "ranking_by": "关注分数",
            "board_size": 50,
            "horizon_days": 5,
            "positive_return": 0.03,
            "captured_at": "2026-04-14 15:00:00",
        },
        "board": pd.DataFrame([{"symbol": "000002"}]),
    }
    (review_dir / f"snapshot_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_utokenz_20260411.pkl").write_bytes(
        pickle.dumps(older_snapshot)
    )
    (review_dir / f"snapshot_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_uatoken_20260414.pkl").write_bytes(
        pickle.dumps(newer_snapshot)
    )

    older_review = {
        "meta": {
            "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
            "board_date": "2026-04-11",
            "review_date": "2026-04-14",
            "horizon_days": 5,
            "positive_return": 0.03,
            "ranking_by": "关注分数",
            "board_size": 50,
        },
        "summary": {"board_date": "2026-04-11", "review_date": "2026-04-14", "review_count": 1},
        "details": pd.DataFrame([{"symbol": "000001", "direction_hit": 1.0}]),
    }
    newer_review = {
        "meta": {
            "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
            "board_date": "2026-04-14",
            "review_date": "2026-04-15",
            "horizon_days": 5,
            "positive_return": 0.03,
            "ranking_by": "关注分数",
            "board_size": 50,
        },
        "summary": {"board_date": "2026-04-14", "review_date": "2026-04-15", "review_count": 1},
        "details": pd.DataFrame([{"symbol": "000002", "direction_hit": 1.0}]),
    }
    (review_dir / f"review_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_utokenz_20260411_20260414.pkl").write_bytes(
        pickle.dumps(older_review)
    )
    (review_dir / f"review_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_uatoken_20260414_20260415.pkl").write_bytes(
        pickle.dumps(newer_review)
    )

    loaded_board, meta = daily_review.load_latest_snapshot_board(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )
    summary = daily_review.load_latest_review_summary(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    assert loaded_board["symbol"].tolist() == ["000002"]
    assert meta["board_date"] == "2026-04-14"
    assert summary["review_date"] == "2026-04-15"


def test_load_latest_snapshot_prefers_higher_cache_version_for_same_board_date(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    lower_payload = {
        "meta": {
            "cache_version": 8,
            "board_date": "2026-04-14",
            "latest_market_data_date": "2026-04-14",
            "ranking_by": "鍏虫敞鍒嗘暟",
            "board_size": 50,
            "horizon_days": 5,
            "positive_return": 0.03,
            "captured_at": "2026-04-14 15:00:00",
        },
        "board": pd.DataFrame([{"symbol": "000001", "attention_score": 80.0}]),
    }
    higher_payload = {
        "meta": {
            "cache_version": 10,
            "board_date": "2026-04-14",
            "latest_market_data_date": "2026-04-14",
            "ranking_by": "鍏虫敞鍒嗘暟",
            "board_size": 50,
            "horizon_days": 5,
            "positive_return": 0.03,
            "captured_at": "2026-04-14 14:00:00",
        },
        "board": pd.DataFrame([{"symbol": "000001", "attention_score": 80.0, "selection_score": 88.0}]),
    }
    (review_dir / "snapshot_v8_h5_r300_b50_utest_20260414.pkl").write_bytes(pickle.dumps(lower_payload))
    (review_dir / "snapshot_v10_h5_r300_b50_utest_20260414.pkl").write_bytes(pickle.dumps(higher_payload))

    board, meta = daily_review.load_latest_snapshot_board(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="鍏虫敞鍒嗘暟",
        board_size=50,
    )

    assert meta["cache_version"] == 10
    assert "selection_score" in board.columns


def test_load_latest_review_prefers_higher_cache_version_for_same_dates(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    lower_payload = {
        "meta": {
            "cache_version": 8,
            "board_date": "2026-04-14",
            "review_date": "2026-04-15",
            "horizon_days": 5,
            "positive_return": 0.03,
            "ranking_by": "鍏虫敞鍒嗘暟",
            "board_size": 50,
        },
        "summary": {"board_date": "2026-04-14", "review_date": "2026-04-15", "review_count": 1, "source": "v8"},
        "details": pd.DataFrame([{"symbol": "000001", "attention_score": 80.0}]),
    }
    higher_payload = {
        "meta": {
            "cache_version": 10,
            "board_date": "2026-04-14",
            "review_date": "2026-04-15",
            "horizon_days": 5,
            "positive_return": 0.03,
            "ranking_by": "鍏虫敞鍒嗘暟",
            "board_size": 50,
        },
        "summary": {"board_date": "2026-04-14", "review_date": "2026-04-15", "review_count": 1, "source": "v10"},
        "details": pd.DataFrame([{"symbol": "000001", "attention_score": 80.0, "selection_score": 88.0}]),
    }
    (review_dir / "review_v8_h5_r300_b50_utest_20260414_20260415.pkl").write_bytes(pickle.dumps(lower_payload))
    (review_dir / "review_v10_h5_r300_b50_utest_20260414_20260415.pkl").write_bytes(pickle.dumps(higher_payload))

    summary = daily_review.load_latest_review_summary(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="鍏虫敞鍒嗘暟",
        board_size=50,
    )
    details = daily_review.load_latest_review_details(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="鍏虫敞鍒嗘暟",
        board_size=50,
    )

    assert summary is not None
    assert summary["source"] == "v10"
    assert "selection_score" in details.columns


def test_load_latest_review_bundle_links_snapshot_by_review_board_date(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    snapshot_same_day = {
        "meta": {
            "cache_version": 10,
            "board_date": "2026-04-14",
            "latest_market_data_date": "2026-04-14",
            "ranking_by": "鍏虫敞鍒嗘暟",
            "board_size": 50,
            "horizon_days": 5,
            "positive_return": 0.03,
            "captured_at": "2026-04-14 14:00:00",
        },
        "board": pd.DataFrame([{"symbol": "000001", "selection_score": 88.0}]),
    }
    snapshot_later_day = {
        "meta": {
            "cache_version": 8,
            "board_date": "2026-04-15",
            "latest_market_data_date": "2026-04-15",
            "ranking_by": "鍏虫敞鍒嗘暟",
            "board_size": 50,
            "horizon_days": 5,
            "positive_return": 0.03,
            "captured_at": "2026-04-15 14:00:00",
        },
        "board": pd.DataFrame([{"symbol": "000001", "attention_score": 80.0}]),
    }
    review_payload = {
        "meta": {
            "cache_version": 10,
            "board_date": "2026-04-14",
            "review_date": "2026-04-15",
            "horizon_days": 5,
            "positive_return": 0.03,
            "ranking_by": "鍏虫敞鍒嗘暟",
            "board_size": 50,
        },
        "summary": {"board_date": "2026-04-14", "review_date": "2026-04-15", "review_count": 1},
        "details": pd.DataFrame([{"symbol": "000001", "selection_score": 88.0}]),
    }
    (review_dir / "snapshot_v10_h5_r300_b50_utest_20260414.pkl").write_bytes(pickle.dumps(snapshot_same_day))
    (review_dir / "snapshot_v8_h5_r300_b50_utest_20260415.pkl").write_bytes(pickle.dumps(snapshot_later_day))
    (review_dir / "review_v10_h5_r300_b50_utest_20260414_20260415.pkl").write_bytes(pickle.dumps(review_payload))

    bundle = daily_review.load_latest_review_bundle(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="鍏虫敞鍒嗘暟",
        board_size=50,
    )

    assert bundle["summary"] is not None
    assert bundle["review_meta"]["board_date"] == "2026-04-14"
    assert bundle["snapshot_meta"]["board_date"] == "2026-04-14"
    assert "selection_score" in bundle["snapshot_board"].columns


def test_run_daily_review_maintenance_creates_review_and_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(daily_review, "load_market_replay_profile", lambda **kwargs: daily_review._default_market_replay_profile())

    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "rank": 1,
                "analysis_date": "2026-04-08",
                "attention_score": 90.0,
                "enhanced_attention_score": 92.0,
                "raw_probability_up": 66.0,
                "enhanced_probability_up": 68.0,
                "probability_up": 70.0,
                "final_rank_score": 91.0,
                "predicted_upside_pct": 9.5,
                "predicted_upside_low_pct": 6.2,
                "predicted_upside_high_pct": 12.8,
                "quant_score": 68.0,
                "sector_score": 73.0,
                "fund_score": 76.0,
                "news_score": 69.0,
                "breakout_quality": 62.0,
                "resonance_quality": 74.0,
                "risk_of_late_entry": 38.0,
                "launch_phase_label": "刚启动",
                "amount": 10.0,
                "candidate_strategy": "策略1",
                "candidate_strategy_label": "策略1·趋势中继",
                "candidate_strategy_short_label": "策略1",
                "market_state_label": "trend",
                "market_stage_proxy": "trend_drive",
                "stage_code": "trend_acceleration",
                "stage_score": 82.0,
                "selection_score": 88.0,
                "selection_confidence": 74.0,
                "technical_adjustment": 4.5,
                "intraday_adjustment": 3.0,
                "backtest_adjustment": 7.0,
                "tomorrow_plan_confidence": 81.0,
                "launch_window_confidence_weight": 0.0,
                "execution_label": "可执行",
                "execution_window": "突破确认型",
                "execution_score": 83.0,
                "execution_confidence": 76.0,
            },
            {
                "symbol": "000002",
                "name": "B",
                "rank": 2,
                "analysis_date": "2026-04-08",
                "attention_score": 84.0,
                "enhanced_attention_score": 83.0,
                "probability_up": 62.0,
                "predicted_upside_pct": 4.2,
                "quant_score": 55.0,
                "sector_score": 48.0,
                "fund_score": 45.0,
                "news_score": 52.0,
                "amount": 8.0,
                "candidate_strategy": "策略2",
                "candidate_strategy_label": "策略2·突破共振",
                "candidate_strategy_short_label": "策略2",
                "market_state_label": "defense",
                "market_stage_proxy": "distribution_risk",
                "stage_code": "distribution_risk",
                "stage_score": 39.0,
                "selection_score": 64.0,
                "selection_confidence": 58.0,
                "technical_adjustment": -3.0,
                "intraday_adjustment": -3.0,
                "backtest_adjustment": -4.0,
                "tomorrow_plan_confidence": 56.0,
                "launch_window_confidence_weight": 0.0,
                "execution_label": "暂不执行",
                "execution_window": "防守等待型",
                "execution_score": 35.0,
                "execution_confidence": 61.0,
            },
        ]
    )
    board.attrs["market_data_date"] = "2026-04-08"
    board.attrs["latest_market_data_date"] = "2026-04-09"
    board.attrs["model_source_label"] = "全市场模型"

    def fake_fetch_daily_history(symbol, start_date=None, end_date=None):
        if symbol == "600519":
            return pd.DataFrame(
                {
                    "date": ["2026-04-08", "2026-04-09"],
                    "close": [100.0, 101.0],
                    "high": [100.5, 101.5],
                }
            )
        if symbol == "000001":
            return pd.DataFrame(
                {
                    "date": ["2026-04-08", "2026-04-09"],
                    "close": [10.0, 10.8],
                    "high": [10.2, 11.0],
                }
            )
        if symbol == "000002":
            return pd.DataFrame(
                {
                    "date": ["2026-04-08", "2026-04-09"],
                    "close": [20.0, 19.6],
                    "high": [20.2, 20.1],
                }
            )
        raise AssertionError(f"unexpected symbol: {symbol}")

    monkeypatch.setattr(daily_review, "fetch_daily_history", fake_fetch_daily_history)

    result = daily_review.run_daily_review_maintenance(
        board,
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    assert result["snapshot_created"] is True
    assert result["new_reviews"] == 1
    assert result["completed_reviews"] == 1
    assert result["latest_summary"]["review_date"] == "2026-04-09"
    assert result["latest_summary"]["review_count"] == 2
    assert result["latest_summary"]["direction_hit_rate_pct"] == 50.0
    assert result["latest_summary"]["calibration_gap_pct"] == 16.0
    assert result["latest_summary"]["direction_brier_score"] > 0.0
    assert result["latest_summary"]["avg_target_progress_pct"] == 100.0
    assert result["latest_summary"]["best_strategy"] == "策略1·趋势中继"
    assert result["latest_summary"]["weakest_strategy"] == "策略2·突破共振"
    assert "策略1·趋势中继" in result["latest_summary"]["strategy_stats"]
    assert result["daily_lightweight_model"]["status"] in {"limited_samples", "ready"}
    assert result["daily_lightweight_model"]["model_parameter_update_allowed"] is False
    assert result["daily_lightweight_model"]["sample_count"] == 2

    profile = daily_review.load_adaptive_rank_profile(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    assert profile["review_days"] == 1
    assert profile["review_stocks"] == 2
    assert profile["best_strategy"] == "策略1·趋势中继"
    assert profile["weakest_strategy"] == "策略2·突破共振"
    assert profile["calibration_scope"] == "rank_score_and_risk_overlay_only"
    assert profile["model_parameter_update_allowed"] is False
    assert "ranking_weight_micro_adjustment" in profile["allowed_calibration_targets"]
    assert abs(sum(profile["weights"].values()) - 1.0) < 0.01

    details = daily_review.load_latest_review_details(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )
    snapshot_board, _ = daily_review.load_latest_snapshot_board(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )
    assert len(details) == 2
    assert "direction_hit" in details.columns
    assert "target_progress_pct" in details.columns
    assert "raw_probability_up" in details.columns
    assert "enhanced_probability_up" in details.columns
    assert "final_rank_score" in details.columns
    assert "stage_code" in details.columns
    assert "stage_score" in details.columns
    assert "selection_score" in details.columns
    assert "predicted_upside_pct" in details.columns
    assert "tomorrow_plan_confidence" in details.columns
    assert "launch_window_confidence_weight" in details.columns
    assert "sector_score" in details.columns
    assert "fund_score" in details.columns
    assert "news_score" in details.columns
    assert "technical_adjustment" in details.columns
    assert "intraday_adjustment" in details.columns
    assert "backtest_adjustment" in details.columns
    assert "execution_score" in details.columns
    assert "launch_phase_label" in details.columns
    assert "candidate_strategy_label" in details.columns
    assert "market_state_label" in details.columns
    assert details.loc[details["symbol"] == "000001", "direction_hit"].iloc[0] == 1.0
    assert details.loc[details["symbol"] == "000001", "raw_probability_up"].iloc[0] == 66.0
    assert details.loc[details["symbol"] == "000001", "enhanced_probability_up"].iloc[0] == 68.0
    assert details.loc[details["symbol"] == "000001", "final_rank_score"].iloc[0] == 91.0
    assert details.loc[details["symbol"] == "000001", "stage_code"].iloc[0] == "trend_acceleration"
    assert details.loc[details["symbol"] == "000001", "stage_score"].iloc[0] == 82.0
    assert details.loc[details["symbol"] == "000001", "selection_score"].iloc[0] == 88.0
    assert details.loc[details["symbol"] == "000001", "predicted_upside_pct"].iloc[0] == 9.5
    assert details.loc[details["symbol"] == "000001", "tomorrow_plan_confidence"].iloc[0] == 81.0
    assert details.loc[details["symbol"] == "000001", "launch_window_confidence_weight"].iloc[0] == 0.0
    assert details.loc[details["symbol"] == "000001", "sector_score"].iloc[0] == 73.0
    assert details.loc[details["symbol"] == "000001", "fund_score"].iloc[0] == 76.0
    assert details.loc[details["symbol"] == "000001", "news_score"].iloc[0] == 69.0
    assert details.loc[details["symbol"] == "000001", "technical_adjustment"].iloc[0] == 4.5
    assert details.loc[details["symbol"] == "000001", "intraday_adjustment"].iloc[0] == 3.0
    assert details.loc[details["symbol"] == "000001", "backtest_adjustment"].iloc[0] == 7.0
    assert details.loc[details["symbol"] == "000001", "execution_score"].iloc[0] == 83.0
    assert details.loc[details["symbol"] == "000001", "launch_phase_label"].iloc[0] == "刚启动"
    assert details.loc[details["symbol"] == "000002", "direction_hit"].iloc[0] == 0.0
    assert details.loc[details["symbol"] == "000001", "candidate_strategy_label"].iloc[0] == "策略1·趋势中继"
    assert "stage_score" in snapshot_board.columns
    assert "selection_score" in snapshot_board.columns
    assert "predicted_upside_pct" in snapshot_board.columns
    assert "tomorrow_plan_confidence" in snapshot_board.columns
    assert "launch_window_confidence_weight" in snapshot_board.columns
    assert "sector_score" in snapshot_board.columns
    assert "technical_adjustment" in snapshot_board.columns
    assert "execution_score" in snapshot_board.columns
    assert snapshot_board.loc[snapshot_board["symbol"] == "000001", "execution_score"].iloc[0] == 83.0
    assert snapshot_board.loc[snapshot_board["symbol"] == "000001", "predicted_upside_pct"].iloc[0] == 9.5
    assert snapshot_board.loc[snapshot_board["symbol"] == "000001", "launch_window_confidence_weight"].iloc[0] == 0.0
    assert snapshot_board.loc[snapshot_board["symbol"] == "000001", "technical_adjustment"].iloc[0] == 4.5
    lightweight_model = daily_review.load_daily_lightweight_backtest_model(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )
    assert lightweight_model["sample_count"] == 2
    assert "strategy" in lightweight_model["panels"]


def test_snapshot_persists_final_rank_score_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)

    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "rank": 1,
                "attention_score": 81.0,
                "enhanced_attention_score": 87.0,
                "predicted_upside_pct": 7.2,
                "tomorrow_plan_confidence": 75.0,
                "sector_score": 68.0,
                "fund_score": 71.0,
                "news_score": 64.0,
                "technical_adjustment": 2.5,
                "intraday_adjustment": 1.0,
                "backtest_adjustment": 3.5,
                "launch_window_confidence_weight": 0.0,
            },
            {
                "symbol": "000002",
                "name": "B",
                "rank": 2,
                "attention_score": 72.0,
                "ranking_score": 79.0,
                "predicted_upside_pct": 3.4,
                "tomorrow_plan_confidence": 59.0,
                "sector_score": 47.0,
                "fund_score": 49.0,
                "news_score": 51.0,
                "technical_adjustment": -1.5,
                "intraday_adjustment": 0.0,
                "backtest_adjustment": -2.0,
                "launch_window_confidence_weight": 0.0,
            },
        ]
    )
    board.attrs["market_data_date"] = "2026-04-10"
    board.attrs["latest_market_data_date"] = "2026-04-10"

    snapshot_path = daily_review.persist_focus_board_snapshot(
        board,
        horizon_days=5,
        positive_return=0.03,
        ranking_by="attention",
        board_size=50,
    )

    assert snapshot_path is not None

    snapshot_board, meta = daily_review.load_latest_snapshot_board(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="attention",
        board_size=50,
    )

    assert meta["board_date"] == "2026-04-10"
    assert "final_rank_score" in snapshot_board.columns
    assert "predicted_upside_pct" in snapshot_board.columns
    assert "tomorrow_plan_confidence" in snapshot_board.columns
    assert "launch_window_confidence_weight" in snapshot_board.columns
    assert "sector_score" in snapshot_board.columns
    assert "fund_score" in snapshot_board.columns
    assert "news_score" in snapshot_board.columns
    assert "technical_adjustment" in snapshot_board.columns
    assert "intraday_adjustment" in snapshot_board.columns
    assert "backtest_adjustment" in snapshot_board.columns
    assert snapshot_board.loc[snapshot_board["symbol"] == "000001", "final_rank_score"].iloc[0] == 87.0
    assert snapshot_board.loc[snapshot_board["symbol"] == "000002", "final_rank_score"].iloc[0] == 79.0
    assert snapshot_board.loc[snapshot_board["symbol"] == "000001", "predicted_upside_pct"].iloc[0] == 7.2
    assert snapshot_board.loc[snapshot_board["symbol"] == "000001", "launch_window_confidence_weight"].iloc[0] == 0.0
    assert snapshot_board.loc[snapshot_board["symbol"] == "000002", "backtest_adjustment"].iloc[0] == -2.0


def test_run_daily_review_maintenance_rolls_profile_over_recent_n_reviews(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(daily_review, "load_market_replay_profile", lambda **kwargs: daily_review._default_market_replay_profile())
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    def write_review(board_date: str, review_date: str, rows: list[dict[str, object]]) -> None:
        payload = {
            "meta": {
                "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
                "board_date": board_date,
                "review_date": review_date,
                "horizon_days": 5,
                "positive_return": 0.03,
                "ranking_by": "attention",
                "board_size": 50,
            },
            "summary": {"board_date": board_date, "review_date": review_date, "review_count": len(rows)},
            "details": pd.DataFrame(rows),
        }
        path = daily_review._review_cache_path(5, 0.03, "attention", 50, board_date, review_date)
        path.write_bytes(pickle.dumps(payload))

    base_columns = {
        "attention_score": 70.0,
        "enhanced_attention_score": 70.0,
        "probability_up": 60.0,
        "quant_score": 55.0,
        "launch_score": 50.0,
        "market_resonance_score": 50.0,
        "intraday_sector_sync_score": 50.0,
        "launch_specialist_score": 50.0,
        "launch_regime_fit_score": 50.0,
        "launch_window_score": 50.0,
        "stage_label": "range_watch",
        "precision_gate_label": "other",
        "launch_window_status": "unknown",
        "intraday_high_return": 0.0,
        "direction_hit": 0.0,
    }
    write_review(
        "2026-04-08",
        "2026-04-09",
        [
            {**base_columns, "symbol": "000001", "rank": 1, "next_day_return": -0.05, "win": 0.0, "hit_target": 0.0},
            {**base_columns, "symbol": "000002", "rank": 2, "next_day_return": -0.03, "win": 0.0, "hit_target": 0.0},
        ],
    )
    write_review(
        "2026-04-10",
        "2026-04-11",
        [
            {**base_columns, "symbol": "000003", "rank": 1, "next_day_return": 0.08, "win": 1.0, "hit_target": 1.0},
            {**base_columns, "symbol": "000004", "rank": 2, "next_day_return": 0.04, "win": 1.0, "hit_target": 1.0},
        ],
    )

    board = pd.DataFrame([{"symbol": "000005", "name": "E", "rank": 1, "attention_score": 80.0}])
    board.attrs["market_data_date"] = "2026-04-12"
    board.attrs["latest_market_data_date"] = "2026-04-12"

    result = daily_review.run_daily_review_maintenance(
        board,
        horizon_days=5,
        positive_return=0.03,
        ranking_by="attention",
        board_size=50,
        rolling_review_days=1,
    )

    profile = result["profile"]
    assert result["rolling_review_days"] == 1
    assert profile["rolling_review_days"] == 1
    assert profile["review_days"] == 1
    assert profile["review_stocks"] == 2
    assert profile["avg_return_pct"] > 0.0
    assert profile["win_rate_pct"] == 100.0


def test_run_daily_review_maintenance_handles_date_as_index_and_column(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(daily_review, "load_market_replay_profile", lambda **kwargs: daily_review._default_market_replay_profile())

    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "rank": 1,
                "analysis_date": "2026-04-09",
                "attention_score": 88.0,
                "probability_up": 72.0,
                "quant_score": 60.0,
                "amount": 10.0,
            }
        ]
    )
    board.attrs["market_data_date"] = "2026-04-09"
    board.attrs["latest_market_data_date"] = "2026-04-10"

    def fake_fetch_daily_history(symbol, start_date=None, end_date=None):
        if symbol == "600519":
            return pd.DataFrame(
                {
                    "date": ["2026-04-09", "2026-04-10"],
                    "close": [100.0, 101.0],
                    "high": [100.3, 101.2],
                }
            )
        if symbol == "000001":
            df = pd.DataFrame(
                {
                    "date": ["2026-04-09", "2026-04-10"],
                    "close": [10.0, 10.5],
                    "high": [10.1, 10.7],
                }
            )
            df.index = pd.Index(pd.to_datetime(df["date"]), name="date")
            return df
        raise AssertionError(f"unexpected symbol: {symbol}")

    monkeypatch.setattr(daily_review, "fetch_daily_history", fake_fetch_daily_history)

    result = daily_review.run_daily_review_maintenance(
        board,
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    assert result["new_reviews"] == 1
    assert result["latest_summary"]["review_date"] == "2026-04-10"


def test_load_review_battle_panels_aggregates_strategy_and_market_state(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        daily_review,
        "load_market_replay_profile",
        lambda **kwargs: {
            "market_replay_days": 64,
            "market_replay_rows": 18000,
            "market_replay_symbols": 260,
            "market_state_stats": {
                "trend": {"support": 4200, "avg_return_pct": 1.6, "intraday_high_return_pct": 2.4, "win_rate_pct": 61.0, "hit_rate_pct": 33.0},
                "defense": {"support": 3600, "avg_return_pct": -0.4, "intraday_high_return_pct": 0.6, "win_rate_pct": 42.0, "hit_rate_pct": 14.0},
            },
            "market_state_edges": {"trend": 0.08, "defense": -0.05},
        },
    )

    payload = {
        "meta": {
            "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
            "board_date": "2026-04-10",
            "review_date": "2026-04-11",
            "horizon_days": 5,
            "positive_return": 0.03,
            "ranking_by": "关注分数",
            "board_size": 50,
        },
        "summary": {"board_date": "2026-04-10", "review_date": "2026-04-11", "review_count": 3},
        "details": pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "name": "A",
                    "rank": 1,
                    "probability_up": 72.0,
                    "next_day_return": 0.06,
                    "intraday_high_return": 0.08,
                    "win": 1.0,
                    "hit_target": 1.0,
                    "direction_hit": 1.0,
                    "candidate_strategy": "策略1",
                    "candidate_strategy_label": "策略1·趋势中继",
                    "market_state_label": "trend",
                },
                {
                    "symbol": "000002",
                    "name": "B",
                    "rank": 2,
                    "probability_up": 68.0,
                    "next_day_return": 0.03,
                    "intraday_high_return": 0.05,
                    "win": 1.0,
                    "hit_target": 1.0,
                    "direction_hit": 1.0,
                    "candidate_strategy": "策略1",
                    "candidate_strategy_label": "策略1·趋势中继",
                    "market_state_label": "trend",
                },
                {
                    "symbol": "000003",
                    "name": "C",
                    "rank": 3,
                    "probability_up": 58.0,
                    "next_day_return": -0.02,
                    "intraday_high_return": 0.01,
                    "win": 0.0,
                    "hit_target": 0.0,
                    "direction_hit": 0.0,
                    "candidate_strategy": "策略2",
                    "candidate_strategy_label": "策略2·突破共振",
                    "market_state_label": "defense",
                },
            ]
        ),
    }
    path = review_dir / f"review_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_token_20260410_20260411.pkl"
    path.write_bytes(pickle.dumps(payload))

    panels = daily_review.load_review_battle_panels(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    strategy_panel = panels["strategy_panel"]
    short_market_state_panel = panels["short_market_state_panel"]
    long_market_state_panel = panels["long_market_state_panel"]
    combo_panel = panels["combo_panel"]

    assert not strategy_panel.empty
    assert strategy_panel.iloc[0]["candidate_strategy_label"] == "策略1·趋势中继"
    assert not short_market_state_panel.empty
    assert "趋势扩散" in set(short_market_state_panel["market_state_display"].tolist())
    assert not long_market_state_panel.empty
    assert "趋势扩散" in set(long_market_state_panel["market_state_display"].tolist())
    assert not combo_panel.empty
    assert panels["meta"]["best_strategy"] == "策略1·趋势中继"
    assert panels["meta"]["weakest_strategy"] == "策略2·突破共振"
    assert "更有效" in panels["meta"]["strategy_effectiveness_summary"]


def test_run_daily_review_maintenance_still_persists_snapshot_without_new_review(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(daily_review, "load_market_replay_profile", lambda **kwargs: daily_review._default_market_replay_profile())

    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "rank": 1,
                "analysis_date": "2026-04-09",
                "attention_score": 90.0,
                "probability_up": 70.0,
                "quant_score": 68.0,
                "amount": 10.0,
            }
        ]
    )
    board.attrs["market_data_date"] = "2026-04-09"
    board.attrs["latest_market_data_date"] = "2026-04-09"

    result = daily_review.run_daily_review_maintenance(
        board,
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    assert result["snapshot_created"] is True
    assert result["new_reviews"] == 0
    assert result["snapshot_path"]


def test_compute_adaptive_rank_score_uses_profile_weights():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "attention_score": 80.0,
                "enhanced_attention_score": 82.0,
                "probability_up": 60.0,
                "quant_score": 55.0,
            },
            {
                "symbol": "000002",
                "attention_score": 70.0,
                "enhanced_attention_score": 72.0,
                "probability_up": 85.0,
                "quant_score": 50.0,
            },
        ]
    )
    profile = {
        "weights": {
            "attention_score": 0.1,
            "probability_up": 0.7,
            "enhanced_attention_score": 0.1,
            "quant_score": 0.1,
        }
    }

    scores = daily_review.compute_adaptive_rank_score(board, profile)

    assert scores.iloc[1] > scores.iloc[0]


def test_compute_adaptive_rank_score_can_use_launch_specialist_columns():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "attention_score": 80.0,
                "enhanced_attention_score": 81.0,
                "probability_up": 60.0,
                "quant_score": 60.0,
                "launch_score": 66.0,
                "market_resonance_score": 58.0,
                "launch_specialist_score": 52.0,
                "launch_regime_fit_score": 50.0,
                "launch_window_score": 54.0,
            },
            {
                "symbol": "000002",
                "attention_score": 80.0,
                "enhanced_attention_score": 81.0,
                "probability_up": 60.0,
                "quant_score": 60.0,
                "launch_score": 66.0,
                "market_resonance_score": 58.0,
                "launch_specialist_score": 78.0,
                "launch_regime_fit_score": 74.0,
                "launch_window_score": 82.0,
            },
        ]
    )

    scores = daily_review.compute_adaptive_rank_score(board, None)

    assert scores.iloc[1] > scores.iloc[0]


def test_compute_adaptive_rank_score_only_applies_light_risk_overlay():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "attention_score": 80.0,
                "enhanced_attention_score": 80.0,
                "probability_up": 62.0,
                "quant_score": 60.0,
                "risk_of_late_entry": 42.0,
                "launch_phase_label": "刚启动",
            },
            {
                "symbol": "000002",
                "attention_score": 80.0,
                "enhanced_attention_score": 80.0,
                "probability_up": 62.0,
                "quant_score": 60.0,
                "risk_of_late_entry": 88.0,
                "launch_phase_label": "已走远",
            },
        ]
    )

    scores = daily_review.compute_adaptive_rank_score(board, None)

    assert scores.iloc[0] > scores.iloc[1]
    assert 0 < scores.iloc[0] - scores.iloc[1] <= 6.0


def test_compute_adaptive_rank_score_penalizes_crowding_and_rewards_long_setup_quality():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "attention_score": 78.0,
                "enhanced_attention_score": 80.0,
                "probability_up": 62.0,
                "quant_score": 65.0,
                "long_setup_quality": 78.0,
                "crowding_risk": 32.0,
                "risk_of_late_entry": 42.0,
                "launch_phase_label": "刚启动",
            },
            {
                "symbol": "000002",
                "attention_score": 78.0,
                "enhanced_attention_score": 80.0,
                "probability_up": 62.0,
                "quant_score": 65.0,
                "long_setup_quality": 58.0,
                "crowding_risk": 82.0,
                "risk_of_late_entry": 42.0,
                "launch_phase_label": "量化拥挤",
            },
        ]
    )

    scores = daily_review.compute_adaptive_rank_score(board, None)

    assert scores.iloc[0] > scores.iloc[1]


def test_load_adaptive_rank_profile_backfills_replay_overlay_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        daily_review,
        "load_market_replay_profile",
        lambda **kwargs: {
            **daily_review._default_market_replay_profile(),
            "market_replay_days": 48,
            "market_replay_rows": 4096,
            "market_replay_symbols": 180,
        },
    )
    review_dir = tmp_path / "daily_focus_board_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    old_profile_payload = {
        "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
        "meta": {
            "ranking_by": "关注分数",
            "board_size": 50,
            "horizon_days": 5,
            "positive_return": 0.03,
        },
        "profile": {
            "weights": dict(daily_review.DEFAULT_PROFILE_WEIGHTS),
            "review_days": 1,
            "review_stocks": 2,
        },
    }
    (review_dir / f"profile_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_ulegacytoken.pkl").write_bytes(
        pickle.dumps(old_profile_payload)
    )

    review_payload = {
        "meta": {
            "cache_version": daily_review.DAILY_REVIEW_CACHE_VERSION,
            "board_date": "2026-04-10",
            "review_date": "2026-04-11",
            "horizon_days": 5,
            "positive_return": 0.03,
            "ranking_by": "关注分数",
            "board_size": 50,
        },
        "summary": {"board_date": "2026-04-10", "review_date": "2026-04-11", "review_count": 2},
        "details": pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "attention_score": 80.0,
                    "enhanced_attention_score": 82.0,
                    "probability_up": 58.0,
                    "quant_score": 76.0,
                    "next_day_return": 0.08,
                    "win": 1.0,
                    "hit_target": 1.0,
                    "stage_label": "趋势主升加速",
                    "precision_gate_label": "90%精度放行",
                },
                {
                    "symbol": "000002",
                    "attention_score": 78.0,
                    "enhanced_attention_score": 76.0,
                    "probability_up": 95.0,
                    "quant_score": 42.0,
                    "next_day_return": -0.05,
                    "win": 0.0,
                    "hit_target": 0.0,
                    "stage_label": "高位分歧派发",
                    "precision_gate_label": "未达90%精度门槛",
                },
            ]
        ),
    }
    review_path = review_dir / f"review_v{daily_review.DAILY_REVIEW_CACHE_VERSION}_h5_r300_b50_ulegacytoken_20260410_20260411.pkl"
    review_path.write_bytes(pickle.dumps(review_payload))

    profile = daily_review.load_adaptive_rank_profile(
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
        board_size=50,
    )

    assert "stage_edges" in profile
    assert "precision_segment_edges" in profile
    assert "probability_bucket_edges" in profile
    assert "quant_bucket_edges" in profile
    assert "launch_bucket_edges" in profile
    assert "resonance_bucket_edges" in profile
    assert "launch_window_bucket_edges" in profile
    assert "launch_window_status_edges" in profile
    assert "strategy_edges" in profile
    assert profile["calibration_scope"] == "rank_score_and_risk_overlay_only"
    assert profile["model_parameter_update_allowed"] is False
    assert profile["market_replay_days"] == 48


def test_derive_market_replay_profile_from_dataset_reconstructs_market_states():
    dataset = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "signal_date": "2026-01-02",
                "future_return": 0.06,
                "target": 1,
                "ret_5": 0.03,
                "ret_20": 0.14,
                "close_vs_ma20": 0.09,
                "breakout_distance_20": 0.01,
                "range_position_20": 0.78,
                "volume_ratio_5": 1.25,
                "upper_shadow_ratio": 0.08,
                "stretch_risk": 6.0,
                "risk_pressure": 135.0,
            },
            {
                "symbol": "000002",
                "signal_date": "2026-01-02",
                "future_return": 0.04,
                "target": 1,
                "ret_5": 0.02,
                "ret_20": 0.11,
                "close_vs_ma20": 0.08,
                "breakout_distance_20": -0.01,
                "range_position_20": 0.73,
                "volume_ratio_5": 1.10,
                "upper_shadow_ratio": 0.10,
                "stretch_risk": 7.0,
                "risk_pressure": 142.0,
            },
            {
                "symbol": "000001",
                "signal_date": "2026-01-03",
                "future_return": 0.03,
                "target": 1,
                "ret_5": 0.02,
                "ret_20": -0.01,
                "close_vs_ma20": 0.01,
                "breakout_distance_20": -0.01,
                "range_position_20": 0.63,
                "volume_ratio_5": 0.96,
                "upper_shadow_ratio": 0.11,
                "stretch_risk": 5.0,
                "risk_pressure": 152.0,
            },
            {
                "symbol": "000002",
                "signal_date": "2026-01-03",
                "future_return": 0.01,
                "target": 0,
                "ret_5": 0.03,
                "ret_20": 0.00,
                "close_vs_ma20": 0.00,
                "breakout_distance_20": -0.02,
                "range_position_20": 0.60,
                "volume_ratio_5": 0.90,
                "upper_shadow_ratio": 0.12,
                "stretch_risk": 5.0,
                "risk_pressure": 156.0,
            },
            {
                "symbol": "000001",
                "signal_date": "2026-01-06",
                "future_return": -0.01,
                "target": 0,
                "ret_5": 0.00,
                "ret_20": 0.03,
                "close_vs_ma20": 0.01,
                "breakout_distance_20": -0.05,
                "range_position_20": 0.49,
                "volume_ratio_5": 0.85,
                "upper_shadow_ratio": 0.16,
                "stretch_risk": 9.0,
                "risk_pressure": 175.0,
            },
            {
                "symbol": "000002",
                "signal_date": "2026-01-06",
                "future_return": 0.00,
                "target": 0,
                "ret_5": -0.01,
                "ret_20": 0.02,
                "close_vs_ma20": 0.00,
                "breakout_distance_20": -0.06,
                "range_position_20": 0.47,
                "volume_ratio_5": 0.88,
                "upper_shadow_ratio": 0.14,
                "stretch_risk": 10.0,
                "risk_pressure": 182.0,
            },
            {
                "symbol": "000001",
                "signal_date": "2026-01-07",
                "future_return": -0.05,
                "target": 0,
                "ret_5": -0.03,
                "ret_20": -0.12,
                "close_vs_ma20": -0.08,
                "breakout_distance_20": 0.00,
                "range_position_20": 0.92,
                "volume_ratio_5": 1.05,
                "upper_shadow_ratio": 0.42,
                "stretch_risk": 26.0,
                "risk_pressure": 240.0,
            },
            {
                "symbol": "000002",
                "signal_date": "2026-01-07",
                "future_return": -0.02,
                "target": 0,
                "ret_5": -0.02,
                "ret_20": -0.09,
                "close_vs_ma20": -0.06,
                "breakout_distance_20": -0.03,
                "range_position_20": 0.90,
                "volume_ratio_5": 1.02,
                "upper_shadow_ratio": 0.39,
                "stretch_risk": 24.0,
                "risk_pressure": 232.0,
            },
        ]
    )

    profile = daily_review._derive_market_replay_profile_from_dataset(dataset, positive_return=0.03)

    assert profile["market_replay_days"] == 4
    assert profile["market_replay_rows"] == 8
    assert profile["market_replay_symbols"] == 2
    assert set(profile["market_state_edges"]).issuperset({"trend", "rebound", "rotation", "defense"})
    assert set(profile["market_stage_proxy_edges"]).issuperset(
        {"trend_drive", "breakout_confirm", "range_watch", "distribution_risk"}
    )


def test_compute_replay_calibrated_scores_can_use_market_replay_layer_only():
    profile = {
        "review_days": 0,
        "review_stocks": 0,
        "market_replay_days": 64,
        "market_replay_rows": 18000,
        "market_replay_symbols": 260,
        "market_state_edges": {"trend": 0.10, "defense": -0.08},
        "market_state_supports": {"trend": 4200, "defense": 3600},
        "market_stage_proxy_edges": {"trend_drive": 0.12, "distribution_risk": -0.09},
        "market_stage_proxy_supports": {"trend_drive": 2600, "distribution_risk": 2100},
    }

    bullish = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 61.0,
            "attention_score": 78.0,
            "enhanced_attention_score": 80.0,
            "quant_score": 72.0,
            "market_ret_5": 0.012,
            "market_ret_20": 0.034,
            "market_close_vs_ma20": 0.028,
            "market_volatility_10": 0.011,
            "market_range_position_20": 0.69,
            "ret_20": 0.14,
            "close_vs_ma20": 0.08,
            "breakout_distance_20": -0.01,
            "range_position_20": 0.71,
            "volume_ratio_5": 1.18,
            "upper_shadow_ratio": 0.10,
            "stretch_risk": 8.0,
            "risk_pressure": 148.0,
        },
        profile,
    )
    risky = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 82.0,
            "attention_score": 81.0,
            "enhanced_attention_score": 83.0,
            "quant_score": 46.0,
            "market_ret_5": -0.018,
            "market_ret_20": -0.024,
            "market_close_vs_ma20": -0.03,
            "market_volatility_10": 0.026,
            "market_range_position_20": 0.38,
            "ret_20": -0.08,
            "close_vs_ma20": 0.06,
            "breakout_distance_20": -0.01,
            "range_position_20": 0.93,
            "volume_ratio_5": 1.01,
            "upper_shadow_ratio": 0.44,
            "stretch_risk": 25.0,
            "risk_pressure": 238.0,
        },
        profile,
    )

    assert bullish["replay_calibration_active"] is True
    assert bullish["probability_up"] > 61.0
    assert bullish["replay_market_state"] == "trend"
    assert bullish["replay_market_stage_proxy"] == "trend_drive"
    assert risky["probability_up"] < 82.0
    assert risky["enhanced_attention_score"] < 83.0
    assert risky["replay_market_state"] == "defense"
    assert risky["replay_market_stage_proxy"] == "distribution_risk"


def test_compute_replay_calibrated_scores_adjusts_probability_and_attention():
    profile = {
        "review_days": 4,
        "review_stocks": 96,
        "stage_edges": {"趋势主升加速": 0.14, "高位分歧派发": -0.12},
        "stage_supports": {"趋势主升加速": 24, "高位分歧派发": 12},
        "precision_segment_edges": {"precision_active": 0.10, "precision_unreached": -0.08},
        "precision_segment_supports": {"precision_active": 20, "precision_unreached": 18},
        "probability_bucket_edges": {"40-60": 0.05, "95-100": -0.10},
        "probability_bucket_supports": {"40-60": 28, "95-100": 16},
        "quant_bucket_edges": {"70-80": 0.08, "0-50": -0.09},
        "quant_bucket_supports": {"70-80": 30, "0-50": 14},
    }

    bullish = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 58.0,
            "attention_score": 80.0,
            "enhanced_attention_score": 82.0,
            "quant_score": 76.0,
            "stage_label": "趋势主升加速",
            "precision_gate_label": "90%精度放行",
        },
        profile,
    )
    risky = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 96.0,
            "attention_score": 84.0,
            "enhanced_attention_score": 83.0,
            "quant_score": 45.0,
            "stage_label": "高位分歧派发",
            "precision_gate_label": "未达90%精度门槛",
        },
        profile,
    )

    assert bullish["replay_calibration_active"] is True
    assert bullish["probability_up"] > 58.0
    assert bullish["attention_score"] > 80.0
    assert risky["probability_up"] < 96.0
    assert risky["enhanced_attention_score"] < 83.0


def test_compute_replay_calibrated_scores_uses_launch_and_resonance_buckets():
    profile = {
        "review_days": 5,
        "review_stocks": 120,
        "stage_edges": {},
        "stage_supports": {},
        "precision_segment_edges": {},
        "precision_segment_supports": {},
        "probability_bucket_edges": {},
        "probability_bucket_supports": {},
        "quant_bucket_edges": {},
        "quant_bucket_supports": {},
        "launch_bucket_edges": {"80-100": 0.10, "0-50": -0.08},
        "launch_bucket_supports": {"80-100": 26, "0-50": 18},
        "resonance_bucket_edges": {"75-100": 0.12, "0-45": -0.09},
        "resonance_bucket_supports": {"75-100": 28, "0-45": 16},
    }

    strong = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 63.0,
            "attention_score": 79.0,
            "enhanced_attention_score": 81.0,
            "quant_score": 68.0,
            "launch_score": 84.0,
            "market_resonance_score": 78.0,
        },
        profile,
    )
    weak = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 63.0,
            "attention_score": 79.0,
            "enhanced_attention_score": 81.0,
            "quant_score": 68.0,
            "launch_score": 42.0,
            "market_resonance_score": 41.0,
        },
        profile,
    )

    assert strong["probability_up"] > 63.0
    assert strong["replay_launch_bucket"] == "80-100"
    assert strong["replay_resonance_bucket"] == "75-100"
    assert weak["probability_up"] < 63.0
    assert weak["replay_launch_bucket"] == "0-50"
    assert weak["replay_resonance_bucket"] == "0-45"


def test_compute_replay_calibrated_scores_uses_launch_window_overlay():
    profile = {
        "review_days": 6,
        "review_stocks": 140,
        "stage_edges": {},
        "stage_supports": {},
        "precision_segment_edges": {},
        "precision_segment_supports": {},
        "probability_bucket_edges": {},
        "probability_bucket_supports": {},
        "quant_bucket_edges": {},
        "quant_bucket_supports": {},
        "launch_bucket_edges": {},
        "launch_bucket_supports": {},
        "resonance_bucket_edges": {},
        "resonance_bucket_supports": {},
        "launch_window_bucket_edges": {"75-100": 0.11, "0-45": -0.08},
        "launch_window_bucket_supports": {"75-100": 30, "0-45": 18},
        "launch_window_status_edges": {"黄金启动窗": 0.13, "高位风险窗": -0.10},
        "launch_window_status_supports": {"黄金启动窗": 24, "高位风险窗": 16},
    }

    strong = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 64.0,
            "attention_score": 79.0,
            "enhanced_attention_score": 81.0,
            "quant_score": 67.0,
            "launch_window_score": 84.0,
            "launch_window_status": "黄金启动窗",
        },
        profile,
    )
    risky = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 64.0,
            "attention_score": 79.0,
            "enhanced_attention_score": 81.0,
            "quant_score": 67.0,
            "launch_window_score": 38.0,
            "launch_window_status": "高位风险窗",
        },
        profile,
    )

    assert strong["probability_up"] > 64.0
    assert strong["replay_launch_window_bucket"] == "75-100"
    assert strong["replay_launch_window_status"] == "黄金启动窗"
    assert risky["probability_up"] < 64.0
    assert risky["enhanced_attention_score"] < 81.0
    assert risky["replay_launch_window_bucket"] == "0-45"
    assert risky["replay_launch_window_status"] == "高位风险窗"
