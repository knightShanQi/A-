import a_share_predictor.daily_review as daily_review


def test_compute_replay_calibrated_scores_outputs_empirical_probability_and_upside():
    profile = {
        "review_days": 6,
        "review_stocks": 160,
        "stage_edges": {"trend": 0.08},
        "stage_supports": {"trend": 24},
        "stage_stats": {"trend": {"win_rate_pct": 68.0, "avg_return_pct": 2.4, "intraday_high_return_pct": 3.8, "hit_rate_pct": 44.0, "support": 24}},
        "precision_segment_edges": {},
        "precision_segment_supports": {},
        "probability_bucket_edges": {"60-70": 0.06},
        "probability_bucket_supports": {"60-70": 32},
        "probability_bucket_stats": {"60-70": {"win_rate_pct": 66.0, "avg_return_pct": 2.2, "intraday_high_return_pct": 3.5, "hit_rate_pct": 40.0, "support": 32}},
        "quant_bucket_edges": {"60-70": 0.03},
        "quant_bucket_supports": {"60-70": 28},
        "quant_bucket_stats": {"60-70": {"win_rate_pct": 61.0, "avg_return_pct": 1.9, "intraday_high_return_pct": 3.0, "hit_rate_pct": 35.0, "support": 28}},
        "launch_bucket_edges": {"70-80": 0.05},
        "launch_bucket_supports": {"70-80": 20},
        "launch_bucket_stats": {"70-80": {"win_rate_pct": 65.0, "avg_return_pct": 2.3, "intraday_high_return_pct": 3.6, "hit_rate_pct": 41.0, "support": 20}},
        "resonance_bucket_edges": {"60-75": 0.04},
        "resonance_bucket_supports": {"60-75": 18},
        "resonance_bucket_stats": {"60-75": {"win_rate_pct": 63.0, "avg_return_pct": 2.0, "intraday_high_return_pct": 3.2, "hit_rate_pct": 38.0, "support": 18}},
        "launch_window_bucket_edges": {"75-100": 0.07},
        "launch_window_bucket_supports": {"75-100": 22},
        "launch_window_bucket_stats": {"75-100": {"win_rate_pct": 70.0, "avg_return_pct": 2.8, "intraday_high_return_pct": 4.1, "hit_rate_pct": 46.0, "support": 22}},
        "launch_window_status_edges": {"黄金启动窗": 0.08},
        "launch_window_status_supports": {"黄金启动窗": 20},
        "launch_window_status_stats": {"黄金启动窗": {"win_rate_pct": 72.0, "avg_return_pct": 3.0, "intraday_high_return_pct": 4.4, "hit_rate_pct": 48.0, "support": 20}},
        "market_replay_days": 64,
        "market_replay_rows": 18000,
        "market_replay_symbols": 260,
        "market_state_edges": {"trend": 0.05},
        "market_state_supports": {"trend": 4200},
        "market_state_stats": {"trend": {"win_rate_pct": 58.0, "avg_return_pct": 1.3, "intraday_high_return_pct": 2.1, "hit_rate_pct": 30.0, "support": 4200}},
        "market_stage_proxy_edges": {"trend_drive": 0.06},
        "market_stage_proxy_supports": {"trend_drive": 2600},
        "market_stage_proxy_stats": {"trend_drive": {"win_rate_pct": 60.0, "avg_return_pct": 1.5, "intraday_high_return_pct": 2.3, "hit_rate_pct": 32.0, "support": 2600}},
    }

    calibrated = daily_review.compute_replay_calibrated_scores(
        {
            "probability_up": 62.0,
            "attention_score": 79.0,
            "enhanced_attention_score": 81.0,
            "predicted_upside_pct": 4.8,
            "predicted_upside_low_pct": 3.6,
            "predicted_upside_high_pct": 6.1,
            "quant_score": 66.0,
            "stage_label": "trend",
            "launch_score": 74.0,
            "market_resonance_score": 68.0,
            "launch_window_score": 82.0,
            "launch_window_status": "黄金启动窗",
            "market_state_label": "trend",
            "market_stage_proxy": "trend_drive",
        },
        profile,
    )

    assert calibrated["probability_up"] > 62.0
    assert calibrated["predicted_upside_pct"] > 0.0
    assert calibrated["predicted_upside_low_pct"] <= calibrated["predicted_upside_pct"]
    assert calibrated["predicted_upside_high_pct"] >= calibrated["predicted_upside_pct"]
    assert calibrated["replay_empirical_probability_pct"] >= 58.0
    assert calibrated["replay_empirical_upside_pct"] > 0.0
    assert calibrated["replay_empirical_intraday_upside_pct"] >= calibrated["replay_empirical_upside_pct"]
