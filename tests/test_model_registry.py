from __future__ import annotations

import pytest

from a_share_predictor.model_registry import build_model_artifact_metadata, validate_model_time_windows


def test_model_artifact_metadata_has_stable_id_and_numeric_metrics():
    metadata = build_model_artifact_metadata(
        model_family="market_wide_ensemble",
        label="h5_p003",
        horizon_days=5,
        positive_return=0.03,
        train_start="2025-01-01",
        train_end="2025-12-31",
        test_start="2026-01-01",
        test_end="2026-03-31",
        feature_schema_version="model_schema_v5",
        feature_columns=["ret_5", "volume_ratio_20"],
        metrics={"roc_auc": 0.61, "note": "diagnostic"},
    )
    repeat = build_model_artifact_metadata(
        model_family="market_wide_ensemble",
        label="h5_p003",
        horizon_days=5,
        positive_return=0.03,
        train_start="2025-01-01",
        train_end="2025-12-31",
        test_start="2026-01-01",
        test_end="2026-03-31",
        feature_schema_version="model_schema_v5",
        feature_columns=["ret_5", "volume_ratio_20"],
        metrics={"roc_auc": 0.70},
    )

    assert metadata.model_id == repeat.model_id
    assert metadata.metrics == {"roc_auc": 0.61}
    assert metadata.to_dict()["feature_columns"] == ["ret_5", "volume_ratio_20"]


def test_model_time_windows_reject_overlap():
    with pytest.raises(ValueError, match="avoid leakage"):
        validate_model_time_windows(
            train_start="2025-01-01",
            train_end="2026-01-02",
            test_start="2026-01-01",
            test_end="2026-03-31",
        )
