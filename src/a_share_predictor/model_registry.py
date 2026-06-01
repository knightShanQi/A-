from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class ModelArtifactMetadata:
    model_id: str
    model_family: str
    label: str
    horizon_days: int
    positive_return: float
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    feature_schema_version: str
    feature_columns: tuple[str, ...]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_columns"] = list(self.feature_columns)
        return payload


def validate_model_time_windows(*, train_start: str, train_end: str, test_start: str, test_end: str) -> None:
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    if train_start_ts > train_end_ts:
        raise ValueError("train_start must be earlier than or equal to train_end")
    if test_start_ts > test_end_ts:
        raise ValueError("test_start must be earlier than or equal to test_end")
    if train_end_ts >= test_start_ts:
        raise ValueError("train/test windows must be strictly time-ordered to avoid leakage")


def build_model_artifact_metadata(
    *,
    model_family: str,
    label: str,
    horizon_days: int,
    positive_return: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    feature_schema_version: str,
    feature_columns: Iterable[str],
    metrics: Mapping[str, object],
) -> ModelArtifactMetadata:
    validate_model_time_windows(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    clean_features = tuple(str(column) for column in feature_columns)
    clean_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(numeric):
            clean_metrics[str(key)] = numeric
    fingerprint_payload = {
        "model_family": model_family,
        "label": label,
        "horizon_days": int(horizon_days),
        "positive_return": float(positive_return),
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "feature_schema_version": feature_schema_version,
        "feature_columns": clean_features,
    }
    model_id = hashlib.sha1(json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
    return ModelArtifactMetadata(
        model_id=model_id,
        model_family=str(model_family),
        label=str(label),
        horizon_days=int(horizon_days),
        positive_return=float(positive_return),
        train_start=str(train_start),
        train_end=str(train_end),
        test_start=str(test_start),
        test_end=str(test_end),
        feature_schema_version=str(feature_schema_version),
        feature_columns=clean_features,
        metrics=clean_metrics,
    )
