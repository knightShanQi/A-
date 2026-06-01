from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


CANONICAL_EVALUATION_ENGINE = "unified_portfolio_nav_v1"
CANONICAL_PRIMARY_METRIC = "annualized_return"
CANONICAL_PRIMARY_SOURCE = "portfolio_daily_nav"
CANONICAL_REQUIRED_METRICS = (
    "ending_equity",
    "cumulative_return",
    "annualized_return",
    "max_drawdown",
    "portfolio_trade_count",
    "portfolio_win_rate",
    "portfolio_avg_net_return",
)


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    engine: str = CANONICAL_EVALUATION_ENGINE
    primary_metric: str = CANONICAL_PRIMARY_METRIC
    primary_source: str = CANONICAL_PRIMARY_SOURCE
    required_metrics: tuple[str, ...] = CANONICAL_REQUIRED_METRICS

    def metadata(self, *, diagnostic_metric_keys: tuple[str, ...] = ()) -> dict[str, object]:
        return {
            "evaluation_engine": self.engine,
            "evaluation_primary_metric": self.primary_metric,
            "evaluation_primary_source": self.primary_source,
            "evaluation_required_metrics": list(self.required_metrics),
            "diagnostic_metric_keys": sorted(set(diagnostic_metric_keys)),
            "diagnostic_metric_warning": "Diagnostic metrics are not decision-grade unless confirmed by portfolio NAV.",
        }


DEFAULT_EVALUATION_POLICY = EvaluationPolicy()


def canonical_evaluation_metadata(*, diagnostic_metric_keys: tuple[str, ...] = ()) -> dict[str, object]:
    return DEFAULT_EVALUATION_POLICY.metadata(diagnostic_metric_keys=diagnostic_metric_keys)


def missing_canonical_metrics(summary: Mapping[str, object]) -> list[str]:
    return [key for key in CANONICAL_REQUIRED_METRICS if key not in summary]


def assert_canonical_evaluation_summary(summary: Mapping[str, object]) -> None:
    missing = missing_canonical_metrics(summary)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Summary is missing canonical portfolio evaluation metrics: {joined}")
    if summary.get("evaluation_engine") != CANONICAL_EVALUATION_ENGINE:
        raise ValueError(
            "Summary must declare evaluation_engine="
            f"{CANONICAL_EVALUATION_ENGINE!r} before it is used for strategy decisions."
        )
