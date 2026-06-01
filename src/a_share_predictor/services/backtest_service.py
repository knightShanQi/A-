from __future__ import annotations

from pathlib import Path
from typing import Any

from ..market_backtest_runner import load_latest_full_market_backtest, run_full_market_backtest


class BacktestService:
    """Application boundary for market-wide portfolio backtests."""

    def run_market_backtest(
        self,
        *,
        date_from: str,
        date_to: str,
        horizon_days: int = 3,
        positive_return: float = 0.10,
        strategy_mode: str = "all",
        top_k: int = 50,
        output_dir: str | Path | None = None,
        force_rebuild: bool = False,
        max_workers: int = 8,
        fast_strategy_backtest: bool = False,
        persist_research: bool = True,
        duckdb_database: str | Path | None = None,
    ) -> dict[str, Any]:
        return run_full_market_backtest(
            date_from=date_from,
            date_to=date_to,
            horizon_days=horizon_days,
            positive_return=positive_return,
            strategy_mode=strategy_mode,
            top_k=top_k,
            output_dir=output_dir,
            force_rebuild=force_rebuild,
            max_workers=max_workers,
            fast_strategy_backtest=fast_strategy_backtest,
            persist_research=persist_research,
            duckdb_database=duckdb_database,
        )

    def load_latest_market_backtest(self, *, result_limit: int = 50) -> dict[str, Any]:
        return load_latest_full_market_backtest(result_limit=result_limit)
