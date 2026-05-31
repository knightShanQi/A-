import pandas as pd

import a_share_predictor.market_backtest_runner as runner


def test_strategy_mode_filters_old_and_new_labels():
    frame = pd.DataFrame(
        [
            {"candidate_strategy": "策略1"},
            {"candidate_strategy": "策略2"},
            {"candidate_strategy": "strategy3"},
            {"candidate_strategy": "dynamic_fallback"},
        ]
    )

    assert runner._strategy_mask(frame, runner._normalize_strategy_mode("old")).tolist() == [True, True, False, False]
    assert runner._strategy_mask(frame, runner._normalize_strategy_mode("new")).tolist() == [False, False, True, False]
