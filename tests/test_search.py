import pandas as pd

from a_share_predictor.dashboard import _resolve_search_candidate
from a_share_predictor.data import search_a_share_universe


def test_search_a_share_universe_matches_code_and_name():
    universe = pd.DataFrame(
        {
            "symbol": ["600519", "300750", "000333"],
            "name": ["贵州茅台", "宁德时代", "美的集团"],
            "name_normalized": ["贵州茅台", "宁德时代", "美的集团"],
        }
    )
    by_code = search_a_share_universe(universe, "600519")
    by_name = search_a_share_universe(universe, "宁德")
    assert by_code.iloc[0]["symbol"] == "600519"
    assert by_name.iloc[0]["symbol"] == "300750"


def test_search_a_share_universe_accepts_common_symbol_formats():
    universe = pd.DataFrame(
        {
            "symbol": ["600519", "300750", "000333"],
            "name": ["贵州茅台", "宁德时代", "美的集团"],
            "name_normalized": ["贵州茅台", "宁德时代", "美的集团"],
        }
    )

    by_prefixed_code = search_a_share_universe(universe, "sh600519")
    by_suffixed_code = search_a_share_universe(universe, "000333.SZ")

    assert by_prefixed_code.iloc[0]["symbol"] == "600519"
    assert by_suffixed_code.iloc[0]["symbol"] == "000333"


def test_search_a_share_universe_matches_full_width_name_variants():
    universe = pd.DataFrame(
        {
            "symbol": ["000725"],
            "name": ["京东方Ａ"],
            "name_normalized": ["京东方A"],
        }
    )

    by_ascii_name = search_a_share_universe(universe, "京东方A")

    assert by_ascii_name.iloc[0]["symbol"] == "000725"


def test_resolve_search_candidate_prefers_exact_symbol_or_name():
    universe = pd.DataFrame(
        {
            "symbol": ["600519", "300750", "000333"],
            "name": ["贵州茅台", "宁德时代", "美的集团"],
            "name_normalized": ["贵州茅台", "宁德时代", "美的集团"],
        }
    )

    matches, symbol_by_code, name_by_code = _resolve_search_candidate(universe, "600519.SH")
    _, symbol_by_name, name_by_name = _resolve_search_candidate(universe, "宁德时代")

    assert len(matches) == 1
    assert symbol_by_code == "600519"
    assert name_by_code == "贵州茅台"
    assert symbol_by_name == "300750"
    assert name_by_name == "宁德时代"


def test_resolve_search_candidate_does_not_return_unknown_symbol():
    universe = pd.DataFrame(
        {
            "symbol": ["600519", "300750", "000333"],
            "name": ["贵州茅台", "宁德时代", "美的集团"],
            "name_normalized": ["贵州茅台", "宁德时代", "美的集团"],
        }
    )

    matches, symbol, name = _resolve_search_candidate(universe, "999999.SH")

    assert matches.empty
    assert symbol is None
    assert name is None
