# =====================================================
# tests/integration/test_backtest_edge_cases.py
#
# Edge-case integration tests for the backtest pipeline:
#   1. Negative-gap symbols must be excluded (gap-down screening)
#   2. Symbols with no intraday parquet must not crash the pipeline
#   3. Capital must always remain positive; statistics dict always present
# =====================================================

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from config.loader import build_config
from data_handler.local import LocalDataHandler
from run.backtest_engine import Backtester

# ---------------------------------------------------------------------------
# Symbols used across this module
# ---------------------------------------------------------------------------

TRADE_DATE   = date(2024, 3, 6)
PREV_DATE    = date(2024, 3, 5)

GAP_UP_SYMBOL   = "GAPPER"     # large gap-up → passes screening
GAP_DOWN_SYMBOL = "GAPDOWN"    # gap-down → must be excluded by screener
NO_INTRA_SYMBOL = "NOINTRA"    # gap-up but no intraday parquet → graceful skip


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_aggregate_parquet(path: Path) -> None:
    """Daily aggregate bars for all three test symbols."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        # GAP_UP: prev close=4.00, trade open=5.50 → +37.5 % gap
        {"symbol": GAP_UP_SYMBOL,   "date": PREV_DATE,  "open": 4.50, "high": 4.80, "low": 3.90,  "close": 4.00,  "volume": 200_000},
        {"symbol": GAP_UP_SYMBOL,   "date": TRADE_DATE, "open": 5.50, "high": 6.20, "low": 5.20,  "close": 5.90,  "volume": 800_000},
        # GAP_DOWN: prev close=10.00, trade open=8.00 → -20 % gap (should be excluded)
        {"symbol": GAP_DOWN_SYMBOL, "date": PREV_DATE,  "open": 10.20, "high": 10.50, "low": 9.80, "close": 10.00, "volume": 300_000},
        {"symbol": GAP_DOWN_SYMBOL, "date": TRADE_DATE, "open": 8.00, "high": 8.50, "low": 7.80,  "close": 8.20,  "volume": 600_000},
        # NO_INTRA: gap-up like GAPPER but we never create an intraday parquet for it
        {"symbol": NO_INTRA_SYMBOL, "date": PREV_DATE,  "open": 2.50, "high": 2.80, "low": 2.30,  "close": 2.00,  "volume": 150_000},
        {"symbol": NO_INTRA_SYMBOL, "date": TRADE_DATE, "open": 3.20, "high": 3.50, "low": 3.00,  "close": 3.30,  "volume": 700_000},
    ]
    pd.DataFrame(records).to_parquet(path, index=False)


def _make_intraday_parquet(path: Path, symbol: str) -> None:
    """120 synthetic 1-min bars for *symbol* on TRADE_DATE."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base_ts = pd.Timestamp("2024-03-06 09:00:00", tz="UTC")
    base_price = 5.50 if symbol == GAP_UP_SYMBOL else 3.20

    # Embed three clear waves so PatternAnalyzer produces a valid signal
    closes: list[float] = []
    closes += [base_price + i * 0.020 for i in range(8)]
    closes += [closes[-1] - 0.02, closes[-1] - 0.04]        # pullback 1
    closes += [closes[-1] + i * 0.025 for i in range(8)]    # wave 2
    closes += [closes[-1] - 0.02, closes[-1] - 0.04]        # pullback 2
    closes += [closes[-1] + i * 0.030 for i in range(8)]    # wave 3
    closes += [closes[-1] - 0.02, closes[-1] - 0.04]        # cool-down
    last = closes[-1]
    for _ in range(120 - len(closes)):
        last += 0.005
        closes.append(round(last, 4))

    rows = []
    for i, c in enumerate(closes):
        c = round(c, 4)
        rows.append({
            "symbol":    symbol,
            "timestamp": base_ts + pd.Timedelta(minutes=i),
            "open":      round(c - 0.005, 4),
            "high":      round(c + 0.005, 4),
            "low":       round(c - 0.010, 4),
            "close":     c,
            "volume":    30_000 + i * 100,
        })
    pd.DataFrame(rows).to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("bt_edge")

    _make_aggregate_parquet(root / "daily_aggregates" / "2024" / "2024-03.parquet")
    # Only GAPPER gets intraday data; NO_INTRA intentionally has none
    _make_intraday_parquet(root / "ticker_data" / "2024" / "03" / "2024-03-06.parquet", GAP_UP_SYMBOL)

    (root / "market_context").mkdir()
    (root / "news").mkdir()
    (root / "reports").mkdir()

    return root


@pytest.fixture(scope="module")
def bt_config(fixture_root):
    overrides = [
        f"backtest.DATA_DIR={fixture_root / 'ticker_data'}",
        f"backtest.BASE_DATA_DIR={fixture_root}",
        f"backtest.NEWS_DATA_DIR={fixture_root / 'news'}",
        f"market_context.CSV_DIR={fixture_root / 'market_context'}",
        f"system.REPORTS_DIR={fixture_root / 'reports'}",
        "backtest.START_DATE=2024-03-06",
        "backtest.END_DATE=2024-03-06",
        "backtest.INITIAL_CAPITAL=10000.0",
        "backtest.IGNORE_CATALYST=True",
        "screening.ENABLE_DAILY_VOLUME_PRESCREEN=False",
        "screening.ENABLE_RELATIVE_VOLUME=False",
        "screening.ENABLE_FLOAT_FILTER=False",
        "screening.ENABLE_MARKETCAP_FILTER=False",
        # Only positive gaps qualify; GAP_DOWN should be excluded
        "screening.MIN_GAP_PERCENT=20.0",
        "screening.MIN_PRICE=1.0",
        "system.USE_FILE_INDEX_CACHE=False",
    ]
    return build_config(cli_overrides=overrides, enable_env_layer=False)


@pytest.fixture(scope="module")
def backtest_results(bt_config, fixture_root):
    data_handler = LocalDataHandler(bt_config, data_dir=str(bt_config.backtest.DATA_DIR))
    backtester   = Backtester(bt_config, data_handler,
                              reports_dir=fixture_root / "reports")
    return backtester.run_backtest(
        datetime(2024, 3, 6),
        datetime(2024, 3, 6),
        initial_capital=10_000.0,
    )


# ---------------------------------------------------------------------------
# TestNegativeGapExcluded
# ---------------------------------------------------------------------------

class TestNegativeGapExcluded:
    def test_gap_down_not_in_pipeline(self, backtest_results):
        """GAP_DOWN_SYMBOL must never appear in any trade record."""
        traded_symbols = {t["symbol"] for t in backtest_results.get("trades", [])}
        assert GAP_DOWN_SYMBOL not in traded_symbols, (
            f"{GAP_DOWN_SYMBOL} appeared in trades despite being a gap-down symbol"
        )


# ---------------------------------------------------------------------------
# TestMissingIntradayData
# ---------------------------------------------------------------------------

class TestMissingIntradayData:
    def test_backtest_completes_without_error(self, backtest_results):
        """Pipeline must not raise even when intraday data is absent for a screened symbol."""
        assert isinstance(backtest_results, dict)

    def test_pipeline_result_has_required_keys(self, backtest_results):
        for key in ("trades", "statistics", "equity_curve", "capital"):
            assert key in backtest_results, f"Missing key: '{key}'"

    def test_no_intraday_symbol_not_traded(self, backtest_results):
        """NO_INTRA_SYMBOL may pass daily screening but must produce no trades."""
        traded_symbols = {t["symbol"] for t in backtest_results.get("trades", [])}
        assert NO_INTRA_SYMBOL not in traded_symbols, (
            f"{NO_INTRA_SYMBOL} should not have been traded (no intraday data)"
        )


# ---------------------------------------------------------------------------
# TestPipelineRobustness
# ---------------------------------------------------------------------------

class TestPipelineRobustness:
    def test_capital_always_positive(self, backtest_results):
        eq = backtest_results.get("equity_curve", [])
        if eq:
            # equity_curve entries may be dicts with a 'capital' key or plain scalars
            values = [v["equity"] if isinstance(v, dict) else v for v in eq]
            assert all(v >= 0 for v in values), "Equity curve went negative"
        assert backtest_results.get("capital", 0) >= 0

    def test_statistics_always_present(self, backtest_results):
        stats = backtest_results.get("statistics")
        assert stats is not None, "statistics key must always be present"
        assert isinstance(stats, dict)
