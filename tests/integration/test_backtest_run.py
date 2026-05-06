# =====================================================
# tests/integration/test_backtest_run.py
#
# Integration test: full backtest run on minimal synthetic fixture data.
# Exercises the complete
#   Backtester → BacktestScreener → UnifiedScreener → PatternAnalyzer → TradeSimulator
# pipeline without requiring real market data files.
# =====================================================

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from config.loader import build_config
from data_handler.local import LocalDataHandler
from run.backtest_engine import Backtester

# ── Constants ─────────────────────────────────────────────────────────────────

TRADE_DATE  = date(2024, 1, 3)   # Wednesday — not a weekend
PREV_DATE   = date(2024, 1, 2)   # Tuesday  — previous trading day
GAP_SYMBOL  = "GAPPER"           # 37.5 % gap-up — easily clears MIN_GAP_PERCENT=20
FLAT_SYMBOL = "FLAT"             # ~0.5 % gap — stays below threshold


# ── Fixture data builders ──────────────────────────────────────────────────────

def _make_aggregate_parquet(path: Path) -> None:
    """Write minimal daily_aggregates/2024/2024-01.parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        # GAPPER prev day: close = 4.00
        {"symbol": GAP_SYMBOL,  "date": PREV_DATE,  "open": 4.50, "high": 4.80, "low": 3.90,  "close": 4.00,  "volume": 200_000},
        # GAPPER trade day: open = 5.50 → gap = (5.50-4.00)/4.00*100 = 37.5 %
        {"symbol": GAP_SYMBOL,  "date": TRADE_DATE, "open": 5.50, "high": 6.20, "low": 5.20,  "close": 5.90,  "volume": 800_000},
        # FLAT: < 1 % gap → should not survive MIN_GAP_PERCENT filter
        {"symbol": FLAT_SYMBOL, "date": PREV_DATE,  "open": 10.00, "high": 10.50, "low": 9.80, "close": 10.00, "volume": 300_000},
        {"symbol": FLAT_SYMBOL, "date": TRADE_DATE, "open": 10.05, "high": 10.20, "low": 9.95, "close": 10.10, "volume": 280_000},
    ]
    pd.DataFrame(records).to_parquet(path, index=False)


def _make_intraday_parquet(path: Path) -> None:
    """
    Write 120 minutes of synthetic GAPPER 1-min bars (04:00 – 06:00 ET).

    Three clear step-up waves are embedded in the first 30 bars to ensure
    PatternAnalyzer.analyze_pattern() detects step_count >= 2 and
    retention_rate >= 35 %, producing a valid pattern signal.

    Wave design (close prices):
        Wave 1  bars  0-7  : 5.50 → 5.64  (+0.020/bar)
        Pull 1  bars  8-9  : 5.62, 5.60
        Wave 2  bars 10-17 : 5.61 → 5.785 (+0.025/bar)   ← higher high ✓
        Pull 2  bars 18-19 : 5.76, 5.74
        Wave 3  bars 20-27 : 5.75 → 5.96  (+0.030/bar)   ← higher high ✓
        Cool    bars 28-29 : 5.94, 5.92   (so bar 27 is a local high)
        Drift   bars 30-119: +0.005/bar continuation
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # 2024-01-03 in January (EST = UTC-5):  04:00 ET = 09:00 UTC
    base_ts = pd.Timestamp("2024-01-03 09:00:00", tz="UTC")
    n_bars  = 120

    closes: list[float] = []
    closes += [5.50 + i * 0.020 for i in range(8)]   # wave 1
    closes += [5.62, 5.60]                             # pullback 1
    closes += [5.61 + i * 0.025 for i in range(8)]   # wave 2
    closes += [5.76, 5.74]                             # pullback 2
    closes += [5.75 + i * 0.030 for i in range(8)]   # wave 3
    closes += [5.94, 5.92]                             # cool-down
    last = closes[-1]
    for _ in range(n_bars - 30):
        last += 0.005
        closes.append(round(last, 4))

    assert len(closes) == n_bars

    rows = []
    for i, c in enumerate(closes):
        c = round(c, 4)
        rows.append({
            "symbol":    GAP_SYMBOL,
            "timestamp": base_ts + pd.Timedelta(minutes=i),
            "open":      round(c - 0.005, 4),
            "high":      round(c + 0.005, 4),
            "low":       round(c - 0.010, 4),
            "close":     c,
            "volume":    30_000 + i * 100,   # gently increasing volume
        })

    pd.DataFrame(rows).to_parquet(path, index=False)


# ── Pytest fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    """Create all fixture data files once per test module; return root Path."""
    root = tmp_path_factory.mktemp("bt_integration")

    _make_aggregate_parquet(root / "daily_aggregates" / "2024" / "2024-01.parquet")
    _make_intraday_parquet(root / "ticker_data" / "2024" / "01" / "2024-01-03.parquet")

    # These directories must exist; empty is fine (BacktestMarketContext logs warning but proceeds)
    (root / "market_context").mkdir()
    (root / "news").mkdir()
    (root / "reports").mkdir()

    return root


@pytest.fixture(scope="module")
def bt_config(fixture_root):
    """Build a minimal backtest config pointing entirely at fixture_root."""
    overrides = [
        f"backtest.DATA_DIR={fixture_root / 'ticker_data'}",
        f"backtest.BASE_DATA_DIR={fixture_root}",
        f"backtest.NEWS_DATA_DIR={fixture_root / 'news'}",
        f"market_context.CSV_DIR={fixture_root / 'market_context'}",
        f"system.REPORTS_DIR={fixture_root / 'reports'}",
        "backtest.START_DATE=2024-01-03",
        "backtest.END_DATE=2024-01-03",
        "backtest.INITIAL_CAPITAL=10000.0",
        "backtest.IGNORE_CATALYST=True",
        # Disable all filters that need real aggregate columns not in our fixture
        "screening.ENABLE_DAILY_VOLUME_PRESCREEN=False",
        "screening.ENABLE_RELATIVE_VOLUME=False",
        "screening.ENABLE_FLOAT_FILTER=False",
        "screening.ENABLE_MARKETCAP_FILTER=False",
        # Ensure only GAPPER qualifies (37.5 % gap > threshold; FLAT ~0.5 % does not)
        "screening.MIN_GAP_PERCENT=20.0",
        "screening.MIN_PRICE=1.0",
        "system.USE_FILE_INDEX_CACHE=False",
    ]
    return build_config(cli_overrides=overrides, enable_env_layer=False)


@pytest.fixture(scope="module")
def backtest_results(bt_config, fixture_root):
    """Run the backtest once; share results across all tests in this module."""
    data_handler = LocalDataHandler(bt_config, data_dir=str(bt_config.backtest.DATA_DIR))
    backtester   = Backtester(bt_config, data_handler,
                              reports_dir=fixture_root / "reports")
    return backtester.run_backtest(
        datetime(2024, 1, 3),
        datetime(2024, 1, 3),
        initial_capital=10_000.0,
    )


# ── Test classes ───────────────────────────────────────────────────────────────

class TestBacktestResultsStructure:
    """Smoke tests: the results dict has the expected shape and types."""

    def test_returns_dict(self, backtest_results):
        assert isinstance(backtest_results, dict)

    def test_has_required_keys(self, backtest_results):
        for key in ("trades", "statistics", "equity_curve", "capital"):
            assert key in backtest_results, f"Missing key: '{key}'"

    def test_trades_is_list(self, backtest_results):
        assert isinstance(backtest_results["trades"], list)

    def test_statistics_is_dict(self, backtest_results):
        assert isinstance(backtest_results["statistics"], dict)

    def test_statistics_has_standard_keys(self, backtest_results):
        stats = backtest_results["statistics"]
        for key in ("total_trades", "win_rate", "net_profit"):
            assert key in stats, f"Missing statistics key: '{key}'"

    def test_equity_curve_non_empty(self, backtest_results):
        # One entry is appended per trading day processed
        assert len(backtest_results["equity_curve"]) >= 1

    def test_capital_is_positive(self, backtest_results):
        assert backtest_results["capital"] > 0


class TestGapScreening:
    """Verifies the gap-screener processes GAPPER and ignores FLAT."""

    def _all_symbols(self, results: dict) -> set[str]:
        """Union of symbols that appeared in diagnostics or trades."""
        diag_symbols  = {d["symbol"] for d in results.get("candidate_diagnostics", [])}
        trade_symbols = {t["symbol"] for t in results.get("trades", [])}
        return diag_symbols | trade_symbols

    def test_gapper_processed_by_pipeline(self, backtest_results):
        """GAPPER must appear in diagnostics (rejected) or trades (entered)."""
        all_syms = self._all_symbols(backtest_results)
        assert GAP_SYMBOL in all_syms, (
            f"GAPPER was never processed — check aggregate fixture. "
            f"Symbols seen: {all_syms}"
        )

    def test_flat_excluded_by_gap_filter(self, backtest_results):
        """FLAT's ~0.5 % gap is below MIN_GAP_PERCENT=20 %; it must not appear anywhere."""
        all_syms = self._all_symbols(backtest_results)
        assert FLAT_SYMBOL not in all_syms, (
            "FLAT should be filtered out before reaching the screening pipeline."
        )
