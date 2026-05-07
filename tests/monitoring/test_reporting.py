"""
Tests for monitoring/reporting.py

Covers:
- _max_drawdown: empty list, monotonic growth, single drop, full recovery
- _sharpe_ratio: empty, flat returns (SD=0 → 0), positive returns
- _sortino_ratio: no downside (returns 0), mixed returns
- _streaks: empty, all wins, all losses, alternating, complex sequence
- _by_exit_reason: counts, win_rate, avg_pnl per reason
- compute_statistics: empty trades, single trade, multiple trades with full fields
- generate_text_report: all expected sections present in output string
"""

import pytest
import math
from datetime import datetime

from monitoring.reporting import (
    _max_drawdown,
    _sharpe_ratio,
    _sortino_ratio,
    _streaks,
    _by_exit_reason,
    compute_statistics,
    generate_text_report,
)


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_empty_returns_zero(self):
        result = _max_drawdown([])
        assert result['max_drawdown'] == 0.0
        assert result['max_dd_duration'] == 0

    def test_monotonic_growth_no_drawdown(self):
        equity = [100, 110, 120, 130, 140]
        result = _max_drawdown(equity)
        assert result['max_drawdown'] == pytest.approx(0.0)

    def test_single_50pct_drop(self):
        equity = [200, 100]
        result = _max_drawdown(equity)
        assert result['max_drawdown'] == pytest.approx(50.0)

    def test_recovery_measures_full_peak_to_trough(self):
        equity = [100, 150, 75, 200]
        result = _max_drawdown(equity)
        # Peak=150, trough=75 → dd = 50%
        assert result['max_drawdown'] == pytest.approx(50.0)

    def test_duration_reflects_bars_below_peak(self):
        equity = [100, 200, 150, 100, 200]
        result = _max_drawdown(equity)
        assert result['max_dd_duration'] >= 2

    def test_single_value_no_drawdown(self):
        result = _max_drawdown([100.0])
        assert result['max_drawdown'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_empty_returns_zero(self):
        assert _sharpe_ratio([]) == 0.0

    def test_none_returns_zero(self):
        assert _sharpe_ratio(None) == 0.0

    def test_flat_returns_zero_sd_returns_zero(self):
        # All same → SD=0 → Sharpe=0
        result = _sharpe_ratio([0.01, 0.01, 0.01])
        assert result == pytest.approx(0.0)

    def test_positive_mean_positive_variance_returns_positive(self):
        returns = [0.01, 0.02, 0.015, 0.025, 0.01]
        result = _sharpe_ratio(returns)
        assert result > 0.0

    def test_negative_mean_returns_negative(self):
        returns = [-0.01, -0.02, -0.015]
        result = _sharpe_ratio(returns)
        assert result < 0.0

    def test_annualization_factor_applied(self):
        # Sharpe = sqrt(252) * mean / std
        import numpy as np
        returns = [0.01, 0.02, 0.015, -0.005, 0.025]
        r = [float(x) for x in returns]
        expected = float(math.sqrt(252) * sum(r) / len(r) / (sum((x - sum(r)/len(r))**2 for x in r) / len(r))**0.5)
        result = _sharpe_ratio(returns)
        assert result == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# _sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_empty_returns_zero(self):
        assert _sortino_ratio([]) == 0.0

    def test_none_returns_zero(self):
        assert _sortino_ratio(None) == 0.0

    def test_all_positive_no_downside_returns_zero(self):
        # No downside → downside_std=0 → Sortino=0
        result = _sortino_ratio([0.01, 0.02, 0.015])
        assert result == pytest.approx(0.0)

    def test_mixed_returns_positive_sortino_when_positive_mean(self):
        returns = [0.02, -0.01, 0.03, -0.005, 0.025]
        result = _sortino_ratio(returns)
        assert result > 0.0

    def test_all_negative_returns_negative_sortino(self):
        returns = [-0.01, -0.02, -0.015]
        result = _sortino_ratio(returns)
        assert result < 0.0


# ---------------------------------------------------------------------------
# _streaks
# ---------------------------------------------------------------------------

class TestStreaks:
    def test_empty_trades_returns_zeros(self):
        assert _streaks([]) == (0, 0)

    def test_all_wins_returns_correct_max_win(self):
        trades = [{'pnl': 100, 'exit_date': datetime(2024, 1, i+1)} for i in range(5)]
        max_win, max_lose = _streaks(trades)
        assert max_win == 5
        assert max_lose == 0

    def test_all_losses_returns_correct_max_lose(self):
        trades = [{'pnl': -100, 'exit_date': datetime(2024, 1, i+1)} for i in range(4)]
        max_win, max_lose = _streaks(trades)
        assert max_win == 0
        assert max_lose == 4

    def test_alternating_max_streak_is_1(self):
        trades = []
        for i in range(6):
            pnl = 100 if i % 2 == 0 else -100
            trades.append({'pnl': pnl, 'exit_date': datetime(2024, 1, i+1)})
        max_win, max_lose = _streaks(trades)
        assert max_win == 1
        assert max_lose == 1

    def test_complex_sequence(self):
        # W W W L W W L L L → max_win=3, max_lose=3
        pnls = [100, 100, 100, -100, 100, 100, -100, -100, -100]
        trades = [
            {'pnl': p, 'exit_date': datetime(2024, 1, i+1)}
            for i, p in enumerate(pnls)
        ]
        max_win, max_lose = _streaks(trades)
        assert max_win == 3
        assert max_lose == 3

    def test_single_win(self):
        trades = [{'pnl': 50, 'exit_date': datetime(2024, 1, 1)}]
        max_win, max_lose = _streaks(trades)
        assert max_win == 1
        assert max_lose == 0


# ---------------------------------------------------------------------------
# _by_exit_reason
# ---------------------------------------------------------------------------

class TestByExitReason:
    def test_counts_correctly(self):
        trades = [
            {'exit_reason': 'stop_loss', 'pnl': -100},
            {'exit_reason': 'stop_loss', 'pnl': -50},
            {'exit_reason': 'trading_window_close', 'pnl': 200},
        ]
        result = _by_exit_reason(trades)
        assert result['stop_loss']['count'] == 2
        assert result['trading_window_close']['count'] == 1

    def test_win_rate_computed_correctly(self):
        trades = [
            {'exit_reason': 'trailing_stop', 'pnl': 100},
            {'exit_reason': 'trailing_stop', 'pnl': -50},
            {'exit_reason': 'trailing_stop', 'pnl': 200},
        ]
        result = _by_exit_reason(trades)
        assert result['trailing_stop']['win_rate'] == pytest.approx(2 / 3 * 100, rel=0.01)

    def test_avg_pnl_computed_correctly(self):
        trades = [
            {'exit_reason': 'stop_loss', 'pnl': -100},
            {'exit_reason': 'stop_loss', 'pnl': -200},
        ]
        result = _by_exit_reason(trades)
        assert result['stop_loss']['avg_pnl'] == pytest.approx(-150.0)

    def test_unknown_reason_handled(self):
        trades = [{'pnl': 100}]  # no exit_reason key
        result = _by_exit_reason(trades)
        assert 'unknown' in result

    def test_empty_trades_returns_empty_dict(self):
        assert _by_exit_reason([]) == {}


# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------

class TestComputeStatistics:
    def _make_trade(self, pnl, exit_reason='trading_window_close', return_pct=None,
                    entry_date=None, exit_date=None):
        if return_pct is None:
            return_pct = pnl / 100.0
        if entry_date is None:
            entry_date = datetime(2024, 1, 2, 9, 35)
        if exit_date is None:
            exit_date = datetime(2024, 1, 2, 10, 0)
        return {
            'symbol': 'TEST',
            'pnl': pnl,
            'return_pct': return_pct,
            'exit_reason': exit_reason,
            'entry_date': entry_date,
            'exit_date': exit_date,
        }

    def test_empty_trades_no_crash(self):
        result = compute_statistics([])
        assert result['total_trades'] == 0
        assert result['win_rate'] == pytest.approx(0.0)
        assert result['net_profit'] == pytest.approx(0.0)

    def test_single_winning_trade(self):
        trades = [self._make_trade(100.0)]
        result = compute_statistics(trades)
        assert result['total_trades'] == 1
        assert result['winning_trades'] == 1
        assert result['losing_trades'] == 0
        assert result['win_rate'] == pytest.approx(100.0)
        assert result['net_profit'] == pytest.approx(100.0)

    def test_single_losing_trade(self):
        trades = [self._make_trade(-50.0)]
        result = compute_statistics(trades)
        assert result['winning_trades'] == 0
        assert result['losing_trades'] == 1
        assert result['win_rate'] == pytest.approx(0.0)

    def test_50pct_win_rate(self):
        trades = [self._make_trade(100), self._make_trade(-100)]
        result = compute_statistics(trades)
        assert result['win_rate'] == pytest.approx(50.0)

    def test_profit_factor_calculated(self):
        trades = [self._make_trade(200), self._make_trade(-100)]
        result = compute_statistics(trades)
        assert result['profit_factor'] == pytest.approx(2.0)

    def test_profit_factor_infinite_when_no_losses(self):
        trades = [self._make_trade(100), self._make_trade(200)]
        result = compute_statistics(trades)
        assert result['profit_factor'] == float('inf')

    def test_max_drawdown_computed_from_equity_curve(self):
        equity_curve = [
            {'date': '2024-01-01', 'equity': 100_000},
            {'date': '2024-01-02', 'equity': 110_000},
            {'date': '2024-01-03', 'equity': 90_000},   # 18.2% drawdown from 110k
        ]
        result = compute_statistics(
            [], equity_curve=equity_curve,
            initial_capital=100_000, final_capital=90_000
        )
        assert result['max_drawdown'] > 0

    def test_sharpe_computed_from_daily_returns(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.01]
        result = compute_statistics([], daily_returns=returns)
        assert isinstance(result['sharpe_ratio'], float)

    def test_best_and_worst_trade_identified(self):
        trades = [
            self._make_trade(500, 'trailing_stop'),
            self._make_trade(-200, 'stop_loss'),
            self._make_trade(100, 'trading_window_close'),
        ]
        result = compute_statistics(trades)
        assert result['best_trade']['pnl'] == pytest.approx(500.0)
        assert result['worst_trade']['pnl'] == pytest.approx(-200.0)

    def test_exit_reasons_breakdown_present(self):
        trades = [
            self._make_trade(100, 'trailing_stop'),
            self._make_trade(-50, 'stop_loss'),
        ]
        result = compute_statistics(trades)
        assert 'exit_reasons' in result
        assert 'trailing_stop' in result['exit_reasons']
        assert 'stop_loss' in result['exit_reasons']

    def test_annualized_return_nonzero_with_trading_days(self):
        result = compute_statistics(
            [],
            initial_capital=100_000,
            final_capital=110_000,
            trading_days=252,
        )
        assert result['annualized_return'] != pytest.approx(0.0)

    def test_all_required_keys_present(self):
        result = compute_statistics([])
        required = [
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'gross_profit', 'gross_loss', 'net_profit', 'avg_win', 'avg_loss',
            'profit_factor', 'expectancy', 'total_return', 'max_drawdown',
            'sharpe_ratio', 'exit_reasons',
        ]
        for key in required:
            assert key in result, f"Missing: {key}"


# ---------------------------------------------------------------------------
# generate_text_report
# ---------------------------------------------------------------------------

class TestGenerateTextReport:
    def _make_stats(self):
        return {
            'initial_capital': 100_000,
            'final_capital': 110_000,
            'total_return': 10.0,
            'annualized_return': 10.0,
            'max_drawdown': 5.0,
            'total_trades': 20,
            'win_rate': 60.0,
            'profit_factor': 1.8,
            'net_profit': 10_000,
            'avg_win': 800,
            'avg_loss': 400,
            'avg_hold_time_minutes': 45.0,
            'max_hold_time_minutes': 90.0,
            'min_hold_time_minutes': 5.0,
            'exit_reasons': {
                'stop_loss': {'count': 8, 'win_rate': 0.0, 'avg_pnl': -400.0},
                'trailing_stop': {'count': 12, 'win_rate': 100.0, 'avg_pnl': 800.0},
            },
        }

    def test_report_contains_title(self):
        text = generate_text_report(self._make_stats(), title="TEST REPORT")
        assert "TEST REPORT" in text

    def test_report_contains_capital_figures(self):
        text = generate_text_report(self._make_stats())
        assert "100,000" in text or "100000" in text

    def test_report_contains_trade_stats(self):
        text = generate_text_report(self._make_stats())
        assert "Win Rate" in text
        assert "Profit Factor" in text

    def test_report_contains_hold_times(self):
        text = generate_text_report(self._make_stats())
        assert "Hold" in text

    def test_report_contains_exit_reasons(self):
        text = generate_text_report(self._make_stats())
        assert "stop_loss" in text
        assert "trailing_stop" in text

    def test_report_is_string(self):
        result = generate_text_report(self._make_stats())
        assert isinstance(result, str)

    def test_report_with_empty_stats_no_crash(self):
        text = generate_text_report({})
        assert isinstance(text, str)
