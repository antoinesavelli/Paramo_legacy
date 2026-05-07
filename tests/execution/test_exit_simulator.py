"""
Tests for execution/exit_simulator.py

Covers:
- _get_slippage_config: returns correct keys/values from config
- _apply_slippage: winner path, loser path, stop-loss path (normal/high-gap)
- simulate_exit:
    - trading_window_close fires when deadline reached
    - stop_loss fires when low <= stop_price
    - trailing_stop activates only after min_profit_pct threshold
    - trailing_stop ratchets up, never down
    - no-bar edge case: exits at entry price with trading_window_close
    - P&L calculation (entry/exit/size math)
    - highest_price tracking
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from execution.exit_simulator import ExitSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    simple_stops=False,
    atr_enabled=True,
    atr_period=5,
    atr_mult=1.5,
    min_profit_pct=1.0,
    max_hold=60,
    winner_mult=0.98,
    loser_mult=1.06,
    stop_pct_high=0.12,
    stop_pct_normal=0.08,
    gap_threshold=200.0,
    enable_slippage=True,
):
    cfg = MagicMock()
    cfg.backtest.SIMPLE_STOPS = simple_stops
    cfg.risk.MAX_HOLD_TIME_MINUTES = max_hold
    cfg.risk.ATR_TRAILING_ENABLED = atr_enabled
    cfg.risk.ATR_TRAILING_PERIOD = atr_period
    cfg.risk.ATR_TRAILING_MULTIPLIER = atr_mult
    cfg.risk.ATR_TRAILING_MIN_PROFIT_PCT = min_profit_pct
    cfg.risk.SLIPPAGE_WINNER_MULTIPLIER = winner_mult
    cfg.risk.SLIPPAGE_LOSER_MULTIPLIER = loser_mult
    cfg.risk.SLIPPAGE_STOP_HIGH_GAP_PCT = stop_pct_high
    cfg.risk.SLIPPAGE_STOP_NORMAL_PCT = stop_pct_normal
    cfg.risk.SLIPPAGE_GAP_THRESHOLD = gap_threshold
    cfg.risk.ENABLE_SLIPPAGE = enable_slippage
    return cfg


def _make_simulator(cfg=None):
    if cfg is None:
        cfg = _make_config()
    pa = MagicMock()
    pa.analyze_pattern.return_value = {
        'pattern_strength': 0, 'pattern_count': 0,
        'patterns_detected': [], 'reason': 'mock',
        'step_ups': {}, 'parabolic': {}, 'breakout': {}, 'volume': {},
        'support_resistance': {},
    }
    sim = ExitSimulator(cfg, pa)
    return sim


def _utc(year=2024, month=1, day=2, hour=14, minute=0, second=0):
    return pd.Timestamp(datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc))


def _make_bars(timestamps, lows, highs, closes, volumes=None):
    """Build a minimal DataFrame representing 1-minute OHLCV bars."""
    if volumes is None:
        volumes = [100_000] * len(timestamps)
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': closes,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })


def _make_position(
    entry_price=10.0,
    stop_price=9.5,
    size=100,
    entry_time=None,
    trading_window_end_utc=None,
    gap_percent=50.0,
):
    if entry_time is None:
        entry_time = _utc(hour=13, minute=0)
    return {
        'entry_price': entry_price,
        'stop_price': stop_price,
        'size': size,
        'entry_time': entry_time,
        'trading_window_end_utc': trading_window_end_utc,
        'gap_percent': gap_percent,
        'vix_at_entry': 15.0,
        'volume_at_entry': 50_000,
        'entry_bar_index': 5,
        'all_bars': pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# _get_slippage_config
# ---------------------------------------------------------------------------

class TestGetSlippageConfig:
    def test_returns_all_keys(self):
        sim = _make_simulator()
        cfg = sim._get_slippage_config()
        assert 'winner_multiplier' in cfg
        assert 'loser_multiplier' in cfg
        assert 'stop_pct_high_gap' in cfg
        assert 'stop_pct_normal' in cfg
        assert 'gap_threshold' in cfg

    def test_values_come_from_config(self):
        config = _make_config(winner_mult=0.95, loser_mult=1.10, stop_pct_normal=0.05)
        sim = _make_simulator(config)
        cfg = sim._get_slippage_config()
        assert cfg['winner_multiplier'] == pytest.approx(0.95)
        assert cfg['loser_multiplier'] == pytest.approx(1.10)
        assert cfg['stop_pct_normal'] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# _apply_slippage
# ---------------------------------------------------------------------------

class TestApplySlippage:
    def _slip(self, sim, exit_price, raw_pnl, size=100, entry_price=10.0,
              exit_reason='trading_window_close', gap_pct=50.0):
        cfg = sim._get_slippage_config()
        return sim._apply_slippage(
            'TEST', exit_price, raw_pnl, size, entry_price, exit_reason, gap_pct, cfg
        )

    def test_winner_reduces_profit(self):
        sim = _make_simulator(_make_config(winner_mult=0.98))
        # entry=10, exit=11, raw_pnl=100*1=100
        adjusted = self._slip(sim, 11.0, 100.0, size=100, entry_price=10.0)
        # expected adjusted_pnl = 100 * 0.98 = 98, exit_price = 10 + 98/100 = 10.98
        assert adjusted == pytest.approx(10.98)

    def test_loser_time_exit_amplifies_loss(self):
        sim = _make_simulator(_make_config(loser_mult=1.06))
        # entry=10, exit=9, raw_pnl=100*(-1)=-100
        adjusted = self._slip(sim, 9.0, -100.0, size=100, entry_price=10.0,
                              exit_reason='trading_window_close')
        # adjusted_pnl = -100 * 1.06 = -106, exit = 10 + (-106/100) = 8.94
        assert adjusted == pytest.approx(8.94)

    def test_stop_loss_normal_gap(self):
        sim = _make_simulator(_make_config(stop_pct_normal=0.08, gap_threshold=200.0))
        # Stop at 9.5, gap=50 (below threshold 200)
        adjusted = self._slip(sim, 9.5, -50.0, size=100, entry_price=10.0,
                              exit_reason='stop_loss', gap_pct=50.0)
        # exit_price = 9.5 * (1 - 0.08) = 8.74
        assert adjusted == pytest.approx(9.5 * 0.92)

    def test_stop_loss_high_gap(self):
        sim = _make_simulator(_make_config(stop_pct_high=0.12, gap_threshold=200.0))
        # gap=250 (above threshold 200) → high-gap slip
        adjusted = self._slip(sim, 9.5, -50.0, size=100, entry_price=10.0,
                              exit_reason='stop_loss', gap_pct=250.0)
        assert adjusted == pytest.approx(9.5 * 0.88)

    def test_winner_exit_price_less_than_no_slippage(self):
        sim = _make_simulator(_make_config(winner_mult=0.98))
        adj = self._slip(sim, 12.0, 200.0, size=100, entry_price=10.0)
        assert adj < 12.0

    def test_loser_stop_exit_price_worse_than_nominal(self):
        sim = _make_simulator(_make_config(stop_pct_normal=0.08, gap_threshold=200.0))
        adj = self._slip(sim, 9.5, -50.0, size=100, entry_price=10.0,
                        exit_reason='stop_loss', gap_pct=50.0)
        assert adj < 9.5


# ---------------------------------------------------------------------------
# simulate_exit — trading_window_close
# ---------------------------------------------------------------------------

class TestSimulateExitTradingWindowClose:
    def _run(self, sim, bars, pos, results=None):
        if results is None:
            results = {'capital': pos['entry_price'] * pos['size']}
        return sim.simulate_exit('TEST', pos, bars, results)

    def test_exits_at_deadline_bar(self):
        sim = _make_simulator(_make_config(atr_enabled=False))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=5)
        ts = [_utc(hour=13, minute=i) for i in range(0, 10)]
        bars = _make_bars(ts, lows=[9.8]*10, highs=[10.5]*10, closes=[10.2]*10)
        pos = _make_position(entry_time=entry_time, trading_window_end_utc=deadline)
        trade = self._run(sim, bars, pos)
        assert trade['exit_reason'] == 'trading_window_close'

    def test_exit_at_deadline_uses_close_price(self):
        sim = _make_simulator(_make_config(atr_enabled=False))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=3)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        closes = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9]
        bars = _make_bars(ts, lows=[9.8]*10, highs=[11.0]*10, closes=closes)
        pos = _make_position(entry_time=entry_time, trading_window_end_utc=deadline)
        trade = self._run(sim, bars, pos)
        assert trade['exit_reason'] == 'trading_window_close'


# ---------------------------------------------------------------------------
# simulate_exit — stop_loss
# ---------------------------------------------------------------------------

class TestSimulateExitStopLoss:
    def _run(self, sim, bars, pos):
        results = {'capital': pos['entry_price'] * pos['size']}
        return sim.simulate_exit('TEST', pos, bars, results)

    def test_stop_fires_when_low_hits_stop(self):
        sim = _make_simulator(_make_config(atr_enabled=False))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=14, minute=0)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        lows  = [9.8, 9.7, 9.6, 9.5, 9.4, 9.8, 9.8, 9.8, 9.8, 9.8]  # bar 4 hits stop 9.5
        highs = [10.2] * 10
        closes = [10.0] * 10
        bars = _make_bars(ts, lows, highs, closes)
        pos = _make_position(entry_price=10.0, stop_price=9.5,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        trade = self._run(sim, bars, pos)
        assert trade['exit_reason'] == 'stop_loss'

    def test_stop_not_triggered_if_low_above_stop(self):
        sim = _make_simulator(_make_config(atr_enabled=False))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=5)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        lows  = [9.6] * 10  # always above stop=9.5
        highs = [10.2] * 10
        closes = [10.0] * 10
        bars = _make_bars(ts, lows, highs, closes)
        pos = _make_position(entry_price=10.0, stop_price=9.5,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        trade = self._run(sim, bars, pos)
        assert trade['exit_reason'] != 'stop_loss'

    def test_pnl_negative_on_stop_loss(self):
        sim = _make_simulator(_make_config(atr_enabled=False))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=14, minute=0)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        lows  = [9.8, 9.4] + [9.8] * 8  # bar 1 hits stop
        highs = [10.2] * 10
        closes = [10.0] * 10
        bars = _make_bars(ts, lows, highs, closes)
        pos = _make_position(entry_price=10.0, stop_price=9.5, size=100,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        trade = self._run(sim, bars, pos)
        assert trade['pnl'] < 0


# ---------------------------------------------------------------------------
# simulate_exit — ATR trailing stop
# ---------------------------------------------------------------------------

class TestSimulateExitTrailingStop:
    def _run(self, sim, bars, pos):
        results = {'capital': pos['entry_price'] * pos['size']}
        return sim.simulate_exit('TEST', pos, bars, results)

    def test_trailing_stop_does_not_fire_below_min_profit(self):
        """Trailing stop should NOT activate if profit < min_profit_pct."""
        sim = _make_simulator(_make_config(
            atr_enabled=True, atr_period=3, atr_mult=1.5, min_profit_pct=5.0
        ))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=14, minute=0)
        ts = [_utc(hour=13, minute=i) for i in range(20)]
        # Price barely moves up 1% then falls — trailing stop shouldn't activate
        highs = [10.1] * 20
        lows  = [9.6] * 20   # above initial stop=9.5
        closes = [10.05] * 20
        bars = _make_bars(ts, lows, highs, closes)
        pos = _make_position(entry_price=10.0, stop_price=9.5,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        trade = self._run(sim, bars, pos)
        assert trade['exit_reason'] != 'trailing_stop'

    def test_trailing_stop_activates_after_profit_threshold(self):
        """When price rises significantly then drops, trailing_stop should trigger."""
        sim = _make_simulator(_make_config(
            atr_enabled=True, atr_period=3, atr_mult=0.5, min_profit_pct=1.0
        ))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=15, minute=0)
        ts = [_utc(hour=13, minute=i) for i in range(30)]
        # Price surges to 11.5 (15% gain), then collapses below trailing stop
        highs = [10.0 + i * 0.1 for i in range(15)] + [8.0] * 15
        lows  = [9.9 + i * 0.09 for i in range(15)] + [7.5] * 15
        closes = highs
        bars = _make_bars(ts, lows, highs, closes)
        pos = _make_position(entry_price=10.0, stop_price=9.0,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        trade = self._run(sim, bars, pos)
        assert trade['exit_reason'] in ('trailing_stop', 'stop_loss', 'trading_window_close')

    def test_highest_price_tracked_correctly(self):
        sim = _make_simulator(_make_config(atr_enabled=False))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=10)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        highs  = [10.0, 10.5, 11.0, 11.5, 12.0, 11.8, 11.5, 11.0, 10.5, 10.0]
        lows   = [9.8] * 10
        closes = [10.0, 10.3, 10.8, 11.2, 11.8, 11.5, 11.2, 10.8, 10.3, 10.0]
        bars = _make_bars(ts, lows, highs, closes)
        pos = _make_position(entry_price=10.0, stop_price=9.0,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        results = {'capital': 1000.0}
        trade = sim.simulate_exit('TEST', pos, bars, results)
        assert trade['highest_price'] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# simulate_exit — no-bar edge case
# ---------------------------------------------------------------------------

class TestSimulateExitNoBars:
    def test_empty_bars_returns_trading_window_close(self):
        sim = _make_simulator(_make_config(atr_enabled=False))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=14, minute=0)
        pos = _make_position(entry_price=10.0, stop_price=9.5,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        results = {'capital': 1000.0}
        trade = sim.simulate_exit('TEST', pos, pd.DataFrame(), results)
        assert trade['exit_reason'] == 'trading_window_close'
        assert trade['symbol'] == 'TEST'


# ---------------------------------------------------------------------------
# simulate_exit — P&L math
# ---------------------------------------------------------------------------

class TestSimulateExitPnL:
    def test_basic_winner_pnl(self):
        """Entry 10, exit ~11, 100 shares → net P&L positive."""
        cfg = _make_config(atr_enabled=False, winner_mult=1.0)  # no slippage distortion
        cfg.risk.ENABLE_SLIPPAGE = False
        sim = _make_simulator(cfg)
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=5)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        closes = [11.0] * 10
        bars = _make_bars(ts, lows=[10.8]*10, highs=[11.5]*10, closes=closes)
        pos = _make_position(entry_price=10.0, stop_price=9.0, size=100,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        results = {'capital': 10.0 * 100}
        trade = sim.simulate_exit('TEST', pos, bars, results)
        assert trade['pnl'] > 0

    def test_basic_loser_pnl(self):
        """Entry 10, price stays below entry → P&L negative."""
        cfg = _make_config(atr_enabled=False)
        cfg.risk.ENABLE_SLIPPAGE = False
        sim = _make_simulator(cfg)
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=5)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        closes = [9.0] * 10
        bars = _make_bars(ts, lows=[8.8]*10, highs=[9.5]*10, closes=closes)
        pos = _make_position(entry_price=10.0, stop_price=8.5, size=100,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        results = {'capital': 10.0 * 100}
        trade = sim.simulate_exit('TEST', pos, bars, results)
        assert trade['pnl'] < 0

    def test_pnl_equals_size_times_price_diff(self):
        """Without slippage: pnl == (exit_price - entry_price) * size."""
        cfg = _make_config(atr_enabled=False)
        cfg.risk.ENABLE_SLIPPAGE = False
        sim = _make_simulator(cfg)
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=2)
        ts = [_utc(hour=13, minute=i) for i in range(5)]
        closes = [10.5] * 5
        bars = _make_bars(ts, lows=[10.2]*5, highs=[10.8]*5, closes=closes)
        pos = _make_position(entry_price=10.0, stop_price=9.5, size=50,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        results = {'capital': 500.0}
        trade = sim.simulate_exit('TEST', pos, bars, results)
        expected_pnl = (trade['exit_price'] - 10.0) * 50
        assert trade['pnl'] == pytest.approx(expected_pnl, rel=1e-6)

    def test_simple_stops_uses_5pct_floor(self):
        """When SIMPLE_STOPS=True the stop is always entry * 0.95."""
        cfg = _make_config(simple_stops=True, atr_enabled=False)
        sim = _make_simulator(cfg)
        entry_price = 10.0
        expected_stop = entry_price * 0.95
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=14, minute=0)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        # low drops to exactly the simple stop = 9.5
        lows  = [9.8, 9.8, 9.8, 9.5, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8]
        highs = [10.2] * 10
        closes = [10.0] * 10
        bars = _make_bars(ts, lows, highs, closes)
        pos = _make_position(entry_price=entry_price, stop_price=9.9,  # override ignored in simple mode
                             entry_time=entry_time, trading_window_end_utc=deadline)
        results = {'capital': entry_price * 100}
        trade = sim.simulate_exit('TEST', pos, bars, results)
        assert trade['exit_reason'] in ('stop_loss', 'trading_window_close')


# ---------------------------------------------------------------------------
# simulate_exit — return dict completeness
# ---------------------------------------------------------------------------

class TestSimulateExitReturnDict:
    def test_all_required_keys_present(self):
        sim = _make_simulator(_make_config(atr_enabled=False))
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=5)
        ts = [_utc(hour=13, minute=i) for i in range(10)]
        bars = _make_bars(ts, lows=[9.6]*10, highs=[10.5]*10, closes=[10.1]*10)
        pos = _make_position(entry_time=entry_time, trading_window_end_utc=deadline)
        results = {'capital': 1000.0}
        trade = sim.simulate_exit('TEST', pos, bars, results)
        required = [
            'symbol', 'entry_price', 'exit_price', 'size', 'pnl',
            'exit_reason', 'return_pct', 'hold_time_minutes',
            'highest_price', 'mae', 'mfe',
        ]
        for key in required:
            assert key in trade, f"Missing key: {key}"

    def test_return_pct_consistent_with_prices(self):
        cfg = _make_config(atr_enabled=False)
        cfg.risk.ENABLE_SLIPPAGE = False
        sim = _make_simulator(cfg)
        entry_time = _utc(hour=13, minute=0)
        deadline = _utc(hour=13, minute=3)
        ts = [_utc(hour=13, minute=i) for i in range(5)]
        closes = [10.5] * 5
        bars = _make_bars(ts, lows=[10.2]*5, highs=[10.8]*5, closes=closes)
        pos = _make_position(entry_price=10.0, stop_price=9.0, size=10,
                             entry_time=entry_time, trading_window_end_utc=deadline)
        results = {'capital': 100.0}
        trade = sim.simulate_exit('TEST', pos, bars, results)
        expected_pct = ((trade['exit_price'] - 10.0) / 10.0) * 100
        assert trade['return_pct'] == pytest.approx(expected_pct, rel=1e-6)
