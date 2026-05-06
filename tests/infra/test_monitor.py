"""
Tests for core/monitor.py — DB initialisation, trade recording, and
in-memory metric aggregation. Uses an in-memory SQLite database so no
files are written to disk.
"""
import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from monitoring.monitor import Monitor


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def monitor(tmp_path):
    """Return a Monitor wired to a temp-dir SQLite file."""
    cfg = MagicMock()
    cfg.system.DATABASE_PATH = str(tmp_path / "test_trading.db")
    return Monitor(config=cfg)


def _trade(symbol="AAPL", pnl=100.0, exit_reason="stop"):
    return {
        "symbol": symbol,
        "entry_time": "2024-01-02T09:35:00",
        "exit_time": "2024-01-02T10:05:00",
        "entry_price": 150.0,
        "exit_price": 152.0 if pnl > 0 else 148.0,
        "position_size": 10,
        "pnl": pnl,
        "exit_reason": exit_reason,
    }


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

class TestDatabaseInit:
    def test_tables_created(self, monitor):
        with sqlite3.connect(monitor.db_path) as conn:
            tables = {
                r[0] for r in
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert {"trades", "system_logs", "performance_metrics", "order_audit"} <= tables

    def test_init_idempotent(self, monitor):
        """Calling _init_database a second time must not raise."""
        monitor._init_database()   # should be a no-op


# ---------------------------------------------------------------------------
# record_trade / DB persistence
# ---------------------------------------------------------------------------

class TestRecordTrade:
    def test_trade_persisted_to_db(self, monitor):
        monitor.record_trade(_trade("TSLA", pnl=50.0))
        with sqlite3.connect(monitor.db_path) as conn:
            rows = conn.execute("SELECT symbol, pnl FROM trades").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "TSLA"
        assert abs(rows[0][1] - 50.0) < 0.001

    def test_multiple_trades_persist(self, monitor):
        for sym in ("AAPL", "NVDA", "MSFT"):
            monitor.record_trade(_trade(sym, pnl=10.0))
        with sqlite3.connect(monitor.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert count == 3

    def test_missing_key_does_not_crash_monitor(self, monitor):
        """record_trade should catch exceptions and not propagate."""
        bad_trade = {"symbol": "X"}   # missing required fields
        monitor.record_trade(bad_trade)  # must not raise


# ---------------------------------------------------------------------------
# _update_metrics (in-memory aggregation)
# ---------------------------------------------------------------------------

class TestUpdateMetrics:
    def test_winning_trade_increments_wins(self, monitor):
        monitor._update_metrics(_trade(pnl=200.0))
        assert monitor.metrics["winning_trades"] == 1
        assert monitor.metrics["total_pnl"] == pytest.approx(200.0)

    def test_losing_trade_increments_losses(self, monitor):
        monitor._update_metrics(_trade(pnl=-150.0))
        assert monitor.metrics["losing_trades"] == 1
        assert monitor.metrics["total_pnl"] == pytest.approx(-150.0)

    def test_best_and_worst_trade_tracked(self, monitor):
        monitor._update_metrics(_trade(pnl=300.0))
        monitor._update_metrics(_trade(pnl=-100.0))
        assert monitor.metrics["best_trade"] == pytest.approx(300.0)
        assert monitor.metrics["worst_trade"] == pytest.approx(-100.0)

    def test_win_streak_increments(self, monitor):
        for _ in range(3):
            monitor._update_metrics(_trade(pnl=10.0))
        assert monitor.metrics["current_streak"] == 3
        assert monitor.metrics["max_streak"] == 3

    def test_loss_resets_win_streak(self, monitor):
        monitor._update_metrics(_trade(pnl=10.0))
        monitor._update_metrics(_trade(pnl=10.0))
        monitor._update_metrics(_trade(pnl=-5.0))
        # streak flips to -1 after a loss
        assert monitor.metrics["current_streak"] < 0

    def test_total_trades_count(self, monitor):
        monitor._update_metrics(_trade(pnl=10.0))
        monitor._update_metrics(_trade(pnl=-5.0))
        assert monitor.metrics["total_trades"] == 2