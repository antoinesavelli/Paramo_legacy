# =====================================================
# test_live_trading_dry_run.py - TradeExecutor + place_order with mocked API
# =====================================================

import time
import pytest
from unittest.mock import MagicMock, patch, call
from config.config import TradingConfig
from config.loader import apply_overrides_immutable
from strategy.risk_manager import RiskManager
from execution.trade_executor import TradeExecutor, place_order, _TERMINAL_STATES


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_order(order_id="ord-1", status="filled", filled_avg_price="100.00", qty="10"):
    o = MagicMock()
    o.id = order_id
    o.status = status
    o.filled_avg_price = filled_avg_price
    o.qty = qty
    return o


def _make_api(
    submit_order_returns=None,
    get_order_returns=None,
    positions=None,
    equity: float = 50_000.0,
    buying_power: float = 50_000.0,
):
    api = MagicMock()

    acct = MagicMock()
    acct.equity = str(equity)
    acct.buying_power = str(buying_power)
    acct.portfolio_value = str(equity)
    api.get_account.return_value = acct
    api.list_positions.return_value = positions or []

    if submit_order_returns is not None:
        api.submit_order.return_value = submit_order_returns
    if get_order_returns is not None:
        api.get_order.return_value = get_order_returns

    return api


def _make_executor(api=None, overrides: list = None) -> TradeExecutor:
    cfg = apply_overrides_immutable(TradingConfig(), overrides or [])
    if api is None:
        api = _make_api()
    rm = RiskManager(cfg, api)
    return TradeExecutor(cfg, api, rm)


# ---------------------------------------------------------------------------
# TestPlaceOrder
# ---------------------------------------------------------------------------

class TestPlaceOrder:
    def test_immediate_fill(self):
        order = _make_order(status="filled")
        api = _make_api(submit_order_returns=order, get_order_returns=order)
        logger = MagicMock()

        result = place_order(api, logger, symbol="AAPL", qty=10, side="buy",
                             type="market", time_in_force="day")
        assert result is not None
        assert result.status == "filled"

    def test_submission_failure_returns_none(self):
        api = MagicMock()
        api.submit_order.side_effect = Exception("connection refused")
        logger = MagicMock()

        result = place_order(api, logger, symbol="AAPL", qty=10, side="buy",
                             type="market", time_in_force="day")
        assert result is None

    def test_timeout_returns_last_known_order(self):
        pending = _make_order(status="pending_new")
        api = _make_api(submit_order_returns=pending, get_order_returns=pending)
        logger = MagicMock()

        # Very short timeout — will fire before any fill
        result = place_order(api, logger, symbol="AAPL", qty=10, side="buy",
                             type="limit", time_in_force="ioc",
                             limit_price=101.0, timeout=0.05, poll_interval=0.1)
        # Timed-out but still returns the last known order object
        assert result is not None
        assert result.status == "pending_new"

    def test_transient_poll_error_does_not_abort(self):
        """A single get_order failure should not abort the poll loop."""
        filled = _make_order(status="filled")
        api = MagicMock()
        api.submit_order.return_value = _make_order(status="pending_new")
        # First poll raises; second succeeds
        api.get_order.side_effect = [Exception("network blip"), filled]
        logger = MagicMock()

        result = place_order(api, logger, symbol="AAPL", qty=5, side="buy",
                             type="market", time_in_force="day",
                             timeout=2.0, poll_interval=0.05)
        assert result is not None
        assert result.status == "filled"


# ---------------------------------------------------------------------------
# TestExecuteEntry
# ---------------------------------------------------------------------------

class TestExecuteEntry:
    def _signal(self, symbol="AAPL", entry=100.0, stop=95.0, **kw):
        return {"symbol": symbol, "entry_price": entry, "stop_price": stop, **kw}

    def test_successful_entry(self):
        filled = _make_order(status="filled", filled_avg_price="100.50")
        api = _make_api(submit_order_returns=filled, get_order_returns=filled,
                        equity=50_000.0, buying_power=50_000.0)
        # stop order
        stop_ord = _make_order(order_id="stop-1", status="accepted")
        api.submit_order.side_effect = [filled, stop_ord]
        api.get_order.return_value = filled

        executor = _make_executor(api)
        result = executor.execute_entry(self._signal())

        assert result["success"] is True
        assert result["symbol"] == "AAPL"
        assert result["entry_price"] == pytest.approx(100.50)

    def test_entry_rejected_by_risk_manager(self):
        api = _make_api(equity=50_000.0, buying_power=50_000.0)
        executor = _make_executor(api)
        # stop above entry → risk check rejects
        result = executor.execute_entry(self._signal(entry=100.0, stop=110.0))
        assert result["success"] is False

    def test_unfilled_entry_returns_failure(self):
        canceled = _make_order(status="canceled")
        api = _make_api(submit_order_returns=canceled, get_order_returns=canceled,
                        equity=50_000.0, buying_power=50_000.0)
        executor = _make_executor(api)
        result = executor.execute_entry(self._signal())
        assert result["success"] is False
        assert "filled" in result["reason"].lower() or "canceled" in result["reason"].lower()

    def test_reentry_blocked_when_disabled(self):
        api = _make_api(equity=50_000.0, buying_power=50_000.0)
        executor = _make_executor(api, overrides=[("risk.ENABLE_REENTRY", "False")])
        result = executor.execute_entry(self._signal(is_reentry=True))
        assert result["success"] is False
        assert "reentry" in result["reason"].lower()


# ---------------------------------------------------------------------------
# TestExecuteExit
# ---------------------------------------------------------------------------

class TestExecuteExit:
    def _seed_active_trade(self, executor, symbol="AAPL",
                           entry_price=100.0, position_size=10):
        from datetime import datetime, timedelta
        executor.active_trades[symbol] = {
            "entry_time": datetime.now() - timedelta(minutes=5),
            "max_hold_time": datetime.now() + timedelta(hours=1),
            "entry_price": entry_price,
            "position_size": position_size,
            "stop_price": entry_price * 0.95,
            "stop_order_id": "stop-1",
            "entry_order_id": "entry-1",
            "highest_price": entry_price,
            "recent_bars": __import__("pandas").DataFrame(),
            "reentry_count": 0,
            "is_reentry": False,
        }

    def test_successful_exit(self):
        filled_exit = _make_order(status="filled", filled_avg_price="105.00")
        api = _make_api(submit_order_returns=filled_exit, get_order_returns=filled_exit,
                        equity=50_000.0, buying_power=50_000.0)
        executor = _make_executor(api)
        self._seed_active_trade(executor)

        result = executor.execute_exit("AAPL", reason="time_limit")
        assert result["success"] is True
        assert result["exit_price"] == pytest.approx(105.0)
        assert result["pnl"] == pytest.approx((105.0 - 100.0) * 10)

    def test_exit_no_active_position(self):
        executor = _make_executor()
        result = executor.execute_exit("FAKE", reason="manual")
        assert result["success"] is False
        assert "position" in result["reason"].lower()

    def test_exit_appends_to_trade_history(self):
        filled_exit = _make_order(status="filled", filled_avg_price="98.00")
        api = _make_api(submit_order_returns=filled_exit, get_order_returns=filled_exit,
                        equity=50_000.0, buying_power=50_000.0)
        executor = _make_executor(api)
        self._seed_active_trade(executor)

        executor.execute_exit("AAPL", reason="stop_hit")
        assert len(executor.trade_history) == 1
        assert executor.trade_history[0]["exit_reason"] == "stop_hit"
        assert "AAPL" not in executor.active_trades


# ---------------------------------------------------------------------------
# TestCloseAllPositions
# ---------------------------------------------------------------------------

class TestCloseAllPositions:
    def test_closes_all_active_trades(self):
        filled_exit = _make_order(status="filled", filled_avg_price="50.00")
        api = _make_api(submit_order_returns=filled_exit, get_order_returns=filled_exit,
                        equity=50_000.0, buying_power=50_000.0)
        executor = _make_executor(api)

        from datetime import datetime, timedelta
        for sym in ["AAPL", "TSLA", "NVDA"]:
            executor.active_trades[sym] = {
                "entry_time": datetime.now(),
                "max_hold_time": datetime.now() + timedelta(hours=1),
                "entry_price": 50.0,
                "position_size": 5,
                "stop_price": 47.0,
                "stop_order_id": None,
                "entry_order_id": None,
                "highest_price": 50.0,
                "recent_bars": __import__("pandas").DataFrame(),
                "reentry_count": 0,
                "is_reentry": False,
            }

        results = []
        for sym in list(executor.active_trades.keys()):
            results.append(executor.execute_exit(sym, reason="eod"))

        assert all(r["success"] for r in results)
        assert len(executor.active_trades) == 0
