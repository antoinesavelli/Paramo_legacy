# =====================================================
# trade_executor.py - Trade Execution
# =====================================================

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from config.config import TradingConfig
from strategy.risk_manager import RiskManager, calc_atr
from utils.logging import get_logger

# Terminal order states — polling stops when any of these is reached.
_TERMINAL_STATES = frozenset({"filled", "canceled", "expired", "rejected", "pending_cancel"})


def place_order(api, logger, poll_interval: float = 0.2, timeout: float = 5.0, **kwargs):
    """
    Submit an order and poll until it reaches a terminal state.

    Replaces the old time.sleep(1) with a bounded poll loop so we react
    to fills as fast as Alpaca acknowledges them (~200 ms typical) while
    still bounding the worst-case wait to `timeout` seconds.

    Args:
        api:           Alpaca REST client.
        logger:        Bound logger for this call site.
        poll_interval: Seconds between get_order() polls (default 0.2 s).
        timeout:       Maximum seconds to wait before returning whatever
                       state the order is in (default 5.0 s).
        **kwargs:      Forwarded verbatim to api.submit_order().

    Returns:
        The most recent Order object, or None if submission itself failed.
    """
    try:
        order = api.submit_order(**kwargs)
    except Exception as e:
        logger.error("Order submission failed: %s", e)
        return None

    deadline = time.monotonic() + timeout

    while True:
        try:
            order = api.get_order(order.id)
        except Exception as e:
            logger.warning("get_order poll failed for %s: %s", order.id, e)
            # Don't abort — a transient network hiccup shouldn't kill the order.
            # Fall through to sleep and try again.

        if order.status in _TERMINAL_STATES:
            return order

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Order %s timed out after %.1fs — status=%s symbol=%s",
                order.id, timeout, order.status, kwargs.get("symbol", "?"),
            )
            return order  # Return as-is; caller checks .status

        time.sleep(min(poll_interval, remaining))


class TradeExecutor:
    """Handles order placement, execution, and re-entry management"""

    def __init__(self, config: TradingConfig, api, risk_manager: RiskManager, market_context=None):
        self.config = config
        self.api = api
        self.risk_manager = risk_manager
        self.market_context = market_context
        self.logger = get_logger(__name__, component="executor")
        self.active_trades = {}
        self.trade_history = []
        self.reentry_candidates = {}
        self.reentry_history = {}

        # Pull poll settings once so every place_order call uses config values.
        self._poll_interval = config.system.ORDER_POLL_INTERVAL_SECONDS
        self._poll_timeout = config.system.ORDER_FILL_TIMEOUT_SECONDS

    def _place(self, **kwargs):
        """Internal shortcut — always uses config-driven poll settings."""
        return place_order(
            self.api,
            self.logger,
            poll_interval=self._poll_interval,
            timeout=self._poll_timeout,
            **kwargs,
        )

    def _emit_order_audit(self, event: str, symbol: str, order=None, **extra) -> None:
        """
        Emit a structured [ORDER_AUDIT] log line at every order lifecycle point.

        Lines are captured by the file handler in setup_logging and can be
        parsed post-session for regulatory review or debugging.  No DB
        connection is required here — use Monitor.audit_order_event() to also
        persist events to SQLite when a Monitor instance is available.
        """
        order_id = getattr(order, "id", None) if order else None
        status   = getattr(order, "status", None) if order else None
        price    = getattr(order, "filled_avg_price", None) if order else None
        qty      = getattr(order, "qty", None) if order else None
        extra_str = " ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
        self.logger.info(
            "[ORDER_AUDIT] event=%s symbol=%s order_id=%s status=%s qty=%s price=%s %s",
            event, symbol, order_id, status, qty, price, extra_str,
        )

    # ------------------------------------------------------------------
    # All existing call sites: replace place_order(self.api, self.logger, ...)
    #                          with     self._place(...)
    # ------------------------------------------------------------------

    def execute_entry(self, signal: Dict) -> Dict:
        """Execute entry order with market context-adjusted sizing"""
        try:
            symbol = signal['symbol']
            entry_price = signal['entry_price']
            stop_price = signal['stop_price']
            is_reentry = signal.get('is_reentry', False)

            if is_reentry:
                if not self.config.risk.ENABLE_REENTRY:
                    self.logger.info("Re-entry blocked for %s: ENABLE_REENTRY=False", symbol)
                    return {'success': False, 'reason': 'reentry_disabled'}
                reentry_count = self.reentry_history.get(symbol, 0)
                if reentry_count >= self.config.risk.MAX_REENTRIES_PER_STOCK:
                    self.logger.info("Re-entry blocked for %s: max reentries reached (%d)", symbol, reentry_count)
                    return {'success': False, 'reason': 'max_reentries_reached'}

            market_adjustment = 1.0
            if self.market_context:
                try:
                    market_adjustment = self.market_context.get_position_size_adjustment()
                    env = self.market_context.market_indicators.get('trading_environment', 'neutral')
                    score = self.market_context.market_indicators.get('market_score', 50)
                    self.logger.info(
                        "Market context for %s: env=%s score=%.1f adjustment=%.2fx",
                        symbol, env, score, market_adjustment,
                    )
                except Exception as e:
                    self.logger.warning("Failed to get market adjustment: %s, using 1.0x", e)

            self.logger.info(
                "Entry requested for %s reentry=%s entry=%.2f stop=%.2f",
                symbol, is_reentry, entry_price, stop_price,
            )

            risk_check = self.risk_manager.check_entry_risk(symbol, entry_price, stop_price, market_adjustment)
            if not risk_check['approved']:
                self.logger.warning("Entry rejected for %s: %s", symbol, risk_check['reason'])
                return {'success': False, 'reason': risk_check['reason']}

            position_size = risk_check['position_size']

            # Live only: price limit slightly above ask to ensure fill urgency.
            # This is NOT slippage simulation — it is limit order aggressiveness.
            aggression = self.config.risk.LIMIT_ORDER_AGGRESSION_PCT

            entry_order = self._place(
                symbol=symbol, qty=position_size, side='buy',
                type='limit', time_in_force='ioc',
                limit_price=round(entry_price * (1 + aggression), 2),
            )
            self._emit_order_audit("ENTRY_ATTEMPT_1", symbol, entry_order, order_type="limit_ioc")

            if not entry_order or entry_order.status != 'filled':
                if entry_order and entry_order.status not in ('canceled', 'expired', 'rejected'):
                    try:
                        self.api.cancel_order(entry_order.id)
                    except Exception:
                        pass

                high_gap = signal.get('gap_percent', 0) >= self.config.risk.SLIPPAGE_GAP_THRESHOLD
                aggression2 = (
                    self.config.risk.LIMIT_ORDER_HIGH_GAP_AGGRESSION_PCT
                    if high_gap else aggression
                )
                order_type = 'market' if is_reentry else 'limit'
                entry_order = self._place(
                    symbol=symbol, qty=position_size, side='buy',
                    type=order_type, time_in_force='day',
                    limit_price=round(entry_price * (1 + aggression2), 2) if order_type == 'limit' else None,
                )
                self._emit_order_audit("ENTRY_ATTEMPT_2", symbol, entry_order, order_type=order_type)

            if entry_order and entry_order.status == 'filled':
                filled_price = float(entry_order.filled_avg_price)
                self._emit_order_audit("ENTRY_FILLED", symbol, entry_order, filled_price=filled_price)
                stop_order = self._place(
                    symbol=symbol, qty=position_size, side='sell',
                    type='stop', time_in_force='gtc', stop_price=stop_price,
                )
                self._track_new_position(symbol, filled_price, position_size, stop_price, stop_order, entry_order, is_reentry)
                return {
                    'success': True,
                    'symbol': symbol,
                    'entry_price': filled_price,
                    'position_size': position_size,
                    'stop_price': stop_price,
                    'is_reentry': is_reentry,
                }

            return {'success': False, 'reason': f"Order not filled: {getattr(entry_order, 'status', 'unknown')}"}

        except Exception as e:
            self.logger.error("Error executing entry for %s: %s", signal['symbol'], e)
            return {'success': False, 'reason': str(e)}

    def _track_new_position(self, symbol, filled_price, position_size, stop_price, stop_order, entry_order, is_reentry):
        """Track new position with ATR trailing stops only (no profit targets, no min hold time)."""
        # Calculate max hold time
        max_hold_time = datetime.now() + timedelta(minutes=self.config.risk.MAX_HOLD_TIME_MINUTES)
        
        # Get recent bars for ATR trailing stop
        try:
            recent_bars = self.api.get_bars(symbol, '1Min', limit=50).df
        except Exception:  # NOTE: FIXED: Added except clause
            recent_bars = pd.DataFrame()
        
        self.active_trades[symbol] = {
            'entry_time': datetime.now(),
            'max_hold_time': max_hold_time,
            'entry_price': filled_price,
            'position_size': position_size,
            'stop_price': stop_price,
            'stop_order_id': stop_order.id if stop_order else None,
            'entry_order_id': entry_order.id if entry_order else None,
            'highest_price': filled_price,
            'recent_bars': recent_bars,
            'reentry_count': self.reentry_history.get(symbol, 0),
            'is_reentry': is_reentry
        }
        
        if is_reentry:
            self.reentry_history[symbol] = self.reentry_history.get(symbol, 0) + 1
            self.logger.info(  # noqa: G004
                f"RE-ENTRY #{self.reentry_history[symbol]} for {symbol}: "
                f"{position_size} shares @ ${filled_price:.2f}, "
                f"stop=${stop_price:.2f}, max_hold={self.config.risk.MAX_HOLD_TIME_MINUTES}min"
            )
        else:
            self.logger.info(  # noqa: G004
                f"Entry for {symbol}: {position_size} shares @ ${filled_price:.2f}, "
                f"stop=${stop_price:.2f}, max_hold={self.config.risk.MAX_HOLD_TIME_MINUTES}min"
            )

    def execute_exit(self, symbol: str, reason: str = "manual") -> Dict:
        """Execute exit order (no minimum hold time enforcement)"""
        try:
            if symbol not in self.active_trades:
                return {'success': False, 'reason': 'No active position'}
            
            trade = self.active_trades[symbol]
            position_size = trade['position_size']
            
            # Cancel existing stop order
            if trade.get('stop_order_id'):
                try:
                    self.api.cancel_order(trade['stop_order_id'])
                except Exception as e:
                    self.logger.warning("Failed to cancel stop for %s: %s", symbol, e, exc_info=True)
            
            # Execute market exit
            exit_order = place_order(
                self.api, self.logger,
                symbol=symbol,
                qty=position_size,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            if exit_order and exit_order.status == 'filled':
                exit_price = float(exit_order.filled_avg_price)
                self._emit_order_audit("EXIT_FILLED", symbol, exit_order, reason=reason)
                pnl = (exit_price - trade['entry_price']) * position_size
                pnl_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
                
                trade_record = {
                    'symbol': symbol,
                    'entry_time': trade['entry_time'],
                    'exit_time': datetime.now(),
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': reason,
                    'is_reentry': trade.get('is_reentry', False),
                    'reentry_count': trade.get('reentry_count', 0)
                }
                
                self.trade_history.append(trade_record)
                del self.active_trades[symbol]
                
                self.logger.info(  # noqa: G004
                    f"EXIT {symbol}: ${exit_price:.2f} | "
                    f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
                    f"Reason: {reason}"
                )
                
                return {
                    'success': True,
                    'symbol': symbol,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': reason
                }
            
            return {'success': False, 'reason': f"Exit order not filled: {getattr(exit_order, 'status', 'unknown')}"}
            
        except Exception as e:
            self.logger.error("Error executing exit for %s: %s", symbol, e)
            return {'success': False, 'reason': str(e)}

    def update_active_positions(self):
        """Update all active positions - check stops, time limits, and trailing stops"""
        for symbol in list(self.active_trades.keys()):
            try:
                self._update_position(symbol)
            except Exception as e:
                self.logger.error("Error updating position for %s: %s", symbol, e, exc_info=True)

    def _update_position(self, symbol: str):
        """Update individual position"""
        trade = self.active_trades[symbol]
        
        # Get current price
        try:
            quote = self.api.get_latest_quote(symbol)
            current_price = float(quote.ap)  # Ask price for exits
        except Exception as e:
            self.logger.warning("Failed to get quote for %s: %s", symbol, e, exc_info=True)
            return
        
        # Check time limit
        if datetime.now() >= trade['max_hold_time']:
            self.execute_exit(symbol, reason="time_limit")
            return
        
        # Update trailing stop if enabled
        if self.config.risk.ATR_TRAILING_ENABLED:
            self._update_trailing_stop(symbol, current_price)

    def _update_trailing_stop(self, symbol: str, current_price: float):
        """Update ATR-based trailing stop using shared calc_atr helper."""
        trade = self.active_trades[symbol]
        if current_price > trade['highest_price']:
            trade['highest_price'] = current_price

        profit_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
        if profit_pct < self.config.risk.ATR_TRAILING_MIN_PROFIT_PCT:
            return

        period = self.config.risk.ATR_TRAILING_PERIOD
        if not trade['recent_bars'].empty and len(trade['recent_bars']) >= period:
            atr = calc_atr(trade['recent_bars'], period)
            new_stop = current_price - (atr * self.config.risk.ATR_TRAILING_MULTIPLIER)

            if new_stop > trade['stop_price']:
                old_stop = trade['stop_price']
                trade['stop_price'] = new_stop

                try:
                    if trade.get('stop_order_id'):
                        self.api.cancel_order(trade['stop_order_id'])

                    new_stop_order = place_order(
                        self.api, self.logger,
                        symbol=symbol,
                        qty=trade['position_size'],
                        side='sell',
                        type='stop',
                        time_in_force='gtc',
                        stop_price=new_stop
                    )

                    if new_stop_order:
                        trade['stop_order_id'] = new_stop_order.id
                        self.logger.info(  # noqa: G004
                            f"Trailing stop updated for {symbol}: "
                            f"${old_stop:.2f} -> ${new_stop:.2f} "
                            f"(profit: {profit_pct:.2f}%)"
                        )
                except Exception as e:
                    self.logger.error("Failed to update trailing stop for %s: %s", symbol, e, exc_info=True)

    def close_all_positions(self):
        """Close all active positions"""
        self.logger.info("Closing all positions (%d active)", len(self.active_trades))
        for symbol in list(self.active_trades.keys()):
            self.execute_exit(symbol, reason="eod_close")

    def get_active_positions(self) -> List[Dict]:
        """Get list of active positions"""
        return [
            {
                'symbol': symbol,
                'entry_price': trade['entry_price'],
                'entry_time': trade['entry_time'],
                'position_size': trade['position_size'],
                'stop_price': trade['stop_price']
            }
            for symbol, trade in self.active_trades.items()
        ]