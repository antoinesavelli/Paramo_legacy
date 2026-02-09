# =====================================================
# trade_executor.py - Trade Execution
# =====================================================

import time
import pandas as pd  # ✅ ADDED: Missing import
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from config.config import TradingConfig  # ✅ FIXED: Added .config
from strategy.risk_manager import RiskManager
from utils.logging import get_logger

def log_api_error(logger, msg, exc):
    logger.error(f"{msg}: {exc}")

def place_order(api, logger, **kwargs):
    """Helper to place an order and handle exceptions."""
    try:
        order = api.submit_order(**kwargs)
        time.sleep(1)
        return api.get_order(order.id)
    except Exception as e:
        log_api_error(logger, "Order placement failed", e)
        return None

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

    def execute_entry(self, signal: Dict) -> Dict:
        """Execute entry order with market context-adjusted sizing"""
        try:
            symbol = signal['symbol']
            entry_price = signal['entry_price']
            stop_price = signal['stop_price']
            is_reentry = signal.get('is_reentry', False)

            # Get market context adjustment
            market_adjustment = 1.0
            if self.market_context:
                try:
                    market_adjustment = self.market_context.get_position_size_adjustment()
                    env = self.market_context.market_indicators.get('trading_environment', 'neutral')
                    score = self.market_context.market_indicators.get('market_score', 50)
                    self.logger.info(
                        f"Market context for {symbol}: env={env}, "
                        f"score={score:.1f}, adjustment={market_adjustment:.2f}x"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to get market adjustment: {e}, using 1.0x")
                    market_adjustment = 1.0

            self.logger.info(f"Entry requested for {symbol} reentry={is_reentry} entry={entry_price:.2f} stop={stop_price:.2f}")
            
            # Pass market adjustment to risk check
            risk_check = self.risk_manager.check_entry_risk(symbol, entry_price, stop_price, market_adjustment)
            
            if not risk_check['approved']:
                self.logger.warning(f"Entry rejected for {symbol}: {risk_check['reason']}")
                return {'success': False, 'reason': risk_check['reason']}
            position_size = risk_check['position_size']

            entry_order = place_order(
                self.api, self.logger,
                symbol=symbol, qty=position_size, side='buy',
                type='limit', time_in_force='ioc', limit_price=entry_price * 1.001
            )
            if not entry_order or entry_order.status != 'filled':
                try:
                    self.api.cancel_order(entry_order.id)
                except Exception:
                    pass
                order_type = 'market' if is_reentry else 'limit'
                entry_order = place_order(
                    self.api, self.logger,
                    symbol=symbol, qty=position_size, side='buy',
                    type=order_type, time_in_force='day',
                    limit_price=entry_price * 1.002 if order_type == 'limit' else None
                )
            if entry_order and entry_order.status == 'filled':
                filled_price = float(entry_order.filled_avg_price)
                stop_order = place_order(
                    self.api, self.logger,
                    symbol=symbol, qty=position_size, side='sell',
                    type='stop', time_in_force='gtc', stop_price=stop_price
                )
                self._track_new_position(symbol, filled_price, position_size, stop_price, stop_order, entry_order, is_reentry)
                return {
                    'success': True,
                    'symbol': symbol,
                    'entry_price': filled_price,
                    'position_size': position_size,
                    'stop_price': stop_price,
                    'is_reentry': is_reentry
                }
            return {'success': False, 'reason': f"Order not filled: {getattr(entry_order, 'status', 'unknown')}"}
        except Exception as e:
            log_api_error(self.logger, f"Error executing entry for {signal['symbol']}", e)
            return {'success': False, 'reason': str(e)}

    def _track_new_position(self, symbol, filled_price, position_size, stop_price, stop_order, entry_order, is_reentry):
        """Track new position with ATR trailing stops only (no profit targets, no min hold time)."""
        # Calculate max hold time
        max_hold_time = datetime.now() + timedelta(minutes=self.config.risk.MAX_HOLD_TIME_MINUTES)
        
        # Get recent bars for ATR trailing stop
        try:
            recent_bars = self.api.get_bars(symbol, '1Min', limit=50).df
        except Exception:  # ✅ FIXED: Added except clause
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
            self.logger.info(
                f"RE-ENTRY #{self.reentry_history[symbol]} for {symbol}: "
                f"{position_size} shares @ ${filled_price:.2f}, "
                f"stop=${stop_price:.2f}, max_hold={self.config.risk.MAX_HOLD_TIME_MINUTES}min"
            )
        else:
            self.logger.info(
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
                    self.logger.warning(f"Failed to cancel stop for {symbol}: {e}")
            
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
                
                self.logger.info(
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
            log_api_error(self.logger, f"Error executing exit for {symbol}", e)
            return {'success': False, 'reason': str(e)}

    def update_active_positions(self):
        """Update all active positions - check stops, time limits, and trailing stops"""
        for symbol in list(self.active_trades.keys()):
            try:
                self._update_position(symbol)
            except Exception as e:
                self.logger.error(f"Error updating position for {symbol}: {e}")

    def _update_position(self, symbol: str):
        """Update individual position"""
        trade = self.active_trades[symbol]
        
        # Get current price
        try:
            quote = self.api.get_latest_quote(symbol)
            current_price = float(quote.ap)  # Ask price for exits
        except Exception as e:
            self.logger.warning(f"Failed to get quote for {symbol}: {e}")
            return
        
        # Check time limit
        if datetime.now() >= trade['max_hold_time']:
            self.execute_exit(symbol, reason="time_limit")
            return
        
        # Update trailing stop if enabled
        if self.config.risk.ATR_TRAILING_ENABLED:
            self._update_trailing_stop(symbol, current_price)

    def _update_trailing_stop(self, symbol: str, current_price: float):
        """Update ATR-based trailing stop"""
        trade = self.active_trades[symbol]
        
        # Update highest price
        if current_price > trade['highest_price']:
            trade['highest_price'] = current_price
        
        # Calculate profit percentage
        profit_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
        
        # Only activate trailing stop if minimum profit reached
        if profit_pct < self.config.risk.ATR_TRAILING_MIN_PROFIT_PCT:
            return
        
        # Calculate ATR if we have bars
        if not trade['recent_bars'].empty and len(trade['recent_bars']) >= self.config.risk.ATR_TRAILING_PERIOD:
            atr = self._calculate_atr(trade['recent_bars'], self.config.risk.ATR_TRAILING_PERIOD)
            new_stop = current_price - (atr * self.config.risk.ATR_TRAILING_MULTIPLIER)
            
            # Only raise the stop, never lower it
            if new_stop > trade['stop_price']:
                old_stop = trade['stop_price']
                trade['stop_price'] = new_stop
                
                # Update stop order
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
                        self.logger.info(
                            f"Trailing stop updated for {symbol}: "
                            f"${old_stop:.2f} -> ${new_stop:.2f} "
                            f"(profit: {profit_pct:.2f}%)"
                        )
                except Exception as e:
                    self.logger.error(f"Failed to update trailing stop for {symbol}: {e}")

    def _calculate_atr(self, bars: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range"""
        try:
            high = bars['high'].values
            low = bars['low'].values
            close = bars['close'].values
            
            tr1 = high - low
            tr2 = abs(high - close[:-1])  # Previous close
            tr3 = abs(low - close[:-1])
            
            tr = pd.DataFrame({'tr1': tr1[:-1], 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return float(atr)
        except Exception as e:
            self.logger.warning(f"ATR calculation failed: {e}")
            return 0.0

    def close_all_positions(self):
        """Close all active positions"""
        self.logger.info(f"Closing all positions ({len(self.active_trades)} active)")
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