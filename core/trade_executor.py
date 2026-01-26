# =====================================================
# trade_executor.py - Trade Execution
# =====================================================

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from config import TradingConfig
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
        except:
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
                return {'success': False, 'reason': 'Position not found'}
            
            self.logger.info(f"Exit requested for {symbol} reason={reason}")
            position_info = self.active_trades[symbol]
            
            try:
                self.api.cancel_order(position_info['stop_order_id'])
            except Exception:
                pass
            
            exit_order = place_order(
                self.api, self.logger,
                symbol=symbol, qty=position_info['position_size'],
                side='sell', type='market', time_in_force='day'
            )
            
            if exit_order and exit_order.status == 'filled':
                exit_price = float(exit_order.filled_avg_price)
                pnl = (exit_price - position_info['entry_price']) * position_info['position_size']
                
                # Log time held
                time_held = (datetime.now() - position_info['entry_time']).total_seconds() / 60.0
                self.logger.info(f"{symbol} held for {time_held:.1f} minutes")
                
                trade_record = self._record_trade(symbol, position_info, exit_price, pnl, reason)
                
                # Clean up re-entry candidates if any exist
                if symbol in self.reentry_candidates:
                    del self.reentry_candidates[symbol]
                    
                del self.active_trades[symbol]
                self.risk_manager.daily_pnl += pnl
                self.logger.info(f"Exit executed for {symbol}: ${pnl:.2f} P&L ({reason})")
                
                return {
                    'success': True,
                    'symbol': symbol,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': reason
                }
            
            return {'success': False, 'reason': f"Exit order not filled: {getattr(exit_order, 'status', 'unknown')}"}
        except Exception as e:
            log_api_error(self.logger, f"Error executing exit for {symbol}", e)
            return {'success': False, 'reason': str(e)}

    def _record_trade(self, symbol, position_info, exit_price, pnl, reason):
        trade_record = {
            'symbol': symbol,
            'entry_time': position_info['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': position_info['entry_price'],
            'exit_price': exit_price,
            'position_size': position_info['position_size'],
            'pnl': pnl,
            'exit_reason': reason,
            'was_reentry': position_info.get('is_reentry', False),
            'reentry_number': position_info.get('reentry_count', 0)
        }
        self.trade_history.append(trade_record)
        return trade_record

    def _track_for_reentry(self, symbol: str, exit_price: float, position_info: Dict):
        self.reentry_candidates[symbol] = {
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'reentry_count': position_info['reentry_count'],
            'high_since_exit': exit_price,
            'monitoring': True,
            'original_entry': position_info['entry_price']
        }
        self.logger.info(f"Tracking {symbol} for re-entry opportunity (count: {position_info['reentry_count']}/{self.config.reentry.MAX_REENTRIES_PER_STOCK})")

    def check_reentry_opportunities(self, quotes: Dict) -> List[Dict]:
        """Check all re-entry candidates for setup conditions"""
        reentry_signals = []
        for symbol in list(self.reentry_candidates.keys()):
            if symbol not in quotes:
                continue
            candidate = self.reentry_candidates[symbol]
            current_price = quotes[symbol]['last']
            current_volume = quotes[symbol]['volume']
            time_since_exit = (datetime.now() - candidate['exit_time']).total_seconds()
            if time_since_exit < 300:
                continue
            if current_price > candidate['high_since_exit']:
                candidate['high_since_exit'] = current_price
            pullback = candidate['high_since_exit'] - current_price
            if (self.config.reentry.MIN_PULLBACK_FOR_REENTRY <= pullback <= self.config.reentry.MAX_PULLBACK_FOR_REENTRY):
                if current_volume > quotes[symbol].get('avg_volume', 500000) * 2:
                    if current_price > candidate['high_since_exit'] - pullback * 0.5:
                        reentry_signals.append({
                            'symbol': symbol,
                            'entry_price': current_price,
                            'is_reentry': True,
                            'reentry_number': candidate['reentry_count'] + 1,
                            'pattern_strength': 75,
                            'pullback_amount': pullback
                        })
                        del self.reentry_candidates[symbol]
            elif pullback > self.config.reentry.MAX_PULLBACK_FOR_REENTRY * 2:
                del self.reentry_candidates[symbol]
                self.logger.info(f"Removed {symbol} from re-entry candidates (excessive pullback: ${pullback:.2f})")
            elif time_since_exit > 1800:
                del self.reentry_candidates[symbol]
                self.logger.debug(f"Removed {symbol} from re-entry candidates (timeout)")
        if reentry_signals:
            self.logger.info(f"Re-entry signals generated: {len(reentry_signals)}")
        return reentry_signals

    def check_max_hold_time(self):
        """Check and execute time-based exits for positions exceeding max hold time."""
        current_time = datetime.now()
        for symbol, position in list(self.active_trades.items()):
            try:
                max_hold_time = position.get('max_hold_time')
                if max_hold_time and current_time >= max_hold_time:
                    time_held_minutes = (current_time - position['entry_time']).total_seconds() / 60.0
                    self.logger.info(f"Max hold time reached for {symbol} ({time_held_minutes:.1f} minutes)")
                    self.execute_exit(symbol, reason="max_hold_time")
            except Exception as e:
                log_api_error(self.logger, f"Error checking max hold time for {symbol}", e)

    def update_stops(self):
        """Update ATR trailing stops for all positions"""
        for symbol, position in self.active_trades.items():
            try:
                quote = self.api.get_last_quote(symbol)
                current_price = quote.askprice
                
                # Update recent bars for ATR calculation
                try:
                    position['recent_bars'] = self.api.get_bars(symbol, '1Min', limit=50).df
                except:
                    pass
                
                new_stop = self.risk_manager.update_trailing_stop(symbol, current_price, position)
                if new_stop and new_stop > position['stop_price']:
                    try:
                        self.api.cancel_order(position['stop_order_id'])
                    except Exception:
                        pass
                    
                    stop_order = place_order(
                        self.api, self.logger,
                        symbol=symbol, qty=position['position_size'],
                        side='sell', type='stop', time_in_force='gtc', stop_price=new_stop
                    )
                    position['stop_price'] = new_stop
                    position['stop_order_id'] = stop_order.id if stop_order else None
                    self.logger.info(f"Stop updated for {symbol}: ${new_stop:.2f}")
            except Exception as e:
                log_api_error(self.logger, f"Error updating stop for {symbol}", e)

    def get_reentry_stats(self) -> Dict:
        """Get statistics about re-entry performance"""
        reentry_trades = [t for t in self.trade_history if t.get('was_reentry', False)]
        if not reentry_trades:
            return {'total': 0, 'profitable': 0, 'avg_pnl': 0}
        profitable = [t for t in reentry_trades if t['pnl'] > 0]
        return {
            'total': len(reentry_trades),
            'profitable': len(profitable),
            'win_rate': len(profitable) / len(reentry_trades) * 100,
            'avg_pnl': sum(t['pnl'] for t in reentry_trades) / len(reentry_trades),
            'by_number': self._analyze_by_reentry_number(reentry_trades)
        }

    def _analyze_by_reentry_number(self, trades: List[Dict]) -> Dict:
        by_number = {}
        for trade in trades:
            num = trade.get('reentry_number', 1)
            if num not in by_number:
                by_number[num] = {'count': 0, 'pnl': 0}
            by_number[num]['count'] += 1
            by_number[num]['pnl'] += trade['pnl']
        return {k: {'count': v['count'], 'avg_pnl': v['pnl']/v['count']} for k, v in by_number.items()}