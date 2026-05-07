# =====================================================
# exit_simulator.py - Exit Logic and Position Management
# =====================================================

from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
from utils.logging import get_logger
from strategy.risk_manager import calc_atr_trailing_stop
from utils.trade_metrics import TradeMetrics


class ExitSimulator:
    """Handles exit simulation with ATR trailing stops and slippage."""
    
    def __init__(self, config, pattern_analyzer):
        self.config = config
        self.pattern_analyzer = pattern_analyzer
        self.logger = get_logger(__name__, component="exit_sim")
        self.metrics = TradeMetrics(config, pattern_analyzer)
    
    def simulate_exit(
        self, 
        symbol: str,
        pos: Dict,
        bars_fwd: pd.DataFrame, 
        results: Dict
    ) -> Optional[Dict]:
        """
        Simulate exit with ATR trailing stop, max hold time, and trading window cutoff.
        
        Args:
            symbol: Stock symbol
            pos: Position dictionary with entry details
            bars_fwd: Forward-looking bars after entry
            results: Results dictionary for capital updates
            
        Returns:
            Trade dictionary or None
        """
        entry_price = pos['entry_price']
        entry_time = pos['entry_time']
        size = pos['size']
        gap_pct = pos.get('gap_percent', 0.0)
        trading_window_end_utc = pos.get('trading_window_end_utc')
        
        # Get stored context for trades.csv
        vix_at_entry = pos.get('vix_at_entry')
        volume_at_entry = pos.get('volume_at_entry')
        entry_bar_index = pos.get('entry_bar_index')
        all_bars = pos.get('all_bars', pd.DataFrame())

        # Calculate initial stop
        if self.config.backtest.SIMPLE_STOPS:
            stop_price = entry_price * 0.95  # 5% stop loss
        else:
            stop_price = pos['stop_price']

        # Determine absolute deadline
        if trading_window_end_utc is not None:
            absolute_deadline = trading_window_end_utc
            self.logger.debug(
                f"{symbol}: Trading window deadline set to "
                f"{TradeMetrics.to_est(absolute_deadline).strftime('%H:%M:%S')} ET"
            )
        else:
            max_hold_minutes = self.config.risk.MAX_HOLD_TIME_MINUTES
            absolute_deadline = entry_time + timedelta(minutes=max_hold_minutes)
        
        # ATR trailing stop parameters
        atr_enabled = getattr(self.config.risk, 'ATR_TRAILING_ENABLED', True)
        atr_period = getattr(self.config.risk, 'ATR_TRAILING_PERIOD', 10)
        atr_mult = getattr(self.config.risk, 'ATR_TRAILING_MULTIPLIER', 1.5)
        min_profit_pct = getattr(self.config.risk, 'ATR_TRAILING_MIN_PROFIT_PCT', 1.0) / 100.0

        # Slippage configuration
        slippage_config = self._get_slippage_config()

        exit_reason = None
        exit_price = None
        exit_time = None
        highest_price = entry_price
        trailing_stop = stop_price
        
        # Track max profit price and time
        max_profit_price = entry_price
        max_profit_time = entry_time

        # Check each bar for exit conditions
        for idx, row in bars_fwd.iterrows():
            low = float(row['low'])
            high = float(row['high'])
            close = float(row['close'])
            ts = pd.Timestamp(row['timestamp'])
            
            # Update highest price and max profit tracking
            if high > highest_price:
                highest_price = high
                max_profit_price = high
                max_profit_time = ts
            
            # PRIORITY 1: ABSOLUTE TRADING WINDOW DEADLINE
            if ts >= absolute_deadline:
                exit_reason = 'trading_window_close'
                exit_price = close
                exit_time = ts
                
                current_pnl_pct = (close - entry_price) / entry_price
                self.logger.info(
                    f"{symbol} TRADING WINDOW CLOSED: "
                    f"entry={TradeMetrics.to_est(entry_time).strftime('%H:%M:%S')}, "
                    f"deadline={TradeMetrics.to_est(absolute_deadline).strftime('%H:%M:%S')}, "
                    f"exit={TradeMetrics.to_est(ts).strftime('%H:%M:%S')}, "
                    f"P&L={current_pnl_pct*100:+.2f}% - FORCED EXIT"
                )
                break
            
            # PRIORITY 2: Fixed stop loss
            if low <= stop_price:
                exit_reason = 'stop_loss'
                exit_price = stop_price
                exit_time = ts
                break
            
            # PRIORITY 3: ATR trailing stop
            if atr_enabled:
                profit_pct = (highest_price - entry_price) / entry_price
                
                if profit_pct >= min_profit_pct:
                    bars_for_atr = bars_fwd.loc[:idx].tail(atr_period + 5)
                    
                    if len(bars_for_atr) >= atr_period:
                        new_trailing_stop = calc_atr_trailing_stop(
                            bars=bars_for_atr,
                            highest_price=highest_price,
                            atr_period=atr_period,
                            atr_mult=atr_mult,
                            min_stop_distance_pct=0.01
                        )
                        
                        if new_trailing_stop > trailing_stop:
                            trailing_stop = max(new_trailing_stop, entry_price + 0.01)
                            
                            self.logger.debug(
                                f"{symbol} ATR trailing stop updated: "
                                f"highest=${highest_price:.2f}, "
                                f"new_stop=${trailing_stop:.2f}"
                            )
        
            # Check if trailing stop was hit
            if exit_reason is None and low <= trailing_stop and trailing_stop > stop_price:
                exit_reason = 'trailing_stop'
                exit_price = trailing_stop
                exit_time = ts

        # Handle case where loop ended without exit
        if exit_reason is None:
            exit_price, exit_time, exit_reason = self._handle_no_exit(
                symbol, bars_fwd, entry_price, entry_time, absolute_deadline
            )

        # Final sanity check
        if exit_price is None or exit_time is None:
            self.logger.error(
                f"{symbol}: Failed to determine exit. "
                f"Entry={TradeMetrics.to_est(entry_time).strftime('%H:%M:%S')}, "
                f"Deadline={TradeMetrics.to_est(absolute_deadline).strftime('%H:%M:%S')}, "
                f"Bars={len(bars_fwd)}"
            )
            exit_reason = 'trading_window_close'
            exit_price = entry_price
            exit_time = absolute_deadline

        # Apply slippage adjustments
        original_exit_price = exit_price
        raw_pnl = (exit_price - entry_price) * size
        exit_price = self._apply_slippage(
            symbol, exit_price, raw_pnl, size, entry_price, exit_reason, gap_pct, slippage_config
        )
    
        # Final P&L calculation with slippage
        pnl = (exit_price - entry_price) * size
        results['capital'] += exit_price * size

        # Calculate hold time
        hold_time_minutes = (exit_time - entry_time).total_seconds() / 60.0 if exit_time and entry_time else 0

        # Calculate profit erosion
        profit_erosion_pct = TradeMetrics.calculate_profit_erosion_pct(entry_price, max_profit_price, exit_price)
        
        # Calculate pattern strength evolution
        pattern_strength_5min, pattern_strength_at_exit = self._calculate_pattern_evolution(
            symbol, pos, all_bars, entry_bar_index, exit_time
        )

        # Convert timestamps to EST
        entry_est = TradeMetrics.to_est(pos['entry_time'])
        exit_est = TradeMetrics.to_est(exit_time)
        max_profit_time_est = TradeMetrics.to_est(max_profit_time)

        # Return trade dictionary
        return {
            'symbol': symbol,
            'entry_date': entry_est,
            'exit_date': exit_est,
            'entry_date_str': TradeMetrics.format_date_for_csv(entry_est),
            'entry_time': TradeMetrics.format_time_for_csv(entry_est),
            'exit_date_str': TradeMetrics.format_date_for_csv(exit_est),
            'exit_time': TradeMetrics.format_time_for_csv(exit_est),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_price_before_slippage': original_exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_before_slippage': raw_pnl,
            'exit_reason': exit_reason,
            'return_pct': ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0,
            'days_held': 0,
            'hold_time_minutes': hold_time_minutes,
            'highest_price': highest_price,
            'mae': min(0, ((min(bars_fwd['low']) - entry_price) / entry_price * 100)) if not bars_fwd.empty else 0,
            'mfe': max(0, ((max(bars_fwd['high']) - entry_price) / entry_price * 100)) if not bars_fwd.empty else 0,
            # NEW: Additional columns for trades.csv
            'max_profit_price': max_profit_price,
            'max_profit_time': TradeMetrics.format_timestamp_for_csv(max_profit_time_est),
            'profit_erosion_pct': profit_erosion_pct,
            'volume_at_entry': volume_at_entry,
            'pattern_strength_5min': pattern_strength_5min,
            'pattern_strength_at_exit': pattern_strength_at_exit,
        }
    
    def _get_slippage_config(self) -> Dict:
        """Get slippage configuration from config."""
        return {
            'winner_multiplier': getattr(self.config.risk, 'SLIPPAGE_WINNER_MULTIPLIER', 0.98),
            'loser_multiplier': getattr(self.config.risk, 'SLIPPAGE_LOSER_MULTIPLIER', 1.06),
            'stop_pct_high_gap': getattr(self.config.risk, 'SLIPPAGE_STOP_HIGH_GAP_PCT', 0.12),
            'stop_pct_normal': getattr(self.config.risk, 'SLIPPAGE_STOP_NORMAL_PCT', 0.08),
            'gap_threshold': getattr(self.config.risk, 'SLIPPAGE_GAP_THRESHOLD', 200.0)
        }
    
    def _handle_no_exit(
        self,
        symbol: str,
        bars_fwd: pd.DataFrame,
        entry: float,
        entry_time: pd.Timestamp,
        absolute_deadline: pd.Timestamp
    ) -> tuple:
        """Handle case where no exit condition was met during loop."""
        if bars_fwd.empty:
            # No bars exist after entry - force exit at deadline
            self.logger.warning(
                f"{symbol}: No bars available after entry at "
                f"{TradeMetrics.to_est(entry_time).strftime('%H:%M:%S')}. "
                f"Force exit at trading window close="
                f"{TradeMetrics.to_est(absolute_deadline).strftime('%H:%M:%S')} with entry price"
            )
            return entry, absolute_deadline, 'trading_window_close'
        
        last_bar = bars_fwd.iloc[-1]
        last_ts = pd.Timestamp(last_bar['timestamp'])
        last_close = float(last_bar['close'])
        
        # Find bar closest to deadline
        bars_before_deadline = bars_fwd[bars_fwd['timestamp'] <= absolute_deadline]
        
        if not bars_before_deadline.empty:
            deadline_bar = bars_before_deadline.iloc[-1]
            exit_price = float(deadline_bar['close'])
            exit_time = pd.Timestamp(deadline_bar['timestamp'])
            
            deadline_pnl_pct = (exit_price - entry) / entry
            self.logger.info(
                f"{symbol}: Position closed at trading window deadline: "
                f"{TradeMetrics.to_est(exit_time).strftime('%H:%M:%S')}, "
                f"P&L={deadline_pnl_pct*100:+.2f}%"
            )
            return exit_price, exit_time, 'trading_window_close'
        else:
            # All bars are after deadline (edge case)
            self.logger.warning(
                f"{symbol}: All bars after deadline. Force exit at "
                f"{TradeMetrics.to_est(absolute_deadline).strftime('%H:%M:%S')}"
            )
            return float(bars_fwd.iloc[0]['close']), absolute_deadline, 'trading_window_close'
    
    def _apply_slippage(
        self,
        symbol: str,
        exit_price: float,
        raw_pnl: float,
        size: int,
        entry_price: float,
        exit_reason: str,
        gap_pct: float,
        slippage_config: Dict
    ) -> float:
        """Apply slippage adjustments based on trade outcome."""
        original_exit_price = exit_price
        
        if raw_pnl > 0:
            # Winner - reduce profit
            adjusted_pnl = raw_pnl * slippage_config['winner_multiplier']
            exit_price = entry_price + (adjusted_pnl / size)
        
            self.logger.debug(
                f"{symbol} WINNER slippage applied: "
                f"original_exit=${original_exit_price:.2f}, "
                f"adjusted_exit=${exit_price:.2f}, "
                f"original_pnl=${raw_pnl:.2f}, "
                f"adjusted_pnl=${adjusted_pnl:.2f} "
                f"({slippage_config['winner_multiplier']*100:.0f}% of profit)"
            )
        else:
            # Loser - apply different slippage based on exit reason
            if exit_reason == 'stop_loss':
                # Stop price slips AWAY from entry (worse fill)
                stop_slippage_pct = (
                    slippage_config['stop_pct_high_gap'] 
                    if gap_pct > slippage_config['gap_threshold'] 
                    else slippage_config['stop_pct_normal']
                )
            
                exit_price = original_exit_price * (1 - stop_slippage_pct)
                adjusted_pnl = (exit_price - entry_price) * size
            
                self.logger.debug(
                    f"{symbol} STOP LOSS slippage applied: "
                    f"gap={gap_pct:.1f}%, "
                    f"original_exit=${original_exit_price:.2f}, "
                    f"adjusted_exit=${exit_price:.2f}, "
                    f"original_pnl=${raw_pnl:.2f}, "
                    f"adjusted_pnl=${adjusted_pnl:.2f} "
                    f"({stop_slippage_pct*100:.0f}% worse)"
                )
            else:
                # Time exit or other - modest slippage multiplier
                adjusted_pnl = raw_pnl * slippage_config['loser_multiplier']
                exit_price = entry_price + (adjusted_pnl / size)
            
                self.logger.debug(
                    f"{symbol} LOSER slippage applied (reason={exit_reason}): "
                    f"original_exit=${original_exit_price:.2f}, "
                    f"adjusted_exit=${exit_price:.2f}, "
                    f"original_pnl=${raw_pnl:.2f}, "
                    f"adjusted_pnl=${adjusted_pnl:.2f} "
                    f"({slippage_config['loser_multiplier']*100:.0f}% of loss)"
                )
        
        return exit_price
    
    def _apply_slippage_to_exit(self, base_exit_price: float, is_winner: bool, is_stop_loss: bool, gap_percent: float) -> float:
        """
        Apply slippage simulation to exit price based on trade outcome.
        
        Args:
            base_exit_price: Theoretical exit price without slippage
            is_winner: True if trade is profitable
            is_stop_loss: True if exiting via stop loss
            gap_percent: Gap percentage for the stock
            
        Returns:
            Adjusted exit price after slippage
        """
        # NOTE: Check if slippage is enabled
        if not self.config.risk.ENABLE_SLIPPAGE:
            return base_exit_price
        
        # Apply slippage based on exit type
        if is_stop_loss:
            # Stop loss slippage (most severe)
            if gap_percent > self.config.risk.SLIPPAGE_GAP_THRESHOLD:
                slippage_pct = self.config.risk.SLIPPAGE_STOP_HIGH_GAP_PCT
            else:
                slippage_pct = self.config.risk.SLIPPAGE_STOP_NORMAL_PCT
            
            # Price slips AWAY from entry (reduces exit price for longs)
            adjusted_price = base_exit_price * (1 - slippage_pct)
            
            self.logger.debug(
                f"Stop loss slippage applied: {base_exit_price:.2f} → {adjusted_price:.2f} "
                f"({slippage_pct:.1%} penalty, gap={gap_percent:.0f}%)"
            )
            return adjusted_price
        
        elif is_winner:
            # Winner slippage (reduce profit)
            adjusted_price = base_exit_price * self.config.risk.SLIPPAGE_WINNER_MULTIPLIER
            
            self.logger.debug(
                f"Winner slippage applied: {base_exit_price:.2f} → {adjusted_price:.2f} "
                f"({(1-self.config.risk.SLIPPAGE_WINNER_MULTIPLIER):.1%} profit reduction)"
            )
            return adjusted_price
        
        else:
            # Loser slippage (increase loss) - for time-based exits while underwater
            # Note: This makes the exit price WORSE (lower for longs)
            loss_multiplier = self.config.risk.SLIPPAGE_LOSER_MULTIPLIER
            adjusted_price = base_exit_price * (2.0 - loss_multiplier)
            
            self.logger.debug(
                f"Loser slippage applied: {base_exit_price:.2f} → {adjusted_price:.2f} "
                f"({(loss_multiplier-1):.1%} loss increase)"
            )
            return adjusted_price
    
    def _calculate_pattern_evolution(
        self,
        symbol: str,
        pos: Dict,
        all_bars: pd.DataFrame,
        entry_bar_index: Optional[int],
        exit_time: pd.Timestamp
    ) -> tuple:
        """Calculate pattern strength at 5-minute and exit."""
        pattern_strength_5min = None
        pattern_strength_at_exit = None
        
        if not all_bars.empty and entry_bar_index is not None:
            try:
                # Pattern strength 5 minutes after entry (entry + 5 bars)
                five_min_index = min(entry_bar_index + 5, len(all_bars) - 1)
                bars_5min = all_bars.iloc[:five_min_index + 1]
                
                if len(bars_5min) >= 5:
                    warmup = (self.config.session.PREMARKET_WARMUP_MINUTES 
                             if pos.get('is_premarket') 
                             else self.config.session.REGULAR_WARMUP_MINUTES)
                    
                    pa_5min = self.pattern_analyzer.analyze_pattern(
                        symbol, 
                        bars=bars_5min.tail(warmup),
                        is_premarket=pos.get('is_premarket', False)
                    )
                    pattern_strength_5min = pa_5min.get('pattern_strength', None)
                
                # Pattern strength at exit
                exit_bars = all_bars[all_bars['timestamp'] <= exit_time]
                if not exit_bars.empty:
                    exit_bar_index = exit_bars.index[-1]
                    bars_at_exit = all_bars.loc[:exit_bar_index]
                    
                    if len(bars_at_exit) >= 5:
                        warmup = (self.config.session.PREMARKET_WARMUP_MINUTES 
                                 if pos.get('is_premarket') 
                                 else self.config.session.REGULAR_WARMUP_MINUTES)
                        
                        pa_exit = self.pattern_analyzer.analyze_pattern(
                            symbol,
                            bars=bars_at_exit.tail(warmup),
                            is_premarket=pos.get('is_premarket', False)
                        )
                        pattern_strength_at_exit = pa_exit.get('pattern_strength', None)
                        
            except Exception as e:
                self.logger.debug(f"{symbol}: Error calculating pattern strength evolution: {e}")
        
        return pattern_strength_5min, pattern_strength_at_exit
