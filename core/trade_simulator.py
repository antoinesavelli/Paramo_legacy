# =====================================================
# trade_simulator.py - Trade Simulation and Position Management
# =====================================================

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from utils.logging import get_logger
from core.risk_manager import calc_atr_stop, calc_position_size_percentage, calc_atr_trailing_stop
from market_context.scoring import position_size_adjustment_from_indicators
from screener.backtest import CandidateSignal


class TradeSimulator:
    """Handles trade execution, position management, and exit simulation for backtesting."""
    
    def __init__(self, config, data_handler, pattern_analyzer, market_context=None):
        self.config = config
        self.data_handler = data_handler
        self.pattern_analyzer = pattern_analyzer
        self.market_context = market_context
        self.logger = get_logger(__name__, component="trade_sim")
        self.positions: Dict[str, Dict] = {}
    
    def process_signals(
        self, 
        day: datetime, 
        signals: List[CandidateSignal],
        results: Dict,
        session_start_utc: pd.Timestamp,
        session_end_utc: pd.Timestamp,
        premarket_enabled: bool
    ) -> tuple[int, int]:
        """Process candidate signals and simulate trades with market context."""
        
        # ✅ Get market context for the day
        market_adjustment = 1.0
        if self.market_context:
            try:
                self.market_context.update_market_context(day)
                market_adjustment = position_size_adjustment_from_indicators(
                    self.market_context.market_indicators,
                    self.config.market_context
                )
                env = self.market_context.market_indicators.get('trading_environment', 'neutral')
                score = self.market_context.market_indicators.get('market_score', 50)
                self.logger.info(
                    f"[MARKET CONTEXT] {day.date()}: env={env}, "
                    f"score={score:.1f}, size_adj={market_adjustment:.2f}x"
                )
            except Exception as e:
                self.logger.warning(f"Failed to get market context: {e}, using 1.0x")
                market_adjustment = 1.0
        
        session_cfg = self.config.session
        warmup = session_cfg.PREMARKET_WARMUP_MINUTES if premarket_enabled else session_cfg.REGULAR_WARMUP_MINUTES
        max_per_day = getattr(self.config.backtest, "MAX_CANDIDATES_PER_DAY", 5)
        
        opened = 0
        rejected = 0

        for idx, sig in enumerate(signals[:max_per_day], 1):
            if len(self.positions) >= self.config.risk.MAX_CONCURRENT_POSITIONS:
                self.logger.info(
                    f"[POSITION LIMIT] Max concurrent positions reached "
                    f"({self.config.risk.MAX_CONCURRENT_POSITIONS})"
                )
                break

            symbol = sig.symbol
            entry_ts = sig.entry_ts
            entry_price = sig.entry_price
            stop_price = sig.stop_price
            gap_pct = sig.gap_percent
            pattern_strength = sig.pattern_strength
            is_premarket = sig.meta.get('is_premarket', False)

            self.logger.debug(f"[{idx}/{min(len(signals), max_per_day)}] Processing {symbol}...")

            risk_ps = entry_price - stop_price
            
            # Position sizing using percentage-based calculation
            if is_premarket:
                base_size = self._position_size(entry_price, stop_price, results['capital'], market_adjustment)
                size = max(1, base_size // 2)
            else:
                size = self._position_size(entry_price, stop_price, results['capital'], market_adjustment)
            
            # Validation checks
            if risk_ps <= 0:
                rejected += 1
                results['candidate_diagnostics'].append({
                    "date": pd.Timestamp(day).date(), 
                    "symbol": symbol, 
                    "gap_percent": gap_pct,
                    "phase": "reject", 
                    "reason": "risk_ps_non_positive",
                    # Delineated fields
                    "insufficient_patterns_value": None,
                    "insufficient_patterns_passed": None,
                    "retention_value": None,
                    "retention_passed": None,
                    "accel_value": None,
                    "accel_passed": None,
                    "other_reason": "risk_ps_non_positive"
                })
                continue
                
            if size < 1:
                rejected += 1
                results['candidate_diagnostics'].append({
                    "date": pd.Timestamp(day).date(), 
                    "symbol": symbol, 
                    "gap_percent": gap_pct,
                    "phase": "reject", 
                    "reason": "size_zero",
                    # Delineated fields
                    "insufficient_patterns_value": None,
                    "insufficient_patterns_passed": None,
                    "retention_value": None,
                    "retention_passed": None,
                    "accel_value": None,
                    "accel_passed": None,
                    "other_reason": "size_zero"
                })
                continue

            cost = size * entry_price
            if cost > results['capital']:
                rejected += 1
                results['candidate_diagnostics'].append({
                    "date": pd.Timestamp(day).date(), 
                    "symbol": symbol, 
                    "gap_percent": gap_pct,
                    "phase": "reject", 
                    "reason": "insufficient_capital",
                    # Delineated fields
                    "insufficient_patterns_value": None,
                    "insufficient_patterns_passed": None,
                    "retention_value": None,
                    "retention_passed": None,
                    "accel_value": None,
                    "accel_passed": None,
                    "other_reason": "insufficient_capital"
                })
                continue

            # Get detailed pattern metrics
            pattern_metrics = self._extract_pattern_metrics(
                symbol, 
                session_start_utc, 
                session_end_utc, 
                warmup, 
                premarket_enabled
            )

            # Open position
            self.positions[symbol] = {
                'entry_price': entry_price,
                'stop_price': stop_price,
                'size': size,
                'entry_time': entry_ts,
                'is_premarket': is_premarket,
                'gap_percent': gap_pct,
                'pattern_strength': pattern_strength,
                'pattern_metrics': pattern_metrics
            }
            results['capital'] -= cost
            opened += 1

            # Entry diagnostic with delineated fields
            entry_diagnostic = {
                "date": pd.Timestamp(day).date(), 
                "symbol": symbol, 
                "gap_percent": gap_pct,
                "phase": "entered", 
                "reason": "signal_valid",
                "entry": entry_price, 
                "stop": stop_price, 
                "size": size,
                "is_premarket": is_premarket,
                "risk_per_share": risk_ps,
                "total_risk": risk_ps * size,
                # Delineated fields (all passed for entered trades)
                "insufficient_patterns_value": pattern_metrics.get('pattern_count', 0),
                "insufficient_patterns_passed": True,
                "retention_value": pattern_metrics.get('step_up_retention', 0),
                "retention_passed": True,
                "accel_value": pattern_metrics.get('parabolic_acceleration', 0),
                "accel_passed": True,
                "other_reason": None
            }
            entry_diagnostic.update(pattern_metrics)
            results['candidate_diagnostics'].append(entry_diagnostic)

            # Simulate exits
            all_bars = self.data_handler.get_intraday_bars(
                symbol, '1Min', start=session_start_utc, end=session_end_utc
            )
            bars_fwd = all_bars[all_bars['timestamp'] > entry_ts] if not all_bars.empty else all_bars
            trade = self.simulate_exit(symbol, bars_fwd, day, results)
            
            if trade:
                # Log trade exit
                patterns = pattern_metrics.get('patterns_detected', '')
                self.logger.info(
                    f"[EXIT] {symbol} | "
                    f"Price: ${trade['exit_price']:.2f} | "
                    f"P&L: ${trade['pnl']:+.2f} ({trade['return_pct']:+.2f}%) | "
                    f"Reason: {trade['exit_reason']} | "
                    f"Pattern Score: {pattern_strength:.1f} | "
                    f"Patterns: {patterns}"
                )
                
                trade['is_premarket_entry'] = is_premarket
                trade['gap_percent'] = gap_pct
                trade['pattern_strength'] = pattern_strength
                results['trades'].append(trade)
                
                # Exit diagnostic with delineated fields
                exit_diagnostic = {
                    "date": pd.Timestamp(day).date(), 
                    "symbol": symbol, 
                    "gap_percent": gap_pct,
                    "phase": "exited", 
                    "reason": trade['exit_reason'],
                    "pnl": trade['pnl'],
                    "return_pct": trade['return_pct'],
                    "hold_time_minutes": self._calculate_hold_time(
                        self.positions[symbol]['entry_time'], 
                        trade.get('exit_date')
                    ),
                    # Delineated fields (all passed for exited trades)
                    "insufficient_patterns_value": pattern_metrics.get('pattern_count', 0),
                    "insufficient_patterns_passed": True,
                    "retention_value": pattern_metrics.get('step_up_retention', 0),
                    "retention_passed": True,
                    "accel_value": pattern_metrics.get('parabolic_acceleration', 0),
                    "accel_passed": True,
                    "other_reason": None
                }
                exit_diagnostic.update(pattern_metrics)
                results['candidate_diagnostics'].append(exit_diagnostic)
                
                del self.positions[symbol]

        return opened, rejected
    
    def _extract_pattern_metrics(
        self, 
        symbol: str, 
        session_start_utc: pd.Timestamp, 
        session_end_utc: pd.Timestamp,
        warmup: int,
        is_premarket: bool
    ) -> Dict:
        """Extract detailed pattern metrics for a symbol."""
        bars = self.data_handler.get_intraday_bars(
            symbol, start=session_start_utc, end=session_end_utc
        )
        
        pattern_metrics = {}
        if not bars.empty and len(bars) >= warmup:
            bars_warm = bars.iloc[:warmup].copy()
            pa = self.pattern_analyzer.analyze_pattern(
                symbol, bars=bars_warm, is_premarket=is_premarket
            )
            
            # Extract detailed metrics
            step_up = pa.get('step_ups', {}) or {}
            parabolic = pa.get('parabolic', {}) or {}
            breakout = pa.get('breakout', {}) or {}
            volume = pa.get('volume', {}) or {}
            sr = pa.get('support_resistance', {}) or {}
            
            pattern_metrics = {
                'pattern_strength': pa.get('pattern_strength', 0),
                'pattern_count': pa.get('pattern_count', 0),
                'patterns_detected': '|'.join(pa.get('patterns_detected', [])),
                
                # Step-up metrics
                'step_up_detected': step_up.get('detected', False),
                'step_up_count': step_up.get('step_count', 0),
                'step_up_retention': step_up.get('retention_rate', 0),
                'step_up_total_advance': step_up.get('total_advance', 0),
                'step_up_strength': step_up.get('strength', 0),
                
                # Parabolic metrics
                'parabolic_detected': parabolic.get('detected', False),
                'parabolic_angle': parabolic.get('angle', 0),
                'parabolic_acceleration': parabolic.get('acceleration', 0),
                'parabolic_vol_multiplier': parabolic.get('volume_multiplier', 0),
                'parabolic_strength': parabolic.get('strength', 0),
                
                # Breakout metrics
                'breakout_detected': breakout.get('detected', False),
                'breakout_level': breakout.get('breakout_level', 0),
                'breakout_range_size': breakout.get('range_size', 0),
                'breakout_vol_ratio': breakout.get('volume_ratio', 0),
                'breakout_strength': breakout.get('strength', 0),
                
                # Volume metrics
                'volume_trend': volume.get('volume_trend', 'unknown'),
                'volume_avg': volume.get('avg_volume', 0),
                'volume_recent': volume.get('recent_volume', 0),
                'volume_high_correlation': volume.get('high_volume_correlation', 0),
                'volume_strength': volume.get('strength', 0),
                
                # Support/Resistance metrics
                'sr_current_price': sr.get('current_price', 0),
                'sr_support_levels': len(sr.get('support', [])),
                'sr_resistance_levels': len(sr.get('resistance', [])),
                'sr_strength': sr.get('strength', 0),
            }
        
        return pattern_metrics
    
    def simulate_exit(
        self, 
        symbol: str, 
        bars_fwd: pd.DataFrame, 
        day: datetime, 
        results: Dict
    ) -> Optional[Dict]:
        """Simulate exit with ATR trailing stop, max hold time, and fixed stop only."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        entry = pos['entry_price']
        entry_time = pos['entry_time']
        size = pos['size']

        # Calculate initial stop (NO PROFIT TARGET)
        if self.config.backtest.SIMPLE_STOPS:
            stop = entry * 0.95  # 5% stop loss
        else:
            stop = pos['stop_price']

        # ✅ FIXED: Max hold time is EXACTLY 30 minutes from entry
        max_hold_minutes = self.config.risk.MAX_HOLD_TIME_MINUTES
        max_hold_time = entry_time + timedelta(minutes=max_hold_minutes)

        # ATR trailing stop parameters
        atr_enabled = getattr(self.config.risk, 'ATR_TRAILING_ENABLED', True)
        atr_period = getattr(self.config.risk, 'ATR_TRAILING_PERIOD', 10)
        atr_mult = getattr(self.config.risk, 'ATR_TRAILING_MULTIPLIER', 1.5)
        min_profit_pct = getattr(self.config.risk, 'ATR_TRAILING_MIN_PROFIT_PCT', 1.0) / 100.0

        exit_reason = None
        exit_price = None
        exit_time = None
        highest_price = entry
        trailing_stop = stop

        # Check each bar for exit conditions
        for idx, row in bars_fwd.iterrows():
            low = float(row['low'])
            high = float(row['high'])
            close = float(row['close'])
            ts = pd.Timestamp(row['timestamp'])
            
            # Update highest price
            if high > highest_price:
                highest_price = high
            
            # PRIORITY 1: Max hold time (STRICT - exit on first bar at/after deadline)
            if ts >= max_hold_time:
                exit_reason = 'max_hold_time'
                exit_price = close
                exit_time = ts
                self.logger.debug(
                    f"{symbol} max hold time reached: "
                    f"entry={entry_time}, deadline={max_hold_time}, exit={ts}"
                )
                break
            
            # PRIORITY 2: Fixed stop loss
            if low <= stop:
                exit_reason = 'stop_loss'
                exit_price = stop
                exit_time = ts
                break
            
            # PRIORITY 3: ATR-based trailing stop logic
            if atr_enabled:
                profit_pct = (highest_price - entry) / entry
                
                # Only activate trailing stop after minimum profit
                if profit_pct >= min_profit_pct:
                    # Get recent bars for ATR calculation (last N bars up to current)
                    bars_for_atr = bars_fwd.loc[:idx].tail(atr_period + 5)
                    
                    if len(bars_for_atr) >= atr_period:
                        # Calculate ATR trailing stop
                        new_trailing_stop = calc_atr_trailing_stop(
                            bars=bars_for_atr,
                            highest_price=highest_price,
                            atr_period=atr_period,
                            atr_mult=atr_mult,
                            min_stop_distance_pct=0.01
                        )
                        
                        # Update trailing stop (only move up, never down)
                        if new_trailing_stop > trailing_stop:
                            trailing_stop = max(new_trailing_stop, entry + 0.01)  # Never below breakeven
                            
                            self.logger.debug(
                                f"{symbol} ATR trailing stop updated: "
                                f"highest=${highest_price:.2f}, "
                                f"new_stop=${trailing_stop:.2f}"
                            )
            
            # Check if trailing stop was hit
            if low <= trailing_stop and trailing_stop > stop:
                exit_reason = 'trailing_stop'
                exit_price = trailing_stop
                exit_time = ts
                break

        # ✅ CRITICAL FIX: Handle case where loop ended without exit
        if exit_reason is None:
            if bars_fwd.empty:
                # No bars exist after entry - force exit at deadline with entry price
                exit_reason = 'max_hold_time_no_data'
                exit_price = entry
                exit_time = max_hold_time
                self.logger.warning(
                    f"{symbol}: No bars available after entry at {entry_time}. "
                    f"Force exit at max_hold_time={max_hold_time} with entry price"
                )
            else:
                last_bar = bars_fwd.iloc[-1]
                last_ts = pd.Timestamp(last_bar['timestamp'])
                
                # Check if we ran out of bars before hitting the deadline
                if last_ts < max_hold_time:
                    # ✅ NEW: Exit at the last available bar before deadline
                    exit_reason = 'max_hold_time'
                    exit_price = float(last_bar['close'])
                    exit_time = last_ts
                    self.logger.warning(
                        f"{symbol}: Last bar at {last_ts} is before deadline {max_hold_time}. "
                        f"Exiting at last bar (hold time: {(last_ts - entry_time).total_seconds() / 60.0:.1f} min)"
                    )
                else:
                    # Last bar is after deadline - find closest bar to deadline
                    bars_before_deadline = bars_fwd[bars_fwd['timestamp'] <= max_hold_time]
                    
                    if not bars_before_deadline.empty:
                        deadline_bar = bars_before_deadline.iloc[-1]
                        exit_reason = 'max_hold_time'
                        exit_price = float(deadline_bar['close'])
                        exit_time = pd.Timestamp(deadline_bar['timestamp'])
                        self.logger.debug(
                            f"{symbol}: Exited at closest bar to deadline: {exit_time}"
                        )
                    else:
                        # Edge case: All bars are after the deadline (shouldn't happen with 1-min data)
                        exit_reason = 'max_hold_time'
                        exit_price = float(bars_fwd.iloc[0]['close'])
                        exit_time = pd.Timestamp(bars_fwd.iloc[0]['timestamp'])
                        self.logger.warning(
                            f"{symbol}: All bars after deadline. Using first bar at {exit_time}"
                        )

        # Final sanity check
        if exit_price is None or exit_time is None:
            self.logger.error(
                f"{symbol}: Failed to determine exit. "
                f"Entry={entry_time}, Deadline={max_hold_time}, Bars={len(bars_fwd)}"
            )
            exit_reason = 'max_hold_time_emergency'
            exit_price = entry
            exit_time = max_hold_time

        pnl = (exit_price - entry) * size
        results['capital'] += exit_price * size

        # Calculate actual hold time
        hold_time_minutes = (exit_time - entry_time).total_seconds() / 60.0 if exit_time and entry_time else 0

        # ✅ Log warning if hold time exceeds limit (for debugging)
        if hold_time_minutes > max_hold_minutes + 1:  # Allow 1-minute tolerance for bar timing
            self.logger.warning(
                f"{symbol}: Hold time {hold_time_minutes:.1f} min exceeds "
                f"limit of {max_hold_minutes} min (reason: {exit_reason})"
            )

        return {
            'symbol': symbol,
            'entry_date': pos['entry_time'],
            'exit_date': exit_time,
            'entry_price': entry,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'return_pct': ((exit_price - entry) / entry) * 100 if entry else 0,
            'days_held': 0,
            'hold_time_minutes': hold_time_minutes,
            'highest_price': highest_price,
            'mae': ((entry - min(bars_fwd['low'])) / entry * 100) if not bars_fwd.empty else 0,
            'mfe': ((max(bars_fwd['high']) - entry) / entry * 100) if not bars_fwd.empty else 0,
        }
    
    def close_position(
        self, 
        symbol: str, 
        day: datetime, 
        results: Dict, 
        reason: str
    ) -> Optional[Dict]:
        """Force-close an open position."""
        if symbol not in self.positions:
            return None
        
        session_cfg = self.config.session
        if session_cfg.PREMARKET_ENABLED:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(
                day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern'
            )
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
        
        session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')
        
        bars = self.data_handler.get_intraday_bars(
            symbol, '1Min', start=session_start_utc, end=session_end_utc
        )
        
        pos = self.positions[symbol]
        entry = pos['entry_price']
        size = pos['size']
        gap_pct = pos.get('gap_percent', 0.0)
        pattern_metrics = pos.get('pattern_metrics', {})
        
        if bars is None or bars.empty:
            exit_price = entry
            exit_time = session_end_utc
        else:
            last = bars.iloc[-1]
            exit_price = float(last['close'])
            exit_time = pd.Timestamp(last['timestamp'])
        
        pnl = (exit_price - entry) * size
        results['capital'] += exit_price * size
        
        trade = {
            'symbol': symbol,
            'entry_date': pos['entry_time'],
            'exit_date': exit_time,
            'entry_price': entry,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'exit_reason': reason,
            'return_pct': ((exit_price - entry) / entry) * 100 if entry else 0,
            'days_held': 0,
            'is_premarket_entry': pos.get('is_premarket', False),
            'gap_percent': gap_pct,
            'pattern_strength': pos.get('pattern_strength', 0)
        }
        results.setdefault('trades', []).append(trade)
        
        # Exit diagnostic with delineated fields
        exit_diagnostic = {
            "date": pd.Timestamp(day).date(),
            "symbol": symbol,
            "gap_percent": gap_pct,
            "phase": "exited",
            "reason": reason,
            "pnl": pnl,
            "return_pct": trade['return_pct'],
            # Delineated fields
            "insufficient_patterns_value": pattern_metrics.get('pattern_count', 0),
            "insufficient_patterns_passed": True,
            "retention_value": pattern_metrics.get('step_up_retention', 0),
            "retention_passed": True,
            "accel_value": pattern_metrics.get('parabolic_acceleration', 0),
            "accel_passed": True,
            "other_reason": None
        }
        exit_diagnostic.update(pattern_metrics)
        results['candidate_diagnostics'].append(exit_diagnostic)
        
        del self.positions[symbol]
        return trade
    
    def close_all_positions(self, end_date: datetime, results: Dict) -> None:
        """Force-close all open positions."""
        for symbol in list(self.positions.keys()):
            self.logger.info(f"Force closing position: {symbol}")
            self.close_position(symbol, end_date, results, reason="backtest_end")
    
    def _position_size(self, entry: float, stop: float, cash: float, market_adjustment: float = 1.0) -> int:
        """Calculate position size based on percentage-based risk parameters with market context."""
        return calc_position_size_percentage(
            entry=entry,
            stop=stop,
            account_equity=cash,
            stop_loss_pct=self.config.risk.STOP_LOSS_PERCENT_OF_ACCOUNT,
            max_position_pct=self.config.risk.MAX_POSITION_SIZE_PERCENT,
            market_adjustment=market_adjustment  # ✅ Pass market context
        )
    
    def _calculate_hold_time(self, entry_ts, exit_ts) -> float:
        """Calculate hold time in minutes."""
        try:
            if entry_ts is None or exit_ts is None:
                return 0.0
            entry = pd.Timestamp(entry_ts)
            exit = pd.Timestamp(exit_ts)
            return (exit - entry).total_seconds() / 60.0
        except Exception:
            return 0.0
