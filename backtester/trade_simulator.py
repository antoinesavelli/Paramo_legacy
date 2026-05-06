# =====================================================
# trade_simulator.py - Trade Simulation and Position Management
# =====================================================

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from utils.logging import get_logger
from strategy.risk_manager import calc_position_size_percentage
from market_context.scoring import position_size_adjustment_from_indicators
from screener.backtest import CandidateSignal
from utils.trade_metrics import TradeMetrics
from backtester.exit_simulator import ExitSimulator


class TradeSimulator:
    """Handles trade execution, position management, and exit simulation for backtesting."""
    
    def __init__(self, config, data_handler, pattern_analyzer, market_context=None):
        self.config = config
        self.data_handler = data_handler
        self.pattern_analyzer = pattern_analyzer
        self.market_context = market_context
        self.logger = get_logger(__name__, component="trade_sim")
        self.positions: Dict[str, Dict] = {}
        
        # Initialize helper components
        self.metrics = TradeMetrics(config, pattern_analyzer)
        self.exit_simulator = ExitSimulator(config, pattern_analyzer)
    
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
        
        # Get market context for the day
        market_adjustment, vix_at_entry = self._get_market_context(day)
        
        session_cfg = self.config.session
        warmup = session_cfg.PREMARKET_WARMUP_MINUTES if premarket_enabled else session_cfg.REGULAR_WARMUP_MINUTES
        max_per_day = getattr(self.config.backtest, "MAX_CANDIDATES_PER_DAY", 5)
        
        # Get analysis window end time (hard cutoff for all trades)
        trading_window_end_utc = self._get_trading_window_end(day)
        
        opened = 0
        rejected = 0

        for idx, sig in enumerate(signals[:max_per_day], 1):
            if len(self.positions) >= self.config.risk.MAX_CONCURRENT_POSITIONS:
                self.logger.info(
                    f"[POSITION LIMIT] Max concurrent positions reached "
                    f"({self.config.risk.MAX_CONCURRENT_POSITIONS})"
                )
                break

            # Process individual signal
            signal_result = self._process_signal(
                idx=idx,
                sig=sig,
                results=results,
                session_start_utc=session_start_utc,
                session_end_utc=session_end_utc,
                warmup=warmup,
                premarket_enabled=premarket_enabled,
                market_adjustment=market_adjustment,
                vix_at_entry=vix_at_entry,
                trading_window_end_utc=trading_window_end_utc,
                max_per_day=max_per_day
            )
            
            if signal_result == 'opened':
                opened += 1
            elif signal_result == 'rejected':
                rejected += 1

        return opened, rejected
    
    def _get_market_context(self, day: datetime) -> tuple[float, Optional[float]]:
        """Get market context adjustment and VIX level."""
        market_adjustment = 1.0
        vix_at_entry = None
        
        if self.market_context:
            try:
                self.market_context.update_market_context(day)
                market_adjustment = position_size_adjustment_from_indicators(
                    self.market_context.market_indicators,
                    self.config.market_context
                )
                env = self.market_context.market_indicators.get('trading_environment', 'neutral')
                score = self.market_context.market_indicators.get('market_score', 50)
                
                # Extract VIX level
                vix_data = self.market_context.market_indicators.get('vix_level', {})
                vix_at_entry = vix_data.get('level', None)
                
                self.logger.info(
                    f"[MARKET CONTEXT] {day.date()}: env={env}, "
                    f"score={score:.1f}, VIX={vix_at_entry:.1f}, size_adj={market_adjustment:.2f}x"
                )
            except Exception as e:
                self.logger.warning(f"Failed to get market context: {e}, using 1.0x")
                market_adjustment = 1.0
                vix_at_entry = None
        
        return market_adjustment, vix_at_entry
    
    def _get_trading_window_end(self, day: datetime) -> Optional[pd.Timestamp]:
        """Get trading window end time if enabled."""
        analysis_window_enabled = getattr(self.config.backtest, 'ANALYSIS_WINDOW_ENABLED', False)
        
        if analysis_window_enabled:
            window_end_str = getattr(self.config.backtest, 'ANALYSIS_WINDOW_END_ET', '12:00')
            window_end = datetime.strptime(window_end_str, "%H:%M").time()
            
            trading_window_end_et = pd.Timestamp(
                day.replace(hour=window_end.hour, minute=window_end.minute),
                tz='US/Eastern'
            )
            trading_window_end_utc = trading_window_end_et.tz_convert('UTC')
            
            self.logger.info(
                f"[TRADING WINDOW] Enabled: All positions will be closed by {window_end_str} ET"
            )
            return trading_window_end_utc
        
        return None
    
    def _process_signal(
        self,
        idx: int,
        sig: CandidateSignal,
        results: Dict,
        session_start_utc: pd.Timestamp,
        session_end_utc: pd.Timestamp,
        warmup: int,
        premarket_enabled: bool,
        market_adjustment: float,
        vix_at_entry: Optional[float],
        trading_window_end_utc: Optional[pd.Timestamp],
        max_per_day: int
    ) -> str:
        """
        Process a single signal.
        
        Returns:
            'opened', 'rejected' or 'skipped'
        """
        symbol = sig.symbol
        entry_ts = sig.entry_ts
        entry_price = sig.entry_price
        stop_price = sig.stop_price
        gap_pct = sig.gap_percent
        pattern_strength = sig.pattern_strength
        is_premarket = sig.meta.get('is_premarket', False)

        self.logger.debug(f"[{idx}/{max_per_day}] Processing {symbol}...")

        risk_ps = entry_price - stop_price
        
        # Position sizing
        if is_premarket:
            base_size = self._position_size(entry_price, stop_price, results['capital'], market_adjustment)
            size = max(1, base_size // 2)
        else:
            size = self._position_size(entry_price, stop_price, results['capital'], market_adjustment)
        
        # Format entry timestamp for diagnostics
        entry_date_str = TradeMetrics.format_date_for_csv(entry_ts)
        entry_time_str = TradeMetrics.format_time_for_csv(entry_ts)
        entry_et = TradeMetrics.to_est(entry_ts)
        
        # Get all bars for entry bar identification
        all_bars = self.data_handler.get_intraday_bars(
            symbol, '1Min', start=session_start_utc, end=session_end_utc
        )
        
        # Calculate entry metrics
        entry_metrics = self._calculate_entry_metrics(
            all_bars, entry_ts, entry_et, is_premarket, gap_pct
        )
        
        # Validation checks
        validation_result = self._validate_signal(
            symbol=symbol,
            risk_ps=risk_ps,
            size=size,
            entry_price=entry_price,
            results=results,
            entry_date_str=entry_date_str,
            entry_time_str=entry_time_str,
            gap_pct=gap_pct,
            vix_at_entry=vix_at_entry,
            entry_metrics=entry_metrics
        )
        
        if validation_result is not None:
            return validation_result

        # Apply slippage adjustment if enabled
        gap_threshold = getattr(self.config.risk, 'SLIPPAGE_GAP_THRESHOLD', 200.0)
        if self.config.risk.ENABLE_SLIPPAGE:
            slip_pct = (
                self.config.risk.ENTRY_SLIPPAGE_HIGH_GAP_PCT
                if gap_pct > gap_threshold
                else self.config.risk.ENTRY_SLIPPAGE_PCT
            )
            entry_price = entry_price * (1 + slip_pct)

        # Get detailed pattern metrics
        bars = self.data_handler.get_intraday_bars(
            symbol, start=session_start_utc, end=session_end_utc
        )
        pattern_metrics = self.metrics.extract_pattern_metrics(
            symbol, bars, warmup, premarket_enabled
        )

        # Open position
        self._open_position(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            size=size,
            entry_ts=entry_ts,
            is_premarket=is_premarket,
            gap_pct=gap_pct,
            pattern_strength=pattern_strength,
            pattern_metrics=pattern_metrics,
            trading_window_end_utc=trading_window_end_utc,
            vix_at_entry=vix_at_entry,
            entry_metrics=entry_metrics,
            all_bars=all_bars
        )
        
        results['capital'] -= size * entry_price

        # Create entry diagnostic
        self._create_entry_diagnostic(
            results=results,
            entry_date_str=entry_date_str,
            entry_time_str=entry_time_str,
            symbol=symbol,
            gap_pct=gap_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            size=size,
            is_premarket=is_premarket,
            risk_ps=risk_ps,
            vix_at_entry=vix_at_entry,
            entry_metrics=entry_metrics,
            pattern_metrics=pattern_metrics
        )

        # Simulate exit
        bars_fwd = all_bars[all_bars['timestamp'] > entry_ts] if not all_bars.empty else all_bars
        trade = self.exit_simulator.simulate_exit(symbol, self.positions[symbol], bars_fwd, results)
        
        if trade:
            self._handle_trade_exit(
                trade=trade,
                symbol=symbol,
                pattern_strength=pattern_strength,
                pattern_metrics=pattern_metrics,
                results=results,
                entry_date_str=entry_date_str,
                entry_time_str=entry_time_str,
                gap_pct=gap_pct,
                vix_at_entry=vix_at_entry,
                entry_metrics=entry_metrics,
                is_premarket=is_premarket
            )
            
            del self.positions[symbol]

        return 'opened'
    
    def _calculate_entry_metrics(
        self,
        all_bars: pd.DataFrame,
        entry_ts: pd.Timestamp,
        entry_et: pd.Timestamp,
        is_premarket: bool,
        gap_pct: float
    ) -> Dict:
        """Calculate metrics at entry time."""
        entry_bar_volume = None
        volume_freshness_ratio = 1.0
        entry_bar_index = None
        time_bucket = TradeMetrics.get_time_bucket(entry_et)
        gap_type = "premarket" if is_premarket else "intraday"
        
        if not all_bars.empty:
            # Find the bar closest to entry timestamp
            all_bars_copy = all_bars.copy()
            all_bars_copy['ts_diff'] = (all_bars_copy['timestamp'] - entry_ts).abs()
            entry_bar_idx = all_bars_copy['ts_diff'].idxmin()
            entry_bar_index = all_bars.index.get_loc(entry_bar_idx)
            
            # Get entry bar volume
            entry_bar_volume = int(all_bars.loc[entry_bar_idx, 'volume'])
            
            # Calculate volume freshness ratio
            volume_freshness_ratio = TradeMetrics.calculate_volume_freshness_ratio(all_bars, entry_bar_index)
            
            # Determine gap type
            gap_type = TradeMetrics.get_gap_type(all_bars, entry_bar_index, entry_ts, gap_pct)
        
        return {
            'entry_bar_volume': entry_bar_volume,
            'volume_freshness_ratio': volume_freshness_ratio,
            'entry_bar_index': entry_bar_index,
            'time_bucket': time_bucket,
            'gap_type': gap_type
        }
    
    def _validate_signal(
        self,
        symbol: str,
        risk_ps: float,
        size: int,
        entry_price: float,
        results: Dict,
        entry_date_str: str,
        entry_time_str: str,
        gap_pct: float,
        vix_at_entry: Optional[float],
        entry_metrics: Dict
    ) -> Optional[str]:
        """
        Validate signal. Returns 'rejected' if invalid, None if valid.
        """
        # Risk per share validation
        if risk_ps <= 0:
            self._create_rejection_diagnostic(
                results, entry_date_str, entry_time_str, symbol, gap_pct,
                "risk_ps_non_positive", vix_at_entry, entry_metrics
            )
            return 'rejected'
        
        # Size validation
        if size < 1:
            self._create_rejection_diagnostic(
                results, entry_date_str, entry_time_str, symbol, gap_pct,
                "size_zero", vix_at_entry, entry_metrics
            )
            return 'rejected'
        
        # Capital validation
        cost = size * entry_price
        if cost > results['capital']:
            self._create_rejection_diagnostic(
                results, entry_date_str, entry_time_str, symbol, gap_pct,
                "insufficient_capital", vix_at_entry, entry_metrics
            )
            return 'rejected'
        
        # Volume participation validation
        if entry_metrics['entry_bar_volume'] is not None:
            max_participation = 0.10   # don't consume more than 10% of bar volume
            max_size_by_volume = int(entry_metrics['entry_bar_volume'] * max_participation)
            if size > max_size_by_volume > 0:
                size = max(1, max_size_by_volume)
        
        return None
    
    def _create_rejection_diagnostic(
        self,
        results: Dict,
        entry_date_str: str,
        entry_time_str: str,
        symbol: str,
        gap_pct: float,
        reason: str,
        vix_at_entry: Optional[float],
        entry_metrics: Dict
    ):
        """Create rejection diagnostic entry."""
        results['candidate_diagnostics'].append({
            "date": entry_date_str,
            "time": entry_time_str,
            "symbol": symbol, 
            "gap_percent": gap_pct,
            "phase": "reject", 
            "reason": reason,
            # Additional context columns
            "vix_at_entry": vix_at_entry,
            "time_of_day_bucket": entry_metrics['time_bucket'],
            "gap_type": entry_metrics['gap_type'],
            "volume_at_entry": entry_metrics['entry_bar_volume'],
            "volume_freshness_ratio": entry_metrics['volume_freshness_ratio'],
            # Delineated fields
            "insufficient_patterns_value": None,
            "insufficient_patterns_passed": None,
            "retention_value": None,
            "retention_passed": None,
            "accel_value": None,
            "accel_passed": None,
            "other_reason": reason
        })
    
    def _open_position(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        size: int,
        entry_ts: pd.Timestamp,
        is_premarket: bool,
        gap_pct: float,
        pattern_strength: float,
        pattern_metrics: Dict,
        trading_window_end_utc: Optional[pd.Timestamp],
        vix_at_entry: Optional[float],
        entry_metrics: Dict,
        all_bars: pd.DataFrame
    ):
        """Open a new position."""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'stop_price': stop_price,
            'size': size,
            'entry_time': entry_ts,
            'is_premarket': is_premarket,
            'gap_percent': gap_pct,
            'pattern_strength': pattern_strength,
            'pattern_metrics': pattern_metrics,
            'trading_window_end_utc': trading_window_end_utc,
            # Store context for trades.csv
            'vix_at_entry': vix_at_entry,
            'volume_at_entry': entry_metrics['entry_bar_volume'],
            'entry_bar_index': entry_metrics['entry_bar_index'],
            'all_bars': all_bars
        }
    
    def _create_entry_diagnostic(
        self,
        results: Dict,
        entry_date_str: str,
        entry_time_str: str,
        symbol: str,
        gap_pct: float,
        entry_price: float,
        stop_price: float,
        size: int,
        is_premarket: bool,
        risk_ps: float,
        vix_at_entry: Optional[float],
        entry_metrics: Dict,
        pattern_metrics: Dict
    ):
        """Create entry diagnostic entry."""
        entry_diagnostic = {
            "date": entry_date_str,
            "time": entry_time_str,
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
            # Additional context columns
            "vix_at_entry": vix_at_entry,
            "time_of_day_bucket": entry_metrics['time_bucket'],
            "gap_type": entry_metrics['gap_type'],
            "volume_at_entry": entry_metrics['entry_bar_volume'],
            "volume_freshness_ratio": entry_metrics['volume_freshness_ratio'],
            # Delineated fields
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
    
    def _handle_trade_exit(
        self,
        trade: Dict,
        symbol: str,
        pattern_strength: float,
        pattern_metrics: Dict,
        results: Dict,
        entry_date_str: str,
        entry_time_str: str,
        gap_pct: float,
        vix_at_entry: Optional[float],
        entry_metrics: Dict,
        is_premarket: bool
    ):
        """Handle trade exit logging and diagnostics."""
        # NOTE: Return proceeds to capital (entry cost was debited in _process_signal)
        exit_price = trade['exit_price']
        size = trade['size']
        results['capital'] += exit_price * size

        # Log trade exit
        patterns = pattern_metrics.get('patterns_detected', '')
        self.logger.info(
            f"[EXIT] {symbol} | "
            f"Price: ${exit_price:.2f} | "
            f"P&L: ${trade['pnl']:+.2f} ({trade['return_pct']:+.2f}%) | "
            f"Reason: {trade['exit_reason']} | "
            f"Pattern Score: {pattern_strength:.1f} | "
            f"Patterns: {patterns}"
        )
        
        trade['is_premarket_entry'] = is_premarket
        trade['gap_percent'] = gap_pct
        trade['pattern_strength'] = pattern_strength
        results['trades'].append(trade)
        
        # Format exit timestamp for diagnostics
        exit_date_str = trade.get('exit_date_str', '')
        exit_time_str = trade.get('exit_time_str', '')
        
        # Create exit diagnostic
        exit_diagnostic = {
            "date": entry_date_str,
            "time": entry_time_str,
            "exit_date": exit_date_str,
            "exit_time": exit_time_str,
            "symbol": symbol, 
            "gap_percent": gap_pct,
            "phase": "exited", 
            "reason": trade['exit_reason'],
            "pnl": trade['pnl'],
            "return_pct": trade['return_pct'],
            "hold_time_minutes": TradeMetrics.calculate_hold_time(
                self.positions[symbol]['entry_time'], 
                trade.get('exit_date')
            ),
            # Additional context columns
            "vix_at_entry": vix_at_entry,
            "time_of_day_bucket": entry_metrics['time_bucket'],
            "gap_type": entry_metrics['gap_type'],
            "volume_at_entry": entry_metrics['entry_bar_volume'],
            "volume_freshness_ratio": entry_metrics['volume_freshness_ratio'],
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
        
        # Convert timestamps to EST
        entry_est = TradeMetrics.to_est(pos['entry_time'])
        exit_est = TradeMetrics.to_est(exit_time)
        
        # Store trade
        trade = {
            'symbol': symbol,
            'entry_date': entry_est,
            'exit_date': exit_est,
            'entry_date_str': TradeMetrics.format_date_for_csv(entry_est),
            'entry_time': TradeMetrics.format_time_for_csv(entry_est),
            'exit_date_str': TradeMetrics.format_date_for_csv(exit_est),
            'exit_time': TradeMetrics.format_time_for_csv(exit_est),
            'entry_price': entry,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'exit_reason': reason,
            'return_pct': ((exit_price - entry) / entry) * 100 if entry else 0,
            'days_held': 0,
            'is_premarket_entry': pos.get('is_premarket', False),
            'gap_percent': gap_pct,
            'pattern_strength': pos.get('pattern_strength', 0),
            # Add context columns (with None for forced close)
            'max_profit_price': None,
            'max_profit_time': None,
            'profit_erosion_pct': None,
            'volume_at_entry': pos.get('volume_at_entry'),
            'pattern_strength_5min': None,
            'pattern_strength_at_exit': None,
        }
        results.setdefault('trades', []).append(trade)
        
        # Exit diagnostic
        exit_diagnostic = {
            "date": pd.Timestamp(day).date(),
            "symbol": symbol,
            "gap_percent": gap_pct,
            "phase": "exited",
            "reason": reason,
            "pnl": pnl,
            "return_pct": trade['return_pct'],
            # Add context columns
            "vix_at_entry": pos.get('vix_at_entry'),
            "time_of_day_bucket": None,
            "gap_type": "premarket" if pos.get('is_premarket', False) else "intraday",
            "volume_at_entry": pos.get('volume_at_entry'),
            "volume_freshness_ratio": None,
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
            market_adjustment=market_adjustment
        )
