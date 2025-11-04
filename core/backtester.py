# =====================================================
# backtester.py - Backtesting Logic for Intraday Trading
# =====================================================

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import os
import numpy as np
import math
import random
import time
from utils.logging import get_logger
from data_handler.local import LocalDataHandler
from core.pattern_analyzer import PatternAnalyzer
from utils.reporting import compute_statistics
from news.backtest import NewsIntegrationBacktest
from market_context.backtest import BacktestMarketContext
from core.risk_manager import calc_atr_stop, calc_position_size, calc_profit_target
from screener.backtest import BacktestScreener, CandidateSignal

class Backtester:
    """Intraday backtesting using 1-minute bars from local per-day Parquet files."""

    def __init__(self, config, data_handler: LocalDataHandler, screener=None, pattern_analyzer: Optional[PatternAnalyzer] = None, news_integration: Optional[NewsIntegrationBacktest] = None):
        self.config = config
        self.data_handler = data_handler
        self.pattern_analyzer = pattern_analyzer or PatternAnalyzer(config, data_handler)
        self.logger = get_logger(__name__, component="intraday_bt")
        self.positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.news_integration = news_integration or NewsIntegrationBacktest(config, data_dir=self.config.backtest.NEWS_DATA_DIR)
        self._day_sentiment: Optional[float] = None
        self.market_context_bt = BacktestMarketContext(config)
        self.bt_screener = BacktestScreener(config, data_handler, self.pattern_analyzer, self.news_integration)

    def run_backtest(self, start_date: datetime, end_date: datetime, initial_capital: float = 10000) -> Dict:
        """Run backtest with enhanced progress tracking."""
        
        # ✅ NEW: Track overall progress
        total_start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info(f"BACKTEST START: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        self.logger.info("=" * 80)
        
        results = self._init_results(initial_capital)
        
        # ✅ NEW: Calculate total days and trading days
        cur = start_date
        all_days = []
        while cur <= end_date:
            all_days.append(cur)
            cur += timedelta(days=1)
        
        total_days = len(all_days)
        trading_days_count = 0
        skipped_days_count = 0
        
        self.logger.info(f"Total calendar days to process: {total_days}")
        
        # ✅ NEW: Process each day with progress tracking
        for day_idx, cur_day in enumerate(all_days, 1):
            day_start_time = time.time()
            
            # Skip weekends
            if cur_day.weekday() >= 5:
                skipped_days_count += 1
                self.logger.debug(f"[{day_idx}/{total_days}] {cur_day.date()} - SKIP (weekend)")
                continue
            
            # Check if data exists (fast check)
            day_str = cur_day.strftime('%Y-%m-%d')
            if not self.data_handler.has_data_for_date(day_str):
                skipped_days_count += 1
                self.logger.info(f"[{day_idx}/{total_days}] {cur_day.date()} - SKIP (no data)")
                continue
            
            # ✅ NEW: Log day start
            trading_days_count += 1
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"DAY {trading_days_count} | {cur_day.date()} | Progress: {day_idx}/{total_days} days ({day_idx/total_days*100:.1f}%)")
            self.logger.info(f"Portfolio: ${results['capital']:,.2f} | Open Positions: {len(self.positions)}")
            self.logger.info("=" * 80)
            
            # Load market sentiment
            self._day_sentiment = self._load_market_sentiment(cur_day)
            if self._day_sentiment is not None:
                min_sentiment = getattr(self.config.backtest, "MIN_MARKET_SENTIMENT", 30)
                if self._day_sentiment < min_sentiment:
                    self.logger.info(f"SKIP: Low market sentiment ({self._day_sentiment:.1f} < {min_sentiment})")
                    skipped_days_count += 1
                    continue
            
            # Simulate the trading day
            try:
                self.daily_pnl = 0.0
                self._simulate_intraday_session(cur_day, results)
                
                # ✅ NEW: Log day completion
                day_elapsed = time.time() - day_start_time
                total_elapsed = time.time() - total_start_time
                avg_time_per_day = total_elapsed / day_idx
                days_remaining = total_days - day_idx
                est_time_remaining = avg_time_per_day * days_remaining
                
                self.logger.info(f"Day processed in {day_elapsed:.1f}s | "
                               f"Total time: {total_elapsed:.0f}s | "
                               f"Est. remaining: {est_time_remaining:.0f}s ({est_time_remaining/60:.1f}m)")
                
            except Exception as e:
                self.logger.exception(f"Error processing {cur_day.date()}: {e}")
                # Continue to next day instead of crashing
                continue

        # ✅ NEW: Close remaining positions
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("CLOSING REMAINING POSITIONS")
        self.logger.info("=" * 80)
        
        for symbol in list(self.positions.keys()):
            self.logger.info(f"Force closing position: {symbol}")
            self._close_position(symbol, end_date, results, reason="backtest_end")

        # Calculate statistics
        initial = results['equity_curve'][0]['equity'] if results['equity_curve'] else initial_capital
        final = results['equity_curve'][-1]['equity'] if results['equity_curve'] else initial
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("CALCULATING STATISTICS")
        self.logger.info("=" * 80)
        
        results['statistics'] = compute_statistics(
            trades=results['trades'],
            equity_curve=results['equity_curve'],
            initial_capital=initial,
            final_capital=final,
            daily_returns=None,
            trading_days=trading_days_count
        )

        # Export diagnostics
        diags = results.get('candidate_diagnostics', [])
        if diags:
            self.logger.info(f"Exporting {len(diags)} diagnostic records...")
            diag_df = pd.DataFrame(diags)
            out_fp = os.path.abspath("backtest_candidates.csv")
            try:
                diag_df.to_csv(out_fp, index=False)
                self.logger.info(f"✓ Diagnostics exported: {out_fp}")
            except Exception as e:
                self.logger.error(f"Failed to export diagnostics: {e}")

            # Summary by rejection reason
            try:
                rej = diag_df[diag_df['phase'].isin(['gap_scan_reject', 'reject'])].copy()
                if not rej.empty:
                    grp = rej.groupby('reason')
                    summary = grp.agg(
                        count=('reason', 'size'),
                        unique_symbols=('symbol', pd.Series.nunique),
                        days=('date', pd.Series.nunique)
                    ).reset_index().sort_values('count', ascending=False)

                    summary_fp = os.path.abspath("backtest_reject_summary.csv")
                    summary.to_csv(summary_fp, index=False)
                    
                    self.logger.info("Top rejection reasons:")
                    for _, row in summary.head(10).iterrows():
                        self.logger.info(f"  • {row['reason']}: {row['count']} occurrences")
                    
                    self.logger.info(f"✓ Reject summary exported: {summary_fp}")
            except Exception as e:
                self.logger.error(f"Failed to build reject summary: {e}")

        # ✅ NEW: Final summary
        total_elapsed = time.time() - total_start_time
        stats = results['statistics']
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("BACKTEST COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
        self.logger.info(f"Days processed: {day_idx}/{total_days}")
        self.logger.info(f"  • Trading days: {trading_days_count}")
        self.logger.info(f"  • Skipped days: {skipped_days_count}")
        self.logger.info(f"Total trades: {stats.get('total_trades', 0)}")
        self.logger.info(f"Win rate: {stats.get('win_rate', 0):.1f}%")
        self.logger.info(f"Total return: {stats.get('total_return', 0):.2f}%")
        self.logger.info(f"Max drawdown: {stats.get('max_drawdown', 0):.2f}%")
        self.logger.info(f"Profit factor: {stats.get('profit_factor', 0):.2f}")
        self.logger.info(f"Sharpe ratio: {stats.get('sharpe_ratio', 0):.2f}")
        self.logger.info("=" * 80)
        
        return results

    def _simulate_intraday_session(self, day: datetime, results: Dict) -> None:
        """Simulate full intraday session with progress logging."""
        
        session_start_time = time.time()
        
        # Get session configuration
        session_cfg = self.config.backtest.SESSION
        premarket_enabled = session_cfg.PREMARKET_ENABLED
        
        # Define session boundaries
        if premarket_enabled:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern')
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
        
        session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')

        # ✅ ADDED: Log screening start
        self.logger.info(f"[SCREENING] Starting candidate screening...")
        screen_start = time.time()
        
        # Screen for candidates
        screen_res = self.bt_screener.screen(day)
        
        screen_elapsed = time.time() - screen_start
        self.logger.info(f"[SCREENING] Completed in {screen_elapsed:.1f}s")
        
        diagnostics = screen_res.get("diagnostics", [])
        if diagnostics:
            results['candidate_diagnostics'].extend(diagnostics)

        signals: List[CandidateSignal] = screen_res.get("signals", [])
        self.logger.info(f"[SIGNALS] Found {len(signals)} candidate signals")

        # ✅ ADDED: Process signals with progress
        max_per_day = getattr(self.config.backtest, "MAX_CANDIDATES_PER_DAY", 5)
        opened = 0
        rejected = 0

        for idx, sig in enumerate(signals[:max_per_day], 1):
            if len(self.positions) >= self.config.risk.MAX_CONCURRENT_POSITIONS:
                self.logger.info(f"[POSITION LIMIT] Max concurrent positions reached ({self.config.risk.MAX_CONCURRENT_POSITIONS})")
                break

            symbol = sig.symbol
            entry_ts = sig.entry_ts
            entry_price = sig.entry_price
            stop_price = sig.stop_price
            is_premarket = sig.meta.get('is_premarket', False)

            self.logger.debug(f"[{idx}/{min(len(signals), max_per_day)}] Processing {symbol}...")

            risk_ps = entry_price - stop_price
            
            # Position sizing
            if is_premarket:
                base_size = self._position_size(entry_price, stop_price, results['capital'])
                size = max(1, base_size // 2)
            else:
                size = self._position_size(entry_price, stop_price, results['capital'])
            
            # Validation checks
            if risk_ps <= 0:
                rejected += 1
                results['candidate_diagnostics'].append({
                    "date": pd.Timestamp(day).date(), 
                    "symbol": symbol, 
                    "phase": "reject", 
                    "reason": "risk_ps_non_positive"
                })
                continue
                
            if size < 1:
                rejected += 1
                results['candidate_diagnostics'].append({
                    "date": pd.Timestamp(day).date(), 
                    "symbol": symbol, 
                    "phase": "reject", 
                    "reason": "size_zero"
                })
                continue

            cost = size * entry_price
            if cost > results['capital']:
                rejected += 1
                results['candidate_diagnostics'].append({
                    "date": pd.Timestamp(day).date(), 
                    "symbol": symbol, 
                    "phase": "reject", 
                    "reason": "insufficient_capital"
                })
                continue

            # ✅ ADDED: Log trade entry
            results['capital'] -= cost
            self.positions[symbol] = {
                'entry_time': entry_ts,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'size': size,
                'is_premarket': is_premarket
            }
            opened += 1
            
            self.logger.info(
                f"[ENTRY] {symbol} | "
                f"Price: ${entry_price:.2f} | "
                f"Size: {size} | "
                f"Cost: ${cost:.2f} | "
                f"Stop: ${stop_price:.2f} | "
                f"PM: {is_premarket}"
            )
            
            results['candidate_diagnostics'].append({
                "date": pd.Timestamp(day).date(), 
                "symbol": symbol, 
                "phase": "entered", 
                "entry": entry_price, 
                "stop": stop_price, 
                "size": size,
                "is_premarket": is_premarket
            })

            # Simulate exits
            all_bars = self.data_handler.get_intraday_bars(symbol, '1Min', start=session_start_utc, end=session_end_utc)
            bars_fwd = all_bars[all_bars['timestamp'] > entry_ts] if not all_bars.empty else all_bars
            trade = self._simulate_exits(symbol, bars_fwd, day, results)
            
            if trade:
                # ✅ ADDED: Log trade exit
                self.logger.info(
                    f"[EXIT] {symbol} | "
                    f"Price: ${trade['exit_price']:.2f} | "
                    f"P&L: ${trade['pnl']:+.2f} ({trade['return_pct']:+.2f}%) | "
                    f"Reason: {trade['exit_reason']}"
                )
                
                trade['market_sentiment'] = float(self._day_sentiment) if self._day_sentiment is not None else None
                trade['is_premarket_entry'] = is_premarket
                results['trades'].append(trade)
                results['candidate_diagnostics'].append({
                    "date": pd.Timestamp(day).date(), 
                    "symbol": symbol, 
                    "phase": "exited", 
                    "exit_reason": trade['exit_reason'], 
                    "pnl": trade['pnl']
                })
                del self.positions[symbol]

        # ✅ ADDED: Session summary
        session_elapsed = time.time() - session_start_time
        results['equity_curve'].append({
            'date': day, 
            'equity': results['capital'], 
            'cash': results['capital']
        })
        
        self.logger.info("")
        self.logger.info(f"[SESSION SUMMARY]")
        self.logger.info(f"  • Signals evaluated: {len(signals)}")
        self.logger.info(f"  • Positions opened: {opened}")
        self.logger.info(f"  • Rejected: {rejected}")
        self.logger.info(f"  • Current capital: ${results['capital']:,.2f}")
        self.logger.info(f"  • Open positions: {len(self.positions)}")
        self.logger.info(f"  • Session time: {session_elapsed:.1f}s")

    def _intraday_stop(self, entry: float, bars_warm: pd.DataFrame) -> float:
        return calc_atr_stop(bars_warm, entry, atr_period=14, atr_mult=2.0, fallback_pct=0.03)

    def _position_size(self, entry: float, stop: float, cash: float) -> int:
        return calc_position_size(entry, stop, cash, self.config.risk.MAX_RISK_PER_TRADE, max_value_frac=0.25)

    def _simulate_exits(self, symbol: str, bars_fwd: pd.DataFrame, 
           day: datetime, results: Dict) -> Optional[Dict]:
        """Simplified exit simulation."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        entry = pos['entry_price']
        size = pos['size']

        if self.config.backtest.SIMPLE_STOPS:
            stop = entry * 0.95
            target = entry * 1.02
        else:
            stop = pos['stop_price']
            target = calc_profit_target(entry, self.config.risk.PROFIT_TARGET_CENTS)

        exit_reason = None
        exit_price = None
        exit_time = None

        for _, row in bars_fwd.iterrows():
            low = float(row['low'])
            high = float(row['high'])
            ts = pd.Timestamp(row['timestamp'])
            
            if low <= stop:
                exit_reason = 'stop_loss'
                exit_price = stop
                exit_time = ts
                break
            if high >= target:
                exit_reason = 'profit_target'
                exit_price = target
                exit_time = ts
                break

        if exit_reason is None and not bars_fwd.empty:
            last = bars_fwd.iloc[-1]
            exit_reason = 'eod_close'
            exit_price = float(last['close'])
            exit_time = pd.Timestamp(last['timestamp'])

        if exit_price is None:
            return None

        pnl = (exit_price - entry) * size
        results['capital'] += exit_price * size

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
            'days_held': 0
        }

    def _close_position(self, symbol: str, day: datetime, results: Dict, reason: str) -> Optional[Dict]:
        """Force-close an open position."""
        if symbol not in self.positions:
            return None
        
        session_cfg = self.config.backtest.SESSION
        if session_cfg.PREMARKET_ENABLED:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern')
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
        
        session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')
        
        bars = self.data_handler.get_intraday_bars(symbol, '1Min', start=session_start_utc, end=session_end_utc)
        pos = self.positions[symbol]
        entry = pos['entry_price']
        size = pos['size']
        
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
            'is_premarket_entry': pos.get('is_premarket', False)
        }
        results.setdefault('trades', []).append(trade)
        del self.positions[symbol]
        return trade

    def _init_results(self, initial_capital: float) -> Dict:
        return {
            'trades': [], 
            'equity_curve': [], 
            'statistics': {}, 
            'capital': initial_capital, 
            'candidate_diagnostics': []
        }

    def _load_market_sentiment(self, day: datetime) -> Optional[float]:
        """Load daily market sentiment score."""
        # Implementation unchanged
        return None