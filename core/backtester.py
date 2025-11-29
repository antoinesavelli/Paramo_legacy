# =====================================================
# backtester.py - Backtesting Logic for Intraday Trading
# =====================================================

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import os
import time
import gc
import numpy as np
from utils.logging import get_logger
from data_handler.local import LocalDataHandler
from core.pattern_analyzer import PatternAnalyzer
from core.trade_simulator import TradeSimulator
from utils.reporting import compute_statistics
from news.backtest import NewsIntegrationBacktest
from market_context.backtest import BacktestMarketContext
from screener.backtest import BacktestScreener, CandidateSignal


def _calculate_missed_winner_metrics(diag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missed winner probability scores for rejected trades.
    Compares rejected trades against winner profiles using weighted distance.
    """
    if diag_df.empty or 'phase' not in diag_df.columns:
        return diag_df
    
    df = diag_df.copy()
    
    # Identify winners and losers
    winners = df[(df['phase'] == 'exited') & (df.get('pnl', 0) > 0)]
    losers = df[(df['phase'] == 'exited') & (df.get('pnl', 0) <= 0)]
    rejected = df[df['phase'] == 'reject']
    
    if winners.empty or rejected.empty:
        # No winners to compare against
        df['missed_winner_score'] = 0.0
        df['similarity_to_winners'] = 0.0
        df['likely_missed_winner'] = False
        return df
    
    # Calculate winner profile (average metrics)
    winner_metrics = {
        'avg_angle': winners.get('parabolic_angle', 0).mean() if 'parabolic_angle' in winners.columns else 0,
        'avg_retention': winners.get('step_up_retention', 0).mean() if 'step_up_retention' in winners.columns else 0,
        'avg_steps': winners.get('step_up_steps', 0).mean() if 'step_up_steps' in winners.columns else 0,
        'avg_gap': winners.get('gap_percent', 0).mean() if 'gap_percent' in winners.columns else 0,
        'avg_score': winners.get('pattern_strength', 0).mean() if 'pattern_strength' in winners.columns else 0
    }
    
    # Calculate loser profile
    loser_metrics = {
        'avg_angle': losers.get('parabolic_angle', 0).mean() if 'parabolic_angle' in losers.columns and not losers.empty else 0,
        'avg_retention': losers.get('step_up_retention', 0).mean() if 'step_up_retention' in losers.columns and not losers.empty else 0,
        'avg_steps': losers.get('step_up_steps', 0).mean() if 'step_up_steps' in losers.columns and not losers.empty else 0,
        'avg_gap': losers.get('gap_percent', 0).mean() if 'gap_percent' in losers.columns and not losers.empty else 0
    }
    
    def calculate_similarity_score(row):
        """Calculate weighted distance from winner profile."""
        if pd.isna(row.get('parabolic_angle')) and pd.isna(row.get('step_up_retention')):
            return 0.0
        
        # Extract metrics with safe fallbacks
        angle = row.get('parabolic_angle', 0) if pd.notna(row.get('parabolic_angle')) else 0
        retention = row.get('step_up_retention', 0) if pd.notna(row.get('step_up_retention')) else 0
        steps = row.get('step_up_steps', 0) if pd.notna(row.get('step_up_steps')) else 0
        gap = row.get('gap_percent', 0) if pd.notna(row.get('gap_percent')) else 0
        score = row.get('pattern_strength', 0) if pd.notna(row.get('pattern_strength')) else 0
        
        # Weighted distance calculation (lower = more similar to winners)
        weights = {'angle': 0.25, 'retention': 0.30, 'steps': 0.20, 'gap': 0.15, 'score': 0.10}
        
        distance = 0.0
        if winner_metrics['avg_angle'] != 0:
            distance += weights['angle'] * abs(angle - winner_metrics['avg_angle']) / abs(winner_metrics['avg_angle'])
        if winner_metrics['avg_retention'] != 0:
            distance += weights['retention'] * abs(retention - winner_metrics['avg_retention']) / winner_metrics['avg_retention']
        if winner_metrics['avg_steps'] != 0:
            distance += weights['steps'] * abs(steps - winner_metrics['avg_steps']) / winner_metrics['avg_steps']
        if winner_metrics['avg_gap'] != 0:
            distance += weights['gap'] * abs(gap - winner_metrics['avg_gap']) / winner_metrics['avg_gap']
        if winner_metrics['avg_score'] != 0:
            distance += weights['score'] * abs(score - winner_metrics['avg_score']) / winner_metrics['avg_score']
        
        # Convert distance to similarity score (inverse, 0-100 scale)
        similarity = max(0, 100 - (distance * 20))  # Scale factor of 20 to spread the range
        return similarity
    
    # Calculate similarity for rejected trades only
    df['similarity_to_winners'] = 0.0
    df['missed_winner_score'] = 0.0
    df['likely_missed_winner'] = False
    
    rejected_mask = df['phase'] == 'reject'
    if rejected_mask.any():
        df.loc[rejected_mask, 'similarity_to_winners'] = df[rejected_mask].apply(calculate_similarity_score, axis=1)
        
        # Flag high-similarity rejected trades (threshold: 70+)
        high_similarity_mask = rejected_mask & (df['similarity_to_winners'] >= 70)
        df.loc[high_similarity_mask, 'likely_missed_winner'] = True
        df.loc[high_similarity_mask, 'missed_winner_score'] = df.loc[high_similarity_mask, 'similarity_to_winners']
    
    return df


def _calculate_gate_impact_matrix(diag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate gate impact matrix showing:
    - Count: How many trades blocked by each gate
    - Winner-like %: Percentage matching winner profile
    - Loser-like %: Percentage matching loser profile
    - Net value estimate: Potential PnL if gate removed
    """
    if diag_df.empty or 'reason' not in diag_df.columns:
        return pd.DataFrame()
    
    # Identify winners and losers
    winners = diag_df[(diag_df['phase'] == 'exited') & (diag_df.get('pnl', 0) > 0)]
    losers = diag_df[(diag_df['phase'] == 'exited') & (diag_df.get('pnl', 0) <= 0)]
    # ✅ FIX: Use .copy() to avoid SettingWithCopyWarning
    rejected = diag_df[diag_df['phase'] == 'reject'].copy()
    
    if rejected.empty:
        return pd.DataFrame()
    
    # Winner/Loser profile thresholds
    winner_profile = {
        'min_score': winners.get('pattern_strength', 0).quantile(0.25) if not winners.empty and 'pattern_strength' in winners.columns else 15,
        'min_steps': 2,
        'negative_angle_ok': True,  # Winners can have negative angles
        'min_retention': 40
    }
    
    def matches_winner_profile(row):
        """Check if row matches winner characteristics."""
        score_match = row.get('pattern_strength', 0) >= winner_profile['min_score']
        steps_match = row.get('step_up_steps', 0) >= winner_profile['min_steps']
        retention_match = row.get('step_up_retention', 0) >= winner_profile['min_retention']
        angle = row.get('parabolic_angle', 0)
        angle_match = angle < 0  # Negative angles more common in winners
        
        # At least 3 out of 4 criteria
        matches = sum([score_match, steps_match, retention_match, angle_match])
        return matches >= 3
    
    def matches_loser_profile(row):
        """Check if row matches loser characteristics."""
        score_low = row.get('pattern_strength', 0) < 10
        steps_low = row.get('step_up_steps', 0) < 2
        retention_low = row.get('step_up_retention', 0) < 40
        
        # At least 2 out of 3 criteria
        matches = sum([score_low, steps_low, retention_low])
        return matches >= 2
    
    # Extract rejection reasons (handle complex reasons like "pattern_invalid:low_score,retention")
    # ✅ Now safe because rejected is a copy
    rejected['gate'] = rejected['reason'].apply(lambda x: x.split(':')[0] if isinstance(x, str) else 'unknown')
    
    # Calculate gate impact
    gate_stats = []
    for gate in rejected['gate'].unique():
        gate_trades = rejected[rejected['gate'] == gate]
        count = len(gate_trades)
        
        winner_like = gate_trades.apply(matches_winner_profile, axis=1).sum()
        loser_like = gate_trades.apply(matches_loser_profile, axis=1).sum()
        
        winner_pct = (winner_like / count * 100) if count > 0 else 0
        loser_pct = (loser_like / count * 100) if count > 0 else 0
        
        # Estimate net value: assume winner-like would have avg winner PnL, loser-like would have avg loser PnL
        avg_winner_pnl = winners['pnl'].mean() if not winners.empty and 'pnl' in winners.columns else 0
        avg_loser_pnl = losers['pnl'].mean() if not losers.empty and 'pnl' in losers.columns else 0
        
        est_net_value = (winner_like * avg_winner_pnl) + (loser_like * avg_loser_pnl)
        
        gate_stats.append({
            'gate': gate,
            'count': count,
            'winner_like_count': winner_like,
            'winner_like_pct': round(winner_pct, 1),
            'loser_like_count': loser_like,
            'loser_like_pct': round(loser_pct, 1),
            'est_net_value': round(est_net_value, 2),
            'avg_gap_pct': round(gate_trades.get('gap_percent', 0).mean(), 2) if 'gap_percent' in gate_trades.columns else 0
        })
    
    gate_df = pd.DataFrame(gate_stats).sort_values('count', ascending=False)
    return gate_df


def _identify_false_negative_signatures(diag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify false negative patterns:
    - High score (≥15) but failed one gate
    - Negative angles (like winners)
    - 2-3 steps (like winners)
    - But rejected on retention/acceleration/etc
    """
    if diag_df.empty:
        return pd.DataFrame()
    
    rejected = diag_df[diag_df['phase'] == 'reject'].copy()
    
    if rejected.empty:
        return pd.DataFrame()
    
    # False negative criteria
    false_negatives = rejected[
        (rejected.get('pattern_strength', 0) >= 15) &  # High score
        (
            (rejected.get('parabolic_angle', 0) < 0) |  # Negative angle
            (rejected.get('step_up_steps', 0).between(2, 3))  # 2-3 steps
        ) &
        (
            rejected['reason'].str.contains('retention', case=False, na=False) |
            rejected['reason'].str.contains('acceleration', case=False, na=False) |
            rejected['reason'].str.contains('angle', case=False, na=False)
        )
    ]
    
    # Add signature classification
    if not false_negatives.empty:
        false_negatives = false_negatives.copy()
        false_negatives['fn_signature'] = false_negatives.apply(
            lambda row: f"score={row.get('pattern_strength', 0):.1f}_"
                       f"angle={row.get('parabolic_angle', 0):.1f}_"
                       f"steps={row.get('step_up_steps', 0)}_"
                       f"retention={row.get('step_up_retention', 0):.1f}",
            axis=1
        )
    
    return false_negatives


def _delineate_reason_column(diag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Delineate the 'reason' column into a boolean 'rejected' column.
    Does NOT create individual binary reason_ columns (removed to reduce clutter).
    
    Args:
        diag_df: DataFrame with 'reason' and 'phase' columns
        
    Returns:
        DataFrame with 'rejected' boolean column added
    """
    if 'reason' not in diag_df.columns or 'phase' not in diag_df.columns:
        return diag_df
    
    df = diag_df.copy()
    
    # Add boolean 'rejected' column
    df['rejected'] = (df['phase'] == 'reject').astype(bool)
    
    # Reorder columns: put 'rejected' right after 'phase'
    cols = df.columns.tolist()
    
    if 'phase' in cols:
        phase_idx = cols.index('phase')
        # Insert 'rejected' right after 'phase'
        new_order = cols[:phase_idx+1] + ['rejected'] + [c for c in cols[phase_idx+1:] if c != 'rejected']
        df = df[new_order]
    
    return df


class Backtester:
    """Intraday backtesting orchestrator using 1-minute bars from local per-day Parquet files."""

    def __init__(
        self, 
        config, 
        data_handler: LocalDataHandler, 
        screener=None, 
        pattern_analyzer: Optional[PatternAnalyzer] = None, 
        news_integration: Optional[NewsIntegrationBacktest] = None
    ):
        self.config = config
        self.data_handler = data_handler
        self.pattern_analyzer = pattern_analyzer or PatternAnalyzer(config, data_handler)
        self.logger = get_logger(__name__, component="intraday_bt")
        self.daily_pnl = 0.0
        
        self.news_integration = news_integration or NewsIntegrationBacktest(
            config, data_dir=self.config.backtest.NEWS_DATA_DIR
        )
        self._day_sentiment: Optional[float] = None
        
        # ✅ Initialize market context for backtest
        self.market_context_bt = BacktestMarketContext(config)
        
        self.bt_screener = BacktestScreener(
            config, data_handler, self.pattern_analyzer, self.news_integration
        )
        
        # ✅ Pass market context to trade simulator
        self.trade_simulator = TradeSimulator(
            config, 
            data_handler, 
            self.pattern_analyzer,
            market_context=self.market_context_bt  # ✅ NEW
        )

    def run_backtest(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        initial_capital: float = 10000
    ) -> Dict:
        """Run backtest with enhanced progress tracking."""
        
        total_start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info(f"BACKTEST START: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        self.logger.info("=" * 80)
        
        results = self._init_results(initial_capital)
        
        # Calculate total days and trading days
        cur = start_date
        all_days = []
        while cur <= end_date:
            all_days.append(cur)
            cur += timedelta(days=1)
        
        total_days = len(all_days)
        trading_days_count = 0
        skipped_days_count = 0
        
        self.logger.info(f"Total calendar days to process: {total_days}")
        
        # Process each day with progress tracking
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
            
            # Log day start
            trading_days_count += 1
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(
                f"DAY {trading_days_count} | {cur_day.date()} | "
                f"Progress: {day_idx}/{total_days} days ({day_idx/total_days*100:.1f}%)"
            )
            self.logger.info(
                f"Portfolio: ${results['capital']:,.2f} | "
                f"Open Positions: {len(self.trade_simulator.positions)}"
            )
            self.logger.info("=" * 80)
            
            # Load market sentiment
            self._day_sentiment = self._load_market_sentiment(cur_day)
            if self._day_sentiment is not None:
                min_sentiment = getattr(self.config.backtest, "MIN_MARKET_SENTIMENT", 30)
                if self._day_sentiment < min_sentiment:
                    self.logger.info(
                        f"SKIP: Low market sentiment "
                        f"({self._day_sentiment:.1f} < {min_sentiment})"
                    )
                    skipped_days_count += 1
                    continue
            
            # Simulate the trading day
            try:
                self.daily_pnl = 0.0
                self._simulate_intraday_session(cur_day, results)
                
                # Log day completion
                day_elapsed = time.time() - day_start_time
                total_elapsed = time.time() - total_start_time
                avg_time_per_day = total_elapsed / day_idx
                days_remaining = total_days - day_idx
                est_time_remaining = avg_time_per_day * days_remaining
                
                self.logger.info(
                    f"Day processed in {day_elapsed:.1f}s | "
                    f"Total time: {total_elapsed:.0f}s | "
                    f"Est. remaining: {est_time_remaining:.0f}s ({est_time_remaining/60:.1f}m)"
                )
                
            except Exception as e:
                self.logger.exception(f"Error processing {cur_day.date()}: {e}")
                # Continue to next day instead of crashing
                continue
            
            # Garbage collection
            if self.config.system.FORCE_GARBAGE_COLLECTION:
                if day_idx % self.config.system.GC_FREQUENCY_DAYS == 0:
                    gc.collect()  # Force garbage collection

        # Close remaining positions
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("CLOSING REMAINING POSITIONS")
        self.logger.info("=" * 80)
        
        self.trade_simulator.close_all_positions(end_date, results)

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

        # Export diagnostics with enhanced metrics
        self._export_diagnostics(results)

        # Final summary
        self._log_final_summary(total_start_time, day_idx, trading_days_count, skipped_days_count, results)
        
        return results

    def _simulate_intraday_session(self, day: datetime, results: Dict) -> None:
        """Simulate full intraday session with progress logging."""
        
        session_start_time = time.time()
        
        # Get session configuration
        session_cfg = self.config.session
        premarket_enabled = session_cfg.PREMARKET_ENABLED
        
        # Define session boundaries
        if premarket_enabled:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(
                day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern'
            )
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
        
        session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')

        # Screen for candidates
        self.logger.info(f"[SCREENING] Starting candidate screening...")
        screen_start = time.time()
        
        screen_res = self.bt_screener.screen(day)
        
        screen_elapsed = time.time() - screen_start
        self.logger.info(f"[SCREENING] Completed in {screen_elapsed:.1f}s")
        
        # Filter out __DAY__ summary rows before adding to results
        diagnostics = screen_res.get("diagnostics", [])
        diagnostics = [d for d in diagnostics if d.get('symbol') != '__DAY__']
        
        if diagnostics:
            results['candidate_diagnostics'].extend(diagnostics)

        signals: List[CandidateSignal] = screen_res.get("signals", [])
        self.logger.info(f"[SIGNALS] Found {len(signals)} candidate signals")

        # Process signals using trade simulator
        self.logger.info(f"[ANALYSIS] Analyzing {len(signals)} candidates for patterns...")
        
        opened, rejected = self.trade_simulator.process_signals(
            day, 
            signals, 
            results, 
            session_start_utc, 
            session_end_utc, 
            premarket_enabled
        )

        # Session summary
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
        self.logger.info(f"  • Open positions: {len(self.trade_simulator.positions)}")
        self.logger.info(f"  • Session time: {session_elapsed:.1f}s")

    def _export_diagnostics(self, results: Dict) -> None:
        """Export diagnostics to CSV with advanced pattern analysis metrics."""
        diags = results.get('candidate_diagnostics', [])
        if not diags:
            return
        
        self.logger.info(f"Exporting {len(diags)} diagnostic records...")
        diag_df = pd.DataFrame(diags)
        
        # Filter out __DAY__ summary rows
        original_count = len(diag_df)
        diag_df = diag_df[diag_df['symbol'] != '__DAY__'].copy()
        removed_count = original_count - len(diag_df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} __DAY__ summary rows")
        
        # ✅ NEW: Calculate missed winner metrics
        self.logger.info("Calculating missed winner probability scores...")
        diag_df = _calculate_missed_winner_metrics(diag_df)
        
        # ✅ NEW: Calculate gate impact matrix (export separately)
        self.logger.info("Calculating gate impact matrix...")
        gate_impact_df = _calculate_gate_impact_matrix(diag_df)
        
        # ✅ NEW: Identify false negative signatures
        self.logger.info("Identifying false negative signatures...")
        false_negatives_df = _identify_false_negative_signatures(diag_df)
        
        # Delineate reason column into boolean
        self.logger.info("Delineating reason column...")
        diag_df = _delineate_reason_column(diag_df)
        
        # Export main diagnostics
        out_fp = os.path.abspath("backtest_candidates.csv")
        try:
            diag_df.to_csv(out_fp, index=False)
            self.logger.info(f"✓ Diagnostics exported: {out_fp}")
            
            # Log column summary
            self.logger.info(f"  • Total rows: {len(diag_df):,}")
            self.logger.info(f"  • Total columns: {len(diag_df.columns)}")
            
            # Log rejection breakdown
            if 'rejected' in diag_df.columns:
                reject_count = diag_df['rejected'].sum()
                entered_count = (diag_df['phase'] == 'entered').sum()
                exited_count = (diag_df['phase'] == 'exited').sum()
                self.logger.info(f"  • Phase breakdown:")
                self.logger.info(f"    - Rejected: {reject_count}")
                self.logger.info(f"    - Entered: {entered_count}")
                self.logger.info(f"    - Exited: {exited_count}")
            
            # Log missed winner stats
            if 'likely_missed_winner' in diag_df.columns:
                missed_winners = diag_df['likely_missed_winner'].sum()
                if missed_winners > 0:
                    self.logger.info(f"  • Likely missed winners: {missed_winners}")
                    avg_similarity = diag_df[diag_df['likely_missed_winner']]['similarity_to_winners'].mean()
                    self.logger.info(f"    - Avg similarity score: {avg_similarity:.1f}")
            
        except Exception as e:
            self.logger.error(f"Failed to export diagnostics: {e}")
        
        # ✅ NEW: Export gate impact matrix
        if not gate_impact_df.empty:
            gate_fp = os.path.abspath("backtest_gate_impact.csv")
            try:
                gate_impact_df.to_csv(gate_fp, index=False)
                self.logger.info(f"✓ Gate impact matrix exported: {gate_fp}")
                self.logger.info(f"  • Total gates analyzed: {len(gate_impact_df)}")
                
                # Log top impactful gates
                top_gates = gate_impact_df.nlargest(3, 'count')
                self.logger.info(f"  • Top gates by trade count:")
                for _, row in top_gates.iterrows():
                    self.logger.info(
                        f"    - {row['gate']}: {row['count']} trades, "
                        f"{row['winner_like_pct']:.1f}% winner-like, "
                        f"est. value: ${row['est_net_value']:.2f}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to export gate impact: {e}")
        
        # ✅ NEW: Export false negative signatures
        if not false_negatives_df.empty:
            fn_fp = os.path.abspath("backtest_false_negatives.csv")
            try:
                false_negatives_df.to_csv(fn_fp, index=False)
                self.logger.info(f"✓ False negative signatures exported: {fn_fp}")
                self.logger.info(f"  • Total false negatives: {len(false_negatives_df)}")
                
                # Log common rejection reasons for false negatives
                if 'reason' in false_negatives_df.columns:
                    top_reasons = false_negatives_df['reason'].value_counts().head(3)
                    self.logger.info(f"  • Top rejection reasons:")
                    for reason, count in top_reasons.items():
                        self.logger.info(f"    - {reason}: {count}")
            except Exception as e:
                self.logger.error(f"Failed to export false negatives: {e}")

    def _log_final_summary(
        self, 
        total_start_time: float, 
        day_idx: int, 
        trading_days_count: int, 
        skipped_days_count: int, 
        results: Dict
    ) -> None:
        """Log final backtest summary."""
        total_elapsed = time.time() - total_start_time
        stats = results['statistics']
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("BACKTEST COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
        self.logger.info(f"Days processed: {day_idx}")
        self.logger.info(f"  • Trading days: {trading_days_count}")
        self.logger.info(f"  • Skipped days: {skipped_days_count}")
        self.logger.info(f"Total trades: {stats.get('total_trades', 0)}")
        self.logger.info(f"Win rate: {stats.get('win_rate', 0):.1f}%")
        self.logger.info(f"Total return: {stats.get('total_return', 0):.2f}%")
        self.logger.info(f"Max drawdown: {stats.get('max_drawdown', 0):.2f}%")
        self.logger.info(f"Profit factor: {stats.get('profit_factor', 0):.2f}")
        self.logger.info(f"Sharpe ratio: {stats.get('sharpe_ratio', 0):.2f}")
        self.logger.info("=" * 80)

    def _init_results(self, initial_capital: float) -> Dict:
        """Initialize results dictionary."""
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