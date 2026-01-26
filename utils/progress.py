"""Progress tracking utilities for backtest execution."""

import sys
import time
from datetime import datetime
from typing import Optional


class BacktestProgressTracker:
    """
    Real-time progress tracker for backtest execution.
    
    Displays:
    - Progress bar with percentage
    - Current day being processed
    - Current step (screening, pattern analysis, simulation)
    - Time elapsed and estimated time remaining
    - Trading statistics (signals, trades, capital)
    """
    
    def __init__(self, total_days: int, start_date: datetime, end_date: datetime):
        self.total_days = total_days
        self.start_date = start_date
        self.end_date = end_date
        self.current_day_idx = 0
        self.trading_days_processed = 0
        self.skipped_days = 0
        self.start_time = time.time()
        
        # Current state
        self.current_date: Optional[datetime] = None
        self.current_step: str = "Initializing"
        self.current_substep: str = ""
        
        # Statistics
        self.signals_today = 0
        self.trades_opened_today = 0
        self.total_trades = 0
        self.current_capital = 0.0
        self.open_positions = 0
        
        # Track last update time to avoid excessive refreshes
        self.last_update = 0.0
        self.update_interval = 0.1  # Update every 100ms max
        
    def start_day(self, day: datetime, day_idx: int):
        """Mark the start of a new trading day."""
        self.current_date = day
        self.current_day_idx = day_idx
        self.current_step = "Day Start"
        self.current_substep = ""
        self.signals_today = 0
        self.trades_opened_today = 0
        self._update_display()
    
    def skip_day(self, reason: str = "No data"):
        """Mark a day as skipped."""
        self.skipped_days += 1
        self.current_substep = f"SKIPPED ({reason})"
        self._update_display()
    
    def set_step(self, step: str, substep: str = ""):
        """Update the current processing step."""
        self.current_step = step
        self.current_substep = substep
        self._update_display()
    
    def update_stats(self, signals: int = None, trades_opened: int = None, 
                     capital: float = None, open_positions: int = None):
        """Update trading statistics."""
        if signals is not None:
            self.signals_today = signals
        if trades_opened is not None:
            self.trades_opened_today = trades_opened
            self.total_trades += trades_opened
        if capital is not None:
            self.current_capital = capital
        if open_positions is not None:
            self.open_positions = open_positions
        self._update_display()
    
    def complete_day(self):
        """Mark the current day as complete."""
        self.trading_days_processed += 1
        self.current_step = "Day Complete"
        self._update_display()
    
    def _update_display(self):
        """Update the progress display (rate-limited)."""
        now = time.time()
        if now - self.last_update < self.update_interval and self.current_day_idx < self.total_days:
            return  # Skip update if too soon
        
        self.last_update = now
        
        # Calculate progress
        progress_pct = (self.current_day_idx / self.total_days * 100) if self.total_days > 0 else 0
        
        # Calculate time estimates
        elapsed = now - self.start_time
        if self.current_day_idx > 0:
            avg_time_per_day = elapsed / self.current_day_idx
            remaining_days = self.total_days - self.current_day_idx
            est_remaining = avg_time_per_day * remaining_days
        else:
            est_remaining = 0
        
        # Build progress bar (50 chars wide)
        bar_width = 50
        filled = int(bar_width * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Format date
        date_str = self.current_date.strftime('%Y-%m-%d') if self.current_date else "N/A"
        
        # Build status line
        status_parts = []
        if self.current_step:
            status_parts.append(self.current_step)
        if self.current_substep:
            status_parts.append(self.current_substep)
        status = " | ".join(status_parts) if status_parts else "Processing"
        
        # Build output (multi-line for rich info)
        output = (
            f"\r{bar} {progress_pct:5.1f}% | "
            f"Day {self.current_day_idx}/{self.total_days} | "
            f"{date_str}\n"
            f"  Step: {status:<60}\n"
            f"  Stats: Signals={self.signals_today} Trades={self.trades_opened_today} "
            f"Total={self.total_trades} Capital=${self.current_capital:,.0f} Open={self.open_positions}\n"
            f"  Time: {self._format_time(elapsed)} elapsed | "
            f"{self._format_time(est_remaining)} remaining"
        )
        
        # Clear previous lines and print
        sys.stdout.write('\033[K')  # Clear to end of line
        sys.stdout.write('\033[F' * 3)  # Move up 3 lines
        sys.stdout.write('\033[K')  # Clear line
        sys.stdout.write(output)
        sys.stdout.flush()
    
    def finalize(self, total_trades: int, final_capital: float):
        """Display final completion message."""
        elapsed = time.time() - self.start_time
        
        # Final progress bar (100%)
        bar = '█' * 50
        
        output = (
            f"\r{bar} 100.0% | "
            f"Day {self.total_days}/{self.total_days} | COMPLETE\n"
            f"  Processed: {self.trading_days_processed} trading days, "
            f"{self.skipped_days} skipped\n"
            f"  Results: {total_trades} total trades | "
            f"Final capital: ${final_capital:,.2f}\n"
            f"  Total time: {self._format_time(elapsed)}\n"
        )
        
        sys.stdout.write(output)
        sys.stdout.flush()
        print()  # Add newline
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
