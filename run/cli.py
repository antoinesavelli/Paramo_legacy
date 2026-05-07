# =====================================================
# cli_common.py - Shared CLI helper (simplified; overrides removed)
# =====================================================

import argparse
import logging
import sys
import os
from utils import BacktestLogLevel, enable_backtest_optimization


# NOTE: FIX: Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        # Set UTF-8 mode for stdout/stderr
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
        # Also set environment variable for Python 3.7+
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        pass


class ScreenerWarningFilter(logging.Filter):
    """Filter out repetitive screener warnings from console output."""
    
    SUPPRESSED_MESSAGES = [
        "All candidates filtered by daily volume pre-screen",
        "No candidates meeting gap threshold",
        "No aggregate data for",
        "No previous day data found within 7 days",
        "No gaps calculated for",
    ]
    
    def filter(self, record):
        """Return False to suppress the log record."""
        # Only filter screener.backtest warnings
        if record.name == "screener.backtest" and record.levelno == logging.WARNING:
            msg = record.getMessage()
            # any() short-circuits on first match — efficient at high volume (L11)
            if any(pattern in msg for pattern in self.SUPPRESSED_MESSAGES):
                return False  # Suppress this message
        return True  # Allow all other messages


def add_logging_args(p: argparse.ArgumentParser):
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging level (default: INFO)")
    # NOTE: CHANGED: Default to 'normal' for full diagnostic output
    p.add_argument("--optimize-logging", choices=["normal", "minimal", "silent"],
                   default="normal",
                   help="Optimize logging for backtest performance (default: normal for full diagnostics)")
    p.add_argument("--show-screener-warnings", action="store_true",
                   help="Show repetitive screener warnings in console (default: suppressed, but logged to file)")
    p.add_argument("--log-file", default="run.log",
                   help="Path to log file (default: run.log)")


def configure_logging(level: str, log_file: str = None, optimize: str = None, show_screener_warnings: bool = False):
    """
    Configure logging with proper UTF-8 encoding support.
    
    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        optimize: Optimization level (normal, minimal, silent)
        show_screener_warnings: Whether to show screener warnings in console
    """
    
    # NOTE: Create console handler with UTF-8 support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    
    # Add filter to suppress repetitive screener warnings (unless explicitly requested)
    if not show_screener_warnings:
        console_handler.addFilter(ScreenerWarningFilter())
    
    handlers = [console_handler]
    
    if log_file:
        # NOTE: File handler with explicit UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        handlers.append(file_handler)
    
    # NOTE: Configure root logger with handlers
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True  # NOTE: Force reconfiguration if already configured
    )
    
    # Apply optimization if requested
    if optimize:
        mode_map = {
            "normal": BacktestLogLevel.NORMAL,
            "minimal": BacktestLogLevel.MINIMAL,
            "silent": BacktestLogLevel.SILENT
        }
        enable_backtest_optimization(mode_map.get(optimize.lower(), BacktestLogLevel.NORMAL))