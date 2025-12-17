# =====================================================
# cli_common.py - Shared CLI helper (simplified; overrides removed)
# =====================================================

import argparse
import logging
from utils import BacktestLogLevel, enable_backtest_optimization

def add_logging_args(p: argparse.ArgumentParser):
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging level (default: INFO)")
    # Add new argument for optimizing logging during backtest with default
    p.add_argument("--optimize-logging", choices=["normal", "minimal", "silent"],
                   default="minimal",
                   help="Optimize logging for backtest performance (default: minimal)")

def configure_logging(level: str, log_file: str = None, optimize: str = None):
    import sys
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers
    )
    
    # Apply optimization if requested
    if optimize:
        mode_map = {
            "normal": BacktestLogLevel.NORMAL,
            "minimal": BacktestLogLevel.MINIMAL,
            "silent": BacktestLogLevel.SILENT
        }
        enable_backtest_optimization(mode_map.get(optimize.lower(), BacktestLogLevel.MINIMAL))