# =====================================================
# backtest/__init__.py - Backtest Simulation Engine
# =====================================================

"""
Backtesting simulation engine for intraday trading strategies.

This package contains all components needed to simulate historical trading,
including trade execution, exit logic, position management, and metrics
calculation. The backtester uses historical 1-minute bar data to replay
market conditions and evaluate strategy performance.

Modules:
    backtester: Main orchestrator that runs the backtest loop
    trade_simulator: Signal processing and position management
    exit_simulator: Exit logic with ATR trailing stops and slippage
    trade_metrics: Utility functions for calculations and formatting

Key Features:
    - Intraday position management with time-based exits
    - ATR-based trailing stops for profit protection
    - Realistic slippage simulation (winners, losers, stop losses)
    - Market context integration (VIX, market regime)
    - Comprehensive diagnostics (candidates, trades, gate impact)
    - Pattern evolution tracking (entry, 5-min, exit)
"""

from backtester.core import Backtester
from backtester.trade_simulator import TradeSimulator
from backtester.exit_simulator import ExitSimulator
from utils.trade_metrics import TradeMetrics

__all__ = [
    # Main Components
    'Backtester',
    'TradeSimulator',
    'ExitSimulator',
    'TradeMetrics',
]
