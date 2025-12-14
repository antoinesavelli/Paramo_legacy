"""
Pytest configuration and shared fixtures for the trading system tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field

# Import your actual config classes
from config.config import (
    TradingConfig, BacktestConfig, ScreeningConfig, PatternConfig,
    RiskConfig, SystemConfig, BacktestSessionConfig
)

@pytest.fixture
def mock_config():
    """Create a mock trading configuration for testing."""
    return TradingConfig(
        screening=ScreeningConfig(
            MIN_GAP_PERCENT=5.0,
            MIN_PRICE=1.0,
            MAX_PRICE=20.0,
            MIN_ABSOLUTE_VOLUME=200000
        ),
        risk=RiskConfig(
            MAX_RISK_PER_TRADE=25.0,
            MAX_POSITION_SIZE_PERCENT=4.0,
            MAX_CONCURRENT_POSITIONS=3,
            PROFIT_TARGET_CENTS=40.0
        ),
        backtest=BacktestConfig(
            START_DATE='2024-01-02',
            END_DATE='2024-01-05',
            INITIAL_CAPITAL=1000.0,
            WARMUP_MINUTES=15,
            SESSION=BacktestSessionConfig(
                PREMARKET_ENABLED=True,
                PREMARKET_START_ET="04:00",
                PREMARKET_END_ET="09:30",
                PREMARKET_WARMUP_MINUTES=25,
                PREMARKET_MIN_GAP_PERCENT=3.0,
                PREMARKET_MIN_ABSOLUTE_VOLUME=15000
            )
        )
    )

@pytest.fixture
def sample_bars():
    """Create sample 1-minute bar data for testing."""
    dates = pd.date_range('2024-01-03 09:30', periods=100, freq='1min')
    
    # Simulate a parabolic price movement
    base_price = 10.0
    prices = base_price + np.cumsum(np.random.randn(100) * 0.05 + 0.02)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(100) * 0.01,
        'high': prices + np.abs(np.random.randn(100) * 0.05),
        'low': prices - np.abs(np.random.randn(100) * 0.05),
        'close': prices,
        'volume': np.random.randint(10000, 50000, 100)
    })
    
    return df

@pytest.fixture
def sample_day_data():
    """Create sample full-day data with multiple symbols."""
    symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']
    dates = pd.date_range('2024-01-03 04:00', '2024-01-03 20:00', freq='1min')
    
    data_parts = []
    for symbol in symbols:
        base_price = np.random.uniform(5.0, 15.0)
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.02)
        
        df = pd.DataFrame({
            'symbol': symbol,
            'timestamp': dates,
            'open': prices + np.random.randn(len(dates)) * 0.01,
            'high': prices + np.abs(np.random.randn(len(dates)) * 0.03),
            'low': prices - np.abs(np.random.randn(len(dates)) * 0.03),
            'close': prices,
            'volume': np.random.randint(5000, 50000, len(dates))
        })
        data_parts.append(df)
    
    return pd.concat(data_parts, ignore_index=True)

@pytest.fixture
def mock_data_handler(sample_day_data):
    """Create a mock LocalDataHandler."""
    handler = Mock()
    handler.get_intraday_bars = Mock(return_value=sample_day_data[:100])
    handler.has_data_for_date = Mock(return_value=True)
    handler._file_index = {'2024-01-03': Path('dummy')}
    return handler

@pytest.fixture
def mock_api():
    """Create a mock Alpaca API client."""
    api = Mock()
    
    # Mock order object
    order = Mock()
    order.id = 'order_123'
    order.status = 'filled'
    order.filled_avg_price = 10.50
    
    api.submit_order = Mock(return_value=order)
    api.get_order = Mock(return_value=order)
    api.cancel_order = Mock(return_value=True)
    
    # Mock quote
    quote = Mock()
    quote.askprice = 10.55
    quote.bidprice = 10.45
    api.get_last_quote = Mock(return_value=quote)
    
    return api

@pytest.fixture
def mock_risk_manager():
    """Create a mock RiskManager."""
    rm = Mock()
    rm.check_entry_risk = Mock(return_value={
        'approved': True,
        'position_size': 100,
        'risk_amount': 25.0
    })
    rm.update_trailing_stop = Mock(return_value=None)
    rm.daily_pnl = 0.0
    rm.open_positions = []
    return rm

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory structure for test data."""
    data_dir = tmp_path / "trading_data"
    data_dir.mkdir()
    
    # Create hierarchical structure
    year_dir = data_dir / "2024"
    year_dir.mkdir()
    month_dir = year_dir / "01"
    month_dir.mkdir()
    day_dir = month_dir / "03"
    day_dir.mkdir()
    
    return data_dir

def create_sample_parquet(filepath: Path, symbol: str, date: str):
    """Helper to create sample parquet files for testing."""
    dates = pd.date_range(f'{date} 04:00', f'{date} 20:00', freq='1min')
    base_price = np.random.uniform(5.0, 15.0)
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.02)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(len(dates)) * 0.01,
        'high': prices + np.abs(np.random.randn(len(dates)) * 0.03),
        'low': prices - np.abs(np.random.randn(len(dates)) * 0.03),
        'close': prices,
        'volume': np.random.randint(5000, 50000, len(dates))
    })
    
    df.to_parquet(filepath)
    return df

@pytest.fixture
def populated_data_dir(temp_data_dir):
    """Create a data directory with sample parquet files."""
    year_dir = temp_data_dir / "2024"
    month_dir = year_dir / "01"
    
    # Create files for multiple days
    for day in range(2, 6):
        day_dir = month_dir / f"{day:02d}"
        day_dir.mkdir(exist_ok=True)
        
        # Create daily file with multiple symbols
        date_str = f"2024-01-{day:02d}"
        filename = f"2024{day:02d}03.parquet"
        filepath = day_dir / filename
        
        # Create combined data for all symbols
        symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']
        all_data = []
        for symbol in symbols:
            df = create_sample_parquet(filepath, symbol, date_str)
            df['symbol'] = symbol
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(filepath)
    
    return temp_data_dir
