"""
Tests for GapCalculator to prevent division by zero and ensure correct gap calculations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from data_handler.gap_calculator import GapCalculator

class TestGapCalculator:
    """Test suite for GapCalculator."""
    
    def test_calculate_gaps_basic(self, mock_config):
        """Test basic gap calculation with valid data."""
        # Create mock data
        today_data = pd.DataFrame({
            'symbol': ['AAPL', 'TSLA', 'NVDA'],
            'timestamp': pd.to_datetime(['2024-01-03 09:30'] * 3),
            'open': [110.0, 250.0, 450.0],
            'close': [111.0, 252.0, 455.0]
        })
        
        prev_data = pd.DataFrame({
            'symbol': ['AAPL', 'TSLA', 'NVDA'],
            'timestamp': pd.to_datetime(['2024-01-02 15:59'] * 3),
            'open': [105.0, 240.0, 440.0],
            'close': [108.0, 245.0, 448.0]
        })
        
        def get_day_df(date_str):
            if date_str == '2024-01-03':
                return today_data
            elif date_str == '2024-01-02':
                return prev_data
            return pd.DataFrame()
        
        def get_symbol_day_data(symbol, date_str):
            df = get_day_df(date_str)
            return df[df['symbol'] == symbol] if not df.empty else pd.DataFrame()
        
        file_index = {'2024-01-02': None, '2024-01-03': None}
        
        calc = GapCalculator(get_day_df, get_symbol_day_data, file_index)
        
        result = calc.calculate_gaps(datetime(2024, 1, 3), premarket=False)
        
        assert not result['gaps'].empty
        assert len(result['gaps']) == 3
        assert 'gap_percent' in result['gaps'].columns
        
        # Check AAPL gap: (110 - 108) / 108 * 100 ≈ 1.85%
        aapl_gap = result['gaps'][result['gaps']['symbol'] == 'AAPL']['gap_percent'].iloc[0]
        assert abs(aapl_gap - 1.85) < 0.1
    
    def test_division_by_zero_protection(self):
        """Test that division by zero is prevented when prev_close is 0."""
        today_data = pd.DataFrame({
            'symbol': ['ZERO'],
            'timestamp': pd.to_datetime(['2024-01-03 09:30']),
            'open': [10.0],
            'close': [10.5]
        })
        
        prev_data = pd.DataFrame({
            'symbol': ['ZERO'],
            'timestamp': pd.to_datetime(['2024-01-02 15:59']),
            'open': [0.0],
            'close': [0.0]  # Zero close price
        })
        
        def get_day_df(date_str):
            return today_data if date_str == '2024-01-03' else prev_data
        
        def get_symbol_day_data(symbol, date_str):
            df = get_day_df(date_str)
            return df[df['symbol'] == symbol] if not df.empty else pd.DataFrame()
        
        file_index = {'2024-01-02': None, '2024-01-03': None}
        
        calc = GapCalculator(get_day_df, get_symbol_day_data, file_index)
        result = calc.calculate_gaps(datetime(2024, 1, 3), premarket=False)
        
        # Should filter out zero prices
        assert result['gaps'].empty or 'ZERO' not in result['gaps']['symbol'].values
    
    def test_missing_previous_day(self):
        """Test handling when previous trading day is missing."""
        today_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'timestamp': pd.to_datetime(['2024-01-03 09:30']),
            'open': [110.0],
            'close': [111.0]
        })
        
        def get_day_df(date_str):
            return today_data if date_str == '2024-01-03' else pd.DataFrame()
        
        def get_symbol_day_data(symbol, date_str):
            df = get_day_df(date_str)
            return df[df['symbol'] == symbol] if not df.empty else pd.DataFrame()
        
        file_index = {'2024-01-03': None}
        
        calc = GapCalculator(get_day_df, get_symbol_day_data, file_index)
        result = calc.calculate_gaps(datetime(2024, 1, 3), premarket=False)
        
        assert result['prev_empty'] is True
        assert result['gaps'].empty
    
    def test_extreme_gap_filtering(self):
        """Test that extreme gaps (>10000% or <-100%) are filtered out."""
        today_data = pd.DataFrame({
            'symbol': ['EXTREME'],
            'timestamp': pd.to_datetime(['2024-01-03 09:30']),
            'open': [1000.0],  # 10000% gap
            'close': [1001.0]
        })
        
        prev_data = pd.DataFrame({
            'symbol': ['EXTREME'],
            'timestamp': pd.to_datetime(['2024-01-02 15:59']),
            'open': [10.0],
            'close': [10.0]
        })
        
        def get_day_df(date_str):
            return today_data if date_str == '2024-01-03' else prev_data
        
        def get_symbol_day_data(symbol, date_str):
            df = get_day_df(date_str)
            return df[df['symbol'] == symbol] if not df.empty else pd.DataFrame()
        
        file_index = {'2024-01-02': None, '2024-01-03': None}
        
        calc = GapCalculator(get_day_df, get_symbol_day_data, file_index)
        result = calc.calculate_gaps(datetime(2024, 1, 3), premarket=False)
        
        # Extreme gap should be filtered
        assert result['gaps'].empty or result['gaps']['gap_percent'].iloc[0] < 10000
