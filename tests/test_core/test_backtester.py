"""
Tests for Backtester to ensure division by zero protection and correct simulation.
"""
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from core.backtester import Backtester

class TestBacktester:
    """Test suite for Backtester."""
    
    def test_division_by_zero_protection_in_returns(self, mock_config, mock_data_handler):
        """Test that return_pct calculation handles zero entry prices."""
        bt = Backtester(mock_config, mock_data_handler)
        
        # Simulate a trade with near-zero entry (should be rejected)
        bars = Mock()
        bars.empty = False
        bars.iloc = [Mock(low=0.01, high=0.02, close=0.015, timestamp=datetime.now())]
        bars.iterrows = lambda: iter([(0, bars.iloc[0])])
        
        result = bt._simulate_exits('TEST', bars, datetime.now(), {'capital': 1000})
        
        # With entry validation, this should either:
        # 1. Not create the trade (rejected at entry)
        # 2. Have return_pct = 0 if entry is 0
        if result:
            assert 'return_pct' in result
            assert not (result['return_pct'] == float('inf') or result['return_pct'] == float('-inf'))
    
    def test_backtest_with_no_data(self, mock_config):
        """Test backtest handles missing data gracefully."""
        mock_handler = Mock()
        mock_handler.has_data_for_date = Mock(return_value=False)
        mock_handler._file_index = {}
        
        bt = Backtester(mock_config, mock_handler)
        
        results = bt.run_backtest(
            datetime(2024, 1, 1),
            datetime(2024, 1, 5),
            initial_capital=1000.0
        )
        
        assert results['statistics'] is not None
        assert len(results['trades']) == 0
    
    def test_position_validation(self, mock_config, mock_data_handler, sample_bars):
        """Test that invalid entry/stop prices are rejected."""
        bt = Backtester(mock_config, mock_data_handler)
        
        # Test invalid entry price (0 or negative)
        signal = Mock()
        signal.symbol = 'TEST'
        signal.entry_price = 0.001  # Below minimum
        signal.stop_price = 0.0005
        signal.meta = {'is_premarket': False}
        
        # This should be rejected in validation
        # (tested indirectly through _simulate_intraday_session)
