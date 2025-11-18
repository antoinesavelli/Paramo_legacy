"""
Integration tests for the full backtest pipeline.
"""
import pytest
from datetime import datetime
from data_handler.local import LocalDataHandler
from core.backtester import Backtester

class TestBacktestIntegration:
    """Integration tests for complete backtest flow."""
    
    def test_full_backtest_run(self, mock_config, populated_data_dir):
        """Test a complete backtest run with real data structure."""
        # Create real data handler with test data
        handler = LocalDataHandler(mock_config, data_dir=str(populated_data_dir))
        
        bt = Backtester(mock_config, handler)
        
        results = bt.run_backtest(
            datetime(2024, 1, 2),
            datetime(2024, 1, 5),
            initial_capital=1000.0
        )
        
        # Basic assertions
        assert 'statistics' in results
        assert 'trades' in results
        assert 'equity_curve' in results
        assert results['capital'] > 0  # Should have some capital left
        
        # Check that statistics were calculated
        stats = results['statistics']
        assert 'total_trades' in stats
        assert 'win_rate' in stats
        assert 'total_return' in stats
    
    def test_no_crashes_on_edge_cases(self, mock_config, populated_data_dir):
        """Test that backtest doesn't crash on edge cases."""
        handler = LocalDataHandler(mock_config, data_dir=str(populated_data_dir))
        bt = Backtester(mock_config, handler)
        
        # Should handle gracefully
        try:
            results = bt.run_backtest(
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),  # Same day
                initial_capital=100.0  # Low capital
            )
            success = True
        except Exception as e:
            success = False
            pytest.fail(f"Backtest crashed on edge case: {e}")
        
        assert success
