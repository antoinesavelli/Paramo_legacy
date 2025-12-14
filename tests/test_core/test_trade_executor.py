"""
Tests for TradeExecutor to ensure dictionary iteration safety and correct trade execution.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from core.trade_executor import TradeExecutor

class TestTradeExecutor:
    """Test suite for TradeExecutor."""
    
    def test_execute_entry_success(self, mock_config, mock_api, mock_risk_manager):
        """Test successful trade entry."""
        executor = TradeExecutor(mock_config, mock_api, mock_risk_manager)
        
        signal = {
            'symbol': 'AAPL',
            'entry_price': 150.0,
            'stop_price': 148.0,
            'is_reentry': False
        }
        
        result = executor.execute_entry(signal)
        
        assert result['success'] is True
        assert result['symbol'] == 'AAPL'
        assert 'AAPL' in executor.active_trades
        assert mock_api.submit_order.called
    
    def test_execute_entry_risk_rejected(self, mock_config, mock_api, mock_risk_manager):
        """Test entry rejection due to risk limits."""
        mock_risk_manager.check_entry_risk = Mock(return_value={
            'approved': False,
            'reason': 'Max concurrent positions reached'
        })
        
        executor = TradeExecutor(mock_config, mock_api, mock_risk_manager)
        
        signal = {
            'symbol': 'AAPL',
            'entry_price': 150.0,
            'stop_price': 148.0
        }
        
        result = executor.execute_entry(signal)
        
        assert result['success'] is False
        assert 'AAPL' not in executor.active_trades
    
    def test_execute_exit_success(self, mock_config, mock_api, mock_risk_manager):
        """Test successful trade exit."""
        executor = TradeExecutor(mock_config, mock_api, mock_risk_manager)
        
        # Setup active position
        executor.active_trades['AAPL'] = {
            'entry_time': datetime.now(),
            'entry_price': 150.0,
            'position_size': 100,
            'stop_price': 148.0,
            'stop_order_id': 'stop_123',
            'profit_target': 154.0,
            'reentry_count': 0,
            'is_reentry': False
        }
        
        result = executor.execute_exit('AAPL', reason='profit_target')
        
        assert result['success'] is True
        assert 'AAPL' not in executor.active_trades
        assert len(executor.trade_history) == 1
        assert mock_api.cancel_order.called
    
    def test_dictionary_iteration_safety(self, mock_config, mock_api, mock_risk_manager):
        """Test that dictionary iteration with list() prevents RuntimeError."""
        executor = TradeExecutor(mock_config, mock_api, mock_risk_manager)
        
        # Add multiple positions
        for i, symbol in enumerate(['AAPL', 'TSLA', 'NVDA']):
            executor.active_trades[symbol] = {
                'entry_time': datetime.now(),
                'entry_price': 150.0 + i * 10,
                'position_size': 100,
                'stop_price': 148.0 + i * 10,
                'stop_order_id': f'stop_{i}',
                'profit_target': 154.0 + i * 10,
                'reentry_count': 0,
                'is_reentry': False
            }
        
        # Mock quote with profit target reached
        mock_api.get_last_quote = Mock()
        mock_api.get_last_quote.return_value.askprice = 200.0  # High enough to trigger all targets
        
        # This should not raise RuntimeError even though execute_exit modifies the dict
        try:
            executor.check_profit_targets()
            success = True
        except RuntimeError:
            success = False
        
        assert success, "Dictionary iteration should be protected with list()"
    
    def test_reentry_tracking(self, mock_config, mock_api, mock_risk_manager):
        """Test that re-entry candidates are tracked correctly."""
        executor = TradeExecutor(mock_config, mock_api, mock_risk_manager)
        
        # Execute and exit a trade with profit target
        executor.active_trades['AAPL'] = {
            'entry_time': datetime.now(),
            'entry_price': 150.0,
            'position_size': 100,
            'stop_price': 148.0,
            'stop_order_id': 'stop_123',
            'profit_target': 154.0,
            'reentry_count': 0,
            'is_reentry': False
        }
        
        executor.execute_exit('AAPL', reason='profit_target')
        
        assert 'AAPL' in executor.reentry_candidates
        assert executor.reentry_candidates['AAPL']['monitoring'] is True
