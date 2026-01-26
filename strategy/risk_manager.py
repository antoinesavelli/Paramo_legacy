# =====================================================
# risk_manager.py - Risk Management System
# =====================================================

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from config.config import TradingConfig
from utils.logging import get_logger

def log_api_error(logger, msg, exc):
    logger.error(f"{msg}: {exc}")

def get_account_info(api, logger):
    """Helper to get account info safely."""
    try:
        account = api.get_account()
        return float(account.buying_power), float(account.portfolio_value), float(account.equity)
    except Exception as e:
        log_api_error(logger, "Error getting account info", e)
        return 0, 0, 0

def get_positions(api, logger):
    """Helper to get positions safely."""
    try:
        return api.list_positions()
    except Exception as e:
        log_api_error(logger, "Error getting positions", e)
        return []

# ----------------------------
# Shared pure helpers (live + backtest)
# ----------------------------

def calc_atr(bars: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) with smoothing.
    
    Args:
        bars: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)
        
    Returns:
        Smoothed ATR value
    """
    try:
        if bars is None or bars.empty or len(bars) < period + 1:
            return 0.0
        
        # Calculate True Range
        high = bars['high']
        low = bars['low']
        close = bars['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth ATR using Wilder's method (EMA with alpha = 1/period)
        atr = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        
        return float(atr) if not pd.isna(atr) else 0.0
    except Exception:
        return 0.0


def calc_atr_stop(
    bars: pd.DataFrame,
    entry_price: float,
    atr_period: int = 14,
    atr_mult: float = 2.0,
    fallback_pct: float = 0.03
) -> float:
    """
    Pure ATR-based stop calculator. Returns a stop rounded to 4 decimals.
    - If ATR is available: entry - atr_mult * ATR
    - Else: fallback to percentage stop (entry * (1 - fallback_pct))
    """
    try:
        if bars is not None and not bars.empty and len(bars) >= atr_period + 1:
            atr = calc_atr(bars, period=atr_period)
            if atr > 0:
                return round(entry_price - atr_mult * atr, 4)
    except Exception:
        pass
    return round(entry_price * (1.0 - fallback_pct), 4)


def calc_atr_trailing_stop(
    bars: pd.DataFrame,
    highest_price: float,
    atr_period: int = 10,  # Shorter period for intraday
    atr_mult: float = 1.5,  # Conservative multiplier
    min_stop_distance_pct: float = 0.01  # Minimum 1% stop distance
) -> float:
    """
    Calculate ATR-based trailing stop for intraday trading.
    
    Args:
        bars: Recent price bars (should include current bar)
        highest_price: Highest price achieved since entry
        atr_period: ATR lookback period (5-14 for intraday)
        atr_mult: ATR multiplier (1.0-2.0 for intraday)
        min_stop_distance_pct: Minimum stop distance as % of price
        
    Returns:
        Trailing stop price
    """
    try:
        if bars is None or bars.empty:
            # Fallback to percentage-based if no bars
            return round(highest_price * (1.0 - min_stop_distance_pct), 4)
        
        # Calculate current ATR
        atr = calc_atr(bars, period=atr_period)
        
        if atr <= 0:
            # Fallback if ATR calculation fails
            return round(highest_price * (1.0 - min_stop_distance_pct), 4)
        
        # Calculate ATR-based distance
        atr_distance = atr_mult * atr
        
        # Apply minimum distance constraint
        min_distance = highest_price * min_stop_distance_pct
        distance = max(atr_distance, min_distance)
        
        # Calculate trailing stop
        trailing_stop = highest_price - distance
        
        return round(trailing_stop, 4)
        
    except Exception:
        # Emergency fallback
        return round(highest_price * (1.0 - min_stop_distance_pct), 4)


def calc_position_size_percentage(
    entry: float,
    stop: float,
    account_equity: float,
    stop_loss_pct: float,
    max_position_pct: float,
    market_adjustment: float = 1.0
) -> int:
    """
    Pure position sizing based on percentage of account equity with market context adjustment.
    
    Args:
        entry: Entry price per share
        stop: Stop loss price per share
        account_equity: Total account equity
        stop_loss_pct: Max loss as % of account (e.g., 2.0 for 2%)
        max_position_pct: Max position size as % of account (e.g., 25.0 for 25%)
        market_adjustment: Market context multiplier (0.7 unfavorable, 1.0 neutral, 1.2 favorable)
    
    Returns:
        Number of shares to buy
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    
    # Apply market context adjustment to risk budget
    base_risk_dollars = account_equity * (stop_loss_pct / 100.0)
    adjusted_risk_dollars = base_risk_dollars * market_adjustment
    
    # Calculate position size based on risk
    size_risk_cap = int(adjusted_risk_dollars / risk_per_share)
    
    # Apply market context adjustment to position value limit
    base_position_value = account_equity * (max_position_pct / 100.0)
    adjusted_position_value = base_position_value * market_adjustment
    size_value_cap = int(adjusted_position_value / entry) if entry > 0 else 0
    
    # Return the smaller of the two
    size = min(size_risk_cap, size_value_cap)
    return max(0, size)


class RiskManager:
    """Comprehensive risk management system - percentage-based with ATR trailing stops."""

    def __init__(self, config: TradingConfig, api):
        self.config = config
        self.api = api
        self.logger = get_logger(__name__, component="risk")
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.positions = {}
        self.daily_losses = []
        self.start_of_day_equity = None

    def check_entry_risk(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_price: float,
        market_adjustment: float = 1.0
    ) -> Dict:
        """Validate if new position meets risk parameters (percentage-based with market context)."""
        try:
            buying_power, portfolio_value, equity = get_account_info(self.api, self.logger)
            
            # Check daily loss limit (percentage-based)
            max_daily_loss_dollars = equity * (self.config.risk.MAX_DAILY_LOSS_PERCENT / 100.0)
            if self.daily_pnl <= -max_daily_loss_dollars:
                self.logger.warning(f"Entry blocked for {symbol}: Daily loss limit reached ({self.config.risk.MAX_DAILY_LOSS_PERCENT}% of account)")
                return {'approved': False, 'reason': f'Daily loss limit reached ({self.config.risk.MAX_DAILY_LOSS_PERCENT}%)'}
            
            # Check max concurrent positions
            positions = get_positions(self.api, self.logger)
            if len(positions) >= self.config.risk.MAX_CONCURRENT_POSITIONS:
                self.logger.warning(f"Entry blocked for {symbol}: Max concurrent positions reached")
                return {'approved': False, 'reason': 'Maximum concurrent positions reached'}
            
            # Validate stop price
            price_risk = entry_price - stop_price
            if price_risk <= 0:
                return {'approved': False, 'reason': 'Invalid stop price'}
            
            # Calculate position size with market context adjustment
            position_size = calc_position_size_percentage(
                entry=entry_price,
                stop=stop_price,
                account_equity=equity,
                stop_loss_pct=self.config.risk.STOP_LOSS_PERCENT_OF_ACCOUNT,
                max_position_pct=self.config.risk.MAX_POSITION_SIZE_PERCENT,
                market_adjustment=market_adjustment
            )
            
            # Check if we have enough buying power
            required_capital = position_size * entry_price
            if required_capital > buying_power:
                position_size = int(buying_power / entry_price)
                if position_size < 1:
                    self.logger.warning(f"Entry blocked for {symbol}: Insufficient buying power")
                    return {'approved': False, 'reason': 'Insufficient buying power'}
            
            # Check drawdown
            if self.peak_balance == 0:
                self.peak_balance = equity
            current_drawdown = ((self.peak_balance - equity) / self.peak_balance * 100) if self.peak_balance > 0 else 0
            if current_drawdown >= self.config.risk.MAX_DRAWDOWN_PERCENT:
                self.logger.warning(f"Entry blocked for {symbol}: Max drawdown reached ({current_drawdown:.2f}%)")
                return {'approved': False, 'reason': f'Maximum drawdown reached ({current_drawdown:.2f}%)'}
            
            risk_dollars = position_size * price_risk
            risk_pct_of_account = (risk_dollars / equity) * 100
            
            # Enhanced logging with market context
            self.logger.info(
                f"Entry approved {symbol}: size={position_size}, "
                f"risk=${risk_dollars:.2f} ({risk_pct_of_account:.2f}% of account), "
                f"stop={stop_price:.2f}, market_adj={market_adjustment:.2f}x"
            )
            return {
                'approved': True,
                'position_size': position_size,
                'risk_amount': risk_dollars,
                'risk_percent': risk_pct_of_account,
                'stop_price': stop_price,
                'entry_price': entry_price,
                'market_adjustment': market_adjustment
            }
        except Exception as e:
            log_api_error(self.logger, "Error checking entry risk", e)
            return {'approved': False, 'reason': str(e)}

    def calculate_stop_loss(self, symbol: str, entry_price: float, bars: pd.DataFrame) -> float:
        """Calculate dynamic stop loss based on ATR."""
        try:
            atr_stop = calc_atr_stop(bars, entry_price, atr_period=14, atr_mult=2.0, fallback_pct=0.05)
            
            # Also consider recent support
            recent_low = bars['low'].rolling(20).min().iloc[-1] if (bars is not None and not bars.empty and len(bars) >= 20) else entry_price * 0.95
            technical_stop = float(recent_low) * 0.99

            # Use the more conservative stop (higher value for longs)
            stop_price = max(atr_stop, technical_stop)
            stop_price = round(stop_price, 2)
            
            self.logger.info(f"Stop calculated {symbol}: entry={entry_price:.2f} atr_stop={atr_stop:.2f} tech_stop={technical_stop:.2f} final={stop_price:.2f}")
            return stop_price
        except Exception as e:
            log_api_error(self.logger, "Error calculating stop loss", e)
            return round(entry_price * 0.95, 2)

    def update_trailing_stop(self, symbol: str, current_price: float, position: Dict) -> Optional[float]:
        """
        Update ATR-based trailing stops for profitable positions.
        
        Returns:
            New stop price if updated, None otherwise
        """
        try:
            entry_price = position['entry_price']
            current_stop = position['stop_price']
            highest_price = position.get('highest_price', entry_price)
            
            # Update highest price
            if current_price > highest_price:
                highest_price = current_price
                position['highest_price'] = highest_price
            
            # Don't trail if we're not in profit
            if current_price <= entry_price:
                return None
            
            # Get recent bars for ATR calculation
            bars = position.get('recent_bars')
            
            if bars is None or bars.empty:
                # Fallback to simple trailing if no bars available
                profit_per_share = current_price - entry_price
                
                # Move to breakeven after 20 cents profit
                if profit_per_share >= 0.20 and current_stop < entry_price:
                    return entry_price + 0.01
                
                return None
            
            # Calculate ATR-based trailing stop
            atr_period = getattr(self.config.risk, 'ATR_TRAILING_PERIOD', 10)
            atr_mult = getattr(self.config.risk, 'ATR_TRAILING_MULTIPLIER', 1.5)
            
            new_stop = calc_atr_trailing_stop(
                bars=bars,
                highest_price=highest_price,
                atr_period=atr_period,
                atr_mult=atr_mult,
                min_stop_distance_pct=0.01
            )
            
            # Only update if new stop is higher (tighter)
            if new_stop > current_stop:
                # But never let stop go below breakeven
                new_stop = max(new_stop, entry_price + 0.01)
                
                self.logger.info(
                    f"ATR trailing stop update {symbol}: "
                    f"highest=${highest_price:.2f}, "
                    f"old_stop=${current_stop:.2f}, "
                    f"new_stop=${new_stop:.2f}"
                )
                return round(new_stop, 2)
            
            return None
            
        except Exception as e:
            log_api_error(self.logger, "Error updating trailing stop", e)
            return None

    def check_sector_exposure(self, sector: str) -> bool:
        """Check if adding position would exceed sector limits"""
        try:
            positions = get_positions(self.api, self.logger)
            sector_positions = [p for p in positions if self._get_sector(p.symbol) == sector]
            sector_exposure = len(sector_positions) / max(len(positions), 1)
            return sector_exposure < self.config.risk.MAX_SECTOR_EXPOSURE
        except Exception as e:
            log_api_error(self.logger, "Error checking sector exposure", e)
            return True

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol (would need integration with sector data)"""
        return "Unknown"

    def update_daily_pnl(self):
        """Update daily P&L tracking"""
        try:
            _, _, equity = get_account_info(self.api, self.logger)
            
            if self.start_of_day_equity is None:
                self.start_of_day_equity = equity
                self.daily_pnl = 0.0
            else:
                self.daily_pnl = equity - self.start_of_day_equity
            
            if equity > self.peak_balance:
                self.peak_balance = equity
            
            # Calculate percentage loss
            max_daily_loss_dollars = equity * (self.config.risk.MAX_DAILY_LOSS_PERCENT / 100.0)
            loss_pct_used = (abs(self.daily_pnl) / max_daily_loss_dollars * 100) if self.daily_pnl < 0 else 0
            
            if self.daily_pnl <= -max_daily_loss_dollars * 0.8:
                self.logger.warning(f"Approaching daily loss limit: ${self.daily_pnl:.2f} ({loss_pct_used:.1f}% of {self.config.risk.MAX_DAILY_LOSS_PERCENT}% limit)")
            else:
                self.logger.debug(f"Daily PnL updated: ${self.daily_pnl:.2f}")
        except Exception as e:
            log_api_error(self.logger, "Error updating daily P&L", e)

    def emergency_liquidate_all(self):
        """Emergency liquidation of all positions"""
        try:
            self.logger.critical("EMERGENCY LIQUIDATION INITIATED")
            positions = get_positions(self.api, self.logger)
            for position in positions:
                try:
                    self.api.submit_order(
                        symbol=position.symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    self.logger.info(f"Emergency liquidation order placed for {position.symbol}")
                except Exception as e:
                    log_api_error(self.logger, f"Failed to liquidate {position.symbol}", e)
            try:
                self.api.cancel_all_orders()
            except Exception as e:
                log_api_error(self.logger, "Failed to cancel all orders", e)
        except Exception as e:
            log_api_error(self.logger, "Emergency liquidation failed", e)
