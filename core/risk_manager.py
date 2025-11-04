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
            tr = pd.concat([
                bars['high'] - bars['low'],
                (bars['high'] - bars['close'].shift()).abs(),
                (bars['low'] - bars['close'].shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(atr_period).mean().iloc[-1]
            if not pd.isna(atr):
                return round(entry_price - atr_mult * float(atr), 4)
    except Exception:
        pass
    return round(entry_price * (1.0 - fallback_pct), 4)


def calc_position_size(
    entry: float,
    stop: float,
    cash: float,
    max_risk_per_trade: float,
    max_value_frac: float
) -> int:
    """
    Pure position sizing helper:
    - share count capped by risk budget (max_risk_per_trade)
    - capped by max position value fraction of available cash
    """
    risk_ps = entry - stop
    if risk_ps <= 0:
        return 0
    size_risk_cap = int(max_risk_per_trade / risk_ps)
    max_value = cash * max_value_frac
    size_value_cap = int(max_value / entry) if entry > 0 else 0
    size = min(size_risk_cap, size_value_cap)
    return max(0, size)


def calc_profit_target(entry: float, target_cents: float) -> float:
    """Pure profit target helper."""
    return float(entry) + (float(target_cents) / 100.0)


class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self, config: TradingConfig, api):
        self.config = config
        self.api = api
        self.logger = get_logger(__name__, component="risk")
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.positions = {}
        self.daily_losses = []

    def check_entry_risk(self, symbol: str, entry_price: float, stop_price: float) -> Dict:
        """Validate if new position meets risk parameters"""
        try:
            buying_power, portfolio_value, _ = get_account_info(self.api, self.logger)
            if self.daily_pnl <= -self.config.risk.MAX_DAILY_LOSS:
                self.logger.warning(f"Entry blocked for {symbol}: Daily loss limit reached")
                return {'approved': False, 'reason': 'Daily loss limit reached'}
            positions = get_positions(self.api, self.logger)
            if len(positions) >= self.config.risk.MAX_CONCURRENT_POSITIONS:
                self.logger.warning(f"Entry blocked for {symbol}: Max concurrent positions reached")
                return {'approved': False, 'reason': 'Maximum concurrent positions reached'}
            price_risk = entry_price - stop_price
            if price_risk <= 0:
                return {'approved': False, 'reason': 'Invalid stop price'}
            position_size = int(self.config.risk.MAX_RISK_PER_TRADE / price_risk)
            max_position_value = portfolio_value * (self.config.risk.MAX_POSITION_SIZE_PERCENT / 100)
            max_shares = int(max_position_value / entry_price)
            position_size = min(position_size, max_shares)
            required_capital = position_size * entry_price
            if required_capital > buying_power:
                position_size = int(buying_power / entry_price)
                if position_size < 1:
                    self.logger.warning(f"Entry blocked for {symbol}: Insufficient buying power")
                    return {'approved': False, 'reason': 'Insufficient buying power'}
            current_drawdown = ((self.peak_balance - portfolio_value) / self.peak_balance * 100) if self.peak_balance > 0 else 0
            if current_drawdown >= self.config.risk.MAX_DRAWDOWN_PERCENT:
                self.logger.warning(f"Entry blocked for {symbol}: Max drawdown reached ({current_drawdown:.2f}%)")
                return {'approved': False, 'reason': 'Maximum drawdown reached'}
            self.logger.info(f"Entry approved {symbol}: size={position_size}, risk=${position_size * price_risk:.2f}, stop={stop_price:.2f}")
            return {
                'approved': True,
                'position_size': position_size,
                'risk_amount': position_size * price_risk,
                'stop_price': stop_price,
                'entry_price': entry_price
            }
        except Exception as e:
            log_api_error(self.logger, "Error checking entry risk", e)
            return {'approved': False, 'reason': str(e)}

    def calculate_stop_loss(self, symbol: str, entry_price: float, bars: pd.DataFrame) -> float:
        """Calculate dynamic stop loss based on volatility and support (reuses calc_atr_stop)."""
        try:
            atr_stop = calc_atr_stop(bars, entry_price, atr_period=14, atr_mult=2.0, fallback_pct=0.05)
            recent_low = bars['low'].rolling(20).min().iloc[-1] if (bars is not None and not bars.empty and len(bars) >= 20) else entry_price * 0.95
            technical_stop = float(recent_low) * 0.99

            per_share_risk = entry_price - max(atr_stop, technical_stop)
            if per_share_risk <= 0:
                per_share_risk = entry_price * 0.03  # fallback 3%

            max_theoretical_risk = self.config.risk.MAX_RISK_PER_TRADE / 5  # assume at least 5 shares
            if per_share_risk > max_theoretical_risk:
                adjusted_stop = entry_price - max_theoretical_risk
            else:
                adjusted_stop = entry_price - per_share_risk

            stop_price = max(adjusted_stop, technical_stop, atr_stop)
            stop_price = round(stop_price, 2)
            self.logger.info(f"Stop calculated {symbol}: entry={entry_price:.2f} stop={stop_price:.2f}")
            return stop_price
        except Exception as e:
            log_api_error(self.logger, "Error calculating stop loss", e)
            return round(entry_price * 0.95, 2)

    def update_trailing_stop(self, symbol: str, current_price: float, position: Dict) -> Optional[float]:
        """Update trailing stops for profitable positions"""
        try:
            entry_price = position['entry_price']
            current_stop = position['stop_price']
            if current_price <= entry_price:
                return None
            profit = current_price - entry_price
            if profit >= 0.20 and current_stop < entry_price:
                return entry_price + 0.01
            if profit >= self.config.risk.PROFIT_TARGET_CENTS:
                new_stop = entry_price + (profit * 0.5)
                if new_stop > current_stop:
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
            if hasattr(self, 'start_of_day_equity'):
                self.daily_pnl = equity - self.start_of_day_equity
            else:
                self.start_of_day_equity = equity
                self.daily_pnl = 0.0
            if equity > self.peak_balance:
                self.peak_balance = equity
            if self.daily_pnl <= -self.config.risk.MAX_DAILY_LOSS * 0.8:
                self.logger.warning(f"Approaching daily loss limit: ${self.daily_pnl:.2f}")
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
