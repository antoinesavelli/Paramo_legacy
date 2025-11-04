# =====================================================
# market_context.backtest.py - Market Context Analysis (Backtest)
# =====================================================

from datetime import datetime
import pandas as pd
from typing import Dict
import os
import numpy as np
from utils.logging import get_logger
from market_context.scoring import (
    calculate_market_score,
    classify_environment,
)

class BacktestMarketContext:
    """
    File-based MarketContext for backtesting.
    Reads SPY.csv, VIX.csv, RUT.csv from a directory and computes the same
    key signals used by live MarketContext to gate trading days.
    """

    def __init__(self, config, dirpath: str | None = None):
        self.config = config
        self.dirpath = dirpath or self.config.market_context.CSV_DIR
        self.logger = get_logger(__name__, component="market_context_bt")
        self.market_indicators: Dict = {}
        # Preload dataframes
        self._spy = self._load_csv("SPY.csv", has_commas_in_price=False)
        self._vix = self._load_csv("VIX.csv", has_commas_in_price=False, treat_dash_volume_as_nan=True)
        self._rut = self._load_csv("RUT.csv", has_commas_in_price=True)
        if self._spy.empty or self._vix.empty:
            self.logger.warning(f"Market context CSVs not fully available at {dirpath}")

    def _load_csv(self, name: str, has_commas_in_price: bool, treat_dash_volume_as_nan: bool = False) -> pd.DataFrame:
        fp = os.path.join(self.dirpath, name)
        try:
            df = pd.read_csv(fp, encoding="utf-8")
        except Exception as e:
            self.logger.error(f"Failed to read {fp}: {e}")
            return pd.DataFrame()

        expected = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing = [c for c in expected if c not in df.columns]
        if missing:
            self.logger.error(f"{name} missing columns: {missing}")
            return pd.DataFrame()

        def _clean_num(s: pd.Series, allow_commas: bool) -> pd.Series:
            s = s.astype(str).str.strip()
            if allow_commas:
                s = s.str.replace(",", "", regex=False)
            return pd.to_numeric(s, errors="coerce")

        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_cols:
            df[col] = _clean_num(df[col], allow_commas=has_commas_in_price)

        # Volume: do string operations first, then convert; avoid replace() downcasting
        vol = df['Volume'].astype(str).str.replace(",", "", regex=False).str.strip()
        if treat_dash_volume_as_nan:
            # Treat literal "-" as missing without triggering dtype downcasting warnings
            vol = vol.mask(vol.eq("-"), np.nan)
        df['Volume'] = pd.to_numeric(vol, errors="coerce")

        try:
            df['date'] = pd.to_datetime(df['Date'], format="%b %d, %Y").dt.normalize()
        except Exception:
            df['date'] = pd.to_datetime(df['Date']).dt.normalize()
        df = df.sort_values('date').reset_index(drop=True)
        return df[['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    def _slice_upto(self, df: pd.DataFrame, day: datetime, limit: int) -> pd.DataFrame:
        if df.empty:
            return df
        d = pd.Timestamp(day).normalize()
        sub = df[df['date'] <= d]
        if sub.empty:
            return pd.DataFrame()
        return sub.tail(limit)

    def update_market_context(self, day: datetime) -> Dict:
        """Compute market context using data up to and including 'day'."""
        try:
            spy = self._slice_upto(self._spy, day, 40)
            vix = self._slice_upto(self._vix, day, 2)
            rut = self._slice_upto(self._rut, day, 40)

            context: Dict = {
                'spy_trend': self._spy_trend(spy),
                'rut_trend': self._rut_trend(rut),
                'vix_level': self._vix_level(vix),
                'volume_profile': self._volume_profile(spy),
                'market_breadth': self._breadth_proxy(spy, rut),
                'timestamp': pd.Timestamp(day)
            }

            context['market_score'] = self._calculate_market_score(context)
            context['trading_environment'] = self._classify_environment(context)

            self.market_indicators = context
            return context
        except Exception as e:
            self.logger.error(f"Error updating backtest market context: {e}")
            return self.market_indicators

    def _spy_trend(self, spy: pd.DataFrame) -> Dict:
        mc = self.config.market_context
        if spy is None or spy.empty or len(spy) < mc.SMA_SLOW:
            return {'trend': 'unknown', 'strength': 0, 'momentum': 0}
        s = spy.copy()
        s['sma_fast'] = s['Close'].rolling(mc.SMA_FAST).mean()
        s['sma_slow'] = s['Close'].rolling(mc.SMA_SLOW).mean()
        current = float(s['Close'].iloc[-1])
        sma_f = float(s['sma_fast'].iloc[-1])
        sma_s = float(s['sma_slow'].iloc[-1])
        if current > sma_f > sma_s:
            trend = 'bullish'
            strength = min(100.0, ((current / sma_s - 1.0) * 100.0) * 10.0)
        elif current < sma_f < sma_s:
            trend = 'bearish'
            strength = min(100.0, ((1.0 - current / sma_s) * 100.0) * 10.0)
        else:
            trend, strength = 'neutral', 50.0
        momentum = float((s['Close'].iloc[-1] / s['Close'].iloc[-mc.SMA_FAST] - 1.0) * 100.0) if len(s) >= mc.SMA_FAST + 1 else 0.0
        return {'trend': trend, 'strength': strength, 'momentum': momentum, 'price': current, 'sma_5': sma_f, 'sma_20': sma_s}

    def _rut_trend(self, rut: pd.DataFrame) -> Dict:
        mc = self.config.market_context
        if rut is None or rut.empty or len(rut) < mc.SMA_SLOW:
            return {'trend': 'unknown', 'strength': 0, 'momentum': 0}
        r = rut.copy()
        r['sma_fast'] = r['Close'].rolling(mc.SMA_FAST).mean()
        r['sma_slow'] = r['Close'].rolling(mc.SMA_SLOW).mean()
        current = float(r['Close'].iloc[-1])
        sma_f = float(r['sma_fast'].iloc[-1])
        sma_s = float(r['sma_slow'].iloc[-1])
        if current > sma_f > sma_s:
            trend = 'bullish'
            strength = min(100.0, ((current / sma_s - 1.0) * 100.0) * 10.0)
        elif current < sma_f < sma_s:
            trend = 'bearish'
            strength = min(100.0, ((1.0 - current / sma_s) * 100.0) * 10.0)
        else:
            trend, strength = 'neutral', 50.0
        momentum = float((r['Close'].iloc[-1] / r['Close'].iloc[-mc.SMA_FAST] - 1.0) * 100.0) if len(r) >= mc.SMA_FAST + 1 else 0.0
        return {'trend': trend, 'strength': strength, 'momentum': momentum, 'price': current, 'sma_5': sma_f, 'sma_20': sma_s}

    def _vix_level(self, vix: pd.DataFrame) -> Dict:
        mc = self.config.market_context
        if vix is None or vix.empty:
            return {'level': 20.0, 'classification': 'normal'}
        level = float(vix['Close'].iloc[-1])
        if level < mc.VIX_LOW_MAX:
            classification = 'low'
        elif level < mc.VIX_NORMAL_MAX:
            classification = 'normal'
        elif level < mc.VIX_ELEVATED_MAX:
            classification = 'elevated'
        elif level < mc.VIX_HIGH_MAX:
            classification = 'high'
        else:
            classification = 'extreme'
        return {'level': level, 'classification': classification}

    def _breadth_proxy(self, spy: pd.DataFrame, rut: pd.DataFrame) -> Dict:
        mc = self.config.market_context
        try:
            spy_ret = float(spy['Close'].pct_change().iloc[-1]) if spy is not None and len(spy) >= 2 else 0.0
            rut_ret = float(rut['Close'].pct_change().iloc[-1]) if rut is not None and len(rut) >= 2 else 0.0
            if rut_ret > mc.BREADTH_POSITIVE_RUT_RET_MIN and rut_ret >= spy_ret:
                breadth, ratio = 'positive', 0.65
            elif rut_ret < mc.BREADTH_NEGATIVE_RUT_RET_MAX and rut_ret <= spy_ret:
                breadth, ratio = 'negative', 0.35
            else:
                breadth, ratio = 'neutral', 0.5
            return {'advances': None, 'declines': None, 'ratio': ratio, 'breadth': breadth, 'rut_ret': rut_ret, 'spy_ret': spy_ret}
        except Exception:
            return {'advances': None, 'declines': None, 'ratio': 0.5, 'breadth': 'neutral'}

    def _volume_profile(self, spy: pd.DataFrame) -> Dict:
        if spy is None or spy.empty or len(spy) < 2:
            return {'profile': 'normal', 'relative_volume': 1.0}
        current_volume = float(spy['Volume'].iloc[-1])
        avg_volume = float(spy['Volume'].iloc[:-1].mean())
        rv = current_volume / avg_volume if avg_volume > 0 else 1.0
        profile = 'high' if rv > 1.5 else 'low' if rv < 0.7 else 'normal'
        return {'profile': profile, 'relative_volume': rv, 'current': current_volume, 'average': avg_volume}

    def _calculate_market_score(self, context: Dict) -> float:
        return calculate_market_score(context, self.config.market_context)

    def _classify_environment(self, context: Dict) -> str:
        return classify_environment(context, self.config.market_context)

    def get_intraday_bars(self, symbol, session_date: datetime, data_handler) -> pd.DataFrame:
        """
        Get intraday bars for a given symbol and session date.
        Only works with backtest mode.
        """
        try:
            if data_handler is None or not hasattr(data_handler, "get_intraday_bars"):
                self.logger.error("Invalid data_handler provided for intraday bars retrieval.")
                return pd.DataFrame()

            start = pd.Timestamp(f"{session_date} 04:00:00", tz='US/Eastern').tz_convert('UTC')
            end = pd.Timestamp(f"{session_date} 20:00:00", tz='US/Eastern').tz_convert('UTC')

            bars = data_handler.get_intraday_bars(symbol, start=start, end=end)
            if bars is None or bars.empty:
                self.logger.warning(f"No intraday data found for {symbol} on {session_date}.")
            return bars
        except Exception as e:
            self.logger.error(f"Error retrieving intraday bars for {symbol} on {session_date}: {e}")
            return pd.DataFrame()