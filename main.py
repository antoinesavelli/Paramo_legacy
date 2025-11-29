# =====================================================
# PARAMO:    PARABOLIC MOMENTUM TRADING ALGORITHM
# Alpaca API Integration for Extreme Gap Trading
# =====================================================

from config import TradingConfig
from data_handler.api import APIDataHandler
from data_handler.local import LocalDataHandler
from screener.live import LiveScreener
from core.pattern_analyzer import PatternAnalyzer
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor
from core.monitor import Monitor
from market_context.live import MarketContext
from core.backtester import Backtester
from utils.reporting import compute_statistics, generate_text_report
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import logging
import signal
import sys
import schedule
import time
import os
from utils.logging import setup_logging
from news.live import NewsIntegrationLive
from news.backtest import NewsIntegrationBacktest
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import functools
import time as _time


def _ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def ensure_storage_layout(market_root: str = None, news_root: str = None, logger: logging.Logger = None):
    """
    Ensure required directory structure exists.
    
    Expected structure:
    - <data_dir>/YYYY/MM/YYYYMMDD.parquet (market data)
    - <data_dir>/daily_aggregates/YYYY/YYYYMM.parquet (aggregates)
    - <data_dir>/news/ (news data)
    - <data_dir>/market_context/ (SPY, VIX, RUT CSVs)
    """
    logger = logger or logging.getLogger(__name__)
    
    if market_root:
        _ensure_dir(market_root)
        _ensure_dir(os.path.join(market_root, "daily_aggregates"))
        logger.info(f"Storage ready for market data at: {market_root}")
    
    if news_root:
        _ensure_dir(news_root)
        logger.info(f"Storage ready for news at: {news_root}")


class TradingSystem:
    """Main trading system orchestrator"""

    def __init__(self, config: TradingConfig | None = None):
        self.config = config or TradingConfig()
        setup_logging(level=logging.INFO, log_file="run.log")
        self.logger = logging.getLogger(__name__)
        
        self.api = tradeapi.REST(
            self.config.api.ALPACA_API_KEY,
            self.config.api.ALPACA_SECRET_KEY,
            self.config.api.ALPACA_BASE_URL,
            api_version='v2'
        )
        
        self.data_handler = APIDataHandler(self.config)
        self.screener = LiveScreener(self.config, self.data_handler)
        self.pattern_analyzer = PatternAnalyzer(self.config, self.data_handler)
        self.risk_manager = RiskManager(self.config, self.api)
        
        self.market_context = MarketContext(self.config, self.api)
        
        self.trade_executor = TradeExecutor(
            self.config, 
            self.api, 
            self.risk_manager,
            market_context=self.market_context
        )
        
        self.monitor = Monitor(self.config)
        self.news_integration = NewsIntegrationLive(self.config)
        self.running = False
        self.last_scan_time = None
        self.heartbeat_path = "runtime_heartbeat.txt"
        self.last_heartbeat = datetime.utcnow()
        self.watchdog_interval = 120
        self.max_scan_seconds = 45
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.logger.info("Trading system initialized")

    def _update_heartbeat(self):
        self.last_heartbeat = datetime.utcnow()
        try:
            with open(self.heartbeat_path, "w", encoding="utf-8") as f:
                f.write(self.last_heartbeat.isoformat())
        except Exception:
            pass

    def _watchdog_loop(self):
        while True:
            _time.sleep(self.watchdog_interval)
            now = datetime.utcnow()
            delta = (now - self.last_heartbeat).total_seconds()
            if delta > self.watchdog_interval * 3:
                self.logger.critical(f"Watchdog: system stalled (last heartbeat {delta:.0f}s ago). Initiating shutdown.")
                try:
                    self.shutdown()
                finally:
                    os._exit(1)

    def _signal_handler(self, signum, frame):
        self.logger.info("Shutdown signal received")
        self.shutdown()
        sys.exit(0)

    def start(self):
        self.logger.info("Starting trading system...")
        self.running = True
        if not self._verify_api():
            return
        self._schedule_tasks()
        self._main_loop()

    def _verify_api(self):
        try:
            account = self.api.get_account()
            self.logger.info(f"Connected to Alpaca. Account status: {account.status}")
            if account.trading_blocked:
                self.logger.error("Trading is blocked on this account")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def _schedule_tasks(self):
        schedule.every(self.config.system.SCAN_INTERVAL_SECONDS).seconds.do(self.scan_market)
        schedule.every(5).seconds.do(self.update_positions)
        schedule.every(30).seconds.do(self.check_risk_limits)
        schedule.every(60).seconds.do(self.generate_report)

    def _main_loop(self):
        while self.running:
            try:
                market_status = self.data_handler.get_market_status()
                if market_status['is_open']:
                    schedule.run_pending()
                    time.sleep(1)
                else:
                    self.logger.info("Market is closed. Waiting...")
                    time.sleep(60)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(5)

    def scan_market(self):
        try:
            for attempt in range(3):
                try:
                    self.market_context.update_market_context()
                    break
                except Exception as e:
                    self.logger.error(f"Market context update failed (attempt {attempt+1}): {e}")
                    time.sleep(2)
            if not self.market_context.should_trade():
                self.logger.info("Market conditions unfavorable. Skipping scan.")
                return
            self.logger.debug("Starting market scan...")
            self.risk_manager.update_daily_pnl()
            
            account = self.api.get_account()
            equity = float(account.equity)
            max_daily_loss_dollars = equity * (self.config.risk.MAX_DAILY_LOSS_PERCENT / 100.0)
            
            if self.risk_manager.daily_pnl <= -max_daily_loss_dollars:
                self.logger.warning("Daily loss limit reached. Stopping scans.")
                return
            try:
                candidates = self.screener.run_screen()
            except Exception as e:
                self.logger.error(f"Screener failed: {e}")
                candidates = []
            if not candidates:
                self.logger.debug("No candidates found")
                return
            for candidate in candidates[:5]:
                symbol = candidate['symbol']
                news_impact = None
                for attempt in range(2):
                    try:
                        news_impact = self.news_integration.analyze_news_impact(symbol)
                        break
                    except Exception as e:
                        self.logger.error(f"News analysis failed for {symbol} (attempt {attempt+1}): {e}")
                        time.sleep(1)
                if news_impact is None:
                    self.logger.warning(f"Skipping {symbol}: News data unavailable.")
                    continue
                if not news_impact.get('has_catalyst') or news_impact.get('catalyst_strength', 0) < 30:
                    self.logger.info(f"Skipping {symbol}: No strong catalyst in news.")
                    continue
                if symbol in self.trade_executor.active_trades:
                    continue
                try:
                    pattern_analysis = self.pattern_analyzer.analyze_pattern(symbol)
                except Exception as e:
                    self.logger.error(f"Pattern analysis failed for {symbol}: {e}")
                    continue
                if pattern_analysis.get('valid'):
                    self.logger.info(f"Valid pattern found for {symbol}")
                    try:
                        bars = self.data_handler.get_intraday_bars(symbol, '1Min', 60)
                        quote_data = self.data_handler.get_quote_data([symbol])
                        entry_price = quote_data[symbol]['ask'] if symbol in quote_data else None
                        if entry_price is None:
                            self.logger.warning(f"Entry price unavailable for {symbol}. Skipping.")
                            continue
                        stop_price = self.risk_manager.calculate_stop_loss(symbol, entry_price, bars)
                        signal = {
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'stop_price': stop_price,
                            'pattern_strength': pattern_analysis['pattern_strength'],
                            'timestamp': datetime.now()
                        }
                        result = self.trade_executor.execute_entry(signal)
                        if result.get('success'):
                            self.monitor.log_system_event('INFO', f"Position opened: {symbol}", result)
                    except Exception as e:
                        self.logger.error(f"Trade execution failed for {symbol}: {e}")
            self.last_scan_time = datetime.now()
        except Exception as e:
            self.logger.critical(f"Fatal error in scan_market: {e}")

    def update_positions(self):
        try:
            self.trade_executor.check_max_hold_time()
            self.trade_executor.update_stops()
            
            for symbol, position in list(self.trade_executor.active_trades.items()):
                if position and 'entry_time' in position:
                    if datetime.now() - position['entry_time'] > timedelta(hours=2):
                        self.logger.info(f"Closing {symbol} due to stalled momentum")
                        try:
                            self.trade_executor.execute_exit(symbol, reason="time_stop")
                        except Exception as e:
                            self.logger.error(f"Error executing exit for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    def check_risk_limits(self):
        try:
            self.risk_manager.update_daily_pnl()
            account = self.api.get_account()
            equity = float(account.equity)
            
            if self.risk_manager.peak_balance > 0:
                drawdown = ((self.risk_manager.peak_balance - equity) / self.risk_manager.peak_balance) * 100
                if drawdown >= self.config.risk.MAX_DRAWDOWN_PERCENT * 1.2:
                    self.logger.critical(f"EMERGENCY: Drawdown {drawdown:.2f}% exceeds limits")
                    self.risk_manager.emergency_liquidate_all()
                    self.running = False
            
            max_daily_loss_dollars = equity * (self.config.risk.MAX_DAILY_LOSS_PERCENT / 100.0)
            if self.risk_manager.daily_pnl <= -max_daily_loss_dollars:
                self.logger.warning("Daily loss limit reached. Closing all positions.")
                for symbol in list(self.trade_executor.active_trades.keys()):
                    try:
                        self.trade_executor.execute_exit(symbol, reason="daily_loss_limit")
                    except Exception as e:
                        self.logger.error(f"Error executing exit for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")

    def generate_report(self):
        try:
            trades = getattr(self.trade_executor, "trade_history", [])
            stats = compute_statistics(trades=trades)
            report = generate_text_report(stats, title="LIVE PERFORMANCE")
            self.logger.info(report)
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")

    def shutdown(self):
        self.logger.info("Shutting down trading system...")
        self.running = False
        for symbol in list(self.trade_executor.active_trades.keys()):
            try:
                self.trade_executor.execute_exit(symbol, reason="system_shutdown")
            except Exception as e:
                self.logger.error(f"Error executing exit for {symbol} during shutdown: {e}")
        try:
            self.api.cancel_all_orders()
        except Exception:
            pass
        self.generate_report()
        self.logger.info("Trading system shutdown complete")