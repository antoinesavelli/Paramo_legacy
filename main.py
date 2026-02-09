# =====================================================
# PARAMO:    PARABOLIC MOMENTUM TRADING ALGORITHM
# Alpaca API Integration for Extreme Gap Trading
# =====================================================

from config.config import TradingConfig  # ✅ FIXED: Added .config
from data_handler.api import APIDataHandler
from data_handler.local import LocalDataHandler
from screener.live import LiveScreener
from strategy.pattern_analyzer import PatternAnalyzer
from strategy.risk_manager import RiskManager
from core.trade_executor import TradeExecutor
from core.monitor import Monitor
from market_context.live import MarketContext
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
    - <market_root>/ticker_data/YYYY/MM/YYYY-MM-DD.parquet (market data)
    - T:/trading/daily_aggregates/YYYY/YYYYMM.parquet (aggregates)  # ✅ FIXED: Updated format
    - <market_root>/news_data/ (news data)
    - <market_root>/market_context/ (SPY, VIX, RUT CSVs)
    """
    logger = logger or logging.getLogger(__name__)
    
    if market_root:
        _ensure_dir(market_root)
        # Don't create daily_aggregates subdirectory here - it's at root level now
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
        schedule.every().day.at("08:00").do(self._market_open_tasks)
        schedule.every().day.at("16:00").do(self._market_close_tasks)

    def _market_open_tasks(self):
        self.logger.info("Running market open tasks...")
        try:
            self.market_context.update_market_context()
            if not self.market_context.should_trade():
                self.logger.warning("Market conditions unfavorable - trading disabled for today")
                self.running = False
                return
            self.running = True
        except Exception as e:
            self.logger.error(f"Error in market open tasks: {e}")

    def _market_close_tasks(self):
        self.logger.info("Running market close tasks...")
        try:
            self.trade_executor.close_all_positions()
            self._generate_daily_report()
        except Exception as e:
            self.logger.error(f"Error in market close tasks: {e}")

    def _main_loop(self):
        self.logger.info("Entering main trading loop...")
        while True:
            try:
                self._update_heartbeat()
                schedule.run_pending()
                if self.running:
                    self._run_scan_with_timeout()
                time.sleep(self.config.system.SCAN_INTERVAL_SECONDS)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)

    def _run_scan_with_timeout(self):
        try:
            future = self.executor.submit(self._run_trading_cycle)
            future.result(timeout=self.max_scan_seconds)
        except FuturesTimeout:
            self.logger.error(f"Trading cycle exceeded {self.max_scan_seconds}s timeout")
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")

    def _run_trading_cycle(self):
        try:
            candidates = self.screener.screen()
            for candidate in candidates:
                if candidate['symbol'] not in self.trade_executor.active_trades:
                    signal = self._analyze_pattern(candidate)
                    if signal:
                        self.trade_executor.execute_entry(signal)
            self.trade_executor.update_active_positions()
            self.last_scan_time = datetime.now()
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")

    def _analyze_pattern(self, candidate: Dict) -> Optional[Dict]:
        try:
            pattern = self.pattern_analyzer.analyze_pattern(
                candidate['symbol'],
                is_premarket=False,
                gap_percent=candidate.get('gap_percent', 0)
            )
            if pattern.get('valid', False):
                return {
                    'symbol': candidate['symbol'],
                    'entry_price': pattern['entry_price'],
                    'stop_price': pattern['stop_price'],
                    'pattern_strength': pattern.get('pattern_strength', 0),
                    'gap_percent': candidate.get('gap_percent', 0)
                }
        except Exception as e:
            self.logger.error(f"Pattern analysis error for {candidate['symbol']}: {e}")
        return None

    def _generate_daily_report(self):
        try:
            stats = compute_statistics(
                trades=self.trade_executor.trade_history,
                equity_curve=[],
                initial_capital=10000,
                final_capital=10000,
                daily_returns=None,
                trading_days=1
            )
            report = generate_text_report(stats, title="DAILY TRADING REPORT")
            self.logger.info("\n" + report)
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")

    def shutdown(self):
        self.logger.info("Shutting down trading system...")
        self.running = False
        try:
            self.trade_executor.close_all_positions()
            self.executor.shutdown(wait=True, cancel_futures=True)
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    config = TradingConfig()
    system = TradingSystem(config)
    system.start()