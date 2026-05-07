# =====================================================
# run/live_engine.py - Live Trading System Orchestrator
# =====================================================

from config.config import TradingConfig
from data_handler.api import APIDataHandler
from screener.live import LiveScreener
from strategy.patterns.pattern_analyzer import PatternAnalyzer
from strategy.risk_manager import RiskManager
from execution.trade_executor import TradeExecutor
from monitoring.monitor import Monitor
from market_context.live import MarketContext
from monitoring.reporting import compute_statistics, generate_text_report
import alpaca_trade_api as tradeapi
from datetime import datetime, timezone
from typing import Optional
import logging
import signal
import sys
import schedule
import time
import os
from utils.logging import setup_logging, get_logger
from news.live import NewsIntegrationLive
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import time as _time


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_storage_layout(market_root: str = None, news_root: str = None, logger: logging.Logger = None):
    logger = logger or get_logger(__name__)
    if market_root:
        _ensure_dir(market_root)
        logger.info("Storage ready for market data at: %s", market_root)
    if news_root:
        _ensure_dir(news_root)
        logger.info("Storage ready for news at: %s", news_root)


class TradingSystem:
    """Live trading system orchestrator."""

    def __init__(self, config: TradingConfig | None = None):
        self.config = config or TradingConfig()
        setup_logging(level=logging.INFO, log_file="run.log")
        self.logger = get_logger(__name__)

        self.api = tradeapi.REST(
            self.config.api.ALPACA_API_KEY,
            self.config.api.ALPACA_SECRET_KEY,
            self.config.api.ALPACA_BASE_URL,
            api_version='v2'
        )

        self.data_handler = APIDataHandler(self.config)
        self.pattern_analyzer = PatternAnalyzer(self.config, self.data_handler)
        self.screener = LiveScreener(self.config, self.data_handler, self.pattern_analyzer)
        self.risk_manager = RiskManager(self.config, self.api)
        self.market_context = MarketContext(self.config, self.api)
        self.trade_executor = TradeExecutor(
            self.config, self.api, self.risk_manager,
            market_context=self.market_context
        )
        self.monitor = Monitor(self.config)
        self.news_integration = NewsIntegrationLive(self.config)
        self.running = False
        self.last_scan_time = None
        self.heartbeat_path = "runtime_heartbeat.txt"
        self.last_heartbeat = datetime.now(timezone.utc)
        self.watchdog_interval = 120
        self.max_scan_seconds = 45
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.logger.info("Trading system initialized")

    def _update_heartbeat(self):
        self.last_heartbeat = datetime.now(timezone.utc)
        try:
            with open(self.heartbeat_path, "w", encoding="utf-8") as f:
                f.write(self.last_heartbeat.isoformat())
        except Exception:
            self.logger.debug("Heartbeat write failed", exc_info=True)

    def _watchdog_loop(self):
        while True:
            _time.sleep(self.watchdog_interval)
            now = datetime.now(timezone.utc)
            delta = (now - self.last_heartbeat).total_seconds()
            if delta > self.watchdog_interval * 3:
                self.logger.critical(  # noqa: G004
                    f"Watchdog: system stalled (last heartbeat {delta:.0f}s ago). Initiating shutdown."
                )
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
            self.logger.info("Connected to Alpaca. Account status: %s", account.status)
            if account.trading_blocked:
                self.logger.error("Trading is blocked on this account")
                return False
            return True
        except Exception as e:
            self.logger.error("Failed to connect to Alpaca: %s", e, exc_info=True)
            return False

    def _schedule_tasks(self):
        schedule.every().day.at("08:00").do(self._market_open_tasks)
        schedule.every().day.at("16:00").do(self._market_close_tasks)

    def _market_open_tasks(self):
        self.logger.info("Running market open tasks...")
        try:
            self.market_context.update_market_context()
            indicators = self.market_context.market_indicators
            score = indicators.get('market_score', 'N/A')
            env = indicators.get('trading_environment', 'N/A')
            vix = indicators.get('vix_level', {})
            spy = indicators.get('spy_trend', {})
            rut = indicators.get('rut_trend', {})
            self.logger.info(  # noqa: G004
                f"Market context update complete | Score: {score} | Environment: {env} | "
                f"VIX: {vix.get('level', 'N/A')} ({vix.get('classification', 'N/A')}) | "
                f"SPY trend: {spy.get('trend', 'N/A')} | RUT trend: {rut.get('trend', 'N/A')}"
            )
            if not self.market_context.should_trade():
                self.logger.warning("Trading DISABLED for today | Score: %s | Environment: %s", score, env)
                self.running = False
                return
            self.logger.info("Market conditions acceptable - trading ENABLED")
            self.running = True
        except Exception as e:
            self.logger.error("Error in market open tasks: %s", e, exc_info=True)

    def _market_close_tasks(self):
        self.logger.info("Running market close tasks...")
        try:
            self.trade_executor.close_all_positions()
            self._generate_daily_report()
        except Exception as e:
            self.logger.error("Error in market close tasks: %s", e, exc_info=True)

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
                self.logger.error("Error in main loop: %s", e, exc_info=True)
                time.sleep(60)

    def _run_scan_with_timeout(self):
        try:
            future = self.executor.submit(self._run_trading_cycle)
            future.result(timeout=self.max_scan_seconds)
        except FuturesTimeout:
            self.logger.error("Trading cycle exceeded %ds timeout", self.max_scan_seconds)
        except Exception as e:
            self.logger.error("Error in trading cycle: %s", e, exc_info=True)

    def _run_trading_cycle(self):
        try:
            candidates = self.screener.run_screen()
            max_candidates = self.config.backtest.MAX_CANDIDATES_PER_DAY
            candidates = candidates[:max_candidates]          # NOTE: enforce same cap as backtest
            for candidate in candidates:
                if candidate['symbol'] not in self.trade_executor.active_trades:
                    self.trade_executor.execute_entry(candidate)
            self.trade_executor.update_active_positions()
            self.last_scan_time = datetime.now(timezone.utc)
        except Exception as e:
            self.logger.error("Error in trading cycle: %s", e, exc_info=True)

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
            self.logger.error("Error generating report: %s", e, exc_info=True)

    def shutdown(self):
        self.logger.info("Shutting down trading system...")
        self.running = False
        try:
            self.trade_executor.close_all_positions()
            self.executor.shutdown(wait=True, cancel_futures=True)
        except Exception as e:
            self.logger.error("Error during shutdown: %s", e, exc_info=True)
        self.logger.info("Shutdown complete")