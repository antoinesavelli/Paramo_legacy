# =====================================================
# monitor.py - Performance Monitoring System
# =====================================================

import sqlite3
import json
from datetime import datetime
from config import TradingConfig
from typing import Dict, Optional, List
from utils.logging import get_logger
from utils.reporting import compute_statistics

def log_db_error(logger, msg, exc):
    logger.error(f"{msg}: {exc}")

class Monitor:
    """System monitoring and performance tracking"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = get_logger(__name__, component="monitor")
        self.db_path = config.system.DATABASE_PATH
        self._init_database()
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'current_streak': 0,
            'max_streak': 0
        }

    def _init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_time TIMESTAMP,
                        exit_time TIMESTAMP,
                        entry_price REAL,
                        exit_price REAL,
                        position_size INTEGER,
                        pnl REAL,
                        exit_reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        level TEXT,
                        message TEXT,
                        data TEXT
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metric_name TEXT,
                        metric_value REAL
                    )
                ''')
                conn.commit()
        except Exception as e:
            log_db_error(self.logger, "Error initializing database", e)

    def record_trade(self, trade: Dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (symbol, entry_time, exit_time, entry_price,
                                       exit_price, position_size, pnl, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade['symbol'],
                    trade['entry_time'],
                    trade['exit_time'],
                    trade['entry_price'],
                    trade['exit_price'],
                    trade['position_size'],
                    trade['pnl'],
                    trade['exit_reason']
                ))
                conn.commit()
            self._update_metrics(trade)
            self.logger.info(f"Trade recorded {trade['symbol']} pnl=${trade['pnl']:.2f} reason={trade['exit_reason']}")
        except Exception as e:
            log_db_error(self.logger, "Error recording trade", e)

    def _update_metrics(self, trade: Dict):
        self.metrics['total_trades'] += 1
        self.metrics['total_pnl'] += trade['pnl']
        if trade['pnl'] > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['current_streak'] = max(0, self.metrics['current_streak']) + 1
            self.metrics['max_streak'] = max(self.metrics['max_streak'], self.metrics['current_streak'])
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['current_streak'] = min(0, self.metrics['current_streak']) - 1
        self.metrics['best_trade'] = max(self.metrics['best_trade'], trade['pnl'])
        self.metrics['worst_trade'] = min(self.metrics['worst_trade'], trade['pnl'])

    def get_performance_report(self) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, entry_time, exit_time, entry_price, exit_price, position_size, pnl, exit_reason
                    FROM trades
                    WHERE exit_time >= datetime('now', '-30 days')
                ''')
                rows = cursor.fetchall()

            trades: List[Dict] = []
            for r in rows:
                symbol, entry_time, exit_time, entry_price, exit_price, position_size, pnl, exit_reason = r
                trades.append({
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })

            stats = compute_statistics(trades=trades)
            self.logger.info(f"Performance (30d): trades={stats.get('total_trades',0)} win_rate={stats.get('win_rate',0):.2f}% net=${stats.get('net_profit',0):,.2f}")
            return stats if stats else self.metrics
        except Exception as e:
            log_db_error(self.logger, "Error generating performance report", e)
            return self.metrics

    def log_system_event(self, level: str, message: str, data: Optional[Dict] = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_logs (level, message, data)
                    VALUES (?, ?, ?)
                ''', (level, message, json.dumps(data) if data else None))
                conn.commit()
        except Exception as e:
            log_db_error(self.logger, "Error logging system event", e)