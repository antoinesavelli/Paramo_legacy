-- schema.sql — Paramo trading system database schema
-- Single source of truth for all SQLite tables.
-- Kept in sync with monitoring/monitor.py _init_database().

CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT      NOT NULL,
    entry_time    TIMESTAMP,
    exit_time     TIMESTAMP,
    entry_price   REAL,
    exit_price    REAL,
    position_size INTEGER,
    pnl           REAL,
    exit_reason   TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level     TEXT,
    message   TEXT,
    data      TEXT
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name  TEXT,
    metric_value REAL
);

CREATE TABLE IF NOT EXISTS order_audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    symbol     TEXT,
    order_id   TEXT,
    status     TEXT,
    qty        INTEGER,
    price      REAL,
    side       TEXT,
    order_type TEXT,
    details    TEXT
);

-- Indexes for fast time-range queries on large position histories
CREATE INDEX IF NOT EXISTS idx_trades_symbol_entry
    ON trades (symbol, entry_time);

CREATE INDEX IF NOT EXISTS idx_order_audit_symbol_ts
    ON order_audit (symbol, timestamp);

CREATE INDEX IF NOT EXISTS idx_system_logs_ts
    ON system_logs (timestamp);
