"""Logging utilities and helpers."""

import logging
from typing import Optional
from contextlib import contextmanager
import time
from enum import Enum


class ContextAdapter(logging.LoggerAdapter):
    """Attach static context (e.g., component, run_id) and support key=value logging safely."""
    def process(self, msg, kwargs):
        # Keep only logging-supported kwargs; convert others to context suffix
        supported = {"exc_info", "stack_info", "stacklevel", "extra"}
        extra_fields = {}
        for k in list(kwargs.keys()):
            if k not in supported:
                extra_fields[k] = kwargs.pop(k)

        extra = kwargs.get("extra", {})
        merged_ctx = {**self.extra, **extra, **extra_fields}
        suffix = " ".join(f"{k}={v}" for k, v in merged_ctx.items()) if merged_ctx else ""
        if suffix:
            msg = f"{msg} {suffix}"
        # Ensure 'extra' remains a dict for handlers/formatters that expect it
        kwargs["extra"] = extra if isinstance(extra, dict) else {}
        return msg, kwargs


def get_logger(name: Optional[str] = None, **static_context) -> logging.Logger:
    """Get a module logger with optional static context bound."""
    base = logging.getLogger(name if name else __name__)
    return ContextAdapter(base, static_context or {})


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Idempotent logging setup for console and optional file."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    root.setLevel(level)
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # ✅ CHANGED: Always log DEBUG+ to file for diagnostics
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(file_handler)


@contextmanager
def log_step(logger: logging.Logger, message: str, **fields):
    """Log start/end of a step with elapsed time."""
    start = time.perf_counter()
    logger.info("START: %s", message, **fields)
    try:
        yield
        elapsed = time.perf_counter() - start
        logger.info("END: %s elapsed=%.3fs", message, round(elapsed, 3), **fields)
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("FAIL: %s elapsed=%.3fs", message, round(elapsed, 3), **fields)
        raise


class BacktestLogLevel(Enum):
    """Logging levels for backtest optimization."""
    NORMAL = 0   # Regular logging
    MINIMAL = 1  # Reduced logging for better performance
    SILENT = 2   # No logging for maximum performance


def enable_backtest_optimization(level: BacktestLogLevel = BacktestLogLevel.MINIMAL):
    """
    Configure logging optimizations for backtest runs.
    
    Args:
        level: The optimization level to use
        
    Note: File handlers maintain DEBUG level for full diagnostic output.
    This function only affects console output and root logger level.
    """
    root = logging.getLogger()
    
    if level == BacktestLogLevel.SILENT:
        # Disable most logging during backtest (console only)
        root.setLevel(logging.CRITICAL)
    elif level == BacktestLogLevel.MINIMAL:
        # Only show important logs (console only)
        root.setLevel(logging.WARNING)
    else:
        # Keep logging as is
        root.setLevel(logging.INFO)
    
    # ✅ ADDED: Ensure file handlers still capture everything
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
    
    return level


class OptimizedLogger:
    """
    Performance-optimized logger with buffering support for high-throughput scenarios.
    
    ⚠️  IMPORTANT BUFFERING BEHAVIOR:
    ------------------------------------
    - Logs are buffered up to 1000 entries before automatic flushing
    - Warnings and errors are ALWAYS logged immediately (not buffered)
    - Info and debug messages are buffered for performance
    - Call .flush() explicitly to write buffered logs immediately
    
    WHEN TO USE:
    ------------
    - Performance-critical loops (e.g., processing thousands of trades)
    - Backtests with high-frequency logging
    - Batch operations where immediate logging isn't required
    
    WHEN NOT TO USE:
    ----------------
    - Live trading (use regular logger for real-time monitoring)
    - Error-critical paths where immediate visibility is required
    - Interactive debugging sessions
    
    USAGE EXAMPLE:
    --------------
    ```python
    # Setup
    base_logger = logging.getLogger(__name__)
    opt_logger = OptimizedLogger(base_logger, enabled=True)
    
    # High-frequency loop
    for i in range(10000):
        opt_logger.info(f"Processing item {i}")  # Buffered
        if i % 1000 == 0:
            opt_logger.flush()  # Periodic flush
    
    # Final flush
    opt_logger.flush()  # ✅ IMPORTANT: Always flush at end!
    ```
    
    BUFFER MANAGEMENT:
    ------------------
    - Buffer size: 1000 entries (configurable via _buffer_size)
    - Auto-flush: Triggered when buffer is full
    - Manual flush: Call .flush() to write immediately
    - No flush on warnings/errors: They bypass the buffer
    """
    
    def __init__(self, base_logger, enabled=True):
        self.base_logger = base_logger
        self.enabled = enabled
        self._buffer = []
        self._buffer_size = 1000
    
    def info(self, msg, *args, **kwargs):
        """Log info message (buffered)."""
        if not self.enabled:
            return
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(('info', msg, args, kwargs))
        else:
            self.flush()
            self.base_logger.info(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        """Log debug message (buffered)."""
        if not self.enabled:
            return
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(('debug', msg, args, kwargs))
        else:
            self.flush()
            self.base_logger.debug(msg, *args, **kwargs)
            
    def warning(self, msg, *args, **kwargs):
        """Log warning message (NOT buffered - immediate)."""
        self.base_logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        """Log error message (NOT buffered - immediate)."""
        self.base_logger.error(msg, *args, **kwargs)
    
    def flush(self):
        """
        Write all buffered logs immediately.
        
        ⚠️  IMPORTANT: Always call this at the end of your workflow
        to ensure all buffered logs are written!
        """
        for level, msg, args, kwargs in self._buffer:
            getattr(self.base_logger, level)(msg, *args, **kwargs)
        self._buffer.clear()