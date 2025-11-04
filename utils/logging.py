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
        file_handler.setLevel(level)
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
    """
    if level == BacktestLogLevel.SILENT:
        # Disable most logging during backtest
        logging.getLogger().setLevel(logging.CRITICAL)
    elif level == BacktestLogLevel.MINIMAL:
        # Only show important logs
        logging.getLogger().setLevel(logging.WARNING)
    else:
        # Keep logging as is
        pass
    
    return level


class OptimizedLogger:
    """Logger that can be disabled for performance with buffering support."""
    
    def __init__(self, base_logger, enabled=True):
        self.base_logger = base_logger
        self.enabled = enabled
        self._buffer = []
        self._buffer_size = 1000
    
    def info(self, msg, *args, **kwargs):
        if not self.enabled:
            return
        # Buffer instead of immediate write
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(('info', msg, args, kwargs))
        else:
            self.flush()
            self.base_logger.info(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        if not self.enabled:
            return
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(('debug', msg, args, kwargs))
        else:
            self.flush()
            self.base_logger.debug(msg, *args, **kwargs)
            
    def warning(self, msg, *args, **kwargs):
        # Always log warnings immediately
        self.base_logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        # Always log errors immediately
        self.base_logger.error(msg, *args, **kwargs)
    
    def flush(self):
        """Write buffered logs."""
        for level, msg, args, kwargs in self._buffer:
            getattr(self.base_logger, level)(msg, *args, **kwargs)
        self._buffer.clear()