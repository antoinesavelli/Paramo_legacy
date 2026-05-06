# =====================================================
# analyzer_factory.py - Pattern Analyzer Factory
# =====================================================

from __future__ import annotations

from config import TradingConfig
from strategy.patterns.pattern_analyzer import PatternAnalyzer
from strategy.patterns.analyzer_protocol import PatternAnalyzerProtocol
from utils.logging import get_logger

logger = get_logger(__name__, component="analyzer_factory")


def build_pattern_analyzer(
    config: TradingConfig,
    data_handler,
    is_backtest: bool = False,
) -> PatternAnalyzerProtocol:
    """
    Factory that returns the correct analyzer based on config.

    Rules
    -----
    - ``ENABLED=False``                  → plain PatternAnalyzer (Claude never imported)
    - ``MODE='hard_coded_only'``         → plain PatternAnalyzer (Claude never imported)
    - anything else                      → DualPatternAnalyzer wrapping both

    The DualPatternAnalyzer itself will honour ``ENABLED_IN_BACKTEST`` and
    ``MODE`` at call-time; the factory only decides whether to construct the
    Claude object at all.
    """
    cfg = config.claude_analyzer
    hard_coded = PatternAnalyzer(config, data_handler)

    if not (cfg.ENABLED and cfg.MODE != "hard_coded_only"):
        logger.info(
            f"PatternAnalyzer: using hard-coded only "
            f"(ENABLED={cfg.ENABLED}, MODE='{cfg.MODE}')"
        )
        return hard_coded

    # Only import heavy Claude dependencies when actually needed
    from strategy.patterns.claude_pattern_analyzer import ClaudePatternAnalyzer
    from strategy.patterns.dual_pattern_analyzer import DualPatternAnalyzer

    analyzer = DualPatternAnalyzer(
        hard_coded=hard_coded,
        claude=ClaudePatternAnalyzer(config),
        config=config,
        is_backtest=is_backtest,
    )
    logger.info(
        f"PatternAnalyzer: DualPatternAnalyzer constructed "
        f"(MODE='{cfg.MODE}', CONSENSUS='{cfg.CONSENSUS}', "
        f"is_backtest={is_backtest}, ENABLED_IN_BACKTEST={cfg.ENABLED_IN_BACKTEST})"
    )
    return analyzer