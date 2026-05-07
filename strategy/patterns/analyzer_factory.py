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
    cfg = config.ai_analyzer
    hard_coded = PatternAnalyzer(config, data_handler)

    if not (cfg.ENABLED and cfg.MODE != "hard_coded_only"):
        logger.info(
            "PatternAnalyzer: using hard-coded only "
            "(ENABLED=%s, MODE='%s')",
            cfg.ENABLED, cfg.MODE,
        )
        return hard_coded

    # Only import heavy AI dependencies when actually needed
    from strategy.patterns.ai_pattern_analyzer import AIPatternAnalyzer
    from strategy.patterns.dual_pattern_analyzer import DualPatternAnalyzer

    analyzer = DualPatternAnalyzer(
        hard_coded=hard_coded,
        ai=AIPatternAnalyzer(config),
        config=config,
        is_backtest=is_backtest,
    )
    logger.info(
        "PatternAnalyzer: DualPatternAnalyzer constructed "
        "(MODE='%s', CONSENSUS='%s', is_backtest=%s, ENABLED_IN_BACKTEST=%s)",
        cfg.MODE, cfg.CONSENSUS, is_backtest, cfg.ENABLED_IN_BACKTEST,
    )
    return analyzer