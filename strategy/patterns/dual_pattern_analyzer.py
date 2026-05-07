# =====================================================
# dual_pattern_analyzer.py - Composite Pattern Analyzer
# =====================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, NamedTuple, Optional

import pandas as pd

from strategy.patterns.pattern_analyzer import PatternAnalyzer
from strategy.patterns.ai_pattern_analyzer import AIPatternAnalyzer, AIAnalyzerError
from utils.logging import get_logger

if TYPE_CHECKING:
    from strategy.patterns.analyzer_protocol import PatternAnalyzerProtocol

_VALID_MODES = {"hard_coded_only", "ai_only", "both"}
_VALID_CONSENSUS = {"and", "or", "primary_hard_coded", "primary_ai"}


class _AnalysisArgs(NamedTuple):
    """Bundles the forwarded analyze_pattern call arguments."""
    symbol: str
    bars: Optional[pd.DataFrame]
    cache_key: Optional[str]
    is_premarket: bool
    gap_percent: Optional[float]


class DualPatternAnalyzer:
    """
    Composite analyzer that wraps PatternAnalyzer and ClaudePatternAnalyzer.

    Dispatch and consensus behaviour are controlled entirely by
    ``config.claude_analyzer``.  The screener pipeline needs zero changes —
    this object is injected as a drop-in for ``PatternAnalyzer``.
    """

    def __init__(
        self,
        hard_coded: PatternAnalyzer,
        ai: AIPatternAnalyzer,
        config,
        is_backtest: bool = False,
    ):
        self._hard_coded = hard_coded
        self._ai = ai
        self._cfg = config.ai_analyzer
        self.is_backtest = is_backtest
        self.logger = get_logger(__name__, component="dual_analyzer")

        if self._cfg.MODE not in _VALID_MODES:
            raise ValueError(
                f"AIAnalyzerConfig.MODE must be one of {_VALID_MODES}, "
                f"got '{self._cfg.MODE}'"
            )
        if self._cfg.CONSENSUS not in _VALID_CONSENSUS:
            raise ValueError(
                f"AIAnalyzerConfig.CONSENSUS must be one of {_VALID_CONSENSUS}, "
                f"got '{self._cfg.CONSENSUS}'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_pattern(
        self,
        symbol: str,
        bars: Optional[pd.DataFrame] = None,
        cache_key: Optional[str] = None,
        is_premarket: bool = False,
        gap_percent: Optional[float] = None,
    ) -> Dict:
        req = _AnalysisArgs(symbol, bars, cache_key, is_premarket, gap_percent)
        mode = self._cfg.MODE

        if mode == "hard_coded_only":
            return self._hard_coded.analyze_pattern(*req)

        ai_allowed = self._ai_permitted()

        if mode == "ai_only":
            if not ai_allowed:
                self.logger.info(
                    "%s: AI skipped (backtest guard). Falling back to hard-coded.",
                    symbol
                )
                return self._hard_coded.analyze_pattern(*req)
            return self._run_ai_with_fallback(req, fallback_on_error=True)

        # mode == "both"
        hc_result = self._hard_coded.analyze_pattern(*req)

        if not ai_allowed:
            self.logger.debug(
                "%s: AI skipped (backtest guard). Using hard-coded result.",
                symbol
            )
            hc_result.setdefault("meta", {})["ai_skipped"] = True
            return hc_result

        ai_result = self._run_ai_with_fallback(req, fallback_on_error=False)

        if ai_result is None:
            hc_result.setdefault("meta", {})["ai_error"] = True
            return hc_result

        return self._apply_consensus(hc_result, ai_result, symbol)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ai_permitted(self) -> bool:
        """Return False when AI must be skipped for this call."""
        return not (self.is_backtest and not self._cfg.ENABLED_IN_BACKTEST)

    def _run_ai_with_fallback(
        self,
        req: _AnalysisArgs,
        fallback_on_error: bool,
    ) -> Optional[Dict]:
        """
        Call the AI analyzer.  On AIAnalyzerError:
        - fallback_on_error=True  → run hard-coded and return its result
        - fallback_on_error=False → return None so the caller decides
        """
        try:
            return self._ai.analyze_pattern(*req)
        except AIAnalyzerError as exc:
            if fallback_on_error:
                self.logger.warning(
                    "%s: AIAnalyzerError — %s. Falling back to hard-coded result.",
                    req.symbol, exc
                )
                result = self._hard_coded.analyze_pattern(*req)
                result.setdefault("meta", {})["ai_fallback"] = str(exc)
                return result
            self.logger.warning("%s: AIAnalyzerError — %s.", req.symbol, exc)
            return None

    def _apply_consensus(self, hc: Dict, ai: Dict, symbol: str) -> Dict:
        """Merge two result dicts according to the configured consensus rule."""
        consensus = self._cfg.CONSENSUS
        meta = {"hard_coded": hc, "ai": ai, "consensus_mode": consensus}

        if consensus == "and":
            winner = dict(hc)
            winner["valid"] = hc["valid"] and ai["valid"]

        elif consensus == "or":
            hc_valid = hc["valid"]
            ai_valid = ai["valid"]
            if hc_valid and ai_valid:
                # Both valid — prefer higher pattern strength
                winner = dict(
                    hc if hc.get("pattern_strength", 0) >= ai.get("pattern_strength", 0)
                    else ai
                )
            elif hc_valid:
                winner = dict(hc)
            elif ai_valid:
                winner = dict(ai)
            else:
                winner = dict(hc)

        elif consensus == "primary_hard_coded":
            winner = dict(hc)

        else:  # primary_ai
            winner = dict(ai)

        winner["meta"] = meta
        self.logger.info(
            "%s: consensus=%s → valid=%s hc_valid=%s ai_valid=%s",
            symbol, consensus, winner["valid"], hc["valid"], ai["valid"]
        )
        return winner