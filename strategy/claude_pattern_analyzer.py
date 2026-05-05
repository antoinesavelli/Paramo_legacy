# =====================================================
# claude_pattern_analyzer.py - Claude API Pattern Analyzer
# =====================================================

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from utils.logging import get_logger


# ---------------------------------------------------------------------------
# Sentinel exception – caught by DualPatternAnalyzer for graceful fallback
# ---------------------------------------------------------------------------

class ClaudeAnalyzerError(Exception):
    """Raised on any unrecoverable Claude API / parsing failure."""


# ---------------------------------------------------------------------------
# System prompt (cached on the Anthropic side via cache_control)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert intraday momentum trading analyst specialising in gap-and-go strategies on US equities.

## Your task
Analyse the provided OHLCV bar data and symbol metadata, then score the symbol against four pattern types. Return ONLY a single valid JSON object — no explanation, no markdown fences, no extra text.

## Patterns to detect

### 1. step_up
A series of higher highs where each advance retains at least 35% of its gain before the next leg up. Minimum 2 step-ups required.
- `detected`: bool
- `step_count`: int (number of qualifying step-up legs)
- `retention_rate`: float (average % of each advance retained, 0-100)
- `strength`: float (0-100)

### 2. parabolic
Accelerating upward price action with increasing volume. Fit a quadratic to the last 30 bars; positive acceleration AND recent slope angle > 0 AND recent volume > prior volume.
- `detected`: bool
- `angle`: float (degrees from horizontal)
- `acceleration`: float (quadratic coefficient)
- `volume_multiplier`: float (recent/prior volume ratio)
- `strength`: float (0-100)

### 3. breakout
Price closes above the 20-bar high with volume > 0.5× the 20-bar average volume.
- `detected`: bool
- `breakout_level`: float (the resistance level broken)
- `range_size`: float (high - low of the lookback range)
- `volume_ratio`: float (current / average volume)
- `strength`: float (0-100)

### 4. volume
Overall volume trend scored on increasing vs stable vs decreasing. High-volume correlation bonus if new-high bars carry above-average volume.
- `volume_trend`: str ("increasing" | "stable" | "decreasing" | "unknown")
- `avg_volume`: float
- `recent_volume`: float (last 5 bars average)
- `high_volume_correlation`: float (0-100)
- `strength`: float (0-100)

### 5. support_resistance
Price action relative to identified S/R levels over the last 20-bar windows.
- `support`: list[float]
- `resistance`: list[float]
- `current_price`: float
- `strength`: float (0-100)

## Scoring rubric (confluence weights)
- step_up: 50%
- volume: 25%
- support_resistance: 25%
- parabolic: 0% (score it but does not count toward confluence)
- breakout: 0% (score it but does not count toward confluence)

`pattern_strength` = weighted sum of component strengths, normalised to 0-100.

## Dynamic minimum score threshold
- gap_percent >= 100%  →  min_score = 15.0
- gap_percent >= 50%   →  min_score = 25.0
- gap_percent < 50%    →  min_score = 40.0
`valid` = pattern_strength >= min_score AND at least 1 pattern detected.

## Required JSON output schema
{
  "valid": <bool>,
  "symbol": <str>,
  "pattern_strength": <float 0-100>,
  "patterns_detected": [<str>, ...],
  "pattern_count": <int>,
  "min_score_threshold": <float>,
  "gap_percent": <float or null>,
  "step_ups": { "detected": bool, "step_count": int, "retention_rate": float, "total_advance": float, "strength": float },
  "parabolic": { "detected": bool, "angle": float, "acceleration": float, "volume_multiplier": float, "strength": float, "angle_valid": bool },
  "breakout": { "detected": bool, "breakout_level": float, "range_size": float, "volume_ratio": float, "strength": float },
  "volume": { "volume_trend": str, "avg_volume": float, "recent_volume": float, "high_volume_correlation": float, "strength": float },
  "support_resistance": { "support": [float], "resistance": [float], "current_price": float, "strength": float },
  "timestamp": <ISO-8601 str>,
  "is_premarket_analyzed": <bool>
}
Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# ClaudePatternAnalyzer
# ---------------------------------------------------------------------------

class ClaudePatternAnalyzer:
    """
    Drop-in replacement for PatternAnalyzer that delegates analysis to Claude.

    The system prompt is sent with ``cache_control: {"type": "ephemeral"}`` so
    Anthropic caches it across calls, keeping per-call token cost minimal.
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, component="claude_analyzer")
        self._cfg = config.claude_analyzer

        # Lazy-import so the package is optional in environments that do not
        # install it (e.g., pure backtesting without Claude).
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for ClaudePatternAnalyzer. "
                "Install it with: pip install anthropic"
            ) from exc

    # ------------------------------------------------------------------
    # Public API (mirrors PatternAnalyzer.analyze_pattern exactly)
    # ------------------------------------------------------------------

    def analyze_pattern(
        self,
        symbol: str,
        bars: Optional[pd.DataFrame] = None,
        cache_key: Optional[str] = None,  # accepted for interface compat, unused
        is_premarket: bool = False,
        gap_percent: Optional[float] = None,
    ) -> Dict:
        """Call Claude API and return a result dict matching PatternAnalyzer output."""
        if bars is None or bars.empty or len(bars) < 5:
            raise ClaudeAnalyzerError(f"{symbol}: insufficient bars for Claude analysis")

        user_message = self._build_user_message(symbol, bars, is_premarket, gap_percent)

        try:
            client = self._anthropic.Anthropic()
            response = client.messages.create(
                model=self._cfg.MODEL,
                max_tokens=1024,
                timeout=self._cfg.TIMEOUT_SECONDS,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
        except self._anthropic.APITimeoutError as exc:
            raise ClaudeAnalyzerError(f"{symbol}: Claude API timeout — {exc}") from exc
        except self._anthropic.APIError as exc:
            raise ClaudeAnalyzerError(f"{symbol}: Claude API error — {exc}") from exc

        raw_text = response.content[0].text.strip()
        return self._parse_response(raw_text, symbol, is_premarket, gap_percent)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        symbol: str,
        bars: pd.DataFrame,
        is_premarket: bool,
        gap_percent: Optional[float],
    ) -> str:
        """Serialise bars and metadata into a compact user message."""
        tail = bars.tail(self._cfg.MAX_BARS_TO_SEND).copy()

        # Normalise timestamp column name
        ts_col = "timestamp" if "timestamp" in tail.columns else tail.index.name or "index"
        if ts_col == "index":
            tail = tail.reset_index()
            ts_col = tail.columns[0]

        ohlcv_cols = [ts_col, "open", "high", "low", "close", "volume"]
        available = [c for c in ohlcv_cols if c in tail.columns]
        records: List[Dict] = tail[available].to_dict(orient="records")

        # Convert Timestamps to strings for JSON serialisation
        for row in records:
            if ts_col in row and hasattr(row[ts_col], "isoformat"):
                row[ts_col] = row[ts_col].isoformat()

        gap_str = f"{gap_percent:.2f}" if gap_percent is not None else "null"
        return (
            f"symbol={symbol}\n"
            f"gap_percent={gap_str}\n"
            f"is_premarket={str(is_premarket).lower()}\n"
            f"bars_count={len(records)}\n"
            f"bars={json.dumps(records, separators=(',', ':'))}"
        )

    def _parse_response(
        self,
        raw_text: str,
        symbol: str,
        is_premarket: bool,
        gap_percent: Optional[float],
    ) -> Dict:
        """Parse Claude's JSON response into the standard result dict."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ClaudeAnalyzerError(
                f"{symbol}: Claude returned non-JSON response — {exc}\n"
                f"Raw (first 200 chars): {raw_text[:200]}"
            ) from exc

        # Validate required top-level keys
        required_keys = {
            "valid", "symbol", "pattern_strength", "patterns_detected",
            "pattern_count", "min_score_threshold", "step_ups", "parabolic",
            "breakout", "volume", "support_resistance",
        }
        missing = required_keys - data.keys()
        if missing:
            raise ClaudeAnalyzerError(
                f"{symbol}: Claude response missing keys: {missing}"
            )

        # Normalise / guarantee certain fields
        data["symbol"] = symbol
        data["is_premarket_analyzed"] = is_premarket
        data["gap_percent"] = gap_percent
        data["timestamp"] = data.get("timestamp") or datetime.now().isoformat()

        self.logger.info(
            f"Claude analyzed {symbol}: valid={data['valid']} "
            f"strength={data['pattern_strength']:.2f} "
            f"min_threshold={data['min_score_threshold']:.1f} "
            f"patterns={data['patterns_detected']}"
        )
        return data