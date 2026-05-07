# =====================================================
# ai_pattern_analyzer.py - Ollama AI Pattern Analyzer
# =====================================================
# Delegates OHLCV pattern analysis to a locally-hosted LLM via Ollama.
# The analyzer satisfies the PatternAnalyzerProtocol, making it a
# drop-in replacement for PatternAnalyzer.
#
# Model:   deepseek-r1:7b  (or any model served by the local Ollama instance)
# Runtime: Ollama HTTP API at config.ai_analyzer.OLLAMA_BASE_URL
# =====================================================

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from utils.logging import get_logger


# ---------------------------------------------------------------------------
# Sentinel exception — caught by DualPatternAnalyzer for graceful fallback
# ---------------------------------------------------------------------------

class AIAnalyzerError(Exception):
    """Raised on any unrecoverable AI API or response-parsing failure."""


# ---------------------------------------------------------------------------
# System prompt — describes the analysis task and required JSON schema.
# Kept identical to the original so pattern behaviour is unchanged.
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
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by deepseek-r1."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_block(text: str) -> str:
    """
    Extract a JSON object from text that may contain markdown fences or
    surrounding prose — common with local LLMs that ignore "only JSON" prompts.
    """
    # Try markdown fence first: ```json { ... } ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Fall back to first { ... } block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


# ---------------------------------------------------------------------------
# AIPatternAnalyzer
# ---------------------------------------------------------------------------

class AIPatternAnalyzer:
    """
    Drop-in replacement for PatternAnalyzer that delegates analysis to a
    locally-hosted LLM via the Ollama HTTP API.

    Satisfies PatternAnalyzerProtocol — no changes needed in calling code.
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, component="ai_analyzer")
        self._cfg = config.ai_analyzer

        # Lazy-import requests so environments without it surface a clear error.
        try:
            import requests as _requests
            self._requests = _requests
        except ImportError as exc:
            raise ImportError(
                "requests package is required for AIPatternAnalyzer. "
                "Install it with: pip install requests"
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
        """Call Ollama API and return a result dict matching PatternAnalyzer output."""
        if bars is None or bars.empty or len(bars) < 5:
            raise AIAnalyzerError(f"{symbol}: insufficient bars for AI analysis")

        user_message = self._build_user_message(symbol, bars, is_premarket, gap_percent)

        url = f"{self._cfg.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": self._cfg.MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": 0},  # deterministic output for trading
        }

        try:
            response = self._requests.post(
                url,
                json=payload,
                timeout=self._cfg.TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except self._requests.Timeout as exc:
            raise AIAnalyzerError(
                f"{symbol}: Ollama API timeout after {self._cfg.TIMEOUT_SECONDS}s — {exc}"
            ) from exc
        except self._requests.RequestException as exc:
            raise AIAnalyzerError(
                f"{symbol}: Ollama API request failed ({url}) — {exc}"
            ) from exc

        try:
            raw_text = response.json()["message"]["content"]
        except (KeyError, ValueError) as exc:
            raise AIAnalyzerError(
                f"{symbol}: Unexpected Ollama response structure — {exc}"
            ) from exc

        return self._parse_response(raw_text.strip(), symbol, is_premarket, gap_percent)

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
        """Parse the AI's response into the standard result dict."""
        # deepseek-r1 wraps reasoning in <think> blocks before the answer.
        cleaned = _strip_think_tags(raw_text)
        # Local LLMs sometimes wrap JSON in markdown fences or add prose.
        cleaned = _extract_json_block(cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise AIAnalyzerError(
                f"{symbol}: AI returned non-JSON response — {exc}\n"
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
            raise AIAnalyzerError(
                f"{symbol}: AI response missing keys: {missing}"
            )

        # Normalise / guarantee certain fields
        data["symbol"] = symbol
        data["is_premarket_analyzed"] = is_premarket
        data["gap_percent"] = gap_percent
        data["timestamp"] = data.get("timestamp") or datetime.now().isoformat()

        self.logger.info(
            "AI analyzed %s: valid=%s strength=%.2f min_threshold=%.1f patterns=%s",
            symbol,
            data["valid"],
            data["pattern_strength"],
            data["min_score_threshold"],
            data["patterns_detected"],
        )
        return data
