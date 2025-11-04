# =====================================================
# market_context.scoring.py - Shared Market Context Scoring
# =====================================================

from typing import Dict

def calculate_market_score(context: Dict, mc) -> float:
    """Shared market score calculation (small-cap focused)."""
    score = 50.0  # Start neutral

    spy = context.get('spy_trend', {}) or {}
    rut = context.get('rut_trend', {}) or {}

    # SPY trend (secondary)
    if spy.get('trend') == 'bullish':
        score += getattr(mc, 'SPY_TREND_BULL_WEIGHT', 10.0)
    elif spy.get('trend') == 'bearish':
        score += getattr(mc, 'SPY_TREND_BEAR_WEIGHT', -5.0)

    # RUT trend (primary for small-caps)
    if rut.get('trend') == 'bullish':
        score += getattr(mc, 'RUT_TREND_BULL_WEIGHT', 20.0)
    elif rut.get('trend') == 'bearish':
        score += getattr(mc, 'RUT_TREND_BEAR_WEIGHT', -15.0)

    # RUT leadership bonus (momentum edge over SPY)
    rut_mom = float(rut.get('momentum', 0.0) or 0.0)
    spy_mom = float(spy.get('momentum', 0.0) or 0.0)
    edge = getattr(mc, 'RUT_LEAD_MOMENTUM_EDGE', 0.5)
    if rut.get('trend') == 'bullish' and (rut_mom - spy_mom) >= edge:
        score += getattr(mc, 'RUT_LEAD_BONUS', 5.0)

    # VIX contribution (unchanged)
    vix = context.get('vix_level', {}) or {}
    if vix.get('classification') == 'normal':
        score += 10
    elif vix.get('classification') in ['high', 'extreme']:
        score -= 20
    elif vix.get('classification') == 'low':
        score -= 10

    # Market breadth contribution (already can be RUT-led in backtest)
    breadth = context.get('market_breadth', {}) or {}
    if breadth.get('breadth') == 'positive':
        score += 10
    elif breadth.get('breadth') == 'negative':
        score -= 10

    # Volume profile contribution
    volume = context.get('volume_profile', {}) or {}
    if volume.get('profile') == 'high':
        score += 10

    return max(0.0, min(100.0, float(score)))


def classify_environment(context: Dict, mc) -> str:
    """Shared environment classification based on market_score."""
    score = float(context.get('market_score', 50.0))
    if score >= mc.ENV_FAVORABLE_MIN:
        return 'favorable'
    if score >= mc.ENV_NEUTRAL_MIN:
        return 'neutral'
    return 'unfavorable'


def should_trade_from_indicators(indicators: Dict, mc) -> bool:
    """Shared gating logic based on indicators and config thresholds."""
    if not indicators:
        return True

    vix = indicators.get('vix_level', {}) or {}
    if mc.BLOCK_ON_VIX_EXTREME and vix.get('classification') in ['extreme']:
        return False

    if indicators.get('trading_environment') == 'unfavorable':
        if indicators.get('market_score', 50) < mc.SHOULD_TRADE_MIN_SCORE_IF_UNFAVORABLE:
            return False

    return True


def position_size_adjustment_from_indicators(indicators: Dict, mc) -> float:
    """Shared position size adjustment logic based on environment."""
    if not indicators:
        return 1.0
    env = indicators.get('trading_environment', 'neutral')
    if env == 'favorable':
        return mc.SIZE_ADJ_FAVORABLE
    if env == 'unfavorable':
        return mc.SIZE_ADJ_UNFAVORABLE
    return 1.0