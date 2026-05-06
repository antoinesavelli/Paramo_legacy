"""
Tests for config/loader.py

Covers:
- parse_override_pairs: valid pairs, missing '=', empty list
- _coerce: all supported types (bool, int, float, list, str)
- apply_overrides_immutable: single-level and nested path overrides
- apply_env_layer: env vars with correct prefix applied / ignored otherwise
- validate_config: each validation rule fires correctly
- build_config: no env layer, cli overrides, export round-trip
"""

import json
import os
import pytest
from pathlib import Path

from config.loader import (
    parse_override_pairs,
    _coerce,
    apply_overrides_immutable,
    apply_env_layer,
    validate_config,
    build_config,
    export_effective,
)
from config import TradingConfig


# ---------------------------------------------------------------------------
# parse_override_pairs
# ---------------------------------------------------------------------------

class TestParseOverridePairs:
    def test_single_valid_pair(self):
        result = parse_override_pairs(["screening.MIN_PRICE=3.5"])
        assert result == [("screening.MIN_PRICE", "3.5")]

    def test_multiple_pairs(self):
        result = parse_override_pairs(["a.b=1", "c.d=hello"])
        assert result == [("a.b", "1"), ("c.d", "hello")]

    def test_value_with_equals_sign(self):
        # Only the first '=' is the separator; rest is part of the value.
        result = parse_override_pairs(["key=val=ue"])
        assert result == [("key", "val=ue")]

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="Invalid override"):
            parse_override_pairs(["no_equals_here"])

    def test_none_input_returns_empty(self):
        assert parse_override_pairs(None) == []

    def test_empty_list_returns_empty(self):
        assert parse_override_pairs([]) == []

    def test_strips_whitespace(self):
        result = parse_override_pairs([" screening.MIN_PRICE = 5.0 "])
        assert result == [("screening.MIN_PRICE", "5.0")]


# ---------------------------------------------------------------------------
# _coerce
# ---------------------------------------------------------------------------

class TestCoerce:
    def test_bool_true_variants(self):
        for raw in ("1", "true", "True", "TRUE", "yes", "on"):
            assert _coerce(True, raw) is True

    def test_bool_false_variants(self):
        for raw in ("0", "false", "no", "off", "False"):
            assert _coerce(True, raw) is False

    def test_int_valid(self):
        assert _coerce(10, "42") == 42

    def test_int_invalid_returns_example(self):
        assert _coerce(10, "abc") == 10

    def test_float_valid(self):
        assert _coerce(1.0, "3.14") == pytest.approx(3.14)

    def test_float_invalid_returns_example(self):
        assert _coerce(1.0, "xyz") == 1.0

    def test_list_splits_on_comma(self):
        assert _coerce([], "a, b, c") == ["a", "b", "c"]

    def test_str_passthrough(self):
        assert _coerce("hello", "world") == "world"


# ---------------------------------------------------------------------------
# apply_overrides_immutable
# ---------------------------------------------------------------------------

class TestApplyOverridesImmutable:
    def test_single_level_override(self):
        cfg = TradingConfig()
        result = apply_overrides_immutable(cfg, [("screening.MIN_PRICE", "5.0")])
        assert result.screening.MIN_PRICE == pytest.approx(5.0)

    def test_original_config_unchanged(self):
        cfg = TradingConfig()
        original_price = cfg.screening.MIN_PRICE
        apply_overrides_immutable(cfg, [("screening.MIN_PRICE", "99.0")])
        assert cfg.screening.MIN_PRICE == original_price  # frozen

    def test_multiple_overrides_applied_in_order(self):
        cfg = TradingConfig()
        result = apply_overrides_immutable(cfg, [
            ("screening.MIN_PRICE", "5.0"),
            ("screening.MIN_GAP_PERCENT", "15.0"),
        ])
        assert result.screening.MIN_PRICE == pytest.approx(5.0)
        assert result.screening.MIN_GAP_PERCENT == pytest.approx(15.0)

    def test_unknown_attribute_raises(self):
        cfg = TradingConfig()
        with pytest.raises(AttributeError, match="Unknown config attribute"):
            apply_overrides_immutable(cfg, [("screening.NONEXISTENT_FIELD", "1")])

    def test_bool_override(self):
        cfg = TradingConfig()
        result = apply_overrides_immutable(cfg, [("screening.ENABLE_RELATIVE_VOLUME", "true")])
        assert result.screening.ENABLE_RELATIVE_VOLUME is True

    def test_int_override(self):
        cfg = TradingConfig()
        result = apply_overrides_immutable(cfg, [("system.MAX_RETRIES", "7")])
        assert result.system.MAX_RETRIES == 7


# ---------------------------------------------------------------------------
# apply_env_layer
# ---------------------------------------------------------------------------

class TestApplyEnvLayer:
    def test_matching_prefix_applied(self, monkeypatch):
        monkeypatch.setenv("PARAMO__screening__MIN_PRICE", "8.0")
        cfg = apply_env_layer(TradingConfig())
        assert cfg.screening.MIN_PRICE == pytest.approx(8.0)

    def test_non_matching_prefix_ignored(self, monkeypatch):
        monkeypatch.setenv("OTHER__screening__MIN_PRICE", "999.0")
        cfg = apply_env_layer(TradingConfig())
        assert cfg.screening.MIN_PRICE == TradingConfig().screening.MIN_PRICE

    def test_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("MYAPP__screening__MIN_PRICE", "6.0")
        cfg = apply_env_layer(TradingConfig(), prefix="MYAPP__")
        assert cfg.screening.MIN_PRICE == pytest.approx(6.0)

    def test_no_env_vars_returns_unchanged(self, monkeypatch):
        # Remove any accidentally present PARAMO__ vars
        for key in list(os.environ.keys()):
            if key.startswith("PARAMO__"):
                monkeypatch.delenv(key)
        original = TradingConfig()
        result = apply_env_layer(original)
        assert result == original


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_valid_default_config_passes(self):
        validate_config(TradingConfig())  # must not raise

    def test_negative_min_price_fails(self):
        cfg = apply_overrides_immutable(TradingConfig(), [("screening.MIN_PRICE", "-1.0")])
        with pytest.raises(ValueError, match="MIN_PRICE"):
            validate_config(cfg)

    def test_zero_min_gap_fails(self):
        cfg = apply_overrides_immutable(TradingConfig(), [("screening.MIN_GAP_PERCENT", "0.0")])
        with pytest.raises(ValueError, match="MIN_GAP_PERCENT"):
            validate_config(cfg)

    def test_stop_loss_exceeds_daily_loss_fails(self):
        cfg = apply_overrides_immutable(TradingConfig(), [
            ("risk.STOP_LOSS_PERCENT_OF_ACCOUNT", "10.0"),
            ("risk.MAX_DAILY_LOSS_PERCENT", "5.0"),
        ])
        with pytest.raises(ValueError, match="STOP_LOSS_PERCENT_OF_ACCOUNT cannot exceed"):
            validate_config(cfg)

    def test_max_position_size_over_100_fails(self):
        cfg = apply_overrides_immutable(TradingConfig(), [("risk.MAX_POSITION_SIZE_PERCENT", "101.0")])
        with pytest.raises(ValueError, match="MAX_POSITION_SIZE_PERCENT"):
            validate_config(cfg)

    def test_scan_interval_too_small_fails(self):
        cfg = apply_overrides_immutable(TradingConfig(), [("system.SCAN_INTERVAL_SECONDS", "4")])
        with pytest.raises(ValueError, match="SCAN_INTERVAL_SECONDS"):
            validate_config(cfg)

    def test_multiple_errors_reported_together(self):
        cfg = apply_overrides_immutable(TradingConfig(), [
            ("screening.MIN_PRICE", "-1.0"),
            ("screening.MIN_GAP_PERCENT", "0.0"),
        ])
        with pytest.raises(ValueError) as exc_info:
            validate_config(cfg)
        msg = str(exc_info.value)
        assert "MIN_PRICE" in msg
        assert "MIN_GAP_PERCENT" in msg

    def test_atr_period_zero_when_enabled_fails(self):
        cfg = apply_overrides_immutable(TradingConfig(), [
            ("risk.ATR_TRAILING_ENABLED", "true"),
            ("risk.ATR_TRAILING_PERIOD", "0"),
        ])
        with pytest.raises(ValueError, match="ATR_TRAILING_PERIOD"):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_default_build_succeeds(self):
        cfg = build_config(enable_env_layer=False)
        assert cfg is not None

    def test_cli_override_applied(self):
        cfg = build_config(
            cli_overrides=["screening.MIN_PRICE=7.0"],
            enable_env_layer=False,
        )
        assert cfg.screening.MIN_PRICE == pytest.approx(7.0)

    def test_invalid_cli_override_raises(self):
        with pytest.raises(ValueError):
            build_config(
                cli_overrides=["screening.MIN_PRICE=-5.0"],
                enable_env_layer=False,
            )

    def test_export_writes_valid_json(self, tmp_path):
        export_path = str(tmp_path / "effective.json")
        build_config(enable_env_layer=False, export_path=export_path)
        data = json.loads(Path(export_path).read_text())
        assert "screening" in data
        assert "risk" in data

    def test_export_redacts_api_keys(self, tmp_path):
        export_path = str(tmp_path / "effective.json")
        cfg = apply_overrides_immutable(TradingConfig(), [
            ("api.ALPACA_API_KEY", "my-secret-key"),
        ])
        export_effective(cfg, export_path)
        data = json.loads(Path(export_path).read_text())
        assert data["api"]["ALPACA_API_KEY"] == "***REDACTED***"
        assert "my-secret-key" not in Path(export_path).read_text()