"""
Tests for utils/helpers.py

Covers:
- calculate_hash: determinism, dict vs non-dict, key-order independence
- format_currency / format_percentage: formatting contracts
- safe_divide: normal, zero denominator, custom default
- calculate_compound_return: correct CAGR, edge cases
- is_market_hours: weekday/weekend, within/outside session boundaries
- save_state / load_state: round-trip, corrupt JSON, non-serialisable, missing file
"""

import json
import os
import pytest
from datetime import datetime
from unittest.mock import MagicMock

import pytz

from utils.helpers import (
    calculate_hash,
    format_currency,
    format_percentage,
    safe_divide,
    calculate_compound_return,
    is_market_hours,
    save_state,
    load_state,
)


# ---------------------------------------------------------------------------
# calculate_hash
# ---------------------------------------------------------------------------

class TestCalculateHash:
    def test_same_dict_same_hash(self):
        d = {"a": 1, "b": 2}
        assert calculate_hash(d) == calculate_hash(d)

    def test_key_order_independent(self):
        assert calculate_hash({"a": 1, "b": 2}) == calculate_hash({"b": 2, "a": 1})

    def test_different_dicts_different_hash(self):
        assert calculate_hash({"a": 1}) != calculate_hash({"a": 2})

    def test_non_dict_uses_str(self):
        assert calculate_hash("hello") == calculate_hash("hello")

    def test_returns_string(self):
        assert isinstance(calculate_hash({}), str)

    def test_hash_is_32_chars_md5(self):
        assert len(calculate_hash({"x": 1})) == 32


# ---------------------------------------------------------------------------
# format_currency
# ---------------------------------------------------------------------------

class TestFormatCurrency:
    def test_positive(self):
        assert format_currency(1234.5) == "$1,234.50"

    def test_zero(self):
        assert format_currency(0) == "$0.00"

    def test_negative(self):
        assert format_currency(-99.9) == "$-99.90"

    def test_large_number_has_commas(self):
        assert "," in format_currency(1_000_000)


# ---------------------------------------------------------------------------
# format_percentage
# ---------------------------------------------------------------------------

class TestFormatPercentage:
    def test_two_decimal_places(self):
        assert format_percentage(12.3456) == "12.35%"

    def test_zero(self):
        assert format_percentage(0) == "0.00%"

    def test_negative(self):
        assert format_percentage(-5.5) == "-5.50%"


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------

class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 4) == pytest.approx(2.5)

    def test_zero_denominator_returns_default(self):
        assert safe_divide(10, 0) == 0

    def test_zero_denominator_custom_default(self):
        assert safe_divide(10, 0, default=-1) == -1

    def test_zero_numerator(self):
        assert safe_divide(0, 5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calculate_compound_return
# ---------------------------------------------------------------------------

class TestCalculateCompoundReturn:
    def test_doubled_in_one_period(self):
        # 2x in 1 period → 100% return
        assert calculate_compound_return(100, 200, 1) == pytest.approx(100.0)

    def test_flat_no_growth(self):
        assert calculate_compound_return(100, 100, 5) == pytest.approx(0.0)

    def test_zero_initial_returns_zero(self):
        assert calculate_compound_return(0, 200, 5) == 0

    def test_zero_periods_returns_zero(self):
        assert calculate_compound_return(100, 200, 0) == 0

    def test_negative_initial_returns_zero(self):
        assert calculate_compound_return(-50, 200, 5) == 0

    def test_known_cagr(self):
        # $100 → $121 in 2 years = 10% CAGR
        result = calculate_compound_return(100, 121, 2)
        assert result == pytest.approx(10.0, rel=1e-4)


# ---------------------------------------------------------------------------
# is_market_hours
# ---------------------------------------------------------------------------

def _make_config(premarket_start="04:00", after_hours_end="20:00"):
    config = MagicMock()
    config.session.PREMARKET_START_ET = premarket_start
    config.session.AFTER_HOURS_END_ET = after_hours_end
    return config


def _et(year, month, day, hour, minute):
    eastern = pytz.timezone("US/Eastern")
    return eastern.localize(datetime(year, month, day, hour, minute))


class TestIsMarketHours:
    def test_weekday_within_session(self):
        # Tuesday at 10:00 ET
        ts = _et(2024, 1, 2, 10, 0)
        assert is_market_hours(ts, _make_config()) is True

    def test_weekday_before_premarket(self):
        # Tuesday at 03:59 ET — before premarket starts at 04:00
        ts = _et(2024, 1, 2, 3, 59)
        assert is_market_hours(ts, _make_config()) is False

    def test_weekday_after_hours_end(self):
        # Tuesday at 20:01 ET — after after-hours ends at 20:00
        ts = _et(2024, 1, 2, 20, 1)
        assert is_market_hours(ts, _make_config()) is False

    def test_saturday_rejected(self):
        ts = _et(2024, 1, 6, 10, 0)  # Saturday
        assert is_market_hours(ts, _make_config()) is False

    def test_sunday_rejected(self):
        ts = _et(2024, 1, 7, 10, 0)  # Sunday
        assert is_market_hours(ts, _make_config()) is False

    def test_boundary_premarket_start_inclusive(self):
        ts = _et(2024, 1, 2, 4, 0)
        assert is_market_hours(ts, _make_config()) is True

    def test_boundary_after_hours_end_inclusive(self):
        ts = _et(2024, 1, 2, 20, 0)
        assert is_market_hours(ts, _make_config()) is True


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_round_trip(self, tmp_path):
        fp = str(tmp_path / "state.json")
        data = {"key": "value", "count": 42, "flag": True}
        assert save_state(data, fp) is True
        loaded = load_state(fp)
        assert loaded == data

    def test_saved_file_is_valid_json(self, tmp_path):
        fp = str(tmp_path / "state.json")
        save_state({"x": 1}, fp)
        raw = open(fp).read()
        parsed = json.loads(raw)
        assert parsed == {"x": 1}

    def test_missing_file_returns_none(self, tmp_path):
        result = load_state(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path):
        fp = str(tmp_path / "corrupt.json")
        open(fp, "w").write("{not valid json")
        assert load_state(fp) is None

    def test_non_serialisable_returns_false(self, tmp_path):
        fp = str(tmp_path / "state.json")
        result = save_state({"bad": datetime.now()}, fp)
        assert result is False

    def test_nested_data_preserved(self, tmp_path):
        fp = str(tmp_path / "state.json")
        data = {"nested": {"a": [1, 2, 3], "b": None}}
        save_state(data, fp)
        assert load_state(fp) == data

    def test_overwrite_existing_file(self, tmp_path):
        fp = str(tmp_path / "state.json")
        save_state({"v": 1}, fp)
        save_state({"v": 2}, fp)
        assert load_state(fp) == {"v": 2}