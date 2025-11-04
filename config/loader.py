# =====================================================
# config_loader.py - Immutable configuration override loader
# =====================================================

import os
import json
from dataclasses import is_dataclass, replace, asdict
from typing import List, Tuple, Any
from pathlib import Path
from datetime import time, datetime
from config import TradingConfig

def parse_override_pairs(pairs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}', expected key=value")
        k, v = pair.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out

def _coerce(example: Any, raw: str):
    if isinstance(example, bool):
        return raw.lower() in ("1","true","yes","on")
    if isinstance(example, int):
        try:
            return int(raw)
        except:
            return example
    if isinstance(example, float):
        try:
            return float(raw)
        except:
            return example
    if isinstance(example, list):
        return [x.strip() for x in raw.split(",")]
    return raw

def _apply_path(obj, parts: List[str], value: str):
    if not parts:
        return obj
    attr = parts[0]
    if not hasattr(obj, attr):
        raise AttributeError(f"Unknown config attribute: {attr}")
    current = getattr(obj, attr)
    if len(parts) == 1:
        coerced = _coerce(current, value)
        return replace(obj, **{attr: coerced})
    if not is_dataclass(current):
        raise AttributeError(f"Cannot descend into non-dataclass attribute: {attr}")
    replaced_child = _apply_path(current, parts[1:], value)
    return replace(obj, **{attr: replaced_child})

def apply_overrides_immutable(config: TradingConfig, overrides: List[Tuple[str, str]]) -> TradingConfig:
    cfg = config
    for path, raw in overrides:
        cfg = _apply_path(cfg, path.split("."), raw)
    return cfg

def apply_env_layer(config: TradingConfig, prefix: str = "PARAMO__") -> TradingConfig:
    cfg = config
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        path_parts = k[len(prefix):].split("__")
        cfg = _apply_path(cfg, path_parts, v)
    return cfg

def validate_config(cfg: TradingConfig):
    errs = []
    if cfg.screening.MIN_PRICE >= cfg.screening.MAX_PRICE:
        errs.append("screening.MIN_PRICE must be < screening.MAX_PRICE")
    if cfg.risk.MAX_RISK_PER_TRADE <= 0:
        errs.append("risk.MAX_RISK_PER_TRADE must be > 0")
    if cfg.risk.MAX_RISK_PER_TRADE > cfg.risk.MAX_DAILY_LOSS:
        errs.append("risk.MAX_RISK_PER_TRADE cannot exceed risk.MAX_DAILY_LOSS")
    if cfg.system.SCAN_INTERVAL_SECONDS < 5:
        errs.append("system.SCAN_INTERVAL_SECONDS must be >= 5")
    if errs:
        raise ValueError("Config validation failed:\n - " + "\n - ".join(errs))

def _json_default(obj):
    if isinstance(obj, (time, datetime)):
        # Use ISO format (HH:MM:SS[.ffffff] for time; full ISO for datetime)
        return obj.isoformat()
    # Fallback: let json raise for unknown types (avoids silent errors)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def export_effective(cfg: TradingConfig, path: str):
    data = asdict(cfg)
    # Redact secrets
    if "api" in data:
        for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
            if key in data["api"]:
                data["api"][key] = "***REDACTED***"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")

def build_config(cli_overrides: List[str] = None,
                 enable_env_layer: bool = True,
                 export_path: str | None = None) -> TradingConfig:
    cfg = TradingConfig()
    if enable_env_layer:
        cfg = apply_env_layer(cfg)
    if cli_overrides:
        pairs = parse_override_pairs(cli_overrides)
        cfg = apply_overrides_immutable(cfg, pairs)
    validate_config(cfg)
    if export_path:
        export_effective(cfg, export_path)
    return cfg