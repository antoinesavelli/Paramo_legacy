from pathlib import Path
import sys
# Ensure project root is on sys.path so top-level packages like `config` are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import pandas as pd
from config.config import TradingConfig
from data_handler.local import LocalDataHandler

cfg = TradingConfig()
dh = LocalDataHandler(cfg, data_dir=cfg.backtest.DATA_DIR)

day = pd.Timestamp("2024-01-03")
premarket = True

res = dh.calculate_gaps(day, premarket=premarket)
gaps = res.get("gaps", pd.DataFrame())

print("date:", day.date())
print("total merged rows (all symbols present in both days):", len(gaps))

if not gaps.empty:
    print("gap % stats: min={:.2f} max={:.2f} med={:.2f}".format(
        gaps['gap_percent'].min(), gaps['gap_percent'].max(), gaps['gap_percent'].median()))
    print("\nTop 20 gaps:")
    print(gaps[['symbol','open_price','prev_close','gap_percent','last_price']].sort_values('gap_percent', ascending=False).head(20).to_string(index=False))

# Apply screener filters exactly as code does
screen_cfg = cfg.screening
session_cfg = cfg.backtest.SESSION
min_gap = session_cfg.PREMARKET_MIN_GAP_PERCENT if premarket else screen_cfg.MIN_GAP_PERCENT
min_price = screen_cfg.MIN_PRICE
max_price = screen_cfg.MAX_PRICE

after_gap = gaps[gaps['gap_percent'].abs() >= min_gap]
print("\nrows after gap threshold ({:.1f}%):".format(min_gap), len(after_gap))

after_price = after_gap[(after_gap['last_price'] >= min_price) & (after_gap['last_price'] <= max_price)]
print("rows after price filter ({} <= price <= {}):".format(min_price, max_price), len(after_price))

over_price_count = len(after_gap[after_gap['last_price'] > max_price])
print("rows excluded by MAX_PRICE (>{}):".format(max_price), over_price_count)

# If you care about volume, sample volumes from the original daily file:
print("\nSample last_price distribution:")
print(gaps['last_price'].describe())
