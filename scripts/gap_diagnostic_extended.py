from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
print("total merged rows (both days):", len(gaps))

session_cfg = cfg.backtest.SESSION
screen_cfg = cfg.screening
min_gap = session_cfg.PREMARKET_MIN_GAP_PERCENT if premarket else screen_cfg.MIN_GAP_PERCENT
min_price = screen_cfg.MIN_PRICE
max_price = screen_cfg.MAX_PRICE

after_gap = gaps[gaps['gap_percent'].abs() >= min_gap]
after_price = after_gap[(after_gap['last_price'] >= min_price) & (after_gap['last_price'] <= max_price)]

print("rows after gap threshold ({:.1f}%):".format(min_gap), len(after_gap))
print("rows after price filter ({} <= price <= {}):".format(min_price, max_price), len(after_price))
print("Sample top gaps (after filters):")
print(after_price.sort_values('gap_percent', ascending=False).head(20).to_string(index=False))

# Define session window used by screener (naive eastern times)
if premarket:
    start_dt = pd.Timestamp(day.year, day.month, day.day, 4, 0)
    end_dt = pd.Timestamp(day.year, day.month, day.day, 20, 0)
    warmup = session_cfg.PREMARKET_WARMUP_MINUTES
else:
    start_dt = pd.Timestamp(day.year, day.month, day.day, 9, 30)
    end_dt = pd.Timestamp(day.year, day.month, day.day, 20, 0)
    warmup = getattr(cfg.backtest, "WARMUP_MINUTES", 45)

print(f"\nInspecting up to 50 symbols that passed gap+price (start={start_dt}, end={end_dt}, warmup={warmup}):\n")

for _, row in after_price.sort_values('gap_percent', ascending=False).head(50).iterrows():
    sym = str(row['symbol'])
    lp = row['last_price']
    gp = row['gap_percent']
    # symbol day data (full day)
    day_df = dh.get_symbol_day_data(sym, day.strftime("%Y-%m-%d"))
    total_rows = len(day_df) if not day_df.empty else 0
    total_volume = int(day_df['volume'].sum()) if not day_df.empty and 'volume' in day_df.columns else None

    # intraday bars using same window as screener
    bars = dh.get_intraday_bars(sym, start=start_dt, end=end_dt)
    bars_count = len(bars)
    warmup_rows = len(bars.iloc[:warmup]) if bars_count > 0 else 0
    first_ts = bars['timestamp'].min() if bars_count > 0 else None
    last_ts = bars['timestamp'].max() if bars_count > 0 else None

    print(f"{sym:6s} | price={lp:7.2f} | gap={gp:+6.2f}% | day_rows={total_rows:4d} | day_vol={str(total_volume):>8s} | bars={bars_count:3d} | warmup={warmup_rows:2d} | ts_range={first_ts} -> {last_ts}")
    # optional: show first 3 bars for quick sanity
    if bars_count > 0:
        print(bars[['timestamp','open','high','low','close','volume']].head(3).to_string(index=False))
    print("")
