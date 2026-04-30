"""
Clean up .parquet.backup files from trading data directories.
Run from repo root: python scripts/clean_parquet_backups.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from config.config import TradingConfig

cfg = TradingConfig()

SEARCH_DIRS = [
    Path(cfg.backtest.BASE_DATA_DIR),
    Path(cfg.backtest.DATA_DIR),
    Path(cfg.backtest.NEWS_DATA_DIR),
    Path(cfg.backtest.DAILY_AGGREGATES_DIR),
    Path(cfg.market_context.CSV_DIR),
]

def clean_backups():
    backup_files = []

    for directory in SEARCH_DIRS:
        if not directory.exists():
            print(f"Skipping (not found): {directory}")
            continue
        print(f"Scanning: {directory} ...", flush=True)
        found = list(directory.rglob("*.parquet.backup"))
        print(f"  -> {len(found)} backup(s) found", flush=True)
        backup_files.extend(found)

    total = len(backup_files)
    if total == 0:
        print("No .parquet.backup files found.")
        return

    print(f"\nFound {total} .parquet.backup file(s) total.")
    confirm = input("Delete all? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    failed = 0
    for f in backup_files:
        try:
            f.unlink()
        except Exception as e:
            print(f"  ERROR deleting {f}: {e}")
            failed += 1

    deleted = total - failed
    print(f"Done. Deleted {deleted}/{total} file(s)." + (f" {failed} failed." if failed else ""))

if __name__ == "__main__":
    clean_backups()