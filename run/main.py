# =====================================================
# run.main.py - Mode dispatcher
# =====================================================
"""
Single entry point for both live and backtest modes.
Toggle RUN_MODE in TradingConfig: 'backtest' | 'live'

    python run/main.py
"""

import sys
import logging
from config.loader import build_config


def main():
    import argparse
    from utils.cli_common import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(description="Paramo trading system")
    add_logging_args(parser)
    parser.add_argument("--override", action="append",
                        help="Config override key=value (repeatable). "
                             "Example: --override RUN_MODE=live")
    args = parser.parse_args()

    config = build_config(
        cli_overrides=args.override,
        enable_env_layer=True,
    )

    mode = getattr(config, 'RUN_MODE', None)
    if not mode:
        print("ERROR: RUN_MODE not set in TradingConfig. Add RUN_MODE = 'backtest' or 'live'.")
        sys.exit(1)
    mode = mode.strip().lower()
    logging.getLogger("run").info(f"RUN_MODE = {mode}")

    if mode == 'live':
        from run.live import main as run_live
        sys.exit(run_live())
    elif mode == 'backtest':
        from run.backtest import main as run_backtest
        sys.exit(run_backtest())
    else:
        print(f"Unknown RUN_MODE '{mode}'. Must be 'live' or 'backtest'.")
        sys.exit(1)


if __name__ == "__main__":
    main()