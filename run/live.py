# =====================================================
# run.live.py - Live Run Entry Point
# =====================================================

import signal
import sys
import logging
import argparse
from core.trading_system import TradingSystem
from utils.cli_common import add_logging_args, configure_logging
from config.loader import build_config


def build_parser():
    p = argparse.ArgumentParser(description="Run live trading system")
    add_logging_args(p)
    p.add_argument("--dry-run", action="store_true",
                   help="Initialize components but do not place real orders (requires TradeExecutor support).")
    p.add_argument("--override", action="append",
                   help="Config override key=value (repeatable). Example: --override risk.MAX_DAILY_LOSS_PERCENT=10")
    p.add_argument("--export-config", help="Write effective sanitized config to path", default="artifacts/live/effective_config.json")
    p.add_argument("--no-env-layer", action="store_true", help="Disable environment variable override layer.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level, log_file="run.log")
    logger = logging.getLogger("run_live")

    # Build config (includes validation)
    try:
        config = build_config(
            cli_overrides=args.override,
            enable_env_layer=not args.no_env_layer,
            export_path=args.export_config
        )
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 2
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return 2

    # Verify API keys are set
    if not config.api.ALPACA_API_KEY or not config.api.ALPACA_SECRET_KEY:
        logger.error("Missing required environment variables: ALPACA_API_KEY and/or ALPACA_SECRET_KEY")
        logger.error("Please set these environment variables before running the system.")
        return 2

    system = TradingSystem(config=config)

    if args.dry_run:
        if hasattr(system.trade_executor, "set_dry_run"):
            system.trade_executor.set_dry_run(True)
            logger.info("Enabled dry-run mode (no live orders).")
        else:
            logger.warning("Dry-run requested but TradeExecutor lacks set_dry_run().")

    stop = False
    def _sig_handler(sig, frame):
        nonlocal stop
        if not stop:
            stop = True
            logger.info("Signal received. Initiating shutdown...")
            system.shutdown()
        else:
            logger.warning("Force exiting.")
            sys.exit(1)
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        system.start()
    except Exception as e:
        logger.exception(f"Live system crashed: {e}")
        try:
            system.shutdown()
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())