# =====================================================
# run.backtester.py - Backtest Run Entry Point
# =====================================================

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger("run_backtest")


def main():
    """Main entry point with comprehensive error handling."""
    try:
        from utils.cli_common import add_logging_args, configure_logging

        parser = argparse.ArgumentParser(description="Run intraday backtest")
        add_logging_args(parser)
        parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
        parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
        parser.add_argument("--capital", type=float, help="Override initial capital")
        parser.add_argument("--override", action="append",
                           help="Config override key=value (repeatable)")
        parser.add_argument("--no-storage-init", action="store_true",
                           help="Skip auto creation of storage layout")
        parser.add_argument("--reports-dir", default=None,
                           help="Override reports directory (default: reports/)")
        parser.add_argument("--no-env-layer", action="store_true",
                           help="Disable environment variable override layer")

        args = parser.parse_args()

        from config.loader import build_config, export_effective

        # Build config once — parse, env layer, and validation run exactly once
        config = build_config(
            cli_overrides=args.override,
            enable_env_layer=not args.no_env_layer,
        )

        # Determine reports directory and timestamped run directory
        reports_dir = Path(args.reports_dir if args.reports_dir else config.system.REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = reports_dir / f"backtest_{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Export config before logging is configured so the file is always written
        config_export_path = run_dir / "effective_config.json"
        export_effective(config, str(config_export_path))

        # Configure logging — all output from this point routes through the logger
        log_level = getattr(args, 'log_level', 'INFO')
        optimize = getattr(args, 'optimize_logging', 'minimal')
        show_screener_warnings = getattr(args, 'show_screener_warnings', False)

        log_file = run_dir / "backtest_log.txt"
        configure_logging(
            log_level,
            log_file=str(log_file),
            optimize=optimize,
            show_screener_warnings=show_screener_warnings
        )

        logger.info("=" * 80)
        logger.info("BACKTEST INITIALIZATION")
        logger.info("=" * 80)
        logger.info("Run directory: %s", run_dir.absolute())
        logger.info("Log file: %s", log_file.absolute())
        if not show_screener_warnings:
            logger.info("Console: screener warnings suppressed (see log file for details)")

        # Parse dates and capital
        start_date = datetime.strptime(args.start or config.backtest.START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(args.end or config.backtest.END_DATE, "%Y-%m-%d")
        capital = args.capital if args.capital is not None else config.backtest.INITIAL_CAPITAL

        logger.info("Date range: %s to %s", start_date.date(), end_date.date())
        logger.info("Initial capital: $%s", f"{capital:,.2f}")
        logger.info("Data directory: %s", config.backtest.DATA_DIR)
        logger.info("Fast mode: %s", config.backtest.FAST_MODE)
        logger.info("=" * 80)

        # Initialize data handler
        from data_handler.local import LocalDataHandler
        local_data_handler = LocalDataHandler(config, data_dir=config.backtest.DATA_DIR)

        total_days = (end_date - start_date).days + 1
        logger.info("Total days in range: %d", total_days)
        logger.info("Indexed trading days: %d", len(local_data_handler._file_index))

        # Initialize trading components
        from strategy.pattern_analyzer import PatternAnalyzer
        from backtester.core import Backtester

        pattern_analyzer = PatternAnalyzer(config, local_data_handler)
        bt = Backtester(
            config,
            local_data_handler,
            pattern_analyzer=pattern_analyzer,
            reports_dir=run_dir
        )

        # Run backtest
        logger.info("Starting backtest execution...")
        results = bt.run_backtest(start_date, end_date, initial_capital=capital)
        logger.info("Backtest execution completed")

        # Generate and write reports
        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        from utils.reporting import generate_text_report

        text_report = generate_text_report(results.get("statistics", {}), title="INTRADAY BACKTEST RESULTS")
        report_file = run_dir / "backtest_results.txt"
        report_file.write_text(text_report, encoding="utf-8")
        logger.info("\n%s", text_report)

        if results.get('trades'):
            trades_df = pd.DataFrame(results['trades'])
            trades_file = run_dir / "trades.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info("Trades exported: %s", trades_file)

        if 'statistics' in results:
            stats_file = run_dir / "statistics.json"
            stats_file.write_text(
                json.dumps(results['statistics'], indent=2, default=str),
                encoding="utf-8"
            )
            logger.info("Statistics exported: %s", stats_file)

        logger.info("=" * 80)
        logger.info("All outputs saved to: %s", run_dir.absolute())
        logger.info("  %s", config_export_path.name)
        logger.info("  %s", log_file.name)
        logger.info("  %s", report_file.name)
        if results.get('trades'):
            logger.info("  trades.csv")
        if 'statistics' in results:
            logger.info("  statistics.json")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("Backtest interrupted by user")
        return 130

    except Exception:
        logger.exception("Backtest failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())