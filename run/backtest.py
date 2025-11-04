# =====================================================
# run.backtest.py - Backtest Run Entry Point
# =====================================================

import sys
import argparse
import logging
import traceback
from datetime import datetime, timedelta
import pandas as pd

def main():
    """Main entry point with comprehensive error handling."""
    
    print("="* 80)
    print("STARTING BACKTEST")
    print("=" * 80)
    
    try:
        # Step 1: Import CLI utilities
        print("\n[1/10] Importing CLI utilities...")
        from cli.common import add_logging_args, configure_logging
        print("✓ CLI utilities imported")
        
        # Step 2: Parse arguments
        print("\n[2/10] Parsing command-line arguments...")
        parser = argparse.ArgumentParser(description="Run intraday backtest")
        add_logging_args(parser)
        parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
        parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
        parser.add_argument("--capital", type=float, help="Override initial capital")
        parser.add_argument("--override", action="append",
                           help="Config override key=value (repeatable)")
        parser.add_argument("--no-storage-init", action="store_true",
                           help="Skip auto creation of storage layout")
        parser.add_argument("--export-config", default="artifacts/backtest/effective_config.json",
                           help="Write effective config to path")
        parser.add_argument("--no-env-layer", action="store_true", 
                           help="Disable environment variable override layer")
        
        args = parser.parse_args()
        print("✓ Arguments parsed")
        
        # Step 3: Configure logging
        print("\n[3/10] Configuring logging...")
        log_level = getattr(args, 'log_level', 'INFO')
        optimize = getattr(args, 'optimize_logging', 'minimal')
        configure_logging(log_level, log_file="backtest.log", optimize=optimize)
        logger = logging.getLogger("run_backtest")
        print("✓ Logging configured")
        
        # Step 4: Load configuration
        print("\n[4/10] Loading configuration...")
        from config.loader import build_config
        config = build_config(
            cli_overrides=args.override,
            enable_env_layer=not args.no_env_layer,
            export_path=args.export_config
        )
        print("✓ Configuration loaded")
        
        # Step 5: Parse dates
        print("\n[5/10] Parsing dates...")
        start_date = datetime.strptime(args.start or config.backtest.START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(args.end or config.backtest.END_DATE, "%Y-%m-%d")
        capital = args.capital if args.capital is not None else config.backtest.INITIAL_CAPITAL
        print(f"✓ Dates parsed: {start_date.date()} to {end_date.date()}")
        
        # Step 6: Log configuration
        print("\n[6/10] Logging backtest configuration...")
        logger.info("=" * 80)
        logger.info("BACKTEST INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Initial Capital: ${capital:,.2f}")
        logger.info(f"Data Directory: {config.backtest.DATA_DIR}")
        logger.info(f"Premarket Enabled: {config.backtest.SESSION.PREMARKET_ENABLED}")
        logger.info(f"Fast Mode: {config.backtest.FAST_MODE}")
        logger.info("=" * 80)
        print("✓ Configuration logged")
        
        # Step 7: Initialize data handler
        print("\n[7/10] Initializing data handler...")
        from data_handler.local import LocalDataHandler
        local_data_handler = LocalDataHandler(config, data_dir=config.backtest.DATA_DIR)
        
        total_days = (end_date - start_date).days + 1
        logger.info(f"Total days in range: {total_days}")
        logger.info(f"Indexed trading days: {len(local_data_handler._file_index)}")
        print(f"✓ Data handler initialized ({len(local_data_handler._file_index)} trading days indexed)")
        
        # Step 8: Initialize components
        print("\n[8/10] Initializing trading components...")
        from news.backtest import NewsIntegrationBacktest
        from core.pattern_analyzer import PatternAnalyzer
        from core.backtester import Backtester
        
        news_bt = NewsIntegrationBacktest(config, data_dir=config.backtest.NEWS_DATA_DIR)
        pattern_analyzer = PatternAnalyzer(config, local_data_handler)
        bt = Backtester(config, local_data_handler, pattern_analyzer=pattern_analyzer, news_integration=news_bt)
        print("✓ Components initialized")
        
        # Step 9: Run backtest
        print("\n[9/10] Running backtest...")
        logger.info("Starting backtest execution...")
        logger.info("=" * 80)
        
        results = bt.run_backtest(start_date, end_date, initial_capital=capital)
        print("✓ Backtest execution completed")
        
        # Step 10: Generate report
        print("\n[10/10] Generating report...")
        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        from utils.reporting import generate_text_report
        print("\n" + generate_text_report(results.get("statistics", {}), title="INTRADAY BACKTEST RESULTS"))
        print("✓ Report generated")
        
        print("\n" + "=" * 80)
        print("BACKTEST FINISHED SUCCESSFULLY")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n\n❌ ERROR: Backtest failed!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Also log if logger is available
        try:
            logger = logging.getLogger("run_backtest")
            logger.exception("Backtest failed with exception")
        except:
            pass
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nExiting with code: {exit_code}")
    sys.exit(exit_code)