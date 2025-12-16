
# =====================================================
# parameter_sweep.py - Multi-configuration backtest runner
# =====================================================

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.loader import build_config
from data_handler.local import LocalDataHandler
from news.backtest import NewsIntegrationBacktest
from core.pattern_analyzer import PatternAnalyzer
from core.backtester import Backtester
from utils.logging import get_logger

# Configure logging for sweep
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parameter_sweep.log'),
        logging.StreamHandler()
    ]
)
logger = get_logger(__name__, component="sweep")

# =====================================================
# Parameter Configurations (100% gapper focused)
# =====================================================

PARAMETER_CONFIGS = [
    # Config 1: Aggressive screening + loose patterns
    {
        "name": "aggressive_loose",
        "description": "High gap threshold, loose pattern requirements",
        "overrides": [
            "screening.MIN_GAP_PERCENT=75.0",
            "screening.MIN_RELATIVE_VOLUME=1.5",
            "screening.MIN_ABSOLUTE_VOLUME=150000",
            "pattern.CONFLUENCE_MIN_SCORE=40.0",
            "pattern.PARABOLIC_MIN_ANGLE=35.0",
            "pattern.MIN_STEP_UPS=1",
            "pattern.MIN_ADVANCE_RETENTION=50.0",
        ]
    },
    # Config 2: Moderate screening + moderate patterns
    {
        "name": "moderate_balanced",
        "description": "Balanced gap/volume thresholds with moderate patterns",
        "overrides": [
            "screening.MIN_GAP_PERCENT=60.0",
            "screening.MIN_RELATIVE_VOLUME=2.0",
            "screening.MIN_ABSOLUTE_VOLUME=200000",
            "pattern.CONFLUENCE_MIN_SCORE=50.0",
            "pattern.PARABOLIC_MIN_ANGLE=40.0",
            "pattern.MIN_STEP_UPS=2",
            "pattern.MIN_ADVANCE_RETENTION=55.0",
        ]
    },
    # Config 3: Conservative screening + strict patterns
    {
        "name": "conservative_strict",
        "description": "Very high gap threshold with strict pattern validation",
        "overrides": [
            "screening.MIN_GAP_PERCENT=100.0",
            "screening.MIN_RELATIVE_VOLUME=2.5",
            "screening.MIN_ABSOLUTE_VOLUME=250000",
            "pattern.CONFLUENCE_MIN_SCORE=60.0",
            "pattern.PARABOLIC_MIN_ANGLE=50.0",
            "pattern.MIN_STEP_UPS=2",
            "pattern.MIN_ADVANCE_RETENTION=65.0",
        ]
    },
    # Config 4: Ultra-aggressive (mega gappers only)
    {
        "name": "mega_gappers",
        "description": "150%+ gaps with minimal pattern requirements",
        "overrides": [
            "screening.MIN_GAP_PERCENT=150.0",
            "screening.MIN_RELATIVE_VOLUME=1.0",
            "screening.MIN_ABSOLUTE_VOLUME=100000",
            "pattern.CONFLUENCE_MIN_SCORE=35.0",
            "pattern.PARABOLIC_MIN_ANGLE=30.0",
            "pattern.MIN_STEP_UPS=1",
            "pattern.MIN_ADVANCE_RETENTION=40.0",
        ]
    },
    # Config 5: Volume-focused (lower gaps, high volume)
    {
        "name": "volume_driven",
        "description": "Lower gap threshold but very high volume requirements",
        "overrides": [
            "screening.MIN_GAP_PERCENT=50.0",
            "screening.MIN_RELATIVE_VOLUME=3.0",
            "screening.MIN_ABSOLUTE_VOLUME=300000",
            "pattern.CONFLUENCE_MIN_SCORE=55.0",
            "pattern.PARABOLIC_MIN_ANGLE=45.0",
            "pattern.MIN_STEP_UPS=2",
            "pattern.MIN_ADVANCE_RETENTION=60.0",
        ]
    },
    # Config 6: Price range focused
    {
        "name": "price_focused",
        "description": "Tighter price range with moderate thresholds",
        "overrides": [
            "screening.MIN_GAP_PERCENT=75.0",
            "screening.MIN_PRICE=3.00",
            "screening.MAX_PRICE=15.00",
            "screening.MIN_RELATIVE_VOLUME=2.0",
            "screening.MIN_ABSOLUTE_VOLUME=200000",
            "pattern.CONFLUENCE_MIN_SCORE=50.0",
            "pattern.PARABOLIC_MIN_ANGLE=45.0",
            "pattern.MIN_STEP_UPS=2",
        ]
    },
]

def run_single_backtest(config_spec: Dict[str, Any], 
                       start_date: str, 
                       end_date: str, 
                       initial_capital: float) -> Dict[str, Any]:
    """Run a single backtest with given configuration."""
    
    config_name = config_spec['name']
    logger.info("=" * 80)
    logger.info(f"STARTING CONFIGURATION: {config_name}")
    logger.info(f"Description: {config_spec['description']}")
    logger.info("=" * 80)
    
    try:
        # Build config with overrides
        config = build_config(
            cli_overrides=config_spec['overrides'],
            enable_env_layer=False,
            export_path=f"artifacts/sweep/{config_name}_config.json"
        )
        
        # Log key parameters
        logger.info(f"Gap threshold: {config.screening.MIN_GAP_PERCENT}%")
        logger.info(f"Relative volume: {config.screening.MIN_RELATIVE_VOLUME}x")
        logger.info(f"Absolute volume: {config.screening.MIN_ABSOLUTE_VOLUME:,}")
        logger.info(f"Pattern score threshold: {config.pattern.CONFLUENCE_MIN_SCORE}")
        logger.info(f"Parabolic angle: {config.pattern.PARABOLIC_MIN_ANGLE}°")
        
        # Initialize components
        data_handler = LocalDataHandler(config, data_dir=config.backtest.DATA_DIR)
        news_bt = NewsIntegrationBacktest(config, data_dir=config.backtest.NEWS_DATA_DIR)
        pattern_analyzer = PatternAnalyzer(config, data_handler)
        backtester = Backtester(config, data_handler, pattern_analyzer=pattern_analyzer, news_integration=news_bt)
        
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Run backtest
        results = backtester.run_backtest(start, end, initial_capital=initial_capital)
        
        # Extract key statistics
        stats = results.get('statistics', {})
        
        result_summary = {
            'config_name': config_name,
            'description': config_spec['description'],
            'success': True,
            'stats': {
                'total_trades': stats.get('total_trades', 0),
                'win_rate': stats.get('win_rate', 0.0),
                'total_return': stats.get('total_return', 0.0),
                'max_drawdown': stats.get('max_drawdown', 0.0),
                'profit_factor': stats.get('profit_factor', 0.0),
                'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
                'avg_win': stats.get('avg_win', 0.0),
                'avg_loss': stats.get('avg_loss', 0.0),
                'best_trade': stats.get('best_trade', {}),
                'worst_trade': stats.get('worst_trade', {}),
            },
            'overrides': config_spec['overrides'],
            'trades': results.get('trades', []),
        }
        
        logger.info(f"✓ {config_name} completed: {stats.get('total_trades', 0)} trades, "
                   f"{stats.get('win_rate', 0):.1f}% win rate, "
                   f"{stats.get('total_return', 0):+.2f}% return")
        
        return result_summary
        
    except Exception as e:
        logger.exception(f"✗ {config_name} FAILED: {e}")
        return {
            'config_name': config_name,
            'description': config_spec['description'],
            'success': False,
            'error': str(e),
            'overrides': config_spec['overrides'],
        }

def generate_comparison_report(results: List[Dict[str, Any]], output_path: str):
    """Generate comparative analysis of all configurations."""
    
    logger.info("=" * 80)
    logger.info("GENERATING COMPARISON REPORT")
    logger.info("=" * 80)
    
    # Create summary dataframe
    summary_data = []
    for r in results:
        if not r['success']:
            summary_data.append({
                'Config': r['config_name'],
                'Description': r['description'],
                'Status': 'FAILED',
                'Error': r.get('error', 'Unknown'),
            })
            continue
        
        stats = r['stats']
        summary_data.append({
            'Config': r['config_name'],
            'Description': r['description'][:50],
            'Trades': stats['total_trades'],
            'Win Rate %': f"{stats['win_rate']:.1f}",
            'Total Return %': f"{stats['total_return']:+.2f}",
            'Max DD %': f"{stats['max_drawdown']:.2f}",
            'Profit Factor': f"{stats['profit_factor']:.2f}",
            'Sharpe': f"{stats['sharpe_ratio']:.2f}",
            'Avg Win': f"${stats['avg_win']:.2f}",
            'Avg Loss': f"${stats['avg_loss']:.2f}",
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = Path(output_path).parent / "sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary CSV saved: {csv_path}")
    
    # Save full results to JSON
    json_path = Path(output_path).parent / "sweep_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Full results JSON saved: {json_path}")
    
    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER SWEEP RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("\n" + df.to_string(index=False))
    
    # Find best configurations
    successful = [r for r in results if r['success']]
    if successful:
        logger.info("\n" + "=" * 80)
        logger.info("TOP CONFIGURATIONS")
        logger.info("=" * 80)
        
        # Best by return
        best_return = max(successful, key=lambda x: x['stats']['total_return'])
        logger.info(f"\n🏆 Best Total Return: {best_return['config_name']}")
        logger.info(f"   Return: {best_return['stats']['total_return']:+.2f}%")
        logger.info(f"   Win Rate: {best_return['stats']['win_rate']:.1f}%")
        logger.info(f"   Trades: {best_return['stats']['total_trades']}")
        
        # Best by win rate
        best_wr = max(successful, key=lambda x: x['stats']['win_rate'])
        logger.info(f"\n🎯 Best Win Rate: {best_wr['config_name']}")
        logger.info(f"   Win Rate: {best_wr['stats']['win_rate']:.1f}%")
        logger.info(f"   Return: {best_wr['stats']['total_return']:+.2f}%")
        logger.info(f"   Trades: {best_wr['stats']['total_trades']}")
        
        # Best by sharpe
        best_sharpe = max(successful, key=lambda x: x['stats']['sharpe_ratio'])
        logger.info(f"\n📊 Best Risk-Adjusted (Sharpe): {best_sharpe['config_name']}")
        logger.info(f"   Sharpe: {best_sharpe['stats']['sharpe_ratio']:.2f}")
        logger.info(f"   Return: {best_sharpe['stats']['total_return']:+.2f}%")
        logger.info(f"   Max DD: {best_sharpe['stats']['max_drawdown']:.2f}%")

def main():
    """Main parameter sweep execution."""
    
    print("=" * 80)
    print("PARAMETER SWEEP FOR 100% GAPPER STRATEGY")
    print("=" * 80)
    
    # Configuration
    START_DATE = "2024-01-03"
    END_DATE = "2024-01-31"
    INITIAL_CAPITAL = 1000.0
    OUTPUT_DIR = Path("artifacts/sweep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"Number of configurations: {len(PARAMETER_CONFIGS)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Run all configurations
    results = []
    for i, config_spec in enumerate(PARAMETER_CONFIGS, 1):
        logger.info(f"\n[{i}/{len(PARAMETER_CONFIGS)}] Running configuration: {config_spec['name']}")
        result = run_single_backtest(config_spec, START_DATE, END_DATE, INITIAL_CAPITAL)
        results.append(result)
        
        # Brief pause between runs
        import time
        time.sleep(1)
    
    # Generate comparison report
    generate_comparison_report(results, str(OUTPUT_DIR / "sweep_results.json"))
    
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER SWEEP COMPLETE")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
