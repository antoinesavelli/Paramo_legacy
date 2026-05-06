# =====================================================
# scripts/pattern_optimization/optimize_parameters.py - Parameter Optimization
# =====================================================

"""
Systematic parameter optimization for pattern recognition strategies.

This script performs grid search optimization across multiple parameters:
- PatternConfig: Parabolic thresholds (angle, acceleration, volume multiplier)
- PatternConfig: Step-up parameters (pullback, retention, min steps)
- RiskConfig: Stop loss, hold time, trailing stops
- ScreeningConfig: Gap percentage, price range

All results are saved to timestamped directories with detailed reports.

NEW: Supports parabolic pattern threshold testing with observed data ranges:
  - PARABOLIC_MIN_ANGLE: -0.67° to 1.92° (observed range)
  - PARABOLIC_MIN_ACCELERATION: -0.00123 to 0.000472 (observed range)
  - PARABOLIC_MIN_VOL_MULTIPLIER: 0.13x to 2.45x (observed range)

NEW: parabolic_9hour preset - 18 combinations optimized for 9-hour runtime
  (18 runs × 30 min/year = 540 minutes = 9 hours)
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import json
from itertools import product
from tqdm import tqdm
import traceback

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def generate_parameter_grid(param_definitions: Dict[str, Dict[str, List]]) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from definitions.
    
    Args:
        param_definitions: Dict mapping config sections to {param_name: [values]}
        
    Returns:
        List of parameter combination dictionaries
    """
    # Flatten all parameters with their section prefixes
    all_params = {}
    for section, params in param_definitions.items():
        for param_name, values in params.items():
            key = f"{section}.{param_name}"
            all_params[key] = values
    
    # Generate cartesian product of all parameter values
    param_names = list(all_params.keys())
    param_values = list(all_params.values())
    
    combinations = []
    for combo in product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)
    
    return combinations


def run_single_backtest(
    config_overrides: Dict[str, Any],
    start_date: str,
    end_date: str,
    capital: float,
    data_dir: str,
    reports_base_dir: Path,
    run_id: int,
    total_runs: int,
    silent: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run a single backtest with specific parameter overrides.
    
    Returns:
        Tuple of (results, parameters)
    """
    from config.loader import build_config
    from data_handler.local import LocalDataHandler
    from strategy.pattern_analyzer import PatternAnalyzer
    from backtester.core import Backtester
    from utils.reporting import compute_statistics
    
    # Build config with overrides
    override_strings = [f"{k}={v}" for k, v in config_overrides.items()]
    config = build_config(cli_overrides=override_strings, export_path=None)
    
    # Create run directory
    run_dir = reports_base_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Export config for this run
    config_path = run_dir / "config.json"
    build_config(cli_overrides=override_strings, export_path=str(config_path))
    
    # Initialize components
    local_data_handler = LocalDataHandler(config, data_dir=data_dir)
    pattern_analyzer = PatternAnalyzer(config, local_data_handler)
    
    # Run backtest
    bt = Backtester(
        config, 
        local_data_handler, 
        pattern_analyzer=pattern_analyzer,
        reports_dir=run_dir
    )
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    results = bt.run_backtest(start_dt, end_dt, initial_capital=capital)
    
    # Export results
    if results.get('trades'):
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(run_dir / "trades.csv", index=False)
    
    if results.get('statistics'):
        with open(run_dir / "statistics.json", 'w') as f:
            json.dump(results['statistics'], f, indent=2, default=str)
    
    return results, config_overrides


def main():
    """Main optimization entry point."""
    
    print("=" * 80)
    print("PARAMETER OPTIMIZATION TOOL")
    print("=" * 80)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Systematic parameter optimization for pattern strategies"
    )
    parser.add_argument("--start", default="2024-01-03", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital")
    parser.add_argument("--data-dir", default=r"D:\trading_data", help="Data directory")
    parser.add_argument("--reports-dir", default=r"D:\trading_data\reports", help="Reports directory")
    parser.add_argument("--preset", 
                       choices=['quick', 'standard', 'extensive', 
                               'parabolic_quick', 'parabolic_9hour', 'parabolic_full',
                               'confluence_9hour',  # ⭐ ADD THIS
                               'custom'], 
                       default='standard', 
                       help="Optimization preset")
    parser.add_argument("--custom-params", help="Custom parameters JSON file")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel runs (future)")
    
    args = parser.parse_args()
    
    # Define parameter grids/combinations based on preset
    param_combinations = None  # Will be set by preset
    param_definitions = None   # Will be set by grid-based presets
    
    if args.preset == 'quick':
        param_definitions = {
            'pattern': {
                'MAX_PULLBACK_PERCENT': [30.0, 40.0, 50.0],
                'MIN_ADVANCE_RETENTION': [30.0, 40.0],
            }
        }
    elif args.preset == 'standard':
        param_definitions = {
            'pattern': {
                'MAX_PULLBACK_PERCENT': [30.0, 40.0, 50.0, 60.0],
                'MIN_ADVANCE_RETENTION': [25.0, 30.0, 35.0, 40.0],
                'MIN_STEP_UPS': [1, 2, 3],
            },
            'risk': {
                'STOP_LOSS_PERCENT_OF_ACCOUNT': [3.0, 4.0, 5.0],
                'MAX_HOLD_TIME_MINUTES': [20, 30, 45],
            }
        }
    elif args.preset == 'extensive':
        param_definitions = {
            'pattern': {
                'MAX_PULLBACK_PERCENT': [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
                'MIN_ADVANCE_RETENTION': [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
                'MIN_STEP_UPS': [1, 2, 3],
            },
            'risk': {
                'STOP_LOSS_PERCENT_OF_ACCOUNT': [2.0, 3.0, 4.0, 5.0, 6.0],
                'MAX_HOLD_TIME_MINUTES': [15, 20, 30, 45, 60],
            },
            'screening': {
                'MIN_GAP_PERCENT': [40.0, 50.0, 60.0],
            }
        }
    elif args.preset == 'parabolic_quick':
        param_definitions = {
            'pattern': {
                'PARABOLIC_MIN_ANGLE': [-0.42, 0.0, 1.92],
                'PARABOLIC_MIN_ACCELERATION': [-0.00123, 0.0, 0.000472],
                'PARABOLIC_MIN_VOL_MULTIPLIER': [0.13, 1.0, 1.2],
            }
        }
    elif args.preset == 'parabolic_9hour':
        param_definitions = {
            'pattern': {
                'PARABOLIC_MIN_ANGLE': [-0.67, -0.30, 0.0, 0.5, 1.0, 1.92],
                'PARABOLIC_MIN_VOL_MULTIPLIER': [0.13, 1.0, 1.2]
            }
        }
        print("\n⏱️  [9-HOUR PARABOLIC OPTIMIZATION]")
        print("=" * 80)
        print(f"Combinations: 6 angles × 3 volume multipliers = 18 runs")
        print(f"Estimated runtime: 18 × 30 min/year = 540 minutes (9.0 hours)")
        print(f"\nAngle range: -0.67° to 1.92° (full observed range)")
        print(f"Volume range: 0.13x to 1.2x (permissive to restrictive)")
        print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        estimated_end = datetime.now().timestamp() + (18 * 30 * 60)
        print(f"Estimated completion: {datetime.fromtimestamp(estimated_end).strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    elif args.preset == 'parabolic_full':
        param_definitions = {
            'pattern': {
                'PARABOLIC_MIN_ANGLE': [-0.67, -0.42, -0.16, 0.0, 0.5, 1.0, 1.5, 1.92],
                'PARABOLIC_MIN_ACCELERATION': [-0.00123, -0.0005, 0.0, 0.0002, 0.000472, 0.001],
                'PARABOLIC_MIN_VOL_MULTIPLIER': [0.13, 0.50, 0.75, 1.0, 1.09, 1.2, 1.5],
                'PARABOLIC_MAX_ANGLE': [3.0, 5.0, 10.0],
            }
        }
    elif args.preset == 'confluence_9hour':
        # NOTE: Direct combinations (not a grid)
        param_combinations = [
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.5, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.5, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.5, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.5, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.0, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.5, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.5, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.6, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.4, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.4, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.6, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.6, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.4, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.4, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.6, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.33, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.33, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.34, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.4, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.3, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.3, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.3, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.4, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.3, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.3, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.3, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.4, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.5, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.25, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.25, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.25, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.5, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.25, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
            {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.25, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.25, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.5, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
        ]
        
        print("\n📊 CONFLUENCE PATTERN COMBINATION OPTIMIZATION")
        print("=" * 80)
        print(f"Testing 14 pattern COMBINATIONS (no pure single patterns)")
        print(f"Estimated runtime: 14 × 30 min/year = 420 minutes (7.0 hours)")
        print(f"\nTwo-Pattern Combinations (50/50):")
        print("  1. Step-Up + Parabolic (50/50)")
        print("  2. Step-Up + Volume (50/50)")
        print("  3. Parabolic + Volume (50/50)")
        print(f"\nTwo-Pattern Combinations (weighted):")
        print("  4. Step-Up + Volume (60/40)")
        print("  5. Step-Up + Volume (40/60)")
        print("  6. Step-Up + Parabolic (60/40)")
        print("  7. Step-Up + Parabolic (40/60)")
        print(f"\nThree-Pattern Combinations (equal & weighted):")
        print("  8-14. Various 3-pattern combinations")
        print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        estimated_end = datetime.now().timestamp() + (14 * 30 * 60)
        print(f"Estimated completion: {datetime.fromtimestamp(estimated_end).strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    elif args.preset == 'custom':
        with open(args.custom_params, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            param_combinations = data
            print(f"✓ Loaded {len(param_combinations)} explicit combinations")
        else:
            param_definitions = data
    
    # Generate combinations if using grid-based preset
    if param_combinations is None:
        print(f"\n[1/5] Generating parameter combinations (preset: {args.preset})...")
        param_combinations = generate_parameter_grid(param_definitions)
        total_runs = len(param_combinations)
        print(f"✓ Generated {total_runs} parameter combinations")
        
        print("\nParameter Ranges:")
        for section, params in param_definitions.items():
            print(f"  {section}:")
            for param_name, values in params.items():
                print(f"    {param_name}: {values}")
    else:
        print(f"\n[1/5] Using explicit parameter combinations...")
        total_runs = len(param_combinations)
        print(f"✓ {total_runs} combinations ready")
    
    # Create optimization directory
    print("\n[2/5] Setting up optimization directory...")
    reports_base = Path(args.reports_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_dir = reports_base / f"optimization_{args.preset}_{timestamp}"
    opt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameter definitions
    with open(opt_dir / "parameter_definitions.json", 'w') as f:
        json.dump(param_definitions, f, indent=2)
    
    print(f"✓ Optimization directory: {opt_dir.absolute()}")
    
    # Run optimizations
    print(f"\n[3/5] Running {total_runs} backtests...")
    print("=" * 80)
    results_summary = []
    successful_runs = 0
    failed_runs = 0
    
    # Progress bar for all runs
    with tqdm(total=total_runs, desc="Optimization Progress", unit="run") as pbar:
        for run_id, params in enumerate(param_combinations, start=1):
            try:
                # Format parameters for display
                params_str = ", ".join([f"{k.split('.')[-1]}={v}" for k, v in list(params.items())[:3]])
                pbar.set_postfix_str(f"{params_str}...")
                
                # Run backtest
                results, config_params = run_single_backtest(
                    config_overrides=params,
                    start_date=args.start,
                    end_date=args.end,
                    capital=args.capital,
                    data_dir=args.data_dir,
                    reports_base_dir=opt_dir,
                    run_id=run_id,
                    total_runs=total_runs,
                    silent=True
                )
                
                # Extract key metrics
                stats = results.get('statistics', {})
                summary = {
                    'run_id': run_id,
                    **config_params,
                    'total_trades': stats.get('total_trades', 0),
                    'win_rate': stats.get('win_rate', 0),
                    'net_profit': stats.get('net_profit', 0),
                    'total_return': stats.get('total_return', 0),
                    'profit_factor': stats.get('profit_factor', 0),
                    'max_drawdown': stats.get('max_drawdown', 0),
                    'sharpe_ratio': stats.get('sharpe_ratio', 0),
                    'avg_win': stats.get('avg_win', 0),
                    'avg_loss': stats.get('avg_loss', 0),
                    'expectancy': stats.get('expectancy', 0),
                    'avg_hold_time_minutes': stats.get('avg_hold_time_minutes', 0),
                }
                results_summary.append(summary)
                successful_runs += 1
                
            except Exception as e:
                failed_runs += 1
                error_summary = {
                    'run_id': run_id,
                    **params,
                    'error': str(e),
                    'status': 'FAILED'
                }
                results_summary.append(error_summary)
                
                # Log error details
                error_log = opt_dir / f"run_{run_id:04d}_error.txt"
                with open(error_log, 'w', encoding='utf-8') as f:
                    f.write(f"Run {run_id} failed with error:\n")
                    f.write(str(e))
                    f.write("\n\nTraceback:\n")
                    f.write(traceback.format_exc())
            
            finally:
                pbar.update(1)
    
    print("\n✓ All backtests completed")
    print(f"  Successful: {successful_runs}/{total_runs}")
    print(f"  Failed: {failed_runs}/{total_runs}")
    
    # Generate comparison reports
    print("\n[4/5] Generating comparison reports...")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_summary)
    
    # Save full results
    results_csv = opt_dir / "optimization_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Full results: {results_csv.name}")
    
    # Filter successful runs only
    success_df = results_df[~results_df.get('error', pd.Series([None] * len(results_df))).notna()]
    
    if not success_df.empty:
        # Generate rankings by different metrics
        rankings = {}
        
        metrics_to_rank = [
            ('total_return', False),  # False = higher is better
            ('net_profit', False),
            ('profit_factor', False),
            ('win_rate', False),
            ('sharpe_ratio', False),
            ('max_drawdown', True),  # True = lower is better
            ('expectancy', False),
        ]
        
        for metric, ascending in metrics_to_rank:
            if metric in success_df.columns:
                ranked = success_df.sort_values(metric, ascending=ascending).head(10)
                rankings[metric] = ranked
                
                # Save top 10 for this metric
                metric_file = opt_dir / f"top_10_{metric}.csv"
                ranked.to_csv(metric_file, index=False)
        
        # Generate summary report
        summary_report = []
        summary_report.append("=" * 80)
        summary_report.append("PARAMETER OPTIMIZATION SUMMARY")
        summary_report.append("=" * 80)
        summary_report.append(f"Preset: {args.preset}")
        summary_report.append(f"Total Runs: {total_runs}")
        summary_report.append(f"Successful: {successful_runs}")
        summary_report.append(f"Failed: {failed_runs}")
        summary_report.append(f"Date Range: {args.start} to {args.end}")
        summary_report.append(f"Initial Capital: ${args.capital:,.2f}")
        
        if args.preset == 'parabolic_9hour':
            summary_report.append(f"\n[9-Hour Optimization Completed]")
            summary_report.append(f"Actual runtime: ~{(datetime.now().timestamp() - datetime.strptime(args.start, '%Y-%m-%d').timestamp()) / 60:.0f} minutes")
        
        summary_report.append("")
        
        # Best performers by metric
        for metric, ascending in metrics_to_rank:
            if metric not in success_df.columns:
                continue
            
            best = success_df.sort_values(metric, ascending=ascending).iloc[0]
            direction = "Lowest" if ascending else "Highest"
            summary_report.append(f"{direction} {metric}: {best[metric]:.2f}")
            summary_report.append(f"  Run ID: {int(best['run_id'])}")
            
            # Show key parameters
            param_cols = [col for col in best.index if '.' in col]
            for param in param_cols[:5]:  # Show first 5 parameters
                summary_report.append(f"  {param}: {best[param]}")
            summary_report.append("")
        
        # Statistical summary
        summary_report.append("Statistical Summary (all successful runs):")
        for col in ['total_return', 'win_rate', 'profit_factor', 'max_drawdown', 'sharpe_ratio']:
            if col in success_df.columns:
                summary_report.append(f"  {col}:")
                summary_report.append(f"    Mean: {success_df[col].mean():.2f}")
                summary_report.append(f"    Median: {success_df[col].median():.2f}")
                summary_report.append(f"    Std: {success_df[col].std():.2f}")
                summary_report.append(f"    Min: {success_df[col].min():.2f}")
                summary_report.append(f"    Max: {success_df[col].max():.2f}")
        
        summary_report.append("=" * 80)
        
        # Save and print summary (with UTF-8 encoding to handle special characters)
        summary_text = "\n".join(summary_report)
        summary_file = opt_dir / "optimization_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print("\n" + summary_text)
        print(f"\n✓ Summary report: {summary_file.name}")
    
    # Final output
    print("\n[5/5] Optimization complete!")
    print("=" * 80)
    print(f"All results saved to: {opt_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  • optimization_results.csv (all {total_runs} runs)")
    print(f"  • optimization_summary.txt (best performers)")
    print(f"  • parameter_definitions.json (search space)")
    print(f"  • top_10_*.csv (rankings by metric)")
    print(f"  • run_XXXX/ directories (individual backtest results)")
    
    if args.preset == 'parabolic_9hour':
        print("\n[Next steps]:")
        print("  1. Review optimization_summary.txt for best parameters")
        print("  2. Run analyzer for detailed insights:")
        print(f"     python scripts/pattern_optimization/analyze_optimization.py \\")
        print(f"       \"{opt_dir / 'optimization_results.csv'}\"")
        print("  3. Update config/config.py with optimal parameter values")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nERROR: Optimization failed!")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
