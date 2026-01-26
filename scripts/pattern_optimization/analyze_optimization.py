# =====================================================
# scripts/pattern_optimization/analyze_optimization.py - Optimization Results Analyzer
# =====================================================

"""
Analyze and visualize parameter optimization results.

NEW: Enhanced analysis for parabolic pattern threshold testing including:
  - Threshold impact analysis
  - Range effectiveness
  - Correlation with performance metrics
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_parabolic_thresholds(success_df: pd.DataFrame) -> str:
    """
    Specialized analysis for parabolic pattern threshold optimization.
    
    Args:
        success_df: DataFrame with successful optimization runs
        
    Returns:
        Formatted analysis report
    """
    parabolic_cols = [col for col in success_df.columns if 'PARABOLIC' in col]
    
    if not parabolic_cols:
        return ""
    
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("PARABOLIC THRESHOLD ANALYSIS")
    lines.append("=" * 80)
    
    # Analyze each parabolic parameter
    for param in parabolic_cols:
        lines.append(f"\n### {param}")
        
        # Group by parameter value
        grouped = success_df.groupby(param).agg({
            'total_return': ['mean', 'std', 'count', 'min', 'max'],
            'win_rate': 'mean',
            'total_trades': 'mean',
        })
        
        # Find optimal value
        best_value = grouped['total_return']['mean'].idxmax()
        best_return = grouped['total_return']['mean'].max()
        best_count = grouped.loc[best_value, ('total_return', 'count')]
        
        lines.append(f"\n  Best Value: {best_value}")
        lines.append(f"    Avg Return: {best_return:.2f}%")
        lines.append(f"    Based on: {best_count} runs")
        
        # Show full range performance
        lines.append(f"\n  Performance by Value:")
        for value in sorted(grouped.index):
            mean_ret = grouped.loc[value, ('total_return', 'mean')]
            std_ret = grouped.loc[value, ('total_return', 'std')]
            win_rate = grouped.loc[value, ('win_rate', 'mean')]
            trades = grouped.loc[value, ('total_trades', 'mean')]
            count = grouped.loc[value, ('total_return', 'count')]
            
            marker = " ← BEST" if value == best_value else ""
            lines.append(f"    {value:7.3f}: Return={mean_ret:6.2f}% (±{std_ret:.2f}), "
                        f"WinRate={win_rate:5.1f}%, Trades={trades:5.1f}, N={count}{marker}")
        
        # Correlation analysis
        if len(grouped) > 2:
            correlation = success_df[[param, 'total_return']].corr().iloc[0, 1]
            lines.append(f"\n  Correlation with Return: {correlation:.3f}")
            if abs(correlation) > 0.3:
                direction = "positive" if correlation > 0 else "negative"
                strength = "strong" if abs(correlation) > 0.5 else "moderate"
                lines.append(f"    → {strength.capitalize()} {direction} relationship")
    
    # Multi-parameter interaction analysis
    if 'pattern.PARABOLIC_MIN_ANGLE' in success_df.columns and 'pattern.PARABOLIC_MIN_VOL_MULTIPLIER' in success_df.columns:
        lines.append("\n### Angle × Volume Multiplier Interaction")
        
        pivot = success_df.pivot_table(
            values='total_return',
            index='pattern.PARABOLIC_MIN_ANGLE',
            columns='pattern.PARABOLIC_MIN_VOL_MULTIPLIER',
            aggfunc='mean'
        )
        
        lines.append("\n  Average Return (%) by Angle and Volume Multiplier:")
        lines.append(pivot.to_string())
        
        # Find best combination
        best_combo_idx = success_df['total_return'].idxmax()
        best_combo = success_df.loc[best_combo_idx]
        lines.append(f"\n  Best Combination:")
        lines.append(f"    Angle: {best_combo.get('pattern.PARABOLIC_MIN_ANGLE', 'N/A')}")
        lines.append(f"    Acceleration: {best_combo.get('pattern.PARABOLIC_MIN_ACCELERATION', 'N/A')}")
        lines.append(f"    Vol Mult: {best_combo.get('pattern.PARABOLIC_MIN_VOL_MULTIPLIER', 'N/A')}")
        lines.append(f"    Return: {best_combo['total_return']:.2f}%")
    
    return "\n".join(lines)


def analyze_results(results_csv: str):
    """Analyze optimization results and generate insights."""    
    print("=" * 80)
    print("OPTIMIZATION RESULTS ANALYZER")
    print("=" * 80)
    
    # Load results
    df = pd.read_csv(results_csv)
    
    # Filter successful runs
    if 'error' in df.columns:
        success_df = df[df['error'].isna()].copy()
        failed_df = df[df['error'].notna()]
        print(f"\nTotal runs: {len(df)}")
        print(f"Successful: {len(success_df)}")
        print(f"Failed: {len(failed_df)}")
    else:
        success_df = df.copy()
        print(f"\nTotal runs: {len(df)}")
    
    if success_df.empty:
        print("\nNo successful runs to analyze!")
        return 1
    
    # Identify parameter columns
    param_cols = [col for col in success_df.columns if '.' in col]
    metric_cols = ['total_return', 'net_profit', 'win_rate', 'profit_factor', 
                   'max_drawdown', 'sharpe_ratio', 'expectancy', 'total_trades']
    
    print(f"\nParameter columns: {len(param_cols)}")
    print(f"  {', '.join(param_cols)}")
    print(f"Metric columns: {len([c for c in metric_cols if c in success_df.columns])}")
    
    # Check if this is a parabolic threshold optimization
    is_parabolic_optimization = any('PARABOLIC' in col for col in param_cols)
    
    # Parameter impact analysis
    print("\n" + "=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)
    
    for param in param_cols:
        print(f"\n{param}:")
        
        # Group by parameter value
        grouped = success_df.groupby(param).agg({
            'total_return': ['mean', 'std', 'count'],
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'max_drawdown': 'mean',
        })
        
        print(grouped.to_string())
        
        # Find best value for this parameter
        best_value = grouped['total_return']['mean'].idxmax()
        best_return = grouped['total_return']['mean'].max()
        print(f"\n  → Best value: {best_value} (avg return: {best_return:.2f}%)")
    
    # Parabolic-specific analysis
    if is_parabolic_optimization:
        parabolic_analysis = analyze_parabolic_thresholds(success_df)
        print(parabolic_analysis)
    
    # Correlation analysis
    print("\n" + "=" * 80)
    print("PARAMETER CORRELATIONS WITH PERFORMANCE")
    print("=" * 80)
    
    for metric in ['total_return', 'win_rate', 'profit_factor']:
        if metric not in success_df.columns:
            continue
        
        print(f"\n{metric.upper()} correlations:")
        correlations = success_df[param_cols + [metric]].corr()[metric].drop(metric)
        correlations = correlations.sort_values(ascending=False)
        
        for param, corr in correlations.items():
            if abs(corr) > 0.1:  # Only show meaningful correlations
                direction = "↑" if corr > 0 else "↓"
                strength = ""
                if abs(corr) > 0.5:
                    strength = " (STRONG)"
                elif abs(corr) > 0.3:
                    strength = " (MODERATE)"
                print(f"  {direction} {param}: {corr:.3f}{strength}")
    
    # Top performers
    print("\n" + "=" * 80)
    print("TOP 5 PERFORMERS (by Total Return)")
    print("=" * 80)
    
    top_5 = success_df.nlargest(5, 'total_return')
    
    for idx, row in top_5.iterrows():
        print(f"\nRun {int(row['run_id'])}:")
        print(f"  Total Return: {row['total_return']:.2f}%")
        print(f"  Win Rate: {row['win_rate']:.2f}%")
        print(f"  Profit Factor: {row['profit_factor']:.2f}")
        print(f"  Max Drawdown: {row['max_drawdown']:.2f}%")
        print(f"  Total Trades: {int(row['total_trades'])}")
        print(f"  Parameters:")
        for param in param_cols:
            print(f"    {param}: {row[param]}")
    
    # Risk-adjusted performance
    if 'sharpe_ratio' in success_df.columns:
        print("\n" + "=" * 80)
        print("TOP 5 RISK-ADJUSTED PERFORMERS (by Sharpe Ratio)")
        print("=" * 80)
        
        top_sharpe = success_df.nlargest(5, 'sharpe_ratio')
        
        for idx, row in top_sharpe.iterrows():
            print(f"\nRun {int(row['run_id'])}:")
            print(f"  Sharpe Ratio: {row['sharpe_ratio']:.3f}")
            print(f"  Total Return: {row['total_return']:.2f}%")
            print(f"  Max Drawdown: {row['max_drawdown']:.2f}%")
            print(f"  Win Rate: {row['win_rate']:.2f}%")
    
    # Parameter stability analysis
    print("\n" + "=" * 80)
    print("PARAMETER STABILITY ANALYSIS")
    print("=" * 80)
    print("(How consistent is performance across different parameter values?)")
    
    for param in param_cols:
        grouped_std = success_df.groupby(param)['total_return'].agg(['mean', 'std', 'count'])
        avg_std = grouped_std['std'].mean()
        stability = "STABLE" if avg_std < 10 else "UNSTABLE" if avg_std > 20 else "MODERATE"
        
        print(f"\n{param}:")
        print(f"  Average std deviation: {avg_std:.2f}%")
        print(f"  Interpretation: {stability}")
        print(f"  Recommendation: {'Safe to use any value in range' if stability == 'STABLE' else 'Careful selection required' if stability == 'UNSTABLE' else 'Moderate sensitivity'}")
    
    # Summary recommendations
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    print("\nBest Parameter Values (by highest average return):")
    for param in param_cols:
        grouped = success_df.groupby(param)['total_return'].mean()
        best_value = grouped.idxmax()
        best_return = grouped.max()
        print(f"  {param}: {best_value} → {best_return:.2f}% avg return")
    
    if is_parabolic_optimization:
        print("\nParabolic Threshold Insights:")
        if 'pattern.PARABOLIC_MIN_ANGLE' in param_cols:
            angle_corr = success_df[['pattern.PARABOLIC_MIN_ANGLE', 'total_return']].corr().iloc[0, 1]
            if angle_corr > 0.3:
                print("  • Higher angles tend to perform better → Consider raising MIN_ANGLE")
            elif angle_corr < -0.3:
                print("  • Lower angles (including negative) perform better → Consider lowering MIN_ANGLE")
            else:
                print("  • Angle threshold shows weak correlation → Other factors more important")
    
    print("\n" + "=" * 80)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter optimization results"
    )
    parser.add_argument(
        'results_csv',
        help="Path to optimization_results.csv file"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"ERROR: File not found: {results_path}")
        return 1
    
    return analyze_results(args.results_csv)


if __name__ == "__main__":
    exit(main())
