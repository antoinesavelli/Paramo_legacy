# =====================================================
# reporting.py - Shared Performance Reporting
# =====================================================

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd


def _max_drawdown(equity: List[float]) -> Dict[str, float]:
    if not equity:
        return {"max_drawdown": 0.0, "max_dd_duration": 0}
    peak = equity[0]
    max_dd = 0.0
    max_dd_duration = 0
    current_dd_start = 0
    for i, val in enumerate(equity):
        if val > peak:
            peak = val
            current_dd_start = i
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - current_dd_start
    return {"max_drawdown": round(max_dd, 2), "max_dd_duration": max_dd_duration}


def _sharpe_ratio(daily_returns: Optional[List[float]]) -> float:
    if not daily_returns:
        return 0.0
    r = np.array(daily_returns, dtype=float)
    mu = r.mean()
    sd = r.std()
    return float(np.sqrt(252) * mu / sd) if sd > 0 else 0.0


def _sortino_ratio(daily_returns: Optional[List[float]]) -> float:
    if not daily_returns:
        return 0.0
    r = np.array(daily_returns, dtype=float)
    downside = r[r < 0]
    dd = downside.std() if downside.size > 0 else 0.0
    mu = r.mean()
    return float(np.sqrt(252) * mu / dd) if dd > 0 else 0.0


def _streaks(trades: List[Dict]) -> Tuple[int, int]:
    if not trades:
        return 0, 0
    # Sort by exit date/time to preserve sequence
    sorted_trades = sorted(
        trades,
        key=lambda t: t.get('exit_date') or t.get('exit_time') or datetime.min
    )
    max_win, max_lose = 0, 0
    cur = 0
    for t in sorted_trades:
        pnl = t.get('pnl', 0)
        if pnl > 0:
            cur = cur + 1 if cur >= 0 else 1
        else:
            cur = cur - 1 if cur <= 0 else -1
        max_win = max(max_win, cur)
        max_lose = min(max_lose, cur)
    return max_win, abs(max_lose)


def _parse_dt(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        # Try common formats
        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        for f in fmts:
            try:
                return datetime.strptime(v, f)
            except Exception:
                pass
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            pass
    return None


def _duration_minutes(trade: Dict) -> float:
    try:
        start = trade.get('entry_date') or trade.get('entry_time')
        end = trade.get('exit_date') or trade.get('exit_time')
        start_dt = _parse_dt(start)
        end_dt = _parse_dt(end)
        if not start_dt or not end_dt:
            return 0.0
        return (end_dt - start_dt).total_seconds() / 60.0
    except Exception:
        return 0.0


def _by_exit_reason(trades: List[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for t in trades:
        reason = t.get('exit_reason', 'unknown') or 'unknown'
        d = out.setdefault(reason, {'count': 0, 'wins': 0, 'pnl_sum': 0.0})
        d['count'] += 1
        d['wins'] += 1 if t.get('pnl', 0) > 0 else 0
        d['pnl_sum'] += float(t.get('pnl', 0))
    # finalize with win_rate and avg_pnl
    for r, d in out.items():
        count = d['count'] or 1
        d['win_rate'] = round((d['wins'] / count) * 100.0, 2)
        d['avg_pnl'] = round(d['pnl_sum'] / count, 2)
        del d['wins']
        del d['pnl_sum']
    return out


def compute_statistics(
    trades: List[Dict],
    equity_curve: Optional[List[Dict]] = None,
    initial_capital: Optional[float] = None,
    final_capital: Optional[float] = None,
    daily_returns: Optional[List[float]] = None,
    trading_days: Optional[int] = None,
) -> Dict:
    # Trade stats
    total_trades = len(trades)
    wins = [t for t in trades if t.get('pnl', 0) > 0]
    losses = [t for t in trades if t.get('pnl', 0) <= 0]
    gp = sum(t.get('pnl', 0) for t in wins) if wins else 0.0
    gl = abs(sum(t.get('pnl', 0) for t in losses)) if losses else 0.0
    net = gp - gl
    avg_win = gp / len(wins) if wins else 0.0
    avg_loss = gl / len(losses) if losses else 0.0
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    profit_factor = (gp / gl) if gl > 0 else float('inf')
    # expectancy is the average outcome per trade
    expectancy = (win_rate / 100.0 * avg_win) - ((100.0 - win_rate) / 100.0 * avg_loss)

    # Return % distributions if available
    rets_pct = [float(t.get('return_pct', 0)) for t in trades if 'return_pct' in t]
    avg_ret_pct = float(np.mean(rets_pct)) if rets_pct else 0.0
    med_ret_pct = float(np.median(rets_pct)) if rets_pct else 0.0
    best_ret_pct = float(np.max(rets_pct)) if rets_pct else 0.0
    worst_ret_pct = float(np.min(rets_pct)) if rets_pct else 0.0

    # Equity stats
    eq_series = [e['equity'] for e in equity_curve] if equity_curve else []
    if initial_capital is None:
        initial_capital = eq_series[0] if eq_series else 0.0
    if final_capital is None:
        final_capital = eq_series[-1] if eq_series else initial_capital

    total_return = ((final_capital - initial_capital) / initial_capital * 100.0) if initial_capital else 0.0
    years = (trading_days or 0) / 252.0 if trading_days else 0.0
    annualized_return = (((final_capital / initial_capital) ** (1 / years) - 1) * 100.0) if initial_capital and years > 0 else 0.0
    dd = _max_drawdown(eq_series)
    sharpe = _sharpe_ratio(daily_returns)

    # Notable trades
    best_trade = max(trades, key=lambda t: t.get('pnl', float('-inf'))) if trades else None
    worst_trade = min(trades, key=lambda t: t.get('pnl', float('inf'))) if trades else None
    # Hold time stats
    hold_times = [_duration_minutes(t) for t in trades]
    avg_hold_time = round(np.mean(hold_times), 2) if hold_times else 0.0
    max_hold_time = round(np.max(hold_times), 2) if hold_times else 0.0
    min_hold_time = round(np.min(hold_times), 2) if hold_times else 0.0
    # Exit reasons breakdown
    exit_reasons = _by_exit_reason(trades)

    # NOTE: NEW: MAE/MFE statistics (risk efficiency metrics)
    mae_values = [t.get('mae', 0) for t in trades if 'mae' in t]
    mfe_values = [t.get('mfe', 0) for t in trades if 'mfe' in t]
    
    avg_mae = round(np.mean(mae_values), 2) if mae_values else 0.0
    avg_mfe = round(np.mean(mfe_values), 2) if mfe_values else 0.0
    efficiency = round((avg_mfe / avg_mae * 100), 2) if avg_mae > 0 else 0.0

    return {
        'total_trades': total_trades,
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': round(win_rate, 2),
        'gross_profit': round(gp, 2),
        'gross_loss': round(gl, 2),
        'net_profit': round(net, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
        'expectancy': round(expectancy, 2),
        'avg_return_pct': round(avg_ret_pct, 2),
        'median_return_pct': round(med_ret_pct, 2),
        'best_return_pct': round(best_ret_pct, 2),
        'worst_return_pct': round(worst_ret_pct, 2),
        'total_return': round(total_return, 2),
        'annualized_return': round(annualized_return, 2),
        'max_drawdown': dd['max_drawdown'],
        'max_drawdown_duration_days': dd['max_dd_duration'],
        'initial_capital': round(initial_capital or 0, 2),
        'final_capital': round(final_capital or 0, 2),
        'sharpe_ratio': round(sharpe, 3) if isinstance(sharpe, float) else 0.0,
        'best_trade': {
            'symbol': best_trade.get('symbol'),
            'pnl': round(best_trade.get('pnl', 0), 2),
            'return_pct': round(best_trade.get('return_pct', 0), 2)
        } if best_trade else None,
        'worst_trade': {
            'symbol': worst_trade.get('symbol'),
            'pnl': round(worst_trade.get('pnl', 0), 2),
            'return_pct': round(worst_trade.get('return_pct', 0), 2)
        } if worst_trade else None,
        'avg_hold_time_minutes': avg_hold_time,
        'max_hold_time_minutes': max_hold_time,
        'min_hold_time_minutes': min_hold_time,
        'exit_reasons': exit_reasons,
        'avg_mae': avg_mae,  # Average Max Adverse Excursion
        'avg_mfe': avg_mfe,  # Average Max Favorable Excursion
        'exit_efficiency': efficiency,  # % of favorable move captured
    }


def generate_daily_performance_csv(
    trades: List[Dict],
    equity_curve: List[Dict],
    initial_capital: float,
    output_path: str
) -> None:
    """
    Generate a daily performance CSV with account balance and trading statistics.
    
    Args:
        trades: List of trade dictionaries
        equity_curve: List of daily equity snapshots
        initial_capital: Starting capital
        output_path: Path to save the CSV file
    """
    if not equity_curve:
        return
    
    # Create DataFrame from equity curve
    daily_df = pd.DataFrame(equity_curve)
    
    # Ensure date column is datetime
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Initialize columns
    daily_df['initial_balance'] = 0.0
    daily_df['final_balance'] = daily_df['equity']
    daily_df['daily_pnl'] = 0.0
    daily_df['daily_return_pct'] = 0.0
    daily_df['num_trades'] = 0
    daily_df['num_winners'] = 0
    daily_df['num_losers'] = 0
    daily_df['gross_profit'] = 0.0
    daily_df['gross_loss'] = 0.0
    daily_df['win_rate'] = 0.0
    daily_df['largest_win'] = 0.0
    daily_df['largest_loss'] = 0.0
    daily_df['avg_win'] = 0.0
    daily_df['avg_loss'] = 0.0
    daily_df['cumulative_pnl'] = 0.0
    daily_df['drawdown_pct'] = 0.0
    daily_df['peak_balance'] = initial_capital
    
    # Calculate initial balance (previous day's final balance)
    daily_df['initial_balance'] = daily_df['final_balance'].shift(1).fillna(initial_capital)
    
    # Calculate daily P&L
    daily_df['daily_pnl'] = daily_df['final_balance'] - daily_df['initial_balance']
    daily_df['daily_return_pct'] = (daily_df['daily_pnl'] / daily_df['initial_balance'] * 100).round(2)
    
    # Calculate cumulative P&L
    daily_df['cumulative_pnl'] = daily_df['final_balance'] - initial_capital
    
    # Calculate drawdown
    daily_df['peak_balance'] = daily_df['final_balance'].cummax()
    daily_df['drawdown_pct'] = ((daily_df['peak_balance'] - daily_df['final_balance']) / daily_df['peak_balance'] * 100).round(2)
    
    # Aggregate trades by exit date
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Parse exit dates
        if 'exit_date' in trades_df.columns:
            trades_df['exit_date_parsed'] = pd.to_datetime(trades_df['exit_date'])
        elif 'exit_date_str' in trades_df.columns:
            trades_df['exit_date_parsed'] = pd.to_datetime(trades_df['exit_date_str'])
        else:
            trades_df['exit_date_parsed'] = pd.NaT
        
        # Filter out invalid dates
        trades_df = trades_df[trades_df['exit_date_parsed'].notna()]
        
        if not trades_df.empty:
            # Extract date only (remove time component)
            trades_df['exit_date_only'] = trades_df['exit_date_parsed'].dt.date
            
            # Group by exit date
            daily_trades = trades_df.groupby('exit_date_only').agg(
                num_trades=('pnl', 'count'),
                num_winners=('pnl', lambda x: (x > 0).sum()),
                num_losers=('pnl', lambda x: (x <= 0).sum()),
                gross_profit=('pnl', lambda x: x[x > 0].sum() if (x > 0).any() else 0.0),
                gross_loss=('pnl', lambda x: abs(x[x <= 0].sum()) if (x <= 0).any() else 0.0),
                largest_win=('pnl', lambda x: x.max() if len(x) > 0 else 0.0),
                largest_loss=('pnl', lambda x: x.min() if len(x) > 0 else 0.0),
            ).reset_index()
            
            # Calculate averages
            daily_trades['avg_win'] = daily_trades.apply(
                lambda row: row['gross_profit'] / row['num_winners'] if row['num_winners'] > 0 else 0.0,
                axis=1
            ).round(2)
            
            daily_trades['avg_loss'] = daily_trades.apply(
                lambda row: row['gross_loss'] / row['num_losers'] if row['num_losers'] > 0 else 0.0,
                axis=1
            ).round(2)
            
            # Calculate win rate
            daily_trades['win_rate'] = (
                daily_trades['num_winners'] / daily_trades['num_trades'] * 100
            ).round(2)
            
            # Convert date back to datetime for merging
            daily_trades['date'] = pd.to_datetime(daily_trades['exit_date_only'])
            daily_trades = daily_trades.drop('exit_date_only', axis=1)
            
            # Merge with daily_df
            daily_df = daily_df.merge(
                daily_trades,
                on='date',
                how='left',
                suffixes=('', '_trade')
            )
            
            # Update columns with trade data
            for col in ['num_trades', 'num_winners', 'num_losers', 'gross_profit', 
                       'gross_loss', 'win_rate', 'largest_win', 'largest_loss', 
                       'avg_win', 'avg_loss']:
                if f'{col}_trade' in daily_df.columns:
                    daily_df[col] = daily_df[f'{col}_trade'].fillna(0)
                    daily_df = daily_df.drop(f'{col}_trade', axis=1)
    
    # Round numeric columns
    numeric_cols = [
        'initial_balance', 'final_balance', 'daily_pnl', 'gross_profit', 
        'gross_loss', 'largest_win', 'largest_loss', 'avg_win', 'avg_loss',
        'cumulative_pnl', 'peak_balance'
    ]
    
    for col in numeric_cols:
        if col in daily_df.columns:
            daily_df[col] = daily_df[col].round(2)
    
    # Convert integer columns
    int_cols = ['num_trades', 'num_winners', 'num_losers']
    for col in int_cols:
        if col in daily_df.columns:
            daily_df[col] = daily_df[col].fillna(0).astype(int)
    
    # Format date as YYYY-MM-DD
    daily_df['date'] = daily_df['date'].dt.strftime('%Y-%m-%d')
    
    # Select and order final columns
    final_columns = [
        'date',
        'initial_balance',
        'final_balance',
        'daily_pnl',
        'daily_return_pct',
        'cumulative_pnl',
        'peak_balance',
        'drawdown_pct',
        'num_trades',
        'num_winners',
        'num_losers',
        'win_rate',
        'gross_profit',
        'gross_loss',
        'largest_win',
        'largest_loss',
        'avg_win',
        'avg_loss'
    ]
    
    # Keep only existing columns
    final_columns = [col for col in final_columns if col in daily_df.columns]
    daily_df = daily_df[final_columns]
    
    # Export to CSV
    daily_df.to_csv(output_path, index=False)


def generate_text_report(stats: Dict, title: str = "PERFORMANCE REPORT") -> str:
    lines = []
    lines.append(f"\n=== {title} ===")
    lines.append("Performance")
    lines.append(f"  Initial Capital: ${stats.get('initial_capital', 0):,.2f}")
    lines.append(f"  Final Capital:   ${stats.get('final_capital', 0):,.2f}")
    lines.append(f"  Total Return:    {stats.get('total_return', 0):.2f}%")
    lines.append(f"  Annualized:      {stats.get('annualized_return', 0):.2f}%")
    lines.append(f"  Max Drawdown:    {stats.get('max_drawdown', 0):.2f}%")
    lines.append("")
    lines.append("Trades")
    lines.append(f"  Total Trades:    {stats.get('total_trades', 0)}")
    lines.append(f"  Win Rate:        {stats.get('win_rate', 0):.2f}%")
    lines.append(f"  Profit Factor:   {stats.get('profit_factor', 0)}")
    lines.append(f"  Net Profit:      ${stats.get('net_profit', 0):,.2f}")
    lines.append(f"  Avg Win:         ${stats.get('avg_win', 0):,.2f}")
    lines.append(f"  Avg Loss:        ${stats.get('avg_loss', 0):,.2f}")
    # Hold time stats
    lines.append("")
    lines.append("Hold Times")
    lines.append(f"  Average:  {stats.get('avg_hold_time_minutes', 0):.1f} minutes")
    lines.append(f"  Maximum:  {stats.get('max_hold_time_minutes', 0):.1f} minutes")
    lines.append(f"  Minimum:  {stats.get('min_hold_time_minutes', 0):.1f} minutes")
    # Exit reasons breakdown
    exit_reasons = stats.get('exit_reasons', {})
    if exit_reasons:
        lines.append("")
        lines.append("Exit Reasons")
        for reason, data in sorted(exit_reasons.items(), key=lambda x: x[1]['count'], reverse=True):
            lines.append(f"  {reason}: {data['count']} trades ({data['win_rate']:.1f}% win rate, avg P&L ${data['avg_pnl']:.2f})")
    return "\n".join(lines)
