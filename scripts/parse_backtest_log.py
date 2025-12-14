import re
import csv
import json
from pathlib import Path
from statistics import mean
from typing import List, Dict, Optional

# Paths (adjust as needed)
log_path = Path("backtest.log")  # change to your log path
out_losers_path = Path("losing_trades.csv")
backtest_summary_path = Path("backtest_summary.csv")
reject_summary_path = Path("backtest_reject_summary.csv")
analysis_out_path = Path("backtest_analysis.json")

# Regex to capture EXIT lines in existing logs
exit_re = re.compile(
    r"\[EXIT\]\s+([A-Z0-9]+)\s+\|\s+Price:\s+\$([0-9\.]+)\s+\|\s+P&L:\s+\$(-?[0-9\.]+)\s+\(([-+0-9\.]+%)\)\s+\|\s+Reason:\s+([a-z_]+)",
    re.IGNORECASE,
)

def parse_log_exits(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = exit_re.search(line)
            if m:
                symbol, price, pnl, pct, reason = m.groups()
                rows.append({
                    "symbol": symbol,
                    "exit_price": safe_float(price),
                    "pnl": safe_float(pnl),
                    "pct": pct,
                    "reason": reason,
                    "line": line.strip()
                })
    return rows

def safe_float(v: Optional[str]) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def read_csv_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]

def extract_numeric_field(row: Dict, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            try:
                return float(row[c])
            except Exception:
                # strip $ or % then try
                s = str(row[c]).replace("$", "").replace("%", "").strip()
                try:
                    return float(s)
                except Exception:
                    continue
    return None

def normalize_trade_row(row: Dict) -> Dict:
    # Common column name candidates
    pnl = extract_numeric_field(row, ["pnl", "p&l", "profit", "profit_loss", "net"])
    entry = extract_numeric_field(row, ["entry", "entry_price", "entry_price_usd", "buy_price"])
    exitp = extract_numeric_field(row, ["exit", "exit_price", "exit_price_usd", "sell_price", "price"])
    pct = extract_numeric_field(row, ["pct", "pct_return", "return_pct", "percent", "roi"])
    symbol = row.get("symbol") or row.get("ticker") or row.get("sym") or ""
    reason = row.get("reason") or row.get("exit_reason") or ""
    timestamp = row.get("timestamp") or row.get("time") or ""
    return {
        "symbol": symbol,
        "entry_price": entry,
        "exit_price": exitp,
        "pnl": pnl,
        "pct": pct,
        "reason": reason,
        "timestamp": timestamp,
        **row
    }

def analyze_trades(trades: List[Dict]) -> Dict:
    t = [normalize_trade_row(r) for r in trades]
    wins = [r for r in t if r.get("pnl") is not None and r["pnl"] > 0]
    losses = [r for r in t if r.get("pnl") is not None and r["pnl"] < 0]
    total_trades = len([r for r in t if r.get("pnl") is not None])
    net = sum(r["pnl"] for r in t if r.get("pnl") is not None)
    avg_win = mean([r["pnl"] for r in wins]) if wins else 0.0
    avg_loss = mean([r["pnl"] for r in losses]) if losses else 0.0
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    gross_win = sum(r["pnl"] for r in wins) if wins else 0.0
    gross_loss = sum(r["pnl"] for r in losses) if losses else 0.0
    profit_factor = (gross_win / abs(gross_loss)) if abs(gross_loss) > 0 else float("inf")
    top_losses = sorted(losses, key=lambda x: x["pnl"])[:10]
    top_losses_by_pct = sorted([r for r in t if r.get("pct") is not None], key=lambda x: x["pct"])[:10]
    # Counts per reason and per symbol
    reason_counts = {}
    symbol_loss_counts = {}
    for r in losses:
        reason_counts[r.get("reason") or ""] = reason_counts.get(r.get("reason") or "", 0) + 1
        symbol_loss_counts[r.get("symbol") or ""] = symbol_loss_counts.get(r.get("symbol") or "", 0) + 1

    return {
        "total_trades_with_pnl": total_trades,
        "total_wins": len(wins),
        "total_losses": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "net_pnl": round(net, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "top_losses": top_losses,
        "top_losses_by_pct": top_losses_by_pct,
        "reason_counts": reason_counts,
        "symbol_loss_counts": symbol_loss_counts,
    }

def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

def main():
    # 1) Parse log exits (legacy)
    log_exits = parse_log_exits(log_path)

    # Write losers found in the log
    losers_from_log = [r for r in log_exits if r.get("pnl") is not None and r["pnl"] < 0]
    if losers_from_log:
        save_csv(out_losers_path, losers_from_log, ["symbol", "exit_price", "pnl", "pct", "reason", "line"])
    print(f"Wrote {len(losers_from_log)} losing trades found in log to {out_losers_path}")

    # 2) Read backtest_summary.csv (if available) and analyze
    summary_rows = read_csv_rows(backtest_summary_path)
    analysis = {}
    if summary_rows:
        analysis["from_backtest_summary"] = analyze_trades(summary_rows)
        print("Analyzed trades from", backtest_summary_path)
    else:
        print("No backtest_summary.csv found at", backtest_summary_path)

    # 3) Read backtest_reject_summary.csv (if available)
    reject_rows = read_csv_rows(reject_summary_path)
    if reject_rows:
        # try to aggregate counts if file has 'reason' and 'count' fields, otherwise count rows by reason
        reason_agg = {}
        for r in reject_rows:
            reason = r.get("reason") or r.get("Reason") or r.get("reject_reason") or ""
            count = extract_numeric_field(r, ["count", "occurrences", "qty"]) or 0
            if reason:
                reason_agg[reason] = reason_agg.get(reason, 0) + int(count) if count else reason_agg.get(reason, 0) + 1
        analysis["reject_summary"] = reason_agg
        print("Loaded reject summary from", reject_summary_path)
    else:
        print("No backtest_reject_summary.csv found at", reject_summary_path)

    # 4) Merge log-derived losers into summary analysis (if both exist)
    if summary_rows and log_exits:
        # get set of symbols that lost in log and mark them
        log_loser_symbols = set(r["symbol"] for r in log_exits if r.get("pnl") and r["pnl"] < 0)
        analysis.setdefault("log_loser_symbols", list(sorted(log_loser_symbols)))

    # 5) Save analysis JSON
    with analysis_out_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)
    print("Saved analysis to", analysis_out_path)

if __name__ == "__main__":
    main()
