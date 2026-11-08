#!/usr/bin/env python3
"""Pool the RSI trade lists from several pairs into one statistical test.

    python3 backtest_multi.py           # BTC, ETH, SOL — the default
    python3 backtest_multi.py --symbols BTC,ETH,SOL,XRP --json

Why this exists: 52 trades cannot resolve a 1% mean against a 5% standard
deviation. The T-stat is not low because the edge is weak; it is low because
the sample is small. Running the same rules on three more price paths is the
one legitimate way to buy sample size — the parameters stay frozen at 14 / 30 /
50 / midline-down / 5% and nothing is tuned afterwards.

What it deliberately does NOT do:

* it does not report three T-stats and let you pick the best pair. That would
  be the same p-hacking the split-half test was built to refuse, with a
  different name. There is exactly one statistic printed here: the pooled one.
* it does not let the pairs share capital. Each symbol runs its own $100 ledger
  at $10 an entry, because pooling the *cash* would make every figure depend on
  which asset happened to have a position open. Pooling the *trades* is the
  statistical operation; pooling the accounts is not.

The 200-MA filter is OFF (`use_trend_filter=False`). T5 was measured and lost:
it vetoed 58 of 66 BUY signals and took the headline numbers down with it. This
script tests the baseline that survived.

One honest caveat, printed with the result rather than buried under it: BTC,
ETH and SOL fall together. An oversold reading in one is usually an oversold
reading in all three, so the pooled trades are not independent draws and the
pooled T-stat overstates its own significance. The report therefore also prints
a date-clustered T-stat, which treats each entry day as one observation and
gives the honest degrees of freedom.

Read-only. Every replay is sandboxed by backtest_rsi.run_backtest, and the live
paper ledger's bytes are hashed before and after as proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any

import backtest as bt            # fetch_history, interval_hours
import backtest_mss as bmss      # t_statistic, student_t_two_sided_p
import backtest_rsi as br
import strategy_rsi as rsi
import trade_engine as te

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SYMBOLS = ("BTC", "ETH", "SOL")
INTERVAL = "4h"
DAYS = 365 * 3
SIZE_USD = 10.0
START_USDT = 100.0
USE_TREND_FILTER = False         # T5 measured and rejected — this is the baseline


# The replay's own error type, so one except clause catches a bad pair or a bad
# configuration without catching everything Python can raise.
BacktestError = br.BacktestError


class MultiError(RuntimeError):
    """The pooled test cannot produce a meaningful number."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _ledger_digest() -> str:
    """md5 of the live ledger, or a marker when there is none to protect."""
    try:
        return hashlib.md5(te.LEDGER_PATH.read_bytes()).hexdigest()
    except OSError:
        return "no-ledger"


def trade_pct(trade: dict[str, Any]) -> float:
    """A round trip's return in percent, net of both fees.

    Percent rather than USDT because that is the unit the t statistic has to be
    computed in: every trade is the same $10, so the two are proportional here,
    but a future size change must not silently rescale the test.
    """
    if not trade["usdt_in"]:
        raise MultiError("a trade with no cash in it cannot be scored")
    return (trade["usdt_out"] / trade["usdt_in"] - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# One pair
# --------------------------------------------------------------------------- #

def run_pair(
    symbol: str,
    *,
    bars: int,
    interval: str = INTERVAL,
    size_usd: float = SIZE_USD,
    start_usdt: float = START_USDT,
    use_trend_filter: bool = USE_TREND_FILTER,
) -> dict[str, Any]:
    """Fetch history and replay one symbol. Returns the leg, not a summary."""
    candles = bt.fetch_history(symbol, interval, bars)
    if not candles:
        raise MultiError(f"{symbol}: no candles returned")

    result = br.run_backtest(
        candles,
        pair=symbol,
        interval=interval,
        size_usd=size_usd,
        start_usdt=start_usdt,
        midline_mode=rsi.MIDLINE_MODE,
        stop_pct=rsi.STOP_PCT,
        use_trend_filter=use_trend_filter,
    )
    hold = bt.buy_and_hold(candles)
    _dd = bt.max_drawdown(result["equity_curve"])

    trades = result["trades"]
    returns = [trade_pct(t) for t in trades]
    stats = bmss.t_statistic(returns)
    pnl = sum(t["pnl"] for t in trades)

    return {
        "symbol": symbol,
        "bars": len(candles),
        "first_time": candles[0]["open_time"],
        "last_time": candles[-1]["close_time"],
        "first_price": float(candles[0]["close"]),
        "last_price": float(candles[-1]["close"]),
        "trades": trades,
        "round_trips": len(trades),
        "wins": sum(1 for t in trades if t["pnl"] > 0),
        "losses": sum(1 for t in trades if t["pnl"] <= 0),
        "net_pnl": pnl,
        "net_pnl_pct": pnl / start_usdt * 100.0,
        "fees_paid": float(result["fees_paid"]),
        "gross_pnl": pnl + float(result["fees_paid"]),
        "max_drawdown_usdt": _dd["max_drawdown_usdt"],
        "max_drawdown_pct": _dd["max_drawdown_pct"],
        "bars_in_market": result["bars_in_market"],
        "exposure_pct": result["bars_in_market"] / max(1, result["bars"] - result["warmup"]) * 100.0,
        "buy_hold_pnl": hold["pnl"],
        "buy_hold_pnl_pct": hold["pnl_pct"],
        "stats": stats,
        "warmup": result["warmup"],
    }


# --------------------------------------------------------------------------- #
# The pooled test
# --------------------------------------------------------------------------- #

def pool(legs: list[dict[str, Any]]) -> dict[str, Any]:
    """One trade list, one t test — the whole point of the exercise."""
    if not legs:
        raise MultiError("nothing to pool")

    trades: list[dict[str, Any]] = []
    for leg in legs:
        for trade in leg["trades"]:
            trades.append({**trade, "pair": leg["symbol"]})

    returns = [trade_pct(t) for t in trades]
    stats = bmss.t_statistic(returns)

    pnl = sum(t["pnl"] for t in trades)
    fees = sum(leg["fees_paid"] for leg in legs)
    capital = START_USDT * len(legs)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)

    return {
        "symbols": [leg["symbol"] for leg in legs],
        "trades": trades,
        "n": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": 100.0 * wins / len(trades) if trades else 0.0,
        "net_pnl": pnl,
        "gross_pnl": pnl + fees,
        "fees_paid": fees,
        "capital": capital,
        "net_pnl_on_capital_pct": pnl / capital * 100.0 if capital else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss else None,
        "avg_win": (gross_win / wins) if wins else 0.0,
        "avg_loss": (-gross_loss / losses) if losses else 0.0,
        "best_trade": max((t["pnl"] for t in trades), default=0.0),
        "worst_trade": min((t["pnl"] for t in trades), default=0.0),
        "stats": stats,
        **cluster_stats(trades),
    }


def cluster_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """The same test with each entry *day* as one observation.

    BTC, ETH and SOL are correlated: an oversold RSI in one is usually an
    oversold RSI in all three, and the three trades are then one market event
    counted three times. Collapsing each day's trades to their mean and testing
    across days is the crude, honest correction — it costs degrees of freedom
    and buys independence.
    """
    buckets: dict[str, list[float]] = {}
    for trade in trades:
        buckets.setdefault(_day(trade["entry_time"]), []).append(trade_pct(trade))
    means = [sum(v) / len(v) for v in buckets.values()]
    stats = bmss.t_statistic(means)
    return {
        "clusters": len(buckets),
        "cluster_stats": stats,
        "clusters_per_trade": len(buckets) / len(trades) if trades else 0.0,
    }


def overlap_matrix(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How often two symbols were holding a position on the same day.

    Not a statistic — a diagnostic. If this is high, the pooled T-stat's degrees
    of freedom are fiction and the clustered number is the one to believe.
    """
    days = {leg["symbol"]: {_day(t["entry_time"]) for t in leg["trades"]} for leg in legs}
    out = []
    symbols = [leg["symbol"] for leg in legs]
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            shared = len(days[a] & days[b])
            denom = min(len(days[a]), len(days[b])) or 1
            out.append({
                "pair": f"{a}/{b}",
                "shared_days": shared,
                "pct_of_smaller": 100.0 * shared / denom,
            })
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(value: float | None, places: int = 2, dash: str = "n/a") -> str:
    return dash if value is None else f"{value:,.{places}f}"


def format_report(legs: list[dict[str, Any]], pooled: dict[str, Any],
                  overlap: list[dict[str, Any]], *, size_usd: float = SIZE_USD,
                  interval: str = INTERVAL,
                  use_trend_filter: bool = USE_TREND_FILTER) -> str:
    s = pooled["stats"]
    c = pooled["cluster_stats"]
    lines: list[str] = []

    # The filter state must come from the caller, not a string: a header that
    # says "no trend filter" while the filter is on is a report that lies about
    # its own configuration, and every number under it inherits the lie.
    filter_label = f"MA{rsi.MA_PERIOD} filter on" if use_trend_filter else "no trend filter"
    lines.append(
        f"RSI {rsi.PERIOD}  {rsi.OVERSOLD:g}/{rsi.OVERBOUGHT:g}  "
        f"midline {rsi.MIDLINE:g} ({rsi.MIDLINE_MODE})  |  {filter_label}  |  "
        f"stop {rsi.STOP_PCT:.0%}  |  ${size_usd:,.2f} per entry"
    )
    lines.append(f"interval {interval}  |  {len(legs)} pairs, each on its own "
                 f"${START_USDT:,.2f} ledger  |  TRADING_ENABLED={te.TRADING_ENABLED}")
    lines.append("")

    lines.append("per pair")
    for leg in legs:
        lines.append(
            f"  {leg['symbol']:<5} {_iso(leg['first_time'])} -> {_iso(leg['last_time'])}  "
            f"{leg['bars']:,} bars  price {leg['first_price']:,.2f} -> {leg['last_price']:,.2f} "
            f"({(leg['last_price'] / leg['first_price'] - 1) * 100:+.2f}%)"
        )
        lines.append(
            f"        {leg['round_trips']} round trips  "
            f"{100.0 * leg['wins'] / leg['round_trips'] if leg['round_trips'] else 0:.1f}% win  "
            f"net {leg['net_pnl']:+,.2f} USDT ({leg['net_pnl_pct']:+.2f}%)  "
            f"T {_fmt(leg['stats']['t_stat'], 2)}  "
            f"exposure {leg['exposure_pct']:.1f}%  "
            f"vs hold {leg['net_pnl'] - leg['buy_hold_pnl']:+,.2f}"
        )
    lines.append("")

    lines.append("pooled")
    breakdown = " + ".join(f"{leg['symbol']} {leg['round_trips']}" for leg in legs)
    lines.append(f"  trades            {pooled['n']} round trips  ({breakdown})")
    lines.append(
        f"  net P&L           {pooled['net_pnl']:+,.2f} USDT across "
        f"{len(legs)} x ${START_USDT:,.2f} = ${pooled['capital']:,.2f} "
        f"({pooled['net_pnl_on_capital_pct']:+.2f}%)"
    )
    lines.append(
        f"  before fees       {pooled['gross_pnl']:+,.2f} USDT  "
        f"(fees cost {pooled['fees_paid']:,.2f})"
    )
    lines.append(
        f"  win rate          {pooled['win_rate']:.1f}% "
        f"({pooled['wins']}W / {pooled['losses']}L)  "
        f"profit factor {_fmt(pooled['profit_factor'])}"
    )
    lines.append(
        f"  avg win / loss    {pooled['avg_win']:+,.2f} / {pooled['avg_loss']:+,.2f} USDT  "
        f"best {pooled['best_trade']:+,.2f}  worst {pooled['worst_trade']:+,.2f}"
    )
    lines.append(
        f"  per trade         mean {_fmt(s['mean_pct'], 3)}%  "
        f"stdev {_fmt(s['stdev_pct'], 3)}%"
    )
    if s["t_stat"] is None:
        lines.append("  T-stat            n/a (too few trades, or no variance)")
    else:
        lines.append(
            f"  T-stat            {s['t_stat']:+.2f}   "
            f"p = {s['p_value']:.4f}   df = {s['df']}   n = {s['n']}"
        )
    lines.append("")

    lines.append("correlated assets — the honest degrees of freedom")
    for row in overlap:
        lines.append(
            f"  {row['pair']:<9} {row['shared_days']} entry days shared "
            f"({row['pct_of_smaller']:.0f}% of the smaller leg)"
        )
    lines.append(
        f"  distinct entry days  {pooled['clusters']} for {pooled['n']} trades "
        f"({pooled['clusters_per_trade'] * 100:.0f}%)"
    )
    if c["t_stat"] is None:
        lines.append("  date-clustered T     n/a")
    else:
        lines.append(
            f"  date-clustered T     {c['t_stat']:+.2f}   "
            f"p = {c['p_value']:.4f}   df = {c['df']}"
        )
    lines.append("")

    lines.append(verdict(pooled))
    return "\n".join(lines)


def verdict(pooled: dict[str, Any]) -> str:
    s = pooled["stats"]
    c = pooled["cluster_stats"]
    n = pooled["n"]
    pnl = pooled["net_pnl"]

    if n == 0:
        return "verdict: no trades anywhere — the rules never fired, which is a bug, not a result"
    if s["t_stat"] is None:
        return (f"verdict: {n} trades, but no dispersion to test — every trip returned "
                f"the same amount, which is a bug")

    t = s["t_stat"]
    base = (f"verdict: {n} pooled trades, {pnl:+,.2f} USDT, "
            f"mean {s['mean_pct']:+.3f}% per trip.")

    if abs(t) < 1.0:
        head = " The pooled T-stat of {:+.2f} is indistinguishable from noise even at this sample size.".format(t)
    elif abs(t) < 1.96:
        head = (" The pooled T-stat of {:+.2f} is suggestive but still short of "
                "significance at 5% (|t| >= 1.96).".format(t))
    else:
        direction = "positive" if t > 0 else "negative"
        head = (" The pooled T-stat of {:+.2f} (p = {:.4f}) is statistically {} at "
                "the 5% level.".format(t, s["p_value"], direction))

    if c["t_stat"] is not None and abs(c["t_stat"]) < 1.96 <= abs(t):
        tail = (f" It does not survive the correlated-asset correction: clustered by entry "
                f"day the T-stat is {c['t_stat']:+.2f} over {c['df'] + 1} independent days. "
                f"The raw number is a market event counted {pooled['n'] / max(1, pooled['clusters']):.1f} "
                f"times, not {pooled['n']} independent draws.")
    elif c["t_stat"] is not None:
        tail = (" Clustered by entry day it is {:+.2f} over {} independent days, which "
                "is the number to quote to anyone who knows these assets move together."
                .format(c["t_stat"], c["df"] + 1))
    else:
        tail = ""

    return base + head + tail


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pool RSI trade lists across pairs into one t test (read-only)"
    )
    parser.add_argument("--symbols", default=",".join(SYMBOLS),
                        help=f"comma-separated bases (default {','.join(SYMBOLS)})")
    parser.add_argument("--days", type=float, default=DAYS,
                        help=f"days of history per pair (default {DAYS})")
    parser.add_argument("--bars", type=int, default=None,
                        help="candles per pair (overrides --days)")
    parser.add_argument("--interval", default=INTERVAL, choices=sorted(bt.INTERVAL_HOURS),
                        help=f"candle size (default {INTERVAL})")
    parser.add_argument("--size", type=float, default=SIZE_USD,
                        help=f"USDT per entry (default {SIZE_USD})")
    parser.add_argument("--start", type=float, default=START_USDT,
                        help=f"starting cash per pair (default {START_USDT})")
    parser.add_argument("--trend-filter", dest="use_trend_filter", action="store_true",
                        help="apply the T5 200-MA filter (default: off, the baseline)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a report")
    args = parser.parse_args(argv)

    if te.TRADING_ENABLED:
        print("error: TRADING_ENABLED is True — this is a simulator", file=sys.stderr)
        return 2
    if args.size <= 0 or args.start < args.size:
        print("error: each pair needs starting cash of at least one position", file=sys.stderr)
        return 2

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("error: --symbols named no pairs", file=sys.stderr)
        return 2
    if len(set(symbols)) != len(symbols):
        print("error: --symbols repeats a pair", file=sys.stderr)
        return 2

    bar_hours = bt.interval_hours(args.interval)
    bars = args.bars if args.bars is not None else int(round(args.days * 24 / bar_hours))
    warmup = rsi.PERIOD + 2
    if args.use_trend_filter:
        warmup = max(warmup, rsi.MA_PERIOD)
    if bars < warmup + 1:
        print(f"error: {bars} bars cannot warm up the strategy (need {warmup + 1})",
              file=sys.stderr)
        return 2

    before = _ledger_digest()

    legs: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            leg = run_pair(
                symbol,
                bars=bars,
                interval=args.interval,
                size_usd=args.size,
                start_usdt=args.start,
                use_trend_filter=args.use_trend_filter,
            )
        except (BacktestError, ValueError, MultiError) as exc:
            print(f"error: {symbol}: {exc}", file=sys.stderr)
            return 1
        legs.append(leg)

    pooled = pool(legs)
    overlap = overlap_matrix(legs)
    after = _ledger_digest()

    if args.json:
        payload = {
            "config": {
                "symbols": symbols,
                "interval": args.interval,
                "bars": bars,
                "size_usd": args.size,
                "start_usdt": args.start,
                "use_trend_filter": args.use_trend_filter,
                "rsi": {
                    "period": rsi.PERIOD,
                    "oversold": rsi.OVERSOLD,
                    "overbought": rsi.OVERBOUGHT,
                    "midline": rsi.MIDLINE,
                    "midline_mode": rsi.MIDLINE_MODE,
                    "stop_pct": rsi.STOP_PCT,
                },
            },
            "pairs": [
                {k: v for k, v in leg.items() if k != "trades"} | {
                    "trades": [
                        {k: v for k, v in t.items()} for t in leg["trades"]
                    ]
                }
                for leg in legs
            ],
            "pooled": {k: v for k, v in pooled.items() if k != "trades"},
            "overlap": overlap,
            "ledger_md5": {"before": before, "after": after,
                           "untouched": before == after},
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_report(legs, pooled, overlap, size_usd=args.size,
                            interval=args.interval,
                            use_trend_filter=args.use_trend_filter))
        print()
        print(f"live ledger md5 {before} -> {after}  "
              f"({'UNTOUCHED' if before == after else 'TOUCHED — investigate'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
