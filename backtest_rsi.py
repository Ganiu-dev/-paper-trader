#!/usr/bin/env python3
"""backtest_rsi.py — does buying the panic and selling the euphoria have an edge?

Replays 4h history one bar at a time through strategy_rsi.evaluate(), and
settles every signal with trade_engine.sim_buy() / sim_sell(), passing the
bar's own fill price in. The fees here are therefore the engine's own
arithmetic, not a second copy that can drift away from what the live paper
trader charges.

Read-only. TRADING_ENABLED stays False, and the engine's state files are
pointed at a temporary directory for the duration, so paper_ledger.json and
paper_trades.jsonl are never touched.

    python3 backtest_rsi.py                      # 3 years of 4h candles
    python3 backtest_rsi.py --days 365
    python3 backtest_rsi.py --midline-mode up --trades
    python3 backtest_rsi.py --stop 0.05 --json

Why this file imports its statistics from backtest_mss
------------------------------------------------------
``student_t_two_sided_p`` and ``t_statistic`` already exist and are already
tested (``test_backtest_mss.py`` reproduces published critical values for
them). Two copies of a t distribution is one copy too many: the second drifts,
and then two strategies' T-stats are not comparable because they were not
computed the same way. So the statistics are imported, and only the replay
below is new.

T-stat: what it is and what it is not
-------------------------------------
The T-stat is the mean per-trade return divided by its standard error — "is the
average trade distinguishable from zero?" It is computed on *percent* returns
so a $10 clip and a $100 clip score the same, and it uses a Student t
distribution with n-1 degrees of freedom, because with a few dozen trades the
normal approximation overstates significance.

Two honest caveats. First, trades are not independent draws: a mean-reversion
strategy that re-enters on the next oversold reading produces clustered,
correlated trades, so the effective sample is smaller than n and the real
T-stat is *worse* than this number. Second, a T-stat is a diagnostic, not a
verdict — a 3-year window on one pair is one path through one regime.

Limits of the simulation
------------------------
Entries fill at the signal bar's close. The overbought and midline exits also
fill at the close, which is right — both are close-based rules. The hard stop
fills at the stop price, or at the open on a gap, which is realistic but still
ignores slippage. No spread, no partial fills, no funding. The fee is the only
modelled friction, and it is the one real cost this strategy is most exposed
to: mean reversion trades often and small, so the fee line is where a thin edge
goes to die.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

import backtest as bt          # fetch_history, sandboxed_engine, max_drawdown, interval_hours
import backtest_mss as bmss    # student_t_two_sided_p, t_statistic (one implementation)
import strategy_rsi as rsi
import trade_engine as te

# Re-exported deliberately: there is one t distribution in this project, and it
# lives in backtest_mss. Naming them here lets a test assert the identity rather
# than trusting a comment.
t_statistic = bmss.t_statistic
student_t_two_sided_p = bmss.student_t_two_sided_p

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PAIR = "BNB"
INTERVAL = "4h"
DAYS = 365 * 3
SIZE_USD = 10.0
START_USDT = 100.0


class BacktestError(RuntimeError):
    """The replay cannot produce a meaningful result."""


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #

def run_backtest(
    candles: list[dict[str, Any]],
    *,
    pair: str = PAIR,
    size_usd: float = SIZE_USD,
    period: int = rsi.PERIOD,
    oversold: float = rsi.OVERSOLD,
    overbought: float = rsi.OVERBOUGHT,
    midline: float = rsi.MIDLINE,
    stop_pct: float = rsi.STOP_PCT,
    midline_mode: str = rsi.MIDLINE_MODE,
    start_usdt: float = START_USDT,
    interval: str = INTERVAL,
    rsi_values: list[float | None] | None = None,
) -> dict[str, Any]:
    """Replay candles bar by bar, trading RSI mean reversion with a hard stop.

    One position at a time, no pyramiding, no partial exits — the same shape as
    run_paper_trader.py, so a backtest number means something about the runner.

    ``rsi_values`` is an injection point for tests only: pass a series to prove
    the replay actually reads it. The default recomputes from the closes.
    """
    warmup = period + 2
    if len(candles) < warmup + 1:
        raise BacktestError(f"need at least {warmup + 1} candles, got {len(candles)}")
    if not size_usd > 0:
        raise BacktestError(f"size must be positive, got {size_usd!r}")
    if start_usdt < size_usd:
        raise BacktestError(f"starting cash {start_usdt} cannot fund a {size_usd} entry")
    # Validate the strategy's own settings before the replay starts. Reaching
    # bar 0 and raising from inside evaluate() would mean a typo'd flag burned a
    # ledger reset first, and it reports the wrong exception type to a caller.
    try:
        rsi._validate(period, oversold, overbought, midline, stop_pct, midline_mode)
    except rsi.StrategyError as exc:
        raise BacktestError(str(exc)) from exc

    pair_label = f"{pair}/USDT"
    # The replay writes engine state on every fill, so it must run
    # against a temp ledger. Without this a direct call to
    # run_backtest() truncates the owner's live paper ledger.
    with bt.sandboxed_engine():
        te.reset(starting_usdt=start_usdt)

        # One forward pass for the whole series. rsi[i] is built solely from
        # closes[:i + 1], so precomputing leaks nothing a per-bar rebuild would not.
        # test_backtest_rsi.py proves the precomputed path is identical.
        closes = [float(c["close"]) for c in candles]
        if rsi_values is None:
            rsi_values = rsi.rsi_series(closes, period)
        elif len(rsi_values) != len(candles):
            raise BacktestError(
                f"rsi_values has {len(rsi_values)} entries for {len(candles)} candles"
            )

        cash = float(start_usdt)
        held = 0.0
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        open_entry: dict[str, Any] | None = None
        bars_in_market = 0

        for i in range(warmup, len(candles)):
            window = candles[: i + 1]
            price = float(candles[i]["close"])
            position = None
            if open_entry is not None:
                position = {"entry_price": open_entry["entry_price"]}

            result = rsi.evaluate(
                window,
                position=position,
                period=period,
                oversold=oversold,
                overbought=overbought,
                midline=midline,
                stop_pct=stop_pct,
                midline_mode=midline_mode,
                pair=pair_label,
                interval=interval,
                rsi=rsi_values[: i + 1],
            )
            signal = result["signal"]
            fill: dict[str, Any] | None = None

            if signal == rsi.BUY and open_entry is None:
                if cash < size_usd:
                    # Out of ammo. Recorded, not hidden: a strategy whose edge
                    # depends on trades it could not afford is not a strategy.
                    equity_curve.append(_point(i, candles, cash, held, price))
                    continue
                fill = te.sim_buy(pair_label, size_usd, _price=price)
                cash = float(fill["usdt_after"])
                held = float(fill["position_after"])
                open_entry = {
                    "entry_bar": i,
                    "entry_time": candles[i]["close_time"],
                    "entry_price": price,
                    "usdt_in": float(fill["usdt_amount"]),
                    "quantity": float(fill["token_amount"]),
                    "entry_rsi": result["rsi"],
                    # A BUY verdict carries no stop (there was no position to
                    # protect), so compute the one this entry just took on.
                    "stop_price": rsi.stop_price(price, stop_pct),
                }

            elif signal == rsi.SELL and open_entry is not None:
                exit_price = float(result["fill_price"] if result["fill_price"] else price)
                fill = te.sim_sell(pair_label, held, _price=exit_price)
                cash = float(fill["usdt_after"])
                held = float(fill["position_after"])
                trades.append(
                    {
                        "entry_bar": open_entry["entry_bar"],
                        "entry_time": open_entry["entry_time"],
                        "entry_price": open_entry["entry_price"],
                        "usdt_in": open_entry["usdt_in"],
                        "quantity": open_entry["quantity"],
                        "entry_rsi": open_entry["entry_rsi"],
                        "stop_price": open_entry["stop_price"],
                        "exit_bar": i,
                        "exit_time": candles[i]["close_time"],
                        "exit_price": exit_price,
                        "exit_rsi": result["rsi"],
                        "exit_reason": result["reason"],
                        "usdt_out": float(fill["proceeds_usdt"]),
                        "pnl": float(fill["realized_pnl"]),
                        "bars_held": i - open_entry["entry_bar"],
                    }
                )
                open_entry = None

            if held > 0.0:
                bars_in_market += 1

            equity_curve.append(_point(i, candles, cash, held, price))

        # Realise anything still open at the last close: end on a number, not an
        # unrealised opinion. Same engine path, same fee.
        final_price = float(candles[-1]["close"])
        if open_entry is not None:
            fill = te.sim_sell(pair_label, held, _price=final_price)
            cash = float(fill["usdt_after"])
            trades.append(
                {
                    "entry_bar": open_entry["entry_bar"],
                    "entry_time": open_entry["entry_time"],
                    "entry_price": open_entry["entry_price"],
                    "usdt_in": open_entry["usdt_in"],
                    "quantity": open_entry["quantity"],
                    "entry_rsi": open_entry["entry_rsi"],
                    "stop_price": open_entry["stop_price"],
                    "exit_bar": len(candles) - 1,
                    "exit_time": candles[-1]["close_time"],
                    "exit_price": final_price,
                    "exit_rsi": rsi_values[-1],
                    "exit_reason": "END_OF_DATA",
                    "usdt_out": float(fill["proceeds_usdt"]),
                    "pnl": float(fill["realized_pnl"]),
                    "bars_held": len(candles) - 1 - open_entry["entry_bar"],
                    "forced_close": True,
                }
            )
            held = 0.0
            equity_curve[-1] = _point(len(candles) - 1, candles, cash, 0.0, final_price)

        ledger = te.load_ledger()
    return {
        "pair": pair_label,
        "interval": interval,
        "period": period,
        "oversold": oversold,
        "overbought": overbought,
        "midline": midline,
        "midline_mode": midline_mode,
        "stop_pct": stop_pct,
        "size_usd": size_usd,
        "start_usdt": float(start_usdt),
        "bars": len(candles),
        "first_bar_time": candles[0]["open_time"],
        "last_bar_time": candles[-1]["close_time"],
        "first_price": float(candles[0]["close"]),
        "last_price": final_price,
        "trades": trades,
        "equity_curve": equity_curve,
        "bars_in_market": bars_in_market,
        "final_cash": cash,
        "realized_pnl": float(ledger["realized_pnl"]),
        "fees_paid": float(ledger["fees_paid"]),
        "fills": int(ledger["fills"]),
    }


def _point(i: int, candles: list[dict[str, Any]], cash: float, held: float,
           price: float) -> dict[str, Any]:
    return {
        "bar": i,
        "time": candles[i]["close_time"],
        "price": price,
        "cash": cash,
        "market_value": held * price,
        "equity": cash + held * price,
        "in_market": held > 0.0,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def summarise(result: dict[str, Any], hold: dict[str, Any]) -> dict[str, Any]:
    """Everything the report needs, computed once from the trade list."""
    trades = result["trades"]
    pnls = [float(t["pnl"]) for t in trades]

    # Percent returns, so the T-stat does not measure position size.
    returns_pct = [
        (float(t["usdt_out"]) / float(t["usdt_in"]) - 1.0) * 100.0 for t in trades
    ]
    stats = bmss.t_statistic(returns_pct)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    curve = result["equity_curve"]
    exposure = (
        sum(1 for p in curve if p["in_market"]) / len(curve) * 100.0 if curve else 0.0
    )
    dd = bt.max_drawdown(curve)
    final_equity = float(result["final_cash"])
    total_pnl = final_equity - result["start_usdt"]

    by_reason: dict[str, int] = {}
    pnl_by_reason: dict[str, float] = {}
    for trade in trades:
        reason = trade.get("exit_reason", "UNKNOWN")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        pnl_by_reason[reason] = pnl_by_reason.get(reason, 0.0) + float(trade["pnl"])

    return {
        "bars": result["bars"],
        "days": result["bars"] * bt.interval_hours(result["interval"]) / 24.0,
        "first_time": _iso(result["first_bar_time"]),
        "last_time": _iso(result["last_bar_time"]),
        "start_usdt": result["start_usdt"],
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / result["start_usdt"] * 100.0,
        "round_trips": len(trades),
        "fills": result["fills"],
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "best_trade": max(pnls) if pnls else 0.0,
        "worst_trade": min(pnls) if pnls else 0.0,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "fees_paid": result["fees_paid"],
        "gross_pnl": total_pnl + result["fees_paid"],
        "fee_drag_pct": result["fees_paid"] / result["start_usdt"] * 100.0,
        "max_drawdown_usdt": dd["max_drawdown_usdt"],
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "drawdown_peak_bar": dd["peak_bar"],
        "drawdown_trough_bar": dd["trough_bar"],
        "bars_in_market": result["bars_in_market"],
        "exposure_pct": exposure,
        "buy_hold_pnl": hold["pnl"],
        "buy_hold_pnl_pct": hold["pnl_pct"],
        "difference_pct": total_pnl / result["start_usdt"] * 100.0 - hold["pnl_pct"],
        "avg_bars_held": (sum(t["bars_held"] for t in trades) / len(trades)) if trades else 0.0,
        "exit_reasons": by_reason,
        "pnl_by_exit_reason": pnl_by_reason,
        "t_stat": stats["t_stat"],
        "p_value": stats["p_value"],
        "df": stats["df"],
        "mean_trade_pct": stats["mean_pct"],
        "stdev_trade_pct": stats["stdev_pct"],
        "t_stat_n": stats["n"],
    }


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _fmt(value: float, places: int = 2) -> str:
    return f"{value:,.{places}f}"


def format_report(summary: dict[str, Any], result: dict[str, Any]) -> str:
    s = summary
    stamp = (
        f"backtest_rsi | {result['pair']} {result['interval']} | "
        f"RSI {result['period']}  {result['oversold']:g}/{result['overbought']:g}  "
        f"midline {result['midline']:g} ({result['midline_mode']}) | "
        f"stop {result['stop_pct'] * 100:g}% | ${_fmt(result['size_usd'])} per entry | "
        f"fee {te.FEE_RATE * 100:g}% per side | TRADING_ENABLED={te.TRADING_ENABLED}"
    )

    def pf_text() -> str:
        if s["profit_factor"] is not None:
            return f"{s['profit_factor']:.2f}"
        return "n/a (no losing trades)" if s["round_trips"] else "n/a"

    def t_text() -> str:
        if s["t_stat"] is None:
            return f"n/a ({s['t_stat_n']} trades — too few to test)"
        return f"{s['t_stat']:+.2f}  (p = {s['p_value']:.3f}, df = {s['df']})"

    lines = [
        stamp,
        f"window    {s['first_time']} -> {s['last_time']} UTC  "
        f"({s['bars']:,} bars, {s['days']:,.0f} days)",
        f"price     {result['first_price']:,.4f} -> {result['last_price']:,.4f}  "
        f"({(result['last_price'] / result['first_price'] - 1) * 100:+.2f}%)",
        "",
        "results",
        f"  trades executed   {s['round_trips']} round trips ({s['fills']} fills)",
        f"  win rate          {s['win_rate']:.1f}%  ({s['wins']} won, {s['losses']} lost)",
        f"  net P&L           {s['total_pnl']:+,.2f} USDT  ({s['total_pnl_pct']:+.2f}%)",
        f"    before fees     {s['gross_pnl']:+,.2f} USDT "
        f"   [fees cost {s['fee_drag_pct']:.2f} points]",
        f"  max drawdown      {s['max_drawdown_usdt']:,.2f} USDT  ({s['max_drawdown_pct']:.2f}%)",
        f"  profit factor     {pf_text()}",
        f"  T-stat            {t_text()}",
        "",
        "detail",
        f"  mean / stdev      {s['mean_trade_pct']:+.3f}% / "
        + (f"{s['stdev_trade_pct']:.3f}%" if s["stdev_trade_pct"] is not None else "n/a")
        + " per round trip",
        f"  avg win / loss    {s['avg_win']:+.4f} / {s['avg_loss']:+.4f} USDT",
        f"  best / worst      {s['best_trade']:+.4f} / {s['worst_trade']:+.4f} USDT",
        f"  avg hold          {s['avg_bars_held']:.1f} bars "
        f"({s['avg_bars_held'] * bt.interval_hours(result['interval']):.0f}h)",
        f"  time in market    {s['exposure_pct']:.1f}% of bars "
        f"({s['bars_in_market']:,} of {s['bars']:,})",
        f"  fees paid         {s['fees_paid']:,.4f} USDT",
        f"  buy & hold        {s['buy_hold_pnl']:+,.2f} USDT  ({s['buy_hold_pnl_pct']:+.2f}%)",
        f"  versus hold       {s['difference_pct']:+.2f} points",
    ]

    if s["exit_reasons"]:
        lines += ["", "exits"]
        for reason in sorted(s["exit_reasons"], key=lambda r: -s["exit_reasons"][r]):
            count = s["exit_reasons"][reason]
            pnl = s["pnl_by_exit_reason"].get(reason, 0.0)
            lines.append(f"  {reason:<14} {count:>3} x   {pnl:+.4f} USDT")

    lines += ["", f"verdict   {_verdict(s)}"]
    return "\n".join(lines)


def _verdict(s: dict[str, Any]) -> str:
    """One sentence, and only what the numbers actually support."""
    if s["round_trips"] == 0:
        return "RSI never reached an extreme in this window — nothing to judge."

    t = s["t_stat"]
    if s["total_pnl"] <= 0:
        base = (
            f"the strategy lost {abs(s['total_pnl']):,.2f} USDT over "
            f"{s['round_trips']} round trips at a {s['win_rate']:.1f}% win rate"
        )
    else:
        base = (
            f"the strategy made {s['total_pnl']:,.2f} USDT ({s['total_pnl_pct']:+.2f}%) "
            f"over {s['round_trips']} round trips at a {s['win_rate']:.1f}% win rate"
        )

    if t is None:
        return base + " — too few trades for the T-stat to mean anything."
    if abs(t) < 1.0:
        return base + f", but the T-stat is {t:+.2f}: indistinguishable from noise."
    if abs(t) < 1.96:
        return (
            base + f", and the T-stat of {t:+.2f} is suggestive but short of "
            "significance at 5% (|t| >= 1.96)."
        )
    direction = "positive" if t > 0 else "negative"
    return (
        base + f". The T-stat is {t:+.2f} (p = {s['p_value']:.3f}) — statistically "
        f"{direction} at the 5% level, though trades are not independent draws and "
        "the effective sample is smaller than n."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest RSI mean reversion on historical candles (read-only)"
    )
    parser.add_argument("--symbol", default=PAIR, help=f"base asset (default {PAIR})")
    parser.add_argument("--days", type=float, default=DAYS, help=f"days of history (default {DAYS})")
    parser.add_argument("--bars", type=int, default=None, help="candles of history (overrides --days)")
    parser.add_argument(
        "--interval", default=INTERVAL, choices=sorted(bt.INTERVAL_HOURS),
        help=f"candle size (default {INTERVAL})",
    )
    parser.add_argument("--size", type=float, default=SIZE_USD, help=f"USDT per entry (default {SIZE_USD})")
    parser.add_argument("--period", type=int, default=rsi.PERIOD,
                        help=f"RSI period (default {rsi.PERIOD})")
    parser.add_argument("--oversold", type=float, default=rsi.OVERSOLD,
                        help=f"BUY below this (default {rsi.OVERSOLD:g})")
    parser.add_argument("--overbought", type=float, default=rsi.OVERBOUGHT,
                        help=f"SELL above this (default {rsi.OVERBOUGHT:g})")
    parser.add_argument("--midline", type=float, default=rsi.MIDLINE,
                        help=f"midline exit (default {rsi.MIDLINE:g})")
    parser.add_argument("--midline-mode", default=rsi.MIDLINE_MODE, choices=rsi.MIDLINE_MODES,
                        help=f"which crossing of the midline exits (default {rsi.MIDLINE_MODE})")
    parser.add_argument("--stop", type=float, default=rsi.STOP_PCT,
                        help=f"hard stop fraction (default {rsi.STOP_PCT})")
    parser.add_argument("--start", type=float, default=START_USDT, help="starting cash")
    parser.add_argument("--trades", action="store_true", help="print every round trip")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    if te.TRADING_ENABLED:
        print("error: TRADING_ENABLED is True — this is a simulator", file=sys.stderr)
        return 2
    if not 0 < args.stop < 1:
        print(f"error: --stop must be a fraction in (0, 1), got {args.stop}", file=sys.stderr)
        return 2
    if args.period < 1:
        print("error: --period must be at least 1", file=sys.stderr)
        return 2

    bar_hours = bt.interval_hours(args.interval)
    bars = args.bars if args.bars is not None else int(round(args.days * 24 / bar_hours))
    warmup = args.period + 2

    if bars < warmup + 1:
        print(
            f"error: {bars} bars cannot warm up a {args.period}-period RSI "
            f"(need at least {warmup + 1})",
            file=sys.stderr,
        )
        return 2

    try:
        candles = bt.fetch_history(args.symbol, args.interval, bars)
    except (bt.BacktestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if len(candles) < warmup + 1:
        print(
            f"error: only {len(candles)} usable candles — not enough for a "
            f"{args.period}-period RSI (need {warmup + 1})",
            file=sys.stderr,
        )
        return 1

    # run_backtest() sandboxes itself; this outer one covers buy_and_hold(),
    # which resets the engine too. Nested sandboxes are harmless.
    with bt.sandboxed_engine():
        result = run_backtest(
            candles,
            pair=args.symbol,
            size_usd=args.size,
            period=args.period,
            oversold=args.oversold,
            overbought=args.overbought,
            midline=args.midline,
            stop_pct=args.stop,
            midline_mode=args.midline_mode,
            start_usdt=args.start,
            interval=args.interval,
        )
        hold = bt.buy_and_hold(candles, pair=args.symbol, start_usdt=args.start)

    summary = summarise(result, hold)

    if args.json:
        printable = dict(summary)
        printable["trades"] = result["trades"]
        print(json.dumps(printable, indent=2))
        return 0

    print(format_report(summary, result))

    if args.trades:
        print("\nround trips")
        print(
            f"  {'#':>3}  {'entry':<16}  {'exit':<16}  {'in RSI':>6}  {'out RSI':>7}  "
            f"{'in':>7}  {'out':>8}  {'P&L':>9}  {'bars':>4}  why"
        )
        for n, trade in enumerate(result["trades"], start=1):
            print(
                f"  {n:>3}  {_iso(trade['entry_time']):<16}  "
                f"{_iso(trade['exit_time']):<16}  "
                f"{trade['entry_rsi']:>6.1f}  "
                + (f"{trade['exit_rsi']:>7.1f}" if trade['exit_rsi'] is not None else f"{'n/a':>7}")
                + f"  {trade['usdt_in']:>7.2f}  {trade['usdt_out']:>8.4f}  "
                f"{trade['pnl']:>+9.4f}  {trade['bars_held']:>4}  {trade['exit_reason']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
