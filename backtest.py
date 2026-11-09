#!/usr/bin/env python3
"""backtest.py — does the 10/30 1h crossover actually have an edge?

Replays history one bar at a time through strategy.evaluate(), and settles
every signal with trade_engine.sim_buy() / sim_sell() — at the NEXT bar's open,
never at the close that produced the signal. That matters twice over: the fees
here are the engine's own arithmetic, not a second copy of it that can quietly
drift away from what the live paper trader charges, and the fill price is one
that existed after the decision was made. If the fee model changes in
trade_engine.py, this backtest changes with it.

Read-only. TRADING_ENABLED stays False, and the run points the engine's state
files at a temporary directory, so your paper_ledger.json and
paper_trades.jsonl are never touched.

    python3 backtest.py                      # 1 year of 1h candles
    python3 backtest.py --days 180           # six months
    python3 backtest.py --trades             # also print every round trip
    python3 backtest.py --json               # machine-readable result

What it does NOT model
----------------------
A signal is a decision taken on a closed bar; the fill is the open of the bar
after it. So the fill price is the first price the market offered once the
signal was knowable — no free look at the move that produced the signal, and
no fill at all if that bar never arrives (a signal on the last bar is dropped,
not executed at the close). There is still no slippage, no spread, no partial
fills and no funding; the fee is the only friction. On $10 clips against a
liquid pair that is a fair approximation; it would flatter a large size.

The comparison with buy & hold is deliberately uncomfortable
-------------------------------------------------------------
The strategy risks $10 at a time and sits in cash the rest of the while; buy
& hold puts the entire $100 to work on day one. A raw return comparison
between the two therefore flatters buy & hold by roughly the ratio of exposure,
which is why the report prints the strategy's average exposure and the number
of bars it spent out of the market. Read the two figures as "what each did
with the capital it chose to use", not as "which is better".
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import strategy
import trade_engine as te

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

PAIR = "BNB"
INTERVAL = "1h"
BARS = 8_760            # one year of 1h candles
CHUNK = 1_000           # Binance's per-request ceiling
SIZE_USD = 10.0         # dollars per entry, same as the live runner
START_USDT = 100.0      # same starting capital as the engine
HTTP_TIMEOUT = 15
MAX_CHUNKS = 40         # a loop bound; 40k candles is far more than enough

HOUR_MS = 3_600_000

# Bar length per Binance interval. The backtest counts bars, but a human asks in
# days, and 1095 days is 26,280 hourly bars or 6,570 four-hourly ones — so the
# conversion has to know which interval it is talking about rather than assume 1h.
INTERVAL_HOURS = {
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1.0,
    "2h": 2.0,
    "4h": 4.0,
    "6h": 6.0,
    "8h": 8.0,
    "12h": 12.0,
    "1d": 24.0,
}


def interval_hours(interval: str) -> float:
    """Hours per bar for a Binance interval string, or a loud refusal."""
    try:
        return INTERVAL_HOURS[interval]
    except KeyError:
        raise ValueError(
            f"unsupported interval {interval!r}: "
            f"expected one of {', '.join(sorted(INTERVAL_HOURS))}"
        ) from None


class BacktestError(RuntimeError):
    """History could not be fetched or is too short to mean anything."""


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _parse_rows(rows: list, symbol: str) -> list[dict[str, Any]]:
    """Binance kline rows -> candles, with the same refusals strategy.py makes.

    A row that cannot be read is an error, not a zero: a backtest built on a
    silently-mangled bar looks exactly like a backtest built on a real one.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            raise BacktestError(f"malformed candle for {symbol}: {row!r}"[:300])
        try:
            candle = {
                "open_time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": int(row[6]),
            }
        except (TypeError, ValueError) as exc:
            raise BacktestError(f"malformed candle for {symbol}: {row!r}"[:300]) from exc
        if not candle["close"] > 0:
            raise BacktestError(f"non-positive close for {symbol}: {candle['close']!r}")
        out.append(candle)
    return out


def _get_chunk(
    symbol: str,
    interval: str,
    limit: int,
    end_ms: int,
    _opener=None,
) -> list[list]:
    query = urllib.parse.urlencode(
        {"symbol": symbol, "interval": interval, "limit": limit, "endTime": end_ms}
    )
    request = urllib.request.Request(
        f"{BINANCE_KLINES_URL}?{query}",
        headers={"User-Agent": "paper-trader-backtest/1.0", "Accept": "application/json"},
        method="GET",
    )
    open_url = _opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=HTTP_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise BacktestError(f"Binance HTTP {exc.code} for {symbol}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BacktestError(f"could not reach Binance for {symbol}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BacktestError(f"Binance sent non-JSON for {symbol}: {exc}") from exc

    if not isinstance(payload, list):
        raise BacktestError(f"unexpected payload for {symbol}: {payload!r}"[:300])
    return payload


def fetch_history(
    pair: str = PAIR,
    interval: str = INTERVAL,
    bars: int = BARS,
    *,
    end_ms: int | None = None,
    _opener=None,
) -> list[dict[str, Any]]:
    """Closed candles, oldest first, paged backwards until `bars` are collected.

    Binance caps a request at 1000 candles and takes an ``endTime``, so the
    walk moves the cursor to just before the earliest row each time. It stops
    when the requested count is reached, when Binance runs out of history, or
    when a chunk comes back empty — never in a loop that could spin.
    """
    if bars < 1:
        raise ValueError(f"bars must be at least 1, got {bars!r}")

    symbol = strategy.binance_symbol(pair)     # strict <BASE>/USDT rules
    cursor = _now_ms() if end_ms is None else int(end_ms)

    collected: list[dict[str, Any]] = []
    for _ in range(MAX_CHUNKS):
        remaining = bars - len(collected)
        if remaining <= 0:
            break
        want = min(CHUNK, remaining)
        rows = _get_chunk(symbol, interval, want, cursor, _opener)
        if not rows:
            break

        chunk = _parse_rows(rows, symbol)
        collected.extend(chunk)

        earliest_open = min(c["open_time"] for c in chunk)
        # Step the cursor to just before the oldest bar we just took. Guard
        # against a payload that does not move the window, which would page
        # the same 1000 candles forever.
        next_cursor = earliest_open - 1
        if next_cursor >= cursor:
            break
        cursor = next_cursor

    # Oldest first, one row per open time, forming candle dropped.
    by_open: dict[int, dict[str, Any]] = {}
    for candle in collected:
        by_open[candle["open_time"]] = candle
    candles = [by_open[k] for k in sorted(by_open)]

    now = _now_ms()
    candles = [c for c in candles if c["close_time"] < now]

    if len(candles) < bars:
        # Short history is usually a bad pair or a bad interval, and it makes
        # every figure below quietly less meaningful. Say so out loud.
        print(
            f"warning: asked for {bars} candles, got {len(candles)}",
            file=sys.stderr,
        )

    return candles[-bars:]


# --------------------------------------------------------------------------- #
# Sandbox — never let a backtest touch the live paper ledger
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def sandboxed_engine():
    """Point the engine's state files at a temp dir for the duration."""
    saved = (te.LEDGER_PATH, te.TRADES_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        te.LEDGER_PATH = tmp_path / "paper_ledger.json"
        te.TRADES_PATH = tmp_path / "paper_trades.jsonl"
        try:
            yield tmp_path
        finally:
            te.LEDGER_PATH, te.TRADES_PATH = saved


def sandboxes_engine(func):
    """Make a function that resets the engine safe to call from anywhere.

    A replay resets the engine to open positions. A caller that only wants the
    number — a test, a summary, a parameter sweep — must not have the live
    paper ledger wiped as the price of asking. Nested sandboxes are harmless:
    an inner one saves the outer one's temp paths and restores them.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with sandboxed_engine():
            return func(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #

def _verify_windows(candles: list[dict[str, Any]], fast: int, slow: int) -> None:
    """Prove the sliding window is equivalent to the full prefix.

    ``evaluate`` only looks at the last two averages, so handing it the final
    ``slow + 1`` bars gives the same answer as the whole history at a fraction
    of the cost. That is an optimisation, and an optimisation that changes the
    answer is a bug — so it is checked against the slow path on every run.
    """
    need = slow + 1
    last = len(candles) - 1
    samples = sorted({need, (need + last) // 2, last - 1, last})
    for i in samples:
        if i < need - 1:
            continue
        full = strategy.evaluate(candles[: i + 1], fast=fast, slow=slow)
        window = strategy.evaluate(candles[i - slow: i + 1], fast=fast, slow=slow)
        for field in ("signal", "fast_ma", "slow_ma", "trend"):
            if full[field] != window[field]:
                raise BacktestError(
                    f"windowed evaluation disagrees with the full history at bar {i}: "
                    f"{field} {full[field]!r} vs {window[field]!r}"
                )


@sandboxes_engine
def run_backtest(
    candles: list[dict[str, Any]],
    *,
    pair: str = PAIR,
    size_usd: float = SIZE_USD,
    fast: int = strategy.MA_FAST,
    slow: int = strategy.MA_SLOW,
    start_usdt: float = START_USDT,
    interval: str = INTERVAL,
    verify: bool = True,
) -> dict[str, Any]:
    """Replay `candles` bar by bar, trading the crossover.

    Entry rule and exit rule match run_paper_trader.py exactly: BUY while flat
    spends `size_usd`; SELL while holding liquidates the whole position;
    anything else does nothing. No pyramiding, no partial exits, one position
    at a time.
    """
    need = slow + 1
    if len(candles) < need:
        raise BacktestError(f"need at least {need} candles, got {len(candles)}")
    if len(candles) < need + 1:
        # One bar to decide on, one bar to fill in. A replay that stops on the
        # decision bar can only ever produce zero trades, and reporting that as
        # a result would be a quiet lie.
        raise BacktestError(
            f"need at least {need + 1} candles to fill an order, got {len(candles)}"
        )
    if not size_usd > 0:
        raise BacktestError(f"size must be positive, got {size_usd!r}")
    if start_usdt < size_usd:
        raise BacktestError(f"starting cash {start_usdt} cannot fund a {size_usd} entry")

    if verify:
        _verify_windows(candles, fast, slow)

    pair_label = f"{pair}/USDT"
    te.reset(starting_usdt=start_usdt)

    cash = float(start_usdt)
    held = 0.0
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    open_entry: dict[str, Any] | None = None
    bars_in_market = 0
    skipped_no_next_bar = 0

    # A signal is an order queued on the close of the bar that produced it, and
    # it is worked at the next bar's open. None means nothing is working.
    pending: dict[str, Any] | None = None

    for i in range(need - 1, len(candles)):
        bar = candles[i]
        open_price = float(bar["open"])

        # 1. Work whatever was queued on the previous close, at this bar's
        #    open: the first price that existed once the signal was knowable.
        if pending is not None:
            side = pending["side"]
            if side == "BUY" and held == 0.0:
                fill = te.sim_buy(pair, size_usd, _price=open_price)
                cash = float(fill["usdt_after"])
                held = float(fill["position_after"])
                open_entry = {
                    "signal_bar": pending["signal_bar"],
                    "entry_bar": i,
                    "entry_time": bar["open_time"],
                    "entry_price": open_price,
                    "usdt_in": float(fill["usdt_amount"]),
                    "quantity": float(fill["token_amount"]),
                }
            elif side == "SELL" and held > 0.0:
                fill = te.sim_sell(pair, held, _price=open_price)
                cash = float(fill["usdt_after"])
                held = float(fill["position_after"])
                if open_entry is not None:
                    trades.append(
                        {
                            **open_entry,
                            "exit_signal_bar": pending["signal_bar"],
                            "exit_bar": i,
                            "exit_time": bar["open_time"],
                            "exit_price": open_price,
                            "usdt_out": float(fill["proceeds_usdt"]),
                            "pnl": float(fill["realized_pnl"]),
                            "bars_held": i - open_entry["entry_bar"],
                        }
                    )
                    open_entry = None
            pending = None

        # 2. Decide on this bar's close. The order it produces belongs to the
        #    next bar, so a signal on the final bar is dropped rather than
        #    filled at a close the strategy has already seen.
        window = candles[i - slow: i + 1]
        result = strategy.evaluate(
            window, fast=fast, slow=slow, pair=pair_label, interval=interval
        )
        price = float(bar["close"])
        signal = result["signal"]

        if i + 1 < len(candles):
            if signal == strategy.BUY and held == 0.0:
                pending = {"side": "BUY", "signal_bar": i}
            elif signal == strategy.SELL and held > 0.0:
                pending = {"side": "SELL", "signal_bar": i}
        elif (signal == strategy.BUY and held == 0.0) or (
            signal == strategy.SELL and held > 0.0
        ):
            # The trade existed as an order and never got a bar to fill in.
            # Counted rather than swallowed: it is the one cost of honesty in
            # this model, and on a short replay it can be the whole result.
            skipped_no_next_bar += 1

        if held > 0.0:
            bars_in_market += 1

        market_value = held * price
        equity_curve.append(
            {
                "bar": i,
                "time": bar["close_time"],
                "price": price,
                "cash": cash,
                "market_value": market_value,
                "equity": cash + market_value,
                "in_market": held > 0.0,
            }
        )

    # An order still queued at the final close never gets a bar to fill in, so
    # it is dropped — the alternative is filling at a close the signal was
    # derived from, which is the lookahead this whole model exists to remove.
    # Realise anything still open at the last close though: that is a liquidation
    # of a position already held, not a signal, so it stays. Same fee, same path.
    final_price = float(candles[-1]["close"])
    if held > 0.0:
        fill = te.sim_sell(pair, held, _price=final_price)
        cash = float(fill["usdt_after"])
        if open_entry is not None:
            trades.append(
                {
                    **open_entry,
                    "exit_bar": len(candles) - 1,
                    "exit_time": candles[-1]["close_time"],
                    "exit_price": final_price,
                    "usdt_out": float(fill["proceeds_usdt"]),
                    "pnl": float(fill["realized_pnl"]),
                    "bars_held": len(candles) - 1 - open_entry["entry_bar"],
                    "forced_close": True,
                }
            )
            open_entry = None
        held = 0.0
        equity_curve[-1]["equity"] = cash
        equity_curve[-1]["cash"] = cash
        equity_curve[-1]["market_value"] = 0.0

    ledger = te.load_ledger()
    return {
        "pair": pair_label,
        "interval": interval,
        "fast": fast,
        "slow": slow,
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
        "skipped_no_next_bar": skipped_no_next_bar,
        "fill_model": "next_open",
        "final_cash": cash,
        "realized_pnl": float(ledger["realized_pnl"]),
        "fees_paid": float(ledger["fees_paid"]),
        "fills": int(ledger["fills"]),
    }


@sandboxes_engine
def buy_and_hold(
    candles: list[dict[str, Any]],
    *,
    pair: str = PAIR,
    start_usdt: float = START_USDT,
) -> dict[str, Any]:
    """All the capital in at the first close, everything out at the last.

    Settled through the same engine functions, so both sides of the comparison
    pay the same fee.

    Sandboxes itself (see @sandboxes_engine): it resets the engine to open the
    position, and a caller that only wants the benchmark number — a test, a
    summary — must not have its live paper ledger wiped as the price of asking.
    """
    te.reset(starting_usdt=start_usdt)
    first_price = float(candles[0]["close"])
    last_price = float(candles[-1]["close"])

    buy = te.sim_buy(pair, start_usdt, _price=first_price)
    sell = te.sim_sell(pair, float(buy["token_amount"]), _price=last_price)

    return {
        "entry_price": first_price,
        "exit_price": last_price,
        "usdt_out": float(sell["proceeds_usdt"]),
        "pnl": float(sell["realized_pnl"]),
        "fees_paid": float(buy["fee_usdt"]) + float(sell["fee_usdt"]),
        "pnl_pct": (float(sell["proceeds_usdt"]) / start_usdt - 1.0) * 100.0,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def max_drawdown(equity_curve: list[dict[str, Any]]) -> dict[str, Any]:
    """Biggest peak-to-trough fall in equity, in USDT and in percent."""
    peak = float("-inf")
    worst_usdt = 0.0
    worst_pct = 0.0
    worst_at: int | None = None
    peak_at: int | None = None
    current_peak_bar = 0
    current_peak = float("-inf")

    for point in equity_curve:
        equity = float(point["equity"])
        if equity > current_peak:
            current_peak = equity
            current_peak_bar = point["bar"]
        drop = current_peak - equity
        if drop > worst_usdt:
            worst_usdt = drop
            worst_pct = (drop / current_peak * 100.0) if current_peak else 0.0
            worst_at = point["bar"]
            peak_at = current_peak_bar

    return {
        "max_drawdown_usdt": worst_usdt,
        "max_drawdown_pct": worst_pct,
        "peak_bar": peak_at,
        "trough_bar": worst_at,
    }


def summarise(result: dict[str, Any], hold: dict[str, Any]) -> dict[str, Any]:
    trades = result["trades"]
    pnls = [float(t["pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    final_equity = float(result["final_cash"])
    total_pnl = final_equity - result["start_usdt"]

    curve = result["equity_curve"]
    exposure = (
        sum(1 for p in curve if p["in_market"]) / len(curve) * 100.0 if curve else 0.0
    )
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    dd = max_drawdown(curve)

    return {
        "bars": result["bars"],
        "days": result["bars"] * interval_hours(result["interval"]) / 24.0,
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
        "gross_pnl_pct": (total_pnl + result["fees_paid"]) / result["start_usdt"] * 100.0,
        "fee_drag_pct": result["fees_paid"] / result["start_usdt"] * 100.0,
        "realized_pnl": result["realized_pnl"],
        "max_drawdown_usdt": dd["max_drawdown_usdt"],
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "bars_in_market": result["bars_in_market"],
        "exposure_pct": exposure,
        "buy_hold_pnl": hold["pnl"],
        "buy_hold_pnl_pct": hold["pnl_pct"],
        "difference_pct": (
            total_pnl / result["start_usdt"] * 100.0 - hold["pnl_pct"]
        ),
    }


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _fmt(value: float, places: int = 2) -> str:
    return f"{value:,.{places}f}"


def format_report(summary: dict[str, Any], result: dict[str, Any]) -> str:
    s = summary
    stamp = (
        f"backtest | {result['pair']} {result['interval']} | "
        f"MA {result['fast']}/{result['slow']} | ${_fmt(result['size_usd'])} per entry | "
        f"fee {te.FEE_RATE * 100:g}% per side | TRADING_ENABLED={te.TRADING_ENABLED}"
    )
    window = (
        f"window    {s['first_time']} -> {s['last_time']} UTC  "
        f"({s['bars']:,} bars, {s['days']:,.0f} days)"
    )
    price = (
        f"price     {result['first_price']:,.4f} -> {result['last_price']:,.4f}  "
        f"({(result['last_price'] / result['first_price'] - 1) * 100:+.2f}%)"
    )

    lines = [
        stamp,
        window,
        price,
        "",
        "results",
        f"  trades executed   {s['round_trips']} round trips ({s['fills']} fills)",
        f"  win rate          {s['win_rate']:.1f}%  ({s['wins']} won, {s['losses']} lost)",
        f"  total P&L         {s['total_pnl']:+,.2f} USDT  ({s['total_pnl_pct']:+.2f}%)",
        f"    before fees     {s['gross_pnl']:+,.2f} USDT  ({s['gross_pnl_pct']:+.2f}%)"
        f"   [fees cost {s['fee_drag_pct']:.2f} points]",
        f"  max drawdown      {s['max_drawdown_usdt']:,.2f} USDT  ({s['max_drawdown_pct']:.2f}%)",
        f"  buy & hold        {s['buy_hold_pnl']:+,.2f} USDT  ({s['buy_hold_pnl_pct']:+.2f}%)",
        f"  versus hold       {s['difference_pct']:+.2f} points",
        "",
        "detail",
        f"  starting equity   {s['start_usdt']:,.2f} USDT",
        f"  final equity      {s['final_equity']:,.2f} USDT",
        f"  fees paid         {s['fees_paid']:,.4f} USDT",
        f"  avg win / loss    {s['avg_win']:+.4f} / {s['avg_loss']:+.4f} USDT",
        f"  best / worst      {s['best_trade']:+.4f} / {s['worst_trade']:+.4f} USDT",
        "  profit factor     "
        + (f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "n/a (no losing trades)"),
        f"  time in market    {s['exposure_pct']:.1f}% of bars "
        f"({s['bars_in_market']:,} of {s['bars']:,})",
    ]

    if s["round_trips"] == 0:
        lines += [
            "",
            "verdict   the crossover never fired in this window — nothing to judge.",
        ]
    else:
        lines += [
            "",
            f"verdict   {_verdict(s)}",
        ]

    return "\n".join(lines)


def _verdict(s: dict[str, Any]) -> str:
    """One sentence, and only claims the numbers actually support."""
    pnl = s["total_pnl"]
    hold = s["buy_hold_pnl"]
    exposure = s["exposure_pct"]

    if pnl <= 0:
        return (
            f"the strategy lost {abs(pnl):,.2f} USDT over {s['round_trips']} round trips, "
            f"with a {s['win_rate']:.1f}% win rate — no edge here, just fees and noise."
        )
    if hold > 0 and pnl > hold:
        return (
            f"the strategy made {pnl:,.2f} USDT against hold's {hold:,.2f}, "
            f"but keep in mind it was only {exposure:.0f}% exposed — see below."
        )
    return (
        f"the strategy made {pnl:,.2f} USDT ({s['total_pnl_pct']:+.2f}%) over "
        f"{s['round_trips']} round trips at a {s['win_rate']:.1f}% win rate, "
        f"while buy & hold made {hold:,.2f} USDT fully invested."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest the MA crossover on historical candles (read-only)"
    )
    parser.add_argument("--symbol", default=PAIR, help=f"base asset (default {PAIR})")
    parser.add_argument("--days", type=float, default=None, help="days of history (default 365)")
    parser.add_argument("--bars", type=int, default=None, help="candles of history (overrides --days)")
    parser.add_argument(
        "--interval",
        default=INTERVAL,
        choices=sorted(INTERVAL_HOURS),
        help=f"candle size (default {INTERVAL})",
    )
    parser.add_argument("--size", type=float, default=SIZE_USD, help=f"USDT per entry (default {SIZE_USD})")
    parser.add_argument("--fast", type=int, default=strategy.MA_FAST)
    parser.add_argument("--slow", type=int, default=strategy.MA_SLOW)
    parser.add_argument("--start", type=float, default=START_USDT, help="starting cash")
    parser.add_argument("--trades", action="store_true", help="print every round trip")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    if te.TRADING_ENABLED:
        print("error: TRADING_ENABLED is True — this is a simulator", file=sys.stderr)
        return 2

    bar_hours = interval_hours(args.interval)

    if args.bars is not None:
        bars = args.bars
    elif args.days is not None:
        bars = int(round(args.days * 24 / bar_hours))
    elif args.interval == INTERVAL:
        bars = BARS
    else:
        bars = int(round(365 * 24 / bar_hours))

    if bars < args.slow + 2:
        print(
            f"error: {bars} bars cannot cover a {args.fast}/{args.slow} crossover "
            f"(need at least {args.slow + 2})",
            file=sys.stderr,
        )
        return 2

    try:
        candles = fetch_history(args.symbol, args.interval, bars)
    except (BacktestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if len(candles) < args.slow + 1:
        print(
            f"error: only {len(candles)} usable candles — not enough for {args.fast}/{args.slow}",
            file=sys.stderr,
        )
        return 1

    with sandboxed_engine():
        result = run_backtest(
            candles,
            pair=args.symbol,
            size_usd=args.size,
            fast=args.fast,
            slow=args.slow,
            start_usdt=args.start,
            interval=args.interval,
        )
        hold = buy_and_hold(candles, pair=args.symbol, start_usdt=args.start)

    summary = summarise(result, hold)

    if args.json:
        printable = {k: v for k, v in summary.items()}
        printable["trades"] = result["trades"]
        print(json.dumps(printable, indent=2))
        return 0

    print(format_report(summary, result))

    if args.trades:
        print("\nround trips")
        print(f"  {'#':>3}  {'entry':<16}  {'exit':<16}  {'in':>7}  {'out':>8}  {'P&L':>9}  {'bars':>4}")
        for n, trade in enumerate(result["trades"], start=1):
            print(
                f"  {n:>3}  {_iso(trade['entry_time']):<16}  {_iso(trade['exit_time']):<16}  "
                f"{trade['usdt_in']:>7.2f}  {trade['usdt_out']:>8.4f}  "
                f"{trade['pnl']:>+9.4f}  {trade['bars_held']:>4}"
                + ("  (closed at end)" if trade.get("forced_close") else "")
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
