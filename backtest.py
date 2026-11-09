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
