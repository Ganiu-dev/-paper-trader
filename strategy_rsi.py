#!/usr/bin/env python3
"""strategy_rsi.py — Tree T, part four: mean reversion on RSI.

Trend-following failed here twice. MA crossovers and MSS breakouts both need a
market that keeps going; BNB chops, and both strategies bought the top of each
chop and sold the bottom. So this module bets the other way — that an extreme
reading gets pulled back toward the middle:

    RSI(14) crosses DOWN through 30,
      price above the 200-period MA      ->  BUY   (oversold inside an uptrend)
    RSI(14) crosses UP through 70        ->  SELL  (overbought: sell the euphoria)
    RSI(14) falls back DOWN through 50   ->  SELL  (midline: momentum faded)
    price trades 5% below the entry      ->  SELL  (hard stop: the crash case)
    otherwise                            ->  HOLD

Simulation only. This module reads candles and returns a verdict; it holds no
keys, opens no positions and talks to nothing but the public market-data
endpoint. Acting on the verdict is the runner's or backtester's job.

Event, not state — and why it matters more here than anywhere
-------------------------------------------------------------
"RSI is below 30" is a *state*. It stays true for as long as the market keeps
falling, which on a real crash is days. "RSI has just crossed below 30" is an
*event*, true for exactly one bar. Only the event is a signal. Take the state
as the signal and the strategy buys every single bar of a waterfall — each one
a fresh $10 into a falling knife, all of them eventually stopped out at 5% for
a cumulative loss that has nothing to do with the strategy's actual edge.

Both sides are crossings for the same reason, and ``evaluate`` compares the
last two RSI readings to find them.

Which way does RSI "return to 50"?
----------------------------------
The brief says exit when RSI returns to the midline. Read literally that is
ambiguous, and the two readings are different strategies:

* ``down`` (the default): RSI climbed, then fell back through 50 — the bounce
  stalled. Exit at the midline, near break-even, and keep the capital. This
  lets the trade run to 70 for the full "sell the euphoria" target, and gets
  out flat when momentum dies instead of round-tripping a winner.
* ``up``: RSI normalised back up through 50 after the oversold reading — the
  reversion has happened. Exit there.

``down`` is the default because it is the only one under which the 70 rule is
reachable at all: a long entered below 30 must pass *up* through 50 long before
it can reach 70, so an up-cross exit ends every trade at the midline and the
overbought rule never fires. That would make one of the four stated rules dead
code. ``--midline up`` is there for the other reading; ``either`` takes
whichever comes first.

The hard stop is an order in the book
-------------------------------------
The 5% stop does not wait for a close. It is checked against the bar's low and
fills at the stop price — unless the bar *opened* below it, in which case the
gap is real and the fill is the open, because that is the first price anyone
could have got. ``min(stop, open)`` covers both. A mean-reversion strategy
without this is short a crash.

The trend filter (T5)
---------------------
The split-half test showed the edge is real but small, and that it was made
almost entirely in chop: the strategy beat buy-and-hold by 3.9 points in the
flat test window and lost to it by 207 points in the trending fit window. That
is the opportunity cost of mean reversion — it sits out the rallies.

This module adds one rule, and only to the entry:

    close above the 200-period simple moving average  ->  the BUY is allowed

Everything else is untouched. The exits stay as they were, because the point is
to stop *entering* into downtrends, not to change how a trade is managed once
it exists.

It is deliberately one rule. A filter that also tightened the stop, or moved
the exit, would confound the result: a better number could then come from the
stop rather than the trend, and the test would not be able to say which. One
change at a time is what makes the comparison mean something.

A filter that cannot be evaluated is an error, not a pass. During the first
``ma_period`` bars the average does not exist, and a strategy that treats a
missing trend as "probably fine" would take exactly the trades this change is
meant to remove. ``evaluate`` raises instead, and the backtest starts scoring
only once the average is real.

Wilder's smoothing, and the warm-up
-----------------------------------
RSI uses Wilder's original recursive average: the first value is a simple mean
of the first ``period`` changes, and every value after it is
``(prev * (period - 1) + today) / period``. That makes each reading depend on
every close before it, so a reading is only final once ``period + 1`` closes
exist. ``rsi_series`` returns ``None`` for the warm-up bars rather than a
number built from half a window, and ``evaluate`` refuses to read one.

Usage
-----
    python3 strategy_rsi.py                     # read on BNB/USDT 4h
    python3 strategy_rsi.py --json
    python3 strategy_rsi.py --symbol ETH --interval 1d
    python3 strategy_rsi.py --entry 690.00      # assume long, so the stop is evaluated
    python3 strategy_rsi.py --midline up        # the other reading of "returns to 50"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import strategy

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

INTERVAL = "4h"          # the timeframe this strategy was designed and tested on
CANDLE_LIMIT = 400       # ~66 days of 4h bars: plenty of RSI swings
PERIOD = 14              # Wilder's RSI period
OVERSOLD = 30.0          # BUY when RSI crosses below this
OVERBOUGHT = 70.0        # SELL when RSI crosses above this
MIDLINE = 50.0           # exit when RSI returns here
STOP_PCT = 0.05          # hard stop, as a fraction below entry
MIDLINE_MODE = "down"    # "down" | "up" | "either" — see the docstring
MA_PERIOD = 200          # trend filter: only buy above this many-period SMA
USE_TREND_FILTER = True  # False = the pre-T5 behaviour, for the comparison

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"

#: Why a signal fired. Kept apart from the signal so the report can say whether
#: a loss came from the hard stop or from momentum quietly dying.
REASON_OVERSOLD = "RSI_OVERSOLD"
REASON_OVERBOUGHT = "RSI_OVERBOUGHT"
REASON_MIDLINE = "RSI_MIDLINE"
REASON_STOP_LOSS = "STOP_LOSS"
REASON_TREND_BLOCK = "TREND_FILTER"
REASON_NONE = "NONE"

MIDLINE_MODES = ("down", "up", "either")

class StrategyError(RuntimeError):
    """The candles cannot produce a verdict (bad parameters, not enough data)."""

# --------------------------------------------------------------------------- #
# RSI
# --------------------------------------------------------------------------- #

def _r(value: float, places: int = 8) -> float:
    return round(value, places) + 0.0

def _iso(ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    """RSI from Wilder's two averages.

    Deliberately *not* rounded. Rounding to 2dp before the 30/70 comparison
    turns 29.996 into 30.00 and swallows a real crossing; the rounded value is
    for the report, never for the decision.
    """
    if avg_gain <= 0.0 and avg_loss <= 0.0:
        # A dead-flat window is neither overbought nor oversold. 50 is the
        # neutral answer; 100 or 0 would manufacture an extreme out of silence.
        return 50.0
    if avg_loss <= 0.0:
        return 100.0
    if avg_gain <= 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def rsi_series(closes: list[float], period: int = PERIOD) -> list[float | None]:
    """Wilder's RSI for every bar, oldest first. ``None`` during the warm-up.

    ``rsi[i]`` is computed from ``closes[: i + 1]`` and nothing after it, so the
    series is safe to precompute once for a backtest and read bar by bar —
    which is exactly what ``backtest_rsi.py`` does. The property is asserted
    in ``test_strategy_rsi.py``: the value at bar ``i`` is identical whether
    you pass the whole history or only the prefix up to ``i``.
    """
    if period < 1:
        raise StrategyError(f"period must be at least 1, got {period!r}")

    n = len(closes)
    out: list[float | None] = [None] * n
    if n < period + 1:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = float(closes[i]) - float(closes[i - 1])
        if change > 0.0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)

    for i in range(period + 1, n):
        change = float(closes[i]) - float(closes[i - 1])
        gain = change if change > 0.0 else 0.0
        loss = -change if change < 0.0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)

    return out

def stop_price(entry_price: float, stop_pct: float = STOP_PCT) -> float:
    """The hard stop implied by an entry: ``entry * (1 - stop_pct)``."""
    if not entry_price > 0:
        raise StrategyError(f"entry must be positive, got {entry_price!r}")
    if not 0 < stop_pct < 1:
        raise StrategyError(f"stop_pct must be in (0, 1), got {stop_pct!r}")
    return _r(float(entry_price) * (1.0 - float(stop_pct)))

def moving_average(closes: list[float], period: int = MA_PERIOD) -> list[float | None]:
    """Simple moving average for every bar, oldest first. ``None`` in warm-up.

    Like ``rsi_series``, ``ma[i]`` uses only ``closes[: i + 1]``, so a backtest
    can compute it once and read it bar by bar without looking ahead.

    A simple mean, not an exponential one: the brief says "200-period Moving
    Average", and an EMA weights the last few bars hard enough that a 200-EMA
    can be crossed by a bounce that a 200-SMA would still call a downtrend.
    That difference matters for a filter whose whole job is to be slow.
    """
    if period < 1:
        raise StrategyError(f"ma_period must be at least 1, got {period!r}")

    n = len(closes)
    out: list[float | None] = [None] * n
    if n < period:
        return out

    window = 0.0
    for i in range(period):
        window += float(closes[i])
    out[period - 1] = _r(window / period)

    for i in range(period, n):
        window += float(closes[i]) - float(closes[i - period])
        out[i] = _r(window / period)

    return out

# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #

def _validate(period: int, oversold: float, overbought: float, midline: float,
              stop_pct: float, midline_mode: str, ma_period: int = MA_PERIOD,
              use_trend_filter: bool = USE_TREND_FILTER) -> None:
    if period < 1:
        raise StrategyError(f"period must be at least 1, got {period!r}")
    if not 0.0 < oversold < 100.0:
        raise StrategyError(f"oversold must be in (0, 100), got {oversold!r}")
    if not 0.0 < overbought < 100.0:
        raise StrategyError(f"overbought must be in (0, 100), got {overbought!r}")
    if not oversold < overbought:
        raise StrategyError(
            f"oversold ({oversold}) must be below overbought ({overbought})"
        )
    if not oversold < midline < overbought:
        raise StrategyError(
            f"midline ({midline}) must sit between {oversold} and {overbought}"
        )
    if not 0 < stop_pct < 1:
        raise StrategyError(f"stop_pct must be in (0, 1), got {stop_pct!r}")
    if midline_mode not in MIDLINE_MODES:
        raise StrategyError(
            f"midline_mode must be one of {', '.join(MIDLINE_MODES)}, got {midline_mode!r}"
        )
    if ma_period < 1:
        raise StrategyError(f"ma_period must be at least 1, got {ma_period!r}")
    if use_trend_filter and ma_period < 2:
        # A 1-period "average" is just the close, so the filter would compare
        # the close with itself and always pass. That is not a filter.
        raise StrategyError(
            f"a 1-period moving average is the close itself — "
            f"ma_period must be at least 2 (got {ma_period!r})"
        )

def evaluate(
    candles: list[dict[str, Any]],
    *,
    position: dict[str, Any] | None = None,
    period: int = PERIOD,
    oversold: float = OVERSOLD,
    overbought: float = OVERBOUGHT,
    midline: float = MIDLINE,
    stop_pct: float = STOP_PCT,
    midline_mode: str = MIDLINE_MODE,
    ma_period: int = MA_PERIOD,
    use_trend_filter: bool = USE_TREND_FILTER,
    pair: str | None = None,
    interval: str = INTERVAL,
    rsi: list[float | None] | None = None,
    ma: list[float | None] | None = None,
) -> dict[str, Any]:
    """Turn candles into a verdict: BUY, SELL or HOLD.

    Pure function — no network, no disk. ``position`` is ``None`` when flat, or
    ``{"entry_price": float}`` when long; without it the hard stop cannot be
    evaluated, because the stop depends on where the trade was opened and that
    is not a property of the candles.

    ``rsi`` is an optional precomputed ``rsi_series(closes, period)``. A
    backtest calling this once per bar would otherwise rebuild the whole series
    every bar and turn a linear replay into an O(n^2) one. The caller must pass
    a series computed from *these* closes with *this* period; a length
    mismatch is an error rather than a silent fallback.

    ``ma`` is the same idea for the trend filter's ``moving_average``.

    ``use_trend_filter`` gates *entries only*. A BUY that the 200-MA would
    block comes back as HOLD with reason ``TREND_FILTER``, so a caller can count
    what the filter removed instead of guessing. Exits ignore it entirely: a
    trade already open is managed by the rules that opened it.
    """
    _validate(period, oversold, overbought, midline, stop_pct, midline_mode,
              ma_period, use_trend_filter)

    need = period + 2                      # both RSI readings must be final
    if use_trend_filter:
        need = max(need, ma_period)        # and the average must exist
    if len(candles) < need:
        raise StrategyError(
            f"need at least {need} candles for a {period}-period RSI"
            + (f" and a {ma_period}-period MA" if use_trend_filter else "")
            + f", got {len(candles)}"
        )
    if position is not None:
        entry = float(position.get("entry_price") or 0.0)
        if entry <= 0:
            raise StrategyError(
                f"position needs a positive entry_price, got {position!r}"
            )

    closes = [float(c["close"]) for c in candles]
    if rsi is None:
        rsi = rsi_series(closes, period)
    elif len(rsi) != len(candles):
        raise StrategyError(
            f"precomputed RSI covers {len(rsi)} bars but got {len(candles)} candles "
            "- they must come from this candle series"
        )

    i = len(candles) - 1
    bar = candles[i]
    prev_close = float(candles[i - 1]["close"])
    close = closes[i]
    now = rsi[i]
    prev = rsi[i - 1]
    if now is None or prev is None:
        raise StrategyError(
            f"RSI is still warming up at bar {i}: needs {period + 1} closes, "
            f"have {sum(1 for v in rsi[: i + 1] if v is not None)}"
        )

    ma_now: float | None = None
    if use_trend_filter:
        if ma is None:
            ma = moving_average(closes, ma_period)
        elif len(ma) != len(candles):
            raise StrategyError(
                f"precomputed MA covers {len(ma)} bars but got {len(candles)} candles "
                "- they must come from this candle series"
            )
        ma_now = ma[i]
        if ma_now is None:
            raise StrategyError(
                f"the {ma_period}-period MA is still warming up at bar {i}: "
                f"needs {ma_period} closes, have {i + 1}"
            )

    in_uptrend = ma_now is not None and close > ma_now

    signal, reason = HOLD, REASON_NONE
    fill_price: float | None = None
    stop: float | None = None

    if position is not None:
        entry = float(position["entry_price"])
        # 1. The stop is an order in the book: it fires intrabar. A gap through
        #    it fills at the open, the first price actually available.
        stop = stop_price(entry, stop_pct)
        if float(bar["low"]) <= stop:
            signal, reason = SELL, REASON_STOP_LOSS
            fill_price = min(stop, float(bar["open"]))
        # 2. Overbought: the euphoria target. Needs the close, so it is second.
        elif prev < overbought <= now:
            signal, reason = SELL, REASON_OVERBOUGHT
            fill_price = close
        # 3. Midline: momentum failed to carry. Also close-based.
        elif midline_mode in ("down", "either") and prev >= midline > now:
            signal, reason = SELL, REASON_MIDLINE
            fill_price = close
        elif midline_mode in ("up", "either") and prev <= midline < now:
            signal, reason = SELL, REASON_MIDLINE
            fill_price = close
    else:
        # Buy the panic — only on the crossing bar, never on the state.
        if prev >= oversold > now:
            if use_trend_filter and not in_uptrend:
                # The crossing is real; the trend says do not take it. Reported
                # as a blocked BUY rather than swallowed into a plain HOLD, so
                # the backtest can say how many entries the rule removed.
                signal, reason = HOLD, REASON_TREND_BLOCK
            else:
                signal, reason = BUY, REASON_OVERSOLD
                fill_price = close

    return {
        "signal": signal,
        "reason": reason,
        "pair": pair or "UNKNOWN",
        "interval": interval,
        "price": _r(close),
        "prev_close": _r(prev_close),
        "rsi": round(now, 2),
        "prev_rsi": round(prev, 2),
        "period": period,
        "oversold": oversold,
        "overbought": overbought,
        "midline": midline,
        "midline_mode": midline_mode,
        "stop_pct": stop_pct,
        "ma_period": ma_period if use_trend_filter else None,
        "ma": _r(ma_now) if ma_now is not None else None,
        "above_ma": in_uptrend if use_trend_filter else None,
        "use_trend_filter": use_trend_filter,
        "stop_price": stop,
        "entry_price": float(position["entry_price"]) if position is not None else None,
        "fill_price": _r(float(fill_price)) if fill_price is not None else None,
        "candles": len(candles),
        "candle_open_time": bar["open_time"],
        "candle_close_time": bar["close_time"],
        "candle_close_iso": _iso(bar["close_time"]),
    }

def describe(result: dict[str, Any]) -> str:
    """One terminal line: price, RSI, the thresholds, the stop, and the verdict."""
    stop = f"{result['stop_price']:,.4f}" if result["stop_price"] is not None else "n/a"
    why = "" if result["reason"] == REASON_NONE else f" ({result['reason']})"
    trend = ""
    if result.get("use_trend_filter") and result.get("ma") is not None:
        trend = (
            f"MA{result['ma_period']} {result['ma']:,.2f} "
            f"({'above' if result['above_ma'] else 'below'})  "
        )
    return (
        f"{result['pair']} {result['interval']}  "
        f"price {result['price']:,.4f}  "
        f"RSI{result['period']} {result['rsi']:.2f} "
        f"(prev {result['prev_rsi']:.2f}, "
        f"{result['oversold']:g}/{result['overbought']:g})  "
        f"{trend}"
        f"stop {stop}  "
        f"candle {result['candle_close_iso']}  "
        f"-> {result['signal']}{why}"
    )

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RSI(14) mean-reversion signal (read-only market data)"
    )
    parser.add_argument("--symbol", default="BNB", help="base asset, e.g. BNB (default)")
    parser.add_argument("--interval", default=INTERVAL, help=f"candle size (default {INTERVAL})")
    parser.add_argument("--limit", type=int, default=CANDLE_LIMIT, help="candles to request")
    parser.add_argument("--period", type=int, default=PERIOD, help=f"RSI period (default {PERIOD})")
    parser.add_argument("--oversold", type=float, default=OVERSOLD,
                        help=f"BUY below this (default {OVERSOLD:g})")
    parser.add_argument("--overbought", type=float, default=OVERBOUGHT,
                        help=f"SELL above this (default {OVERBOUGHT:g})")
    parser.add_argument("--midline", type=float, default=MIDLINE,
                        help=f"midline exit level (default {MIDLINE:g})")
    parser.add_argument("--midline-mode", default=MIDLINE_MODE, choices=MIDLINE_MODES,
                        help=f"which crossing of the midline exits (default {MIDLINE_MODE})")
    parser.add_argument("--stop", type=float, default=STOP_PCT,
                        help=f"hard stop as a fraction (default {STOP_PCT})")
    parser.add_argument("--ma-period", type=int, default=MA_PERIOD,
                        help=f"trend-filter moving-average period (default {MA_PERIOD})")
    parser.add_argument("--no-trend-filter", dest="use_trend_filter", action="store_false",
                        help="disable the 200-MA entry filter (the pre-T5 behaviour)")
    parser.add_argument("--entry", type=float, default=None,
                        help="entry price; implies you are long, so the stop is evaluated")
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON")
    args = parser.parse_args(argv)

    try:
        position = {"entry_price": args.entry} if args.entry is not None else None
        # The CLI's default candle count is smaller than the 200-MA warm-up, so
        # widen the request when the filter is on. Otherwise asking for a read
        # would fail on a fetch size rather than say anything about the market.
        limit = args.limit
        if args.use_trend_filter:
            limit = max(limit, args.ma_period + 50)
        candles = strategy.fetch_klines(
            args.symbol, args.interval, limit, closed_only=True
        )
        result = evaluate(
            candles,
            position=position,
            period=args.period,
            oversold=args.oversold,
            overbought=args.overbought,
            midline=args.midline,
            stop_pct=args.stop,
            midline_mode=args.midline_mode,
            ma_period=args.ma_period,
            use_trend_filter=args.use_trend_filter,
            pair=f"{strategy.binance_symbol(args.symbol)[:-4]}/USDT",
            interval=args.interval,
        )
    except (strategy.MarketDataUnavailable, StrategyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(describe(result))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())j
