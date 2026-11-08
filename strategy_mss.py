#!/usr/bin/env python3
"""strategy_mss.py — Tree T, pure price action: Market Structure Shift breakouts.

Reads closed candles and turns them into one decision — BUY, SELL or HOLD —
from market structure rather than an indicator:

    close crosses ABOVE the most recent confirmed Swing High  ->  BUY   (bullish MSS)
    close crosses BELOW the most recent confirmed Swing Low   ->  SELL  (bearish MSS)
    close falls TRAIL_PCT below the peak reached since entry   ->  SELL  (trailing stop)
    otherwise                                                  ->  HOLD

Simulation only. This module reads candles and returns a verdict; it holds no
keys, opens no positions, and talks to nothing but the public market-data
endpoint. Acting on the verdict is the runner's or backtester's job.

The confirmation lag, and why it is the whole ballgame
------------------------------------------------------
A 2-right-bar fractal at bar ``j`` is a candle whose high beats the two bars
before it *and the two bars after it*. Those two bars do not exist yet at bar
``j``. The swing is only knowable once bar ``j + 2`` has closed, so that is
when this module starts using it.

That matters more than anything else in the file. The tempting implementation
is to scan the whole series for swing points, then walk it again asking "has
the close broken the most recent swing high?" — which quietly answers with a
swing high that could not have been confirmed yet. The strategy then buys
breakouts it could never have seen, and the backtest reports an edge that does
not exist. ``swing_levels`` therefore builds the level series bar by bar,
publishing a swing only on the close that confirms it, and
``test_strategy_mss.py`` proves the property directly: the level at bar ``i``
is identical whether you pass the whole history or only ``candles[:i + 1]``.

Event, not state
----------------
"Close is above the swing high" is a *state* that stays true for many bars.
"Close has just crossed above it" is an *event* that is true for one bar.
Only the event is a signal — otherwise every bar of a breakout is a fresh BUY,
and a stop-out would be immediately re-bought at the same price it died at.
``evaluate`` compares the last two closes, the same rule ``strategy.py`` uses
for its moving-average crossing, and for the same reason.

Trailing stop
-------------
The peak is the highest *high* since entry. It is advanced only after the
stop check for the current bar, so a bar that spikes up and then collapses
cannot use its own spike to widen the stop it is about to breach. That is
slightly conservative and refuses to assume an intrabar order it cannot know.

A stop is an order in the book, so it is checked against the bar's low and
fills at the stop level — unless the bar *opened* below it, in which case the
gap is real and the fill is the open. ``min(stop, open)`` covers both.

Usage
-----
    python3 strategy_mss.py                    # read on BNB/USDT 4h
    python3 strategy_mss.py --json
    python3 strategy_mss.py --symbol ETH --interval 1d
    python3 strategy_mss.py --trail 0.05       # a 5% trailing stop
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

INTERVAL = "4h"         # the timeframe this strategy was designed and tested on
CANDLE_LIMIT = 400      # history asked for: 400 x 4h is ~66 days, plenty of swings
SWING_LEFT = 2          # bars that must be lower than the swing candle, before it
SWING_RIGHT = 2         # bars that must be lower than the swing candle, after it
TRAIL_PCT = 0.03        # exit when close/low falls this far from the peak

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"

#: Why a signal fired. Kept separate from the signal so the report can say
#: whether a loss came from the stop or from structure turning over.
REASON_BULLISH_MSS = "BULLISH_MSS"
REASON_BEARISH_MSS = "BEARISH_MSS"
REASON_TRAILING_STOP = "TRAILING_STOP"
REASON_NONE = "NONE"


class StrategyError(RuntimeError):
    """The candles cannot produce a verdict (bad parameters, not enough data)."""


# --------------------------------------------------------------------------- #
# Swings
# --------------------------------------------------------------------------- #

def _r(value: float, places: int = 8) -> float:
    return round(value, places) + 0.0


def _iso(ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_swing_high(candles: list[dict[str, Any]], index: int, left: int = SWING_LEFT,
                  right: int = SWING_RIGHT) -> bool:
    """True if bar `index` is a 2-left / 2-right fractal high.

    Strict on purpose: a tie is *not* a swing. Two bars sharing the high mean
    the market did not reject a level, it paused at one, and counting the
    first as a swing high manufactures a breakout level the market never
    actually turned at.
    """
    if index - left < 0 or index + right >= len(candles):
        return False
    high = float(candles[index]["high"])
    for offset in range(1, left + 1):
        if high <= float(candles[index - offset]["high"]):
            return False
    for offset in range(1, right + 1):
        if high <= float(candles[index + offset]["high"]):
            return False
    return True


def is_swing_low(candles: list[dict[str, Any]], index: int, left: int = SWING_LEFT,
                 right: int = SWING_RIGHT) -> bool:
    """True if bar `index` is a 2-left / 2-right fractal low. Ties do not count."""
    if index - left < 0 or index + right >= len(candles):
        return False
    low = float(candles[index]["low"])
    for offset in range(1, left + 1):
        if low >= float(candles[index - offset]["low"]):
            return False
    for offset in range(1, right + 1):
        if low >= float(candles[index + offset]["low"]):
            return False
    return True


def find_swings(candles: list[dict[str, Any]], left: int = SWING_LEFT,
                right: int = SWING_RIGHT) -> list[dict[str, Any]]:
    """Every confirmed fractal in `candles`, oldest first.

    Each entry carries ``confirm_index`` — the bar whose close made the swing
    knowable. Using the swing before that bar is lookahead; ``swing_levels``
    and the backtester both key off this field, never off ``index`` alone.
    """
    swings: list[dict[str, Any]] = []
    for i in range(len(candles)):
        if is_swing_high(candles, i, left, right):
            swings.append(
                {
                    "kind": "high",
                    "index": i,
                    "confirm_index": i + right,
                    "price": _r(float(candles[i]["high"])),
                    "time": candles[i]["close_time"],
                    "confirm_time": candles[i + right]["close_time"],
                }
            )
        if is_swing_low(candles, i, left, right):
            swings.append(
                {
                    "kind": "low",
                    "index": i,
                    "confirm_index": i + right,
                    "price": _r(float(candles[i]["low"])),
                    "time": candles[i]["close_time"],
                    "confirm_time": candles[i + right]["close_time"],
                }
            )
    swings.sort(key=lambda s: (s["confirm_index"], s["kind"]))
    return swings


def swing_levels(candles: list[dict[str, Any]], left: int = SWING_LEFT,
                 right: int = SWING_RIGHT) -> list[dict[str, Any]]:
    """The most recent *confirmed* swing high/low, as of the close of each bar.

    One entry per bar. ``levels[i]`` holds what a trader could have known at
    the close of bar ``i`` and nothing later. Bar ``i`` can only confirm the
    candidate at ``i - right``, so a single forward pass is enough and there
    is no window that could reach past ``i``.
    """
    if left < 1 or right < 1:
        raise StrategyError(f"swing windows must be >= 1, got left={left} right={right}")

    levels: list[dict[str, Any]] = []
    last_high: dict[str, Any] | None = None
    last_low: dict[str, Any] | None = None

    for i in range(len(candles)):
        candidate = i - right
        if candidate - left >= 0:
            if is_swing_high(candles, candidate, left, right):
                last_high = {
                    "price": _r(float(candles[candidate]["high"])),
                    "index": candidate,
                    "time": candles[candidate]["close_time"],
                }
            if is_swing_low(candles, candidate, left, right):
                last_low = {
                    "price": _r(float(candles[candidate]["low"])),
                    "index": candidate,
                    "time": candles[candidate]["close_time"],
                }

        levels.append(
            {
                "swing_high": last_high["price"] if last_high else None,
                "swing_high_index": last_high["index"] if last_high else None,
                "swing_high_time": last_high["time"] if last_high else None,
                "swing_low": last_low["price"] if last_low else None,
                "swing_low_index": last_low["index"] if last_low else None,
                "swing_low_time": last_low["time"] if last_low else None,
            }
        )

    return levels


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #

def trailing_stop_price(peak_price: float, trail_pct: float = TRAIL_PCT) -> float:
    """The stop level implied by a peak: `peak * (1 - trail_pct)`."""
    if not peak_price > 0:
        raise StrategyError(f"peak must be positive, got {peak_price!r}")
    if not 0 < trail_pct < 1:
        raise StrategyError(f"trail_pct must be in (0, 1), got {trail_pct!r}")
    return _r(float(peak_price) * (1.0 - float(trail_pct)))


def evaluate(
    candles: list[dict[str, Any]],
    *,
    position: dict[str, Any] | None = None,
    trail_pct: float = TRAIL_PCT,
    left: int = SWING_LEFT,
    right: int = SWING_RIGHT,
    pair: str | None = None,
    interval: str = INTERVAL,
    levels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Turn candles into a verdict: BUY, SELL or HOLD.

    Pure function - no network, no disk. ``position`` is ``None`` when flat, or
    ``{"entry_price": float, "peak_price": float}`` when long; without it the
    trailing stop cannot be evaluated, because the stop depends on the peak
    reached since entry and that is not a property of the candles alone.

    ``levels`` is an optional precomputed ``swing_levels(candles, left, right)``.
    A backtest calling this once per bar would otherwise rebuild the whole
    level series every bar and turn a linear replay into an O(n^2) one. The
    caller must pass levels computed from *these* candles with *these* left and
    right values; a length mismatch is an error rather than a silent fallback.
    """
    need = left + right + 1
    if len(candles) < need:
        raise StrategyError(
            f"need at least {need} candles for a {left}/{right} fractal, got {len(candles)}"
        )
    if position is not None:
        peak = float(position.get("peak_price") or 0.0)
        if peak <= 0:
            raise StrategyError(f"position needs a positive peak_price, got {position!r}")

    if levels is None:
        levels = swing_levels(candles, left, right)
    elif len(levels) != len(candles):
        raise StrategyError(
            f"precomputed levels cover {len(levels)} bars but got {len(candles)} candles "
            "- they must come from this candle series"
        )
    i = len(candles) - 1
    bar = candles[i]
    prev_close = float(candles[i - 1]["close"])
    close = float(bar["close"])
    level = levels[i]

    signal, reason = HOLD, REASON_NONE
    fill_price: float | None = None
    stop: float | None = None
    new_peak: float | None = None

    if position is not None:
        # 1. The stop is an order in the book: it fires intrabar, off the peak
        #    as it stood at the previous close. A gap through it fills at the
        #    open, because that is the first price available.
        stop = trailing_stop_price(peak, trail_pct)
        if float(bar["low"]) <= stop:
            signal, reason = SELL, REASON_TRAILING_STOP
            fill_price = min(stop, float(bar["open"]))
        # 2. Structure turning over needs the close, so it is checked second.
        elif (
            level["swing_low"] is not None
            and prev_close >= float(level["swing_low"])
            and close < float(level["swing_low"])
        ):
            signal, reason = SELL, REASON_BEARISH_MSS
            fill_price = close
        else:
            new_peak = _r(max(peak, float(bar["high"])))
    else:
        if (
            level["swing_high"] is not None
            and prev_close <= float(level["swing_high"])
            and close > float(level["swing_high"])
        ):
            signal, reason = BUY, REASON_BULLISH_MSS
            fill_price = close

    return {
        "signal": signal,
        "reason": reason,
        "pair": pair or "UNKNOWN",
        "interval": interval,
        "price": _r(close),
        "prev_close": _r(prev_close),
        "swing_high": level["swing_high"],
        "swing_low": level["swing_low"],
        "stop_price": stop,
        "peak_price": _r(float(new_peak)) if new_peak is not None else (
            _r(float(position["peak_price"])) if position is not None else None
        ),
        "trail_pct": trail_pct,
        "fill_price": _r(float(fill_price)) if fill_price is not None else None,
        "candles": len(candles),
        "candle_open_time": b
