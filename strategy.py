#!/usr/bin/env python3
"""strategy.py — Tree T, part three: the strategy engine.

Reads public 1-hour candles for one pair from Binance and turns them into a
single decision: BUY, SELL or HOLD, from a moving-average crossover.

    fast MA (10) crosses above slow MA (30)  ->  BUY
    fast MA (10) crosses below slow MA (30)  ->  SELL
    otherwise                                ->  HOLD

Simulation only. This module reads candles and returns a verdict; it holds no
keys, opens no positions and talks to nothing but the public market-data
endpoint. Acting on the verdict is the runner's job.

Why a crossover and not "fast above slow"
-----------------------------------------
"Fast is above slow" is a *state*, true for hours on end. "Fast has just
crossed above slow" is an *event*, true for exactly one candle. Only the
event is a trade signal — otherwise a single moving-average reading would be
re-bought on every loop iteration. ``evaluate`` compares the last two closed
candles and reports the crossing, plus the current state as ``trend`` for
display.

The forming candle is dropped
-----------------------------
Binance returns the candle that is still open as the last row. Its close is
just the current price, so it moves until the hour ends — a "crossover" on it
can appear and vanish a dozen times before the candle closes. Using it means
acting on a signal that may never have existed. ``fetch_klines`` requests one
extra candle and drops the open one, so every number here is final.

Usage
-----
    python3 strategy.py                  # human-readable read on BNB/USDT
    python3 strategy.py --json           # the same as a JSON object
    python3 strategy.py --symbol ETH     # any <BASE>/USDT pair
    python3 strategy.py --fast 5 --slow 20
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

INTERVAL = "1h"
CANDLE_LIMIT = 50       # candles of history the strategy asks for
MA_FAST = 10            # periods in the fast moving average
MA_SLOW = 30            # periods in the slow moving average
QUOTE = "USDT"          # the only quote currency this reads
HTTP_TIMEOUT = 10       # seconds

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


class MarketDataUnavailable(RuntimeError):
    """Binance did not give us usable candles."""


class StrategyError(RuntimeError):
    """The candles cannot produce a verdict (bad parameters, not enough data)."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clock(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M")


def _r(value: float, places: int = 8) -> float:
    return round(value, places) + 0.0


def binance_symbol(pair: str) -> str:
    """'BNB' / 'bnbusdt' / 'BNB/USDT' / 'BNB-USDT' -> 'BNBUSDT'.

    Strict on purpose, with the same rules the engine uses: only USDT pairs,
    and nothing is silently repaired. An earlier version split on "-" and
    appended USDT to whatever came first, so 'BTC-EUR' became BTCUSDT — a
    EUR request answered with a dollar quote, which is a wrong answer wearing
    the right shape.
    """
    raw = str(pair).strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("empty symbol")

    if "/" in raw or "-" in raw:
        sep = "/" if "/" in raw else "-"
        parts = raw.split(sep)
        if len(parts) != 2 or not all(parts) or parts[1] != QUOTE:
            raise ValueError(f"unsupported pair {pair!r}: <BASE>/{QUOTE} only")
        base = parts[0]
    elif raw.endswith(QUOTE) and len(raw) > len(QUOTE):
        base = raw[: -len(QUOTE)]
    else:
        base = raw

    if not base.isalnum():
        raise ValueError(f"bad ticker {pair!r}")
    return f"{base}{QUOTE}"


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #

def fetch_klines(
    pair: str = "BNB",
    interval: str = INTERVAL,
    limit: int = CANDLE_LIMIT,
    *,
    closed_only: bool = True,
    _opener=None,
) -> list[dict[str, Any]]:
    """Closed candles for <pair>, oldest first.

    With ``closed_only`` (the default) the still-forming final candle is
    dropped, so the last row is a final close rather than a moving one.
    Raises MarketDataUnavailable on any transport or payload problem — a
    broken feed is never quietly turned into an empty price series.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit!r}")

    pair_symbol = binance_symbol(pair)
    # Ask for one extra so that dropping the open candle still leaves `limit`.
    want = limit + 1 if closed_only else limit
    query = urllib.parse.urlencode(
        {"symbol": pair_symbol, "interval": interval, "limit": want}
    )
    request = urllib.request.Request(
        f"{BINANCE_KLINES_URL}?{query}",
        headers={"User-Agent": "paper-trader/1.0", "Accept": "application/json"},
        method="GET",
    )

    open_url = _opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=HTTP_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise MarketDataUnavailable(f"Binance HTTP {exc.code} for {pair_symbol}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MarketDataUnavailable(f"could not reach Binance for {pair_symbol}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MarketDataUnavailable(f"Binance sent non-JSON for {pair_symbol}: {exc}") from exc

    if not isinstance(payload, list) or not payload:
        raise MarketDataUnavailable(f"unexpected Binance payload for {pair_symbol}: {payload!r}"[:300])

    candles: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            raise MarketDataUnavailable(f"malformed candle for {pair_symbol}: {row!r}"[:300])
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
            raise MarketDataUnavailable(f"malformed candle for {pair_symbol}: {row!r}"[:300]) from exc
        if not candle["close"] > 0:
            raise MarketDataUnavailable(f"non-positive close for {pair_symbol}: {candle['close']!r}")
        candles.append(candle)

    candles.sort(key=lambda c: c["open_time"])
    # Binance does not repeat open times, but a duplicate would silently
    # weight one bar twice in every average, so drop any it sends.
    deduped: list[dict[str, Any]] = []
    for candle in candles:
        if deduped and candle["open_time"] == deduped[-1]["open_time"]:
            continue
        deduped.append(candle)

    if closed_only:
        now = _now_ms()
        deduped = [c for c in deduped if c["close_time"] < now]

    if not deduped:
        raise MarketDataUnavailable(f"no closed candles returned for {pair_symbol}")

    return deduped[-limit:]


# --------------------------------------------------------------------------- #
# Indicator
# --------------------------------------------------------------------------- #

def sma(values: list[float], period: int) -> list[float | None]:
    """Simple moving average, aligned to `values`.

    The first ``period - 1`` entries are None because no full window exists
    yet. Each window is summed fresh rather than carried with a running
    total: 50 values are cheap, and a drifting accumulator would make the
    crossover comparison depend on how many candles came before it.
    """
    if period < 1:
        raise ValueError(f"period must be at least 1, got {period!r}")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    for i in range(period - 1, len(values)):
        out[i] = sum(values[i - period + 1: i + 1]) / period
    return out


def evaluate(
    candles: list[dict[str, Any]],
    *,
    fast: int = MA_FAST,
    slow: int = MA_SLOW,
    pair: str | None = None,
    interval: str = INTERVAL,
) -> dict[str, Any]:
    """Turn candles into a verdict: BUY, SELL or HOLD.

    Pure function — no network, no disk. Needs ``slow + 1`` candles: one for
    each period in the slow average, plus one more so the average exists at
    two consecutive closes and a crossing can be seen.
    """
    if fast < 1 or slow < 1:
        raise StrategyError(f"periods must be positive, got fast={fast!r} slow={slow!r}")
    if fast >= slow:
        raise StrategyError(f"fast period ({fast}) must be shorter than slow period ({slow})")

    need = slow + 1
    if len(candles) < need:
        raise StrategyError(
            f"need at least {need} candles for a {fast}/{slow} crossover, got {len(candles)}"
        )

    closes = [float(c["close"]) for c in candles]
    fast_series = sma(closes, fast)
    slow_series = sma(closes, slow)

    fast_now, slow_now = fast_series[-1], slow_series[-1]
    fast_prev, slow_prev = fast_series[-2], slow_series[-2]
    if None in (fast_now, slow_now, fast_prev, slow_prev):
        raise StrategyError("not enough candles to place both averages at the last two closes")

    # Only a change in the relationship is a signal. "Fast is above slow" has
    # been true for hours by the time you read it; the crossing has not.
    crossed_up = fast_prev <= slow_prev and fast_now > slow_now
    crossed_down = fast_prev >= slow_prev and fast_now < slow_now
    signal = BUY if crossed_up else SELL if crossed_down else HOLD

    if fast_now > slow_now:
        trend = "UP"
    elif fast_now < slow_now:
        trend = "DOWN"
    else:
        trend = "FLAT"

    gap = fast_now - slow_now
    last = candles[-1]
    return {
        "signal": signal,
        "trend": trend,
        "crossed_up": crossed_up,
        "crossed_down": crossed_down,
        "pair": pair or "UNKNOWN",
        "interval": interval,
        "price": _r(closes[-1]),
        "fast": fast,
        "slow": slow,
        "fast_ma": _r(fast_now),
        "slow_ma": _r(slow_now),
        "prev_fast_ma": _r(fast_prev),
        "prev_slow_ma": _r(slow_prev),
        "gap": _r(gap),
        "gap_pct": _r(gap / slow_now * 100, 4) if slow_now else 0.0,
        "candles": len(candles),
        "candle_open_time": last["open_time"],
        "candle_close_time": last["close_time"],
        "candle_close_iso": _iso(last["close_time"]),
        "candle_close_clock": _clock(last["close_time"]),
    }


def describe(result: dict[str, Any]) -> str:
    """One terminal line: price, both averages, trend and verdict."""
    return (
        f"{result['pair']} {result['interval']}  "
        f"price {result['price']:.4f}  "
        f"fast({result['fast']}) {result['fast_ma']:.4f}  "
        f"slow({result['slow']}) {result['slow_ma']:.4f}  "
        f"gap {result['gap']:+.4f} ({result['gap_pct']:+.4f}%)  "
        f"trend {result['trend']}  "
        f"candle {result['candle_close_clock']}  "
        f"-> {result['signal']}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Moving-average crossover signal (read-only market data)"
    )
    parser.add_argument("--symbol", default="BNB", help="base asset, e.g. BNB (default)")
    parser.add_argument("--interval", default=INTERVAL, help=f"candle size (default {INTERVAL})")
    parser.add_argument("--limit", type=int, default=CANDLE_LIMIT, help="candles to request")
    parser.add_argument("--fast", type=int, default=MA_FAST, help=f"fast periods (default {MA_FAST})")
    parser.add_argument("--slow", type=int, default=MA_SLOW, help=f"slow periods (default {MA_SLOW})")
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON")
    parser.add_argument(
        "--include-open", action="store_true", help="use the forming candle too (repaints)"
    )
    args = parser.parse_args(argv)

    try:
        candles = fetch_klines(
            args.symbol,
            args.interval,
            args.limit,
            closed_only=not args.include_open,
        )
        result = evaluate(
            candles, fast=args.fast, slow=args.slow, pair=f"{binance_symbol(args.symbol)[:-4]}/USDT",
            interval=args.interval,
        )
    except (MarketDataUnavailable, StrategyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(describe(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
