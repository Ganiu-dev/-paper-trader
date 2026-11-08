#!/usr/bin/env python3
"""Backtest the MA crossover on real historical candles. Read-only.

Replays Binance 1h candles bar by bar through strategy.evaluate() and settles
every signal through trade_engine.sim_buy()/sim_sell() — the same functions,
the same fee arithmetic the live paper trader charges. There is no second copy
of the fee maths here to quietly drift away from the first one.

Read-only in the strict sense: the run happens inside sandboxed_engine(), which
points trade_engine.LEDGER_PATH and TRADES_PATH at a temp directory, so your
real paper_ledger.json and paper_trades.jsonl are never written. TRADING_ENABLED
stays False and the simulator's fence is still armed underneath.

Usage:
    python3 backtest.py                  # 365 days of BNB/USDT 1h
    python3 backtest.py --days 1095      # three years
    python3 backtest.py --bars 500 --trades
    python3 backtest.py --json

What it does NOT model:
  * Fills at the signal bar's close. The whole bar's move is already behind you
    when the signal fires, so entries are honest but never optimistic.
  * No slippage, spread, partial fills, funding or gaps. The fee is the only
    friction modelled — which is fine for $10 clips and would flatter a large
    size.
  * Buy & hold is the whole starting balance in at the first close. The
    strategy risks $10 at a time. That comparison is deliberately uncomfortable:
    it is the one a real allocator would make.
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import strategy
import trade_engine as te

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
PAIR = "BNB"
INTERVAL = "1h"
BARS = 8_760          # one year of hourly candles
CHUNK = 1_000         # Binance caps a klines request at 1000 rows
SIZE_USD = 10.0
START_USDT = 100.0
HTTP_TIMEOUT = 15
MAX_CHUNKS = 40       # hard stop so a bad cursor cannot loop forever
HOUR_MS = 3_600_000


class BacktestError(Exception):
    """The history could not be read, or the replay would be dishonest."""


# --------------------------------------------------------------------------- #
# Historical candles
# --------------------------------------------------------------------------- #

def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _parse_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise BacktestError(f"expected a list of candles, got {type(rows).__name__}")
    candles: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 7:
            raise BacktestError(f"malformed candle: {str(row)[:300]}")
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
            raise BacktestError(f"unparseable candle {str(row)[:200]}: {exc}") from exc
        if candle["close"] <= 0:
            raise BacktestError(f"non-positive close at {candle['open_time']}: {candle['close']}")
        candles.append(candle)
    return candles


def _get_chunk(
    symbol: str,
    interval: str,
    limit: int,
    end_ms: int | None = None,
    *,
    _opener=None,
) -> list[dict[str, Any]]:
    params = {"symbol": symbol, "interval": interval, "limit": str(limit)}
    if end_ms is not None:
        params["endTime"] = str(end_ms)
    url = f"{BINANCE_KLINES_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "paper-trader-backtest/1.0", "Accept": "application/json"},
        method="GET",
    )
    open_url = _opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=HTTP_TIMEOUT) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:200] if exc.fp else ""
        raise BacktestError(f"Binance HTTP {exc.code} for {symbol}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BacktestError(f"could not reach Binance: {exc}") from exc

    try:
        return _parse_rows(json.loads(payload.decode("utf-8")))
    except json.JSONDecodeError as exc:
        raise BacktestError(f"Binance sent non-JSON: {exc}") from exc


def fetch_history(
    pair: str = PAIR,
    interval: str = INTERVAL,
    bars: int = BARS,
    *,
    end_ms: int | None = None,
    _opener=None,
) -> list[dict[str, Any]]:
    """Page backwards until `bars` closed candles are collected, oldest first.

    Binance returns at most 1000 rows per call, so the cursor walks back by
    setting endTime to one millisecond before the earliest row we already have.
    """
    if bars < 1:
        raise BacktestError(f"bars must be at least 1, got {bars!r}")
    symbol = strategy.binance_symbol(pair)
    cursor = end_ms if end_ms is not None else _now_ms()
    collected: dict[int, dict[str, Any]] = {}
    chunks = 0

    while len(collected) < bars and chunks < MAX_CHUNKS:
        want = min(CHUNK, bars - len(collected))
        rows = _get_chunk(symbol, interval, want, cursor, _opener=_opener)
        chunks += 1
        if not rows:
            break

        for candle in rows:
            collected[candle["open_time"]] = candle

        earliest = m
