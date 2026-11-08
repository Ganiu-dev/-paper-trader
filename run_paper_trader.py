#!/usr/bin/env python3
"""run_paper_trader.py — Tree T, part four: the loop that ties it together.

Every interval this reads the market, asks strategy.py for a verdict, and
acts on it through trade_engine.py:

    BUY  and flat        -> sim_buy(pair, SIZE_USD)
    SELL and holding     -> sim_sell(pair, everything held)
    anything else        -> do nothing, say why

Simulation only. TRADING_ENABLED stays False, so the engine's own guard
refuses to move money even if this script asks it to.

    python3 run_paper_trader.py                # loop forever, 60s apart
    python3 run_paper_trader.py --once         # one check, then exit
    python3 run_paper_trader.py --dry-run      # decide, print, trade nothing
    python3 run_paper_trader.py --interval 300

Why it will not buy twice in an hour
------------------------------------
A crossover is an event on one closed candle, but that candle stays "the last
closed candle" until the next one closes — so the same BUY reads true on every
60-second pass for up to an hour. Two guards stop that becoming a buying
spree: a position check (already long, so nothing to do) and a latch on
(candle close time, signal) that records what has already been acted on. The
position check alone would be enough most of the time; the latch is what makes
the terminal output honest about why it sat still.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import strategy
import trade_engine as te

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PAIR = "BNB"
SIZE_USD = 10.0          # dollars per entry, as specified
INTERVAL_SECONDS = 60    # how often to look
STATE_PATH = Path(__file__).resolve().parent / "paper_runner_state.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt(value: float | None, places: int = 4) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.{places}f}"
    return f"{value:.8f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- #
# Latch — which signal on which candle has already been acted on
# --------------------------------------------------------------------------- #

def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # A missing or unreadable latch costs at most one repeated decision,
        # which the position check still catches. Not worth failing the run.
        return {"acted": []}


def save_state(state: dict[str, Any]) -> None:
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")
    except OSError as exc:
        print(f"warning: could not save runner state: {exc}", file=sys.stderr)


def _key(result: dict[str, Any]) -> str:
    return f"{result['pair']}|{result['candle_close_time']}|{result['signal']}"


# --------------------------------------------------------------------------- #
# One decision
# --------------------------------------------------------------------------- #

def held_amount(pair: str = PAIR) -> float:
    """Tokens currently held in the paper ledger, 0.0 if flat."""
    ledger = te.load_ledger()
    position = ledger["positions"].get(f"{pair}/USDT")
    return float(position["token_amount"]) if position else 0.0


def run_once(
    *,
    pair: str = PAIR,
    size_usd: float = SIZE_USD,
    fast: int = strategy.MA_FAST,
    slow: int = strategy.MA_SLOW,
    dry_run: bool = False,
    state: dict[str, Any] | None = None,
    candles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch, decide, act — once. Returns a record of what happened and why.

    ``candles`` is injectable so this can be tested against fixed data with no
    network and no waiting.
    """
    state = load_state() if state is None else state

    if candles is None:
        candles = strategy.fetch_klines(pair, strategy.INTERVAL, strategy.CANDLE_LIMIT)
    result = strategy.evaluate(
        candles, fast=fast, slow=slow, pair=f"{pair}/USDT", interval=strategy.INTERVAL
    )

    held = held_amount(pair)
    signal = result["signal"]
    action, reason, fill = "NONE", "", None
    key = _key(result)
    already_acted = key in state.get("acted", [])

    if signal == strategy.BUY:
        if held > 0:
            reason = f"already holding {_fmt(held, 8)} {pair} — not pyramiding"
        elif already_acted:
            reason = "this candle's BUY was already acted on"
        elif dry_run:
            action, reason = "WOULD_BUY", f"would spend {_fmt(size_usd)} USDT"
        else:
            try:
                fill = te.sim_buy(pair, size_usd)
                action = "BOUGHT"
                reason = (
                    f"bought {_fmt(fill['token_amount'], 8)} {pair} "
                    f"@ {_fmt(fill['price'])}"
                )
                state.setdefault("acted", []).append(key)
                save_state(state)
            except (te.TradeError, te.PriceUnavailable, RuntimeError) as exc:
                action, reason = "REFUSED", str(exc)

    elif signal == strategy.SELL:
        if held <= 0:
            reason = "flat — nothing to sell"
        elif already_acted:
            reason = "this candle's SELL was already acted on"
        elif dry_run:
            action, reason = "WOULD_SELL", f"would sell {_fmt(held, 8)} {pair}"
        else:
            try:
                fill = te.sim_sell(pair, held)
                action = "SOLD"
                reason = (
                    f"sold {_fmt(fill['token_amount'], 8)} {pair} "
                    f"@ {_fmt(fill['price'])} for {_fmt(fill['proceeds_usdt'])} USDT "
                    f"(P&L {fill['realized_pnl']:+.4f})"
                )
                state.setdefault("acted", []).append(key)
                save_state(state)
            except (te.TradeError, te.PriceUnavailable, RuntimeError) as exc:
                action, reason = "REFUSED", str(exc)

    else:  # HOLD
        if held > 0:
            reason = f"no crossing — keeping {_fmt(held, 8)} {pair}"
        else:
            reason = "no crossing — staying flat"

    return {
        "ts": _now(),
        "signal": signal,
        "action": action,
        "reason": reason,
        "held": held,
        "result": result,
        "fill": fill,
    }


def report(record: dict[str, Any]) -> str:
    """The two lines each pass prints: the market, then the decision."""
    result = record["result"]
    market = strategy.describe(result)
    decision = f"  {record['ts']}  {record['action']:<9} {record['reason']}"
    return f"{market}\n{decision}"


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paper trading loop: MA crossover -> simulated BNB trades"
    )
    parser.add_argument("--symbol", default=PAIR, help=f"base asset (default {PAIR})")
    parser.add_argument("--size", type=float, default=SIZE_USD, help=f"USDT per buy (default {SIZE_USD})")
    parser.add_argument("--fast", type=int, default=strategy.MA_FAST)
    parser.add_argument("--slow", type=int, default=strategy.MA_SLOW)
    parser.add_argument("--interval", type=float, default=INTERVAL_SECONDS, help="seconds between checks")
    parser.add_argument("--once", action="store_true", help="run a single check and exit")
    parser.add_argument("--dry-run", action="store_true", help="decide but place no simulated trades")
    args = parser.parse_args(argv)

    if args.interval <= 0:
        print("error: --interval must be positive", file=sys.stderr)
        return 2

    print(
        f"paper trader | {args.symbol}/USDT {strategy.INTERVAL} | "
        f"MA {args.fast}/{args.slow} | ${_fmt(args.size)} per entry | "
        f"every {args.interval:g}s | TRADING_ENABLED={te.TRADING_ENABLED}"
        + (" | DRY RUN" if args.dry_run else "")
    )
    if te.TRADING_ENABLED:
        print("error: TRADING_ENABLED is True — this is a simulator", file=sys.stderr)
        return 2

    state = load_state()
    passes = 0

    while True:
        passes += 1
        try:
            record = run_once(
                pair=args.symbol,
                size_usd=args.size,
                fast=args.fast,
                slow=args.slow,
                dry_run=args.dry_run,
                state=state,
            )
            print(report(record), flush=True)
        except strategy.MarketDataUnavailable as exc:
            # A feed outage is not a reason to exit: the next pass may work,
            # and stopping would silently end the simulation.
            print(f"  {_now()}  NO_DATA   {exc}", file=sys.stderr, flush=True)
        except strategy.StrategyError as exc:
            print(f"  {_now()}  NO_SIGNAL {exc}", file=sys.stderr, flush=True)

        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\nstopped after {passes} pass(es)")
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nstopped")
        raise SystemExit(0)
