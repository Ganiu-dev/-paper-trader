#!/usr/bin/env python3
"""split_half.py — did the 4h RSI 14 (30/50) edge survive unseen data?

The parameters were chosen on the whole 3-year window, so any T-stat computed
on that window is contaminated: the parameters were allowed to look at the
answer. This script does the honest version.

    fit  (in-sample)     2023-11-01 -> 2024-12-31
    test (out-of-sample) 2025-01-01 -> now

The fit window is replayed on its own and the test window is replayed with the
parameters frozen at 14 / 30 / 50 / midline-down / 5% stop. The tail of the fit
window is carried into the test run as warm-up, so the first scored bar of the
test half already has a Wilder-smoothed RSI instead of starting blind. That tail
is 16 bars for the RSI alone and 200 once the output is gated on a 200-period
average -- see warmup_bars().

Nothing is optimised here. If a number on the test half is good, it is good on
data no parameter ever saw.

Read-only: run_backtest() and buy_and_hold() sandbox themselves, and the live
ledger's bytes are hashed before and after as proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any

import backtest as bt
import backtest_rsi as br
import strategy_rsi as rsi
import trade_engine as te

PAIR = "BNB"
INTERVAL = "4h"

FIT_START = datetime(2023, 11, 1, tzinfo=timezone.utc)
FIT_END = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Frozen settings. Changing any of these after seeing a test result is how a
# split-half test stops being a test.
PERIOD = 14
OVERSOLD = 30.0
OVERBOUGHT = 70.0
MIDLINE = 50.0
MIDLINE_MODE = "down"
STOP_PCT = 0.05
SIZE_USD = 10.0
START_USDT = 100.0

# T5's trend filter. Off by default so this script still reproduces the
# published baseline; --trend-filter turns it on and widens the warm-up to the
# 200 bars the average needs.
USE_TREND_FILTER = False
MA_PERIOD = rsi.MA_PERIOD


def warmup_bars(use_trend_filter: bool = USE_TREND_FILTER,
                ma_period: int = MA_PERIOD) -> int:
    """Bars of history the replay eats before it can trade.

    The RSI needs period + 2; a 200-period MA needs 200. Carrying too few bars
    into the test half would leave the first 184 test bars unable to trade,
    which would quietly shorten the out-of-sample window.
    """
    need = PERIOD + 2
    return max(need, ma_period) if use_trend_filter else need


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _ledger_digest() -> str:
    try:
        return hashlib.md5(te.LEDGER_PATH.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "(no ledger)"


def replay(candles: list[dict[str, Any]], label: str,
           *, use_trend_filter: bool = USE_TREND_FILTER,
           ma_period: int = MA_PERIOD) -> dict[str, Any]:
    result = br.run_backtest(
        candles,
        pair=PAIR,
        size_usd=SIZE_USD,
        period=PERIOD,
        oversold=OVERSOLD,
        overbought=OVERBOUGHT,
        midline=MIDLINE,
        midline_mode=MIDLINE_MODE,
        stop_pct=STOP_PCT,
        ma_period=ma_period,
        use_trend_filter=use_trend_filter,
        start_usdt=START_USDT,
        interval=INTERVAL,
    )
    hold = bt.buy_and_hold(candles, pair=PAIR, start_usdt=START_USDT)
    summary = br.summarise(result, hold)
    summary["_label"] = label
    summary["_first_bar"] = _iso(BARS_SEEN[label][0])
    summary["_last_bar"] = _iso(BARS_SEEN[label][1])
    summary["_price_start"] = BARS_SEEN[label][2]
    summary["_price_end"] = BARS_SEEN[label][3]
    return summary


BARS_SEEN: dict[str, tuple[int, int, float, float]] = {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split-half out-of-sample test of the RSI strategy."
    )
    parser.add_argument(
        "--trend-filter", dest="use_trend_filter", action="store_true",
        help="apply the T5 200-MA entry filter (default: off, the published baseline)",
    )
    parser.add_argument(
        "--ma-period", type=int, default=MA_PERIOD,
        help=f"moving-average period for the trend filter (default {MA_PERIOD})",
    )
    args = parser.parse_args()

    warmup = warmup_bars(args.use_trend_filter, args.ma_period)

    before = _ledger_digest()

    # Fetch enough 4h bars to reach back past the fit window with a full warm-up,
    # so the fit half gets its own pre-window bars rather than eating into 2023-11.
    bars_wanted = (1_260 + int(warmup / 6) + 60) * 6
    candles = bt.fetch_history(PAIR, INTERVAL, bars_wanted)
    if not candles:
        print("error: no candles", file=sys.stderr)
        return 1

    fit_start_ms = _ms(FIT_START)
    fit_end_ms = _ms(FIT_END)

    # Indices by timestamp.
    idx_fit_start = next(
        (i for i, c in enumerate(candles) if c["open_time"] >= fit_start_ms), None
    )
    idx_fit_end = next(
        (i for i, c in enumerate(candles) if c["open_time"] >= fit_end_ms), None
    )
    if idx_fit_start is None or idx_fit_end is None:
        print("error: history does not cover the split boundary", file=sys.stderr)
        return 1

    # Fit half: pre-window warm-up, then 2023-11-01 .. 2024-12-31.
    fit_lo = max(0, idx_fit_start - warmup)
    fit_candles = candles[fit_lo:idx_fit_end]
    fit_scored = candles[idx_fit_start:idx_fit_end]

    # Test half: the tail of the fit window, then 2025-01-01 .. now.
    carry_lo = max(0, idx_fit_end - warmup)
    test_candles = candles[carry_lo:]
    test_scored = candles[idx_fit_end:]

    if len(fit_scored) < warmup * 4 or len(test_scored) < warmup * 4:
        print("error: one half is too short to judge", file=sys.stderr)
        return 1

    BARS_SEEN["fit"] = (
        fit_scored[0]["open_time"], fit_scored[-1]["close_time"],
        float(fit_scored[0]["close"]), float(fit_scored[-1]["close"]),
    )
    BARS_SEEN["test"] = (
        test_scored[0]["open_time"], test_scored[-1]["close_time"],
        float(test_scored[0]["close"]), float(test_scored[-1]["close"]),
    )

    fit = replay(fit_candles, "fit", use_trend_filter=args.use_trend_filter,
                 ma_period=args.ma_period)
    test = replay(test_candles, "test", use_trend_filter=args.use_trend_filter,
                  ma_period=args.ma_period)

    after = _ledger_digest()

    def line(s: dict[str, Any]) -> str:
        t = s["t_stat"]
        p = s["p_value"]
        t_text = (
            "n/a (too few trades)"
            if t is None
            else f"{t:+.2f} (p = {p:.3f}, df = {s['df']})"
        )
        pf = "n/a" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
        return (
            f"{s['_label']:<5} {s['_first_bar']} -> {s['_last_bar']}  "
            f"{s['bars']:,} bars  price {s['_price_start']:,.2f} -> {s['_price_end']:,.2f}  "
            f"({(s['_price_end'] / s['_price_start'] - 1) * 100:+.2f}%)\n"
            f"      round trips {s['round_trips']:>4}   win rate {s['win_rate']:>5.1f}%  "
            f"({s['wins']}W / {s['losses']}L)\n"
            f"      net P&L     {s['total_pnl']:>+8.2f} USDT ({s['total_pnl_pct']:+.2f}%)"
            f"   gross {s['gross_pnl']:+.2f}  fees {s['fees_paid']:.4f}\n"
            f"      T-stat      {t_text}\n"
            f"      trend vetoes {s.get('blocked_by_trend', 0) if s.get('use_trend_filter') else 'n/a (filter off)'}\n"
            f"      max DD      {s['max_drawdown_usdt']:.2f} USDT ({s['max_drawdown_pct']:.2f}%)"
            f"   profit factor {pf}   exposure {s['exposure_pct']:.1f}%\n"
            f"      buy & hold  {s['buy_hold_pnl']:+.2f} USDT ({s['buy_hold_pnl_pct']:+.2f}%)"
            f"   vs hold {s['difference_pct']:+.2f} points\n"
            f"      exits       " + ", ".join(
                f"{k} {v}x ({s['pnl_by_exit_reason'][k]:+.2f})"
                for k, v in sorted(s["exit_reasons"].items(), key=lambda kv: -kv[1])
            )
        )

    print(f"split_half | {PAIR}/USDT {INTERVAL} | RSI {PERIOD} {OVERSOLD:g}/{OVERBOUGHT:g} "
          f"midline {MIDLINE:g} ({MIDLINE_MODE}) | stop {STOP_PCT * 100:g}% | "
          f"${SIZE_USD:.2f} per entry | fee {te.FEE_RATE * 100:g}% per side | "
          f"{f'MA{args.ma_period} filter on' if args.use_trend_filter else 'no trend filter'}")
    print(f"params frozen at fit time — nothing tuned on the test half")
    print(f"warm-up carried into test: last {warmup} bars of the fit window")
    print()
    print(line(fit))
    print()
    print(line(test))
    print()
    print(f"live ledger md5 {before} -> {after}  "
          f"({'UNTOUCHED' if before == after else 'TOUCHED — investigate'})")

    out_name = "split_half_results_trend.json" if args.use_trend_filter \
        else "split_half_results.json"
    with open(out_name, "w", encoding="utf-8") as fh:
        json.dump({"fit": fit, "test": test}, fh, indent=2, default=str)
    print(f"wrote {out_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
