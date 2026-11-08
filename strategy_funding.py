#!/usr/bin/env python3
"""T6: funding-rate carry — the delta-neutral trade, measured rather than assumed.

    python3 strategy_funding.py                     # live snapshot, flag > 0.02%
    python3 strategy_funding.py --threshold 0.0001  # tighter flag
    python3 strategy_funding.py --days 365          # reliability over a year
    python3 strategy_funding.py --json

The trade this reads: perpetual futures have no expiry, so the venue ties them
to spot with a periodic payment between longs and shorts. When the rate is
positive, longs pay shorts. Short the perp, buy the same notional of spot, and
the two legs cancel in price terms — what is left is the funding you collect,
paid every settlement, for as long as the rate stays positive.

That is a different class of edge from T4/T5. RSI mean reversion predicted a
*price*; this predicts a *cash flow*, and the prediction does not depend on
which way the price goes. That is the whole appeal, and it is also where the
honesty has to start: the price legs cancel only if both are filled at the same
size at the same moment, and the carry is only an edge if it exceeds what the
round trip costs to put on and take off.

Three numbers matter and this module refuses to conflate them:

* the **rate per settlement** — what the API returns. It is NOT a daily or an
  annual figure, and reading it as one is the standard way to be off by a
  factor of 1,095.
* the **interval** — how many settlements a year holds. Binance runs 8-hour
  funding on BTC/ETH/SOL (verified: the funding history lands 8.0000h apart),
  but 4-hour on others, and a 4h symbol at the same per-settlement rate earns
  twice as much. The interval is therefore measured from the history rather
  than assumed, and every annualised figure carries it.
* the **APR** — rate x settlements per year. This is the only number that can
  be compared across symbols with different intervals.

The threshold flags are points on a ruler, not a verdict. A rate above 0.02%
per 8h is 21.9% APR before costs, which is worth the trade only if the legs are
cheap to open. A rate below about 0.01% per 8h barely covers two taker fees and
the spread on two instruments.

Read-only: this module has no order path and no key. It reads public market
data and prints what it found. TRADING_ENABLED is asserted False at import, so
it cannot be mistaken for something that trades.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

import strategy           # binance_symbol — one symbol rule for the whole repo
import trade_engine as te

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

FAPI_BASE = "https://fapi.binance.com"
PREMIUM_INDEX_URL = f"{FAPI_BASE}/fapi/v1/premiumIndex"
FUNDING_HISTORY_URL = f"{FAPI_BASE}/fapi/v1/fundingRate"

ASSETS = ("BTC", "ETH", "SOL")

# Per settlement, as a fraction. 0.0002 = 0.02%.
THRESHOLD = 0.0002       # flag: worth looking at
HIGH_RATE = 0.0003       # strong: the rate the brief called "highly positive"

# Only used when the history cannot be read. Every real figure is annualised
# with the interval measured from the data, never with this.
DEFAULT_INTERVAL_HOURS = 8.0

DAYS = 365
HTTP_TIMEOUT = 15
PAGE_LIMIT = 1_000       # Binance's cap on one fundingRate page
MAX_PAGES = 40           # ~13 years at 8h; a guard, not an expectation

# Nothing here moves money, and the assertion makes that structural rather
# than a promise in a docstring.
if te.TRADING_ENABLED:  # pragma: no cover - defensive
    raise RuntimeError("strategy_funding is read-only; TRADING_ENABLED must be False")


class FundingDataUnavailable(RuntimeError):
    """Binance did not give us usable funding data."""


class FundingError(RuntimeError):
    """The data cannot produce a verdict (bad parameters, empty series)."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clock(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%MZ")


def _r(value: float, places: int = 10) -> float:
    return round(value, places) + 0.0


def _get(url: str, params: dict[str, Any] | None = None, *, _opener=None) -> Any:
    """GET JSON from Binance. Raises FundingDataUnavailable on any failure.

    A broken feed is never quietly turned into an empty series or a zero rate:
    a zero funding rate and a funding rate we could not read are different
    facts, and only one of them means the trade is not worth doing.
    """
    target = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    request = urllib.request.Request(
        target,
        headers={"User-Agent": "paper-trader/1.0", "Accept": "application/json"},
        method="GET",
    )
    open_url = _opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise FundingDataUnavailable(f"Binance HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FundingDataUnavailable(f"could not reach Binance: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FundingDataUnavailable(f"Binance sent non-JSON: {exc}") from exc


def settlements_per_year(interval_hours: float) -> float:
    """How many funding payments a year contains, given the interval."""
    if interval_hours <= 0:
        raise FundingError(f"funding interval must be positive, got {interval_hours!r}")
    return 24.0 / interval_hours * 365.0


def annualise(rate: float, interval_hours: float) -> float:
    """A per-settlement rate as a yearly fraction (not a percentage).

    0.0001 per 8h -> 0.1095 a year. This multiplication is the entire reason
    the module exists: a rate that looks negligible per settlement is a
    double-digit annual yield, and a rate that looks small per settlement is
    still nothing.
    """
    return rate * settlements_per_year(interval_hours)


# --------------------------------------------------------------------------- #
# Live snapshot
# --------------------------------------------------------------------------- #

def fetch_premium_index(asset: str, *, _opener=None) -> dict[str, Any]:
    """The venue's current funding state for one perpetual.

    ``lastFundingRate`` is the rate in force now, per settlement — Binance's own
    field name is the source of the "last" confusion, but it is the running
    estimate for the next settlement, not a historical value. Mark and index
    prices come along because their difference (the basis) is the other half of
    the story: positive funding should coincide with the perp trading above
    spot, and when it does not, something is off.
    """
    symbol = strategy.binance_symbol(asset)
    payload = _get(PREMIUM_INDEX_URL, {"symbol": symbol}, _opener=_opener)

    if not isinstance(payload, dict) or not payload:
        raise FundingDataUnavailable(f"unexpected premiumIndex payload for {symbol}: {payload!r}"[:300])
    if "lastFundingRate" not in payload:
        raise FundingDataUnavailable(f"premiumIndex for {symbol} has no funding rate")

    try:
        rate = float(payload["lastFundingRate"])
        mark = float(payload["markPrice"])
        index = float(payload["indexPrice"])
        next_time = int(payload["nextFundingTime"])
        server_time = int(payload["time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FundingDataUnavailable(f"malformed premiumIndex for {symbol}: {payload!r}"[:300]) from exc

    return {
        "symbol": symbol,
        "asset": symbol[: -len("USDT")],
        "rate": _r(rate),
        "mark_price": mark,
        "index_price": index,
        "basis_pct": (mark / index - 1.0) * 100.0 if index else None,
        "next_funding_time": next_time,
        "server_time": server_time,
        "minutes_to_next": (next_time - server_time) / 60_000.0,
    }


def fetch_funding_history(asset: str, days: int = DAYS, *, _opener=None) -> list[dict[str, Any]]:
    """Settled funding rates, oldest first, back ``days`` days.

    Paged by ``startTime`` because one page is 1,000 settlements and a year at
    8h is 1,095. Duplicates are dropped by ``fundingTime``: a rate counted
    twice would weight one settlement twice in every mean below.
    """
    if days < 1:
        raise FundingError(f"days must be at least 1, got {days!r}")

    symbol = strategy.binance_symbol(asset)
    start = _now_ms() - days * 86_400_000

    rows: list[dict[str, Any]] = []
    cursor = start
    for _ in range(MAX_PAGES):
        page = _get(
            FUNDING_HISTORY_URL,
            {"symbol": symbol, "startTime": cursor, "limit": PAGE_LIMIT},
            _opener=_opener,
        )
        if not isinstance(page, list):
            raise FundingDataUnavailable(f"unexpected fundingRate payload for {symbol}: {page!r}"[:300])
        if not page:
            break

        try:
            parsed = [
                {"funding_time": int(r["fundingTime"]), "rate": _r(float(r["fundingRate"]))}
                for r in page
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise FundingDataUnavailable(f"malformed fundingRate for {symbol}: {page[:2]!r}"[:300]) from exc

        rows.extend(parsed)
        if len(page) < PAGE_LIMIT:
            break

        last = parsed[-1]["funding_time"]
        if last < cursor:                      # the window did not move
            break
        cursor = last + 1

    deduped: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: r["funding_time"]):
        if deduped and row["funding_time"] == deduped[-1]["funding_time"]:
            continue
        deduped.append(row)

    if not deduped:
        raise FundingDataUnavailable(f"no funding history returned for {symbol}")
    return deduped


def measure_interval_hours(history: list[dict[str, Any]]) -> float:
    """The funding interval, measured from the gap between settlements.

    Taken as the median gap so one late or duplicated settlement cannot change
    it. This is measured rather than assumed because Binance runs 8-hour
    funding on some symbols and 4-hour on others, and a 4h symbol earning the
    same per-settlement rate earns twice the APR. Guessing 8 for everything
    would silently halve or double every annualised figure in the report.
    """
    if len(history) < 2:
        raise FundingError("need at least two settlements to measure the interval")
    gaps = [
        (history[i]["funding_time"] - history[i - 1]["funding_time"]) / 3_600_000.0
        for i in range(1, len(history))
    ]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        raise FundingError("settlements do not advance in time")
    return float(statistics.median(gaps))


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def snapshot(assets: tuple[str, ...] = ASSETS, *, threshold: float = THRESHOLD,
             high_rate: float = HIGH_RATE, **_kwargs) -> list[dict[str, Any]]:
    """One row per asset: the live rate, its annualised form, and its flag."""
    if threshold < 0:
        raise FundingError(f"threshold must not be negative, got {threshold!r}")
    if not assets:
        raise FundingError("no assets to read")

    rows: list[dict[str, Any]] = []
    for asset in assets:
        row = fetch_premium_index(asset, **_kwargs)
        # The live endpoint does not state the interval, so read one from the
        # recent history. Three settlements are enough and cost one call.
        recent = fetch_funding_history(asset, days=7, **_kwargs)
        row["interval_hours"] = measure_interval_hours(recent)
        row["apr"] = annualise(row["rate"], row["interval_hours"]) * 100.0
        row["flagged"] = row["rate"] >= threshold
        row["strong"] = row["rate"] >= high_rate
        row["carry_per_1000_per_year"] = annualise(row["rate"], row["interval_hours"]) * 1_000.0
        rows.append(row)
    return rows


def history_stats(asset: str, days: int = DAYS, *, threshold: float = THRESHOLD,
                  **_kwargs) -> dict[str, Any]:
    """What the funding rate actually did, over the window.

    A snapshot says what you would earn if today repeated forever. This says how
    often that was true: the share of settlements above the threshold, the mean,
    the extremes, and how often the sign flipped against the trade. Those are the
    numbers that decide whether the carry is an edge or an anecdote.
    """
    history = fetch_funding_history(asset, days=days, **_kwargs)
    interval = measure_interval_hours(history)
    rates = [row["rate"] for row in history]
    per_year = settlements_per_year(interval)

    above = sum(1 for r in rates if r >= threshold)
    negative = sum(1 for r in rates if r < 0)
    mean = sum(rates) / len(rates)
    stdev = statistics.pstdev(rates) if len(rates) > 1 else 0.0

    return {
        "asset": strategy.binance_symbol(asset)[: -len("USDT")],
        "symbol": strategy.binance_symbol(asset),
        "days": days,
        "settlements": len(rates),
        "interval_hours": interval,
        "settlements_per_year": per_year,
        "first_time": history[0]["funding_time"],
        "last_time": history[-1]["funding_time"],
        "mean_rate": _r(mean),
        "median_rate": _r(statistics.median(rates)),
        "stdev_rate": _r(stdev),
        "mean_apr": annualise(mean, interval) * 100.0,
        "median_apr": annualise(statistics.median(rates), interval) * 100.0,
        "max_rate": _r(max(rates)),
        "min_rate": _r(min(rates)),
        "above_threshold": above,
        "above_threshold_pct": 100.0 * above / len(rates),
        "negative": negative,
        "negative_pct": 100.0 * negative / len(rates),
        "threshold": threshold,
        "realised_carry_per_1000": mean * per_year * 1_000.0 * (days / 365.0),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt_rate(rate: float | None, places: int = 4) -> str:
    """A rate as a percentage with enough decimals to see the threshold.

    Four decimals of a percent is 0.0001% resolution, which is the scale the
    0.02% line lives on. Two decimals would round 0.0036% and 0.0085% to the
    same "0.00%" and make three different assets look identical.
    """
    return "n/a" if rate is None else f"{rate * 100:.{places}f}%"


def format_snapshot(rows: list[dict[str, Any]], *, threshold: float = THRESHOLD,
                    high_rate: float = HIGH_RATE) -> str:
    lines: list[str] = []
    lines.append(
        f"T6  funding-rate carry (delta-neutral)  |  flag >= {threshold * 100:.4f}% per settlement"
        f"  |  strong >= {high_rate * 100:.4f}%  |  TRADING_ENABLED={te.TRADING_ENABLED}"
    )
    lines.append("")

    header = (
        f"  {'asset':<5} {'rate/settle':>12} {'APR':>9} {'per $1k/yr':>11} "
        f"{'mark':>12} {'basis':>8} {'next':>7}  flag"
    )
    lines.append(header)
    for row in rows:
        flag = "STRONG" if row["strong"] else ("flag" if row["flagged"] else "-")
        lines.append(
            f"  {row['asset']:<5} {_fmt_rate(row['rate']):>12} "
            f"{row['apr']:>8.2f}% {row['carry_per_1000_per_year']:>10,.2f} "
            f"{row['mark_price']:>12,.2f} "
            f"{(row['basis_pct'] or 0.0):>7.3f}% "
            f"{_clock(row['next_funding_time']):>7}  {flag}"
        )
    lines.append("")

    flagged = [row for row in rows if row["flagged"]]
    intervals = sorted({row["interval_hours"] for row in rows})
    interval_text = ", ".join(f"{i:g}h" for i in intervals)
    lines.append(f"  funding interval (measured)  {interval_text}  "
                 f"= {settlements_per_year(intervals[0]):.0f} settlements a year"
                 if len(intervals) == 1 else
                 f"  funding interval (measured)  {interval_text}  — mixed, APR is per-row")

    if not flagged:
        lines.append(f"  flagged above {threshold * 100:.4f}%: none")
        best = max(rows, key=lambda r: r["rate"])
        lines.append(
            f"  highest is {best['asset']} at {_fmt_rate(best['rate'])} per settlement "
            f"({best['apr']:.2f}% APR), which is {threshold / best['rate']:.1f}x below the flag"
            if best["rate"] > 0 else
            f"  every rate is at or below zero — the carry would be paid, not received"
        )
    else:
        names = ", ".join(f"{row['asset']} {_fmt_rate(row['rate'])}" for row in flagged)
        lines.append(f"  flagged above {threshold * 100:.4f}%: {names}")

    lines.append("")
    lines.append("  APR = rate per settlement x settlements per year. The rate Binance returns is")
    lines.append("  per settlement, NOT daily or annual: reading it as daily understates by ~3x,")
    lines.append("  as annual overstates by ~1,095x. Costs are not modelled here.")
    return "\n".join(lines)


def format_history(stats: list[dict[str, Any]], *, threshold: float = THRESHOLD) -> str:
    lines: list[str] = []
    lines.append(f"reliability  |  settled funding history  |  flag >= {threshold * 100:.4f}%")
    lines.append("")
    lines.append(
        f"  {'asset':<5} {'settles':>8} {'mean':>10} {'mean APR':>10} "
        f"{'median':>10} {'max':>10} {'min':>10} {'> flag':>8} {'< 0':>7}"
    )
    for s in stats:
        lines.append(
            f"  {s['asset']:<5} {s['settlements']:>8,} {_fmt_rate(s['mean_rate']):>10} "
            f"{s['mean_apr']:>9.2f}% {_fmt_rate(s['median_rate']):>10} "
            f"{_fmt_rate(s['max_rate']):>10} {_fmt_rate(s['min_rate']):>10} "
            f"{s['above_threshold_pct']:>7.1f}% {s['negative_pct']:>6.1f}%"
        )
    lines.append("")
    for s in stats:
        lines.append(
            f"  {s['asset']:<5} {_iso(s['first_time'])} -> {_iso(s['last_time'])}  "
            f"{s['settlements']:,} settlements at {s['interval_hours']:g}h  "
            f"carry on $1,000 held the whole window: ${s['realised_carry_per_1000']:+,.2f}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict[str, Any]], stats: list[dict[str, Any]] | None = None,
            *, threshold: float = THRESHOLD) -> str:
    """One sentence, and it is allowed to be 'there is nothing here'."""
    if not rows:
        return "verdict: no rates read — a failed read is not a zero"

    live = [row for row in rows if row["flagged"]]
    if live:
        best = max(live, key=lambda r: r["rate"])
        head = (f"verdict: {len(live)} of {len(rows)} assets are above the flag right now; "
                f"the best is {best['asset']} at {_fmt_rate(best['rate'])} per settlement, "
                f"{best['apr']:.2f}% APR.")
    else:
        best = max(rows, key=lambda r: r["rate"])
        head = (f"verdict: nothing clears the flag right now — the best is {best['asset']} at "
                f"{_fmt_rate(best['rate'])} per settlement, {best['apr']:.2f}% APR.")

    if not stats:
        return head

    covered = [s for s in stats if s["above_threshold_pct"] >= 50.0]
    if covered:
        names = ", ".join(f"{s['asset']} {s['above_threshold_pct']:.0f}%" for s in covered)
        tail = (f" Over the window the flag was cleared on more than half of settlements for "
                f"{names}, so this is a regime rather than a moment.")
    else:
        names = ", ".join(f"{s['asset']} {s['above_threshold_pct']:.0f}%" for s in stats)
        tail = (f" Over the window it was cleared on only {names} of settlements, so today's "
                f"reading is not a standing yield.")
    return head + tail


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read Binance perpetual funding rates (read-only, delta-neutral carry)"
    )
    parser.add_argument("--assets", default=",".join(ASSETS),
                        help=f"comma-separated bases (default {','.join(ASSETS)})")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"flag rate per settlement as a fraction (default {THRESHOLD})")
    parser.add_argument("--high", type=float, default=HIGH_RATE,
                        help=f"strong rate per settlement as a fraction (default {HIGH_RATE})")
    parser.add_argument("--days", type=int, default=DAYS,
                        help=f"history window for the reliability report (default {DAYS})")
    parser.add_argument("--no-history", action="store_true",
                        help="live snapshot only, skip the history calls")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a report")
    args = parser.parse_args(argv)

    assets = tuple(a.strip().upper() for a in args.assets.split(",") if a.strip())
    if not assets:
        print("error: --assets named nothing", file=sys.stderr)
        return 2
    if len(set(assets)) != len(assets):
        print("error: --assets repeats an asset", file=sys.stderr)
        return 2
    if args.threshold < 0 or args.high < 0:
        print("error: thresholds must not be negative", file=sys.stderr)
        return 2

    try:
        rows = snapshot(assets, threshold=args.threshold, high_rate=args.high)
        stats = None if args.no_history else [
            history_stats(asset, days=args.days, threshold=args.threshold) for asset in assets
        ]
    except (FundingDataUnavailable, FundingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(
            {
                "config": {
                    "assets": list(assets),
                    "threshold": args.threshold,
                    "high_rate": args.high,
                    "days": None if args.no_history else args.days,
                    "trading_enabled": te.TRADING_ENABLED,
                },
                "snapshot": rows,
                "history": stats,
                "verdict": verdict(rows, stats, threshold=args.threshold),
            },
            indent=2,
            default=str,
        ))
        return 0

    print(format_snapshot(rows, threshold=args.threshold, high_rate=args.high))
    if stats:
        print()
        print(format_history(stats, threshold=args.threshold))
    print()
    print(verdict(rows, stats, threshold=args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
