#!/usr/bin/env python3
"""Tests for backtest_rsi.py.

The t distribution is not re-tested here — `test_backtest_mss.py` already checks
the regularised incomplete beta against published critical values, and this
module imports that same function instead of carrying a second copy. What is
tested here is the replay: that it reconciles, that it does not cheat, and that
it refuses the inputs it should refuse.

    python3 -m unittest test_backtest_rsi -v

Four things are defended:

1. **The accounting reconciles.** Summed trade P&L equals the change in cash,
   and fees are provably non-zero. A round trip that returns exactly the start
   is either fee-free (a bug) or too good to be true.
2. **The replay does not cheat.** The precomputed RSI prefix must equal a
   per-bar recomputation, and appending future bars must not rewrite a past
   decision.
3. **One position at a time.** No pyramiding, no overlapping holds, nothing
   still open when the data ends.
4. **The guards hold.** Too few candles, a non-positive size, cash that cannot
   fund the entry, a bad midline mode, and the simulation switch must all
   refuse rather than quietly produce a report.

In both `run_backtest` and `summarise`, `round_trips` counts completed round
trips and `fills` counts individual fills, which is twice as many.
"""

from __future__ import annotations

import math
import os
import random
import unittest

import backtest_mss as bmss
import backtest as bt
import backtest_rsi as br
import strategy_rsi as rsi
import trade_engine as te


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def synthetic_candles(n: int = 1200, seed: int = 5, start: float = 100.0,
                      bar_ms: int = 4 * 60 * 60 * 1000) -> list[dict]:
    """Deterministic candles with alternating drift, so panics and rallies both occur."""
    rng = random.Random(seed)
    t0 = 1_600_000_000_000
    price = start
    out = []
    for i in range(n):
        drift = -0.0022 if (i // 90) % 2 == 0 else 0.0026
        price *= math.exp(drift + rng.gauss(0, 0.013))
        high = price * (1 + abs(rng.gauss(0, 0.004)))
        low = price * (1 - abs(rng.gauss(0, 0.004)))
        out.append(
            {
                "open_time": t0 + i * bar_ms,
                "close_time": t0 + (i + 1) * bar_ms - 1,
                "open": price,
                "high": high,
                "low": low,
                "close": price,
                "volume": 1.0,
            }
        )
    return out


def crash_candles(n_before: int = 80, n_crash: int = 10, seed: int = 11) -> list[dict]:
    """A calm drift, then a violent fall — one obvious oversold entry."""
    rng = random.Random(seed)
    t0 = 1_600_000_000_000
    bar_ms = 4 * 60 * 60 * 1000
    price = 100.0
    out = []
    for i in range(n_before + n_crash):
        step = (0.0012 + rng.gauss(0, 0.003)) if i < n_before else -0.06
        price *= math.exp(step)
        out.append(
            {
                "open_time": t0 + i * bar_ms,
                "close_time": t0 + (i + 1) * bar_ms - 1,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1.0,
            }
        )
    return out


def downtrend_then_bounce_candles(n_down: int = 420, start: float = 600.0,
                                  bar_ms: int = 4 * 60 * 60 * 1000) -> list[dict]:
    """A long decline, a bounce that lifts RSI over 30, then a fall back under it.

    The point is a *genuine* oversold crossing while price is still below the
    200-MA — the one signal the filter is supposed to refuse. A market that only
    falls pins RSI at 0 and never crosses, which tests nothing.
    """
    price = start
    closes = []
    for _ in range(n_down):
        price *= 0.995
        closes.append(price)

    # Bounce until RSI is clearly above the oversold line.
    for _ in range(60):
        if (rsi.rsi_series(closes, 14)[-1] or 0.0) > 32.0:
            break
        price *= 1.02
        closes.append(price)

    # Then fall back through 30 while price is still far below the average.
    for _ in range(60):
        series = rsi.rsi_series(closes, 14)
        if series[-1] is not None and series[-2] is not None:
            if series[-2] >= 30.0 > series[-1]:
                break
        price *= 0.98
        closes.append(price)

    t0 = 1_600_000_000_000
    return [
        {
            "open_time": t0 + i * bar_ms,
            "close_time": t0 + (i + 1) * bar_ms - 1,
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1.0,
        }
        for i, c in enumerate(closes)
    ]


def run_backtest(candles, **kwargs):
    """``br.run_backtest`` with the T5 trend filter off.

    These fixtures were written for the RSI rules alone, and a 200-period moving
    average needs 200 bars before it says anything — ``crash_candles`` is 90.
    The filter has its own class below, on series long enough to carry one.
    """
    kwargs.setdefault("use_trend_filter", False)
    return br.run_backtest(candles, **kwargs)


def summarised(candles=None, **kwargs):
    candles = candles if candles is not None else synthetic_candles()
    result = run_backtest(candles, **kwargs)
    return result, br.summarise(result, br.bt.buy_and_hold(candles))


# --------------------------------------------------------------------------- #
# The statistics are borrowed, and that must stay true
# --------------------------------------------------------------------------- #

class TestStatsAreShared(unittest.TestCase):
    def test_t_statistic_is_the_mss_implementation(self):
        """One implementation of the t distribution, not two that drift apart."""
        self.assertIs(br.t_statistic, bmss.t_statistic)

    def test_student_t_is_also_shared(self):
        self.assertIs(br.student_t_two_sided_p, bmss.student_t_two_sided_p)


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #

class TestReplay(unittest.TestCase):
    def setUp(self):
        self.candles = synthetic_candles()
        self.result, self.summary = summarised(self.candles)

    def test_trades_happen(self):
        self.assertGreater(self.summary["round_trips"], 0,
                           "no trades in 1200 bars — the fixture or the entry rule is broken")

    def test_pnl_reconciles_with_cash(self):
        cash_change = self.result["final_cash"] - self.result["start_usdt"]
        self.assertAlmostEqual(self.result["realized_pnl"], cash_change, places=9,
                               msg="summed trade P&L must equal the change in cash")

    def test_fees_are_charged(self):
        self.assertGreater(self.result["fees_paid"], 0.0)
        self.assertGreater(self.summary["gross_pnl"], self.summary["total_pnl"],
                           "net must be worse than gross by the fees")

    def test_fee_drag_matches_fees_paid(self):
        self.assertAlmostEqual(self.summary["fee_drag_pct"], self.result["fees_paid"],
                               places=9)

    def test_two_fills_per_round_trip(self):
        self.assertEqual(self.result["fills"], 2 * self.summary["round_trips"])

    def test_no_position_survives_the_end(self):
        # Every trade recorded must have a matching exit.
        for trade in self.result["trades"]:
            self.assertIn("exit_bar", trade)
            self.assertIsNotNone(trade["exit_bar"])
        self.assertEqual(len(self.result["trades"]), self.summary["round_trips"])

    def test_trades_do_not_overlap(self):
        bars = [(t["entry_bar"], t["exit_bar"]) for t in self.result["trades"]]
        for (_, prev_exit), (next_entry, _) in zip(bars, bars[1:]):
            self.assertLessEqual(prev_exit, next_entry,
                                 "a new position opened before the last one closed")

    def test_exit_reasons_are_known(self):
        known = {rsi.REASON_MIDLINE, rsi.REASON_OVERBOUGHT,
                 rsi.REASON_STOP_LOSS, "END_OF_DATA"}
        for trade in self.result["trades"]:
            self.assertIn(trade["exit_reason"], known)

    def test_stop_never_fills_above_its_level(self):
        for trade in self.result["trades"]:
            if trade["exit_reason"] == rsi.REASON_STOP_LOSS:
                self.assertLessEqual(trade["exit_price"], trade["stop_price"] + 1e-9,
                                     "a stop filled above its own level")

    def test_equity_curve_starts_after_warmup_and_covers_the_rest(self):
        warmup = self.result["warmup"]
        self.assertEqual(len(self.result["equity_curve"]), len(self.candles) - warmup)

    def test_every_entry_fills_at_the_open_of_the_bar_after_the_signal(self):
        # The whole point of the model: the close that produced the signal is
        # already gone by the time it could be acted on.
        self.assertTrue(self.result["trades"], "fixture produced no trades to check")
        for trade in self.result["trades"]:
            self.assertEqual(trade["entry_bar"], trade["signal_bar"] + 1)
            self.assertEqual(
                trade["entry_price"],
                float(self.candles[trade["entry_bar"]]["open"]),
            )

    def test_every_close_based_exit_fills_at_the_next_open(self):
        for trade in self.result["trades"]:
            if trade["exit_reason"] == rsi.REASON_STOP_LOSS:
                # A resting order fires inside its own bar, at its own level.
                self.assertEqual(trade["exit_bar"], trade["exit_signal_bar"])
                continue
            if trade.get("forced_close"):
                # A liquidation of a held position, not a signal.
                continue
            self.assertEqual(trade["exit_bar"], trade["exit_signal_bar"] + 1)
            self.assertEqual(
                trade["exit_price"],
                float(self.candles[trade["exit_bar"]]["open"]),
            )

    def test_the_stop_is_measured_from_the_price_that_was_paid(self):
        for trade in self.result["trades"]:
            expected = rsi.stop_price(
                trade["entry_price"], self.result["stop_pct"]
            )
            self.assertAlmostEqual(trade["stop_price"], expected, places=9)

    def test_equity_curve_is_monotonic_in_bar(self):
        bars = [p["bar"] for p in self.result["equity_curve"]]
        self.assertEqual(bars, sorted(bars))
        self.assertEqual(len(bars), len(set(bars)))

    def test_deterministic(self):
        again = run_backtest(synthetic_candles())
        self.assertEqual(again["realized_pnl"], self.result["realized_pnl"])
        self.assertEqual(len(again["trades"]), len(self.result["trades"]))

    def test_sizing_is_respected(self):
        for trade in self.result["trades"]:
            self.assertAlmostEqual(trade["usdt_in"], 10.0, places=9)

    def test_win_rate_matches_the_trade_list(self):
        wins = sum(1 for t in self.result["trades"] if t["pnl"] > 0)
        self.assertEqual(wins, self.summary["wins"])
        self.assertEqual(len(self.result["trades"]), self.summary["wins"] + self.summary["losses"])
        self.assertAlmostEqual(self.summary["win_rate"],
                               100.0 * wins / len(self.result["trades"]), places=9)


# --------------------------------------------------------------------------- #
# The replay must not cheat
# --------------------------------------------------------------------------- #

class TestNoLookaheadInReplay(unittest.TestCase):
    def test_precomputed_prefix_matches_per_bar_rebuild(self):
        """The licence for computing the RSI series once, up front.

        If slicing ``rsi_series[:i+1]`` ever differed from recomputing the
        series on ``closes[:i+1]``, the backtest would be reading the future.
        """
        candles = synthetic_candles(n=400, seed=9)
        closes = [c["close"] for c in candles]
        full = rsi.rsi_series(closes, rsi.PERIOD)
        for i in range(rsi.PERIOD, len(closes)):
            prefix = rsi.rsi_series(closes[: i + 1], rsi.PERIOD)
            self.assertAlmostEqual(prefix[-1], full[i], places=10,
                                   msg=f"bar {i}: prefix RSI != full-series RSI")

    def test_appending_future_bars_does_not_rewrite_past_trades(self):
        candles = synthetic_candles(n=500, seed=13)
        short = run_backtest(candles)
        long = run_backtest(candles + synthetic_candles(n=300, seed=13))
        # Only compare trades that had already closed in the shorter run.
        closed = [t for t in short["trades"] if t["exit_reason"] != "END_OF_DATA"]
        self.assertTrue(closed, "fixture produced no closed trades")
        for x, y in zip(closed, long["trades"]):
            self.assertEqual(x["entry_bar"], y["entry_bar"])
            self.assertEqual(x["exit_bar"], y["exit_bar"])
            self.assertAlmostEqual(x["pnl"], y["pnl"], places=9)

    def test_derived_rsi_argument_is_used(self):
        """A flat series must change the outcome — otherwise the argument is
        ignored and the prefix-equivalence test above proves nothing."""
        candles = synthetic_candles(n=600, seed=17)
        real = run_backtest(candles)
        flat = run_backtest(candles, rsi_values=[50.0] * len(candles))
        self.assertGreater(len(real["trades"]), 0)
        self.assertEqual(len(flat["trades"]), 0,
                         "a flat RSI of 50 is never oversold, so it must never buy")


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

class TestGuards(unittest.TestCase):
    def test_too_few_candles(self):
        with self.assertRaises(br.BacktestError):
            run_backtest(synthetic_candles(n=10))

    def test_non_positive_size(self):
        with self.assertRaises(br.BacktestError):
            run_backtest(synthetic_candles(), size_usd=0.0)
        with self.assertRaises(br.BacktestError):
            run_backtest(synthetic_candles(), size_usd=-5.0)

    def test_cash_cannot_fund_the_entry(self):
        with self.assertRaises(br.BacktestError):
            run_backtest(synthetic_candles(), size_usd=10.0, start_usdt=5.0)

    def test_bad_midline_mode(self):
        with self.assertRaises(br.BacktestError):
            run_backtest(synthetic_candles(), midline_mode="sideways")

    def test_simulation_must_be_disabled(self):
        original = te.TRADING_ENABLED
        te.TRADING_ENABLED = True
        try:
            with self.assertRaises(Exception):
                run_backtest(synthetic_candles())
        finally:
            te.TRADING_ENABLED = original

    def test_live_ledger_is_never_written(self):
        """Runs must not touch the owner's paper ledger — by content, not by mtime.

        Checking the path is not enough: sandboxed_engine() restores the path on
        exit, so a test that only compared LEDGER_PATH passed while the file was
        still being truncated underneath it. Checking mtime is not enough
        either: it only proves *something* wrote, not what. So snapshot the
        bytes and require them back unchanged.
        """
        if not os.path.exists(te.LEDGER_PATH):
            self.skipTest("no live ledger to protect")

        before_bytes = te.LEDGER_PATH.read_bytes()
        before_trades = te.TRADES_PATH.read_bytes() if te.TRADES_PATH.exists() else None
        before_mtime = os.path.getmtime(te.LEDGER_PATH)

        run_backtest(synthetic_candles())

        self.assertEqual(te.LEDGER_PATH.read_bytes(), before_bytes,
                         "the backtest rewrote the live paper ledger")
        self.assertEqual(os.path.getmtime(te.LEDGER_PATH), before_mtime,
                         "the backtest touched the live paper ledger")
        if before_trades is not None:
            self.assertEqual(te.TRADES_PATH.read_bytes(), before_trades,
                             "the backtest appended to the live trade log")


    def test_buy_and_hold_is_also_sandboxed(self):
        """The benchmark is a calculation, not a paper trade.

        buy_and_hold() resets the engine to open its position. If it is not
        sandboxed, merely asking "what would holding have done?" wipes the
        owner's ledger — and summary(), which every report calls, asks exactly
        that.
        """
        if not os.path.exists(te.LEDGER_PATH):
            self.skipTest("no live ledger to protect")

        before = te.LEDGER_PATH.read_bytes()
        bt.buy_and_hold(synthetic_candles(n=300))
        self.assertEqual(te.LEDGER_PATH.read_bytes(), before,
                         "buy_and_hold() rewrote the live paper ledger")

        br.summarise(run_backtest(synthetic_candles(n=300)),
                     bt.buy_and_hold(synthetic_candles(n=300)))
        self.assertEqual(te.LEDGER_PATH.read_bytes(), before,
                         "summarise() rewrote the live paper ledger")


# --------------------------------------------------------------------------- #
# The stop, in the replay
# --------------------------------------------------------------------------- #

class TestStopInReplay(unittest.TestCase):
    def test_a_crash_is_entered_and_exited(self):
        result = run_backtest(crash_candles(), size_usd=10.0)
        self.assertGreaterEqual(len(result["trades"]), 1)
        for trade in result["trades"]:
            self.assertIsNotNone(trade["exit_bar"])

    def test_stop_caps_the_loss(self):
        result = run_backtest(synthetic_candles(), size_usd=10.0)
        stops = [t for t in result["trades"] if t["exit_reason"] == rsi.REASON_STOP_LOSS]
        self.assertTrue(stops, "fixture produced no stop exits to check")
        for trade in stops:
            # 5% stop plus the two fees, with room for a gap fill below the stop.
            self.assertGreater(trade["pnl"], -0.12 * trade["usdt_in"])


class TestTrendFilterInReplay(unittest.TestCase):
    """The T5 filter, seen from the backtester rather than the strategy.

    strategy_rsi has its own tests for the rule. These check that the replay
    actually applies it, and that turning it off changes nothing else.
    """

    def test_warmup_grows_to_the_ma_period(self):
        result = br.run_backtest(synthetic_candles(n=1200), use_trend_filter=True)
        self.assertEqual(result["warmup"], rsi.MA_PERIOD)
        self.assertEqual(result["ma_period"], rsi.MA_PERIOD)
        self.assertIs(result["use_trend_filter"], True)

    def test_filter_off_reports_no_vetoes(self):
        result = run_backtest(synthetic_candles(n=1200))
        self.assertIsNone(result["ma_period"])
        self.assertEqual(result["blocked_by_trend"], 0)
        self.assertEqual(result["warmup"], rsi.PERIOD + 2)

    def test_every_trade_was_signalled_above_the_average(self):
        # The filter is applied on the signal bar's close — that is the reading
        # it vetoes on. The fill happens one bar later at the open, so checking
        # the average at entry_bar would be reading a bar the decision had not
        # seen when it was made.
        candles = synthetic_candles(n=1200)
        result = br.run_backtest(candles, use_trend_filter=True)
        self.assertTrue(result["trades"], "fixture produced no trades to check")
        ma = rsi.moving_average([c["close"] for c in candles], rsi.MA_PERIOD)
        for trade in result["trades"]:
            level = ma[trade["signal_bar"]]
            self.assertIsNotNone(level, "trade opened before the average existed")
            self.assertGreater(float(candles[trade["signal_bar"]]["close"]), level,
                               "a BUY got through below the 200-MA")

    def test_downtrend_blocks_entries_but_still_counts_them(self):
        """A falling market is exactly what the filter is for.

        Nothing should be bought while price sits under the average, and the
        replay must report how many signals it turned away rather than looking
        like a strategy that simply had no signal.
        """
        candles = downtrend_then_bounce_candles()
        result = br.run_backtest(candles, use_trend_filter=True, size_usd=10.0)
        self.assertEqual(len(result["trades"]), 0,
                         "the filter let a BUY through in a downtrend")
        self.assertGreater(result["blocked_by_trend"], 0,
                           "the filter blocked nothing — is it wired in?")

    def test_exits_are_untouched_by_the_filter(self):
        """A trade already open is managed by the rules that opened it."""
        candles = synthetic_candles(n=1200)
        filtered = br.run_backtest(candles, use_trend_filter=True)
        for trade in filtered["trades"]:
            self.assertIn(trade["exit_reason"],
                          {rsi.REASON_MIDLINE, rsi.REASON_OVERBOUGHT,
                           rsi.REASON_STOP_LOSS, "END_OF_DATA"})

    def test_filter_never_adds_trades(self):
        candles = synthetic_candles(n=1200)
        off = run_backtest(candles)
        on = br.run_backtest(candles, use_trend_filter=True)
        self.assertLessEqual(len(on["trades"]), len(off["trades"]),
                             "the filter created trades it should only remove")

    def test_bad_ma_period_is_refused_before_any_reset(self):
        with self.assertRaises(br.BacktestError):
            br.run_backtest(synthetic_candles(n=600), ma_period=0,
                            use_trend_filter=True)


if __name__ == "__main__":
    unittest.main()
