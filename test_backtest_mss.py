#!/usr/bin/env python3
"""Tests for backtest_mss.py.

Three things are being defended here, and they are the three that would make a
backtest lie to you:

1. The statistics are right. The Student-t p-value is checked against published
   critical values, because a wrong p-value is an invisible wrong answer.
2. The accounting reconciles. Summed trade P&L must equal the change in cash,
   and fees must be provably non-zero — a round trip that returns exactly the
   start is either fee-free (a bug) or too good to be true.
3. The replay does not cheat. Precomputed swing levels must give the same trades
   as recomputing them per bar, and no future bar may affect a past decision.

Everything runs on synthetic deterministic candles: no network, no waiting.
"""

from __future__ import annotations

import math
import random
import unittest
from unittest import mock

import backtest as bt
import backtest_mss as bm
import strategy_mss as mss
import trade_engine as te


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def synthetic_candles(n: int = 900, seed: int = 3, start: float = 100.0,
                      bar_ms: int = 4 * 60 * 60 * 1000) -> list[dict]:
    """Deterministic candles with alternating drift, so both trends appear.

    Reproducible by design: the same seed gives the same series on every
    machine and every Python version that has the same random module.
    """
    rng = random.Random(seed)
    t0 = 1_600_000_000_000
    price = start
    out = []
    for i in range(n):
        drift = 0.0012 if (i // 120) % 2 == 0 else -0.0018
        price *= math.exp(drift + rng.gauss(0, 0.011))
        high = price * (1 + abs(rng.gauss(0, 0.005)))
        low = price * (1 - abs(rng.gauss(0, 0.005)))
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


def run(candles, **kw):
    """Run the backtest in the sandbox, with buy & hold, in one call."""
    with bt.sandboxed_engine():
        result = bm.run_backtest(candles, **kw)
        hold = bt.buy_and_hold(candles, start_usdt=kw.get("start_usdt", 100.0))
    return result, hold


# --------------------------------------------------------------------------- #
# The statistics
# --------------------------------------------------------------------------- #

class TestStudentT(unittest.TestCase):
    """The p-value is checked against textbook critical values."""

    def test_critical_values_are_reproduced(self):
        # (t, df, two-sided p) from a standard t table.
        anchors = [
            (12.706, 1, 0.05),
            (4.303, 2, 0.05),
            (2.228, 10, 0.05),
            (1.812, 10, 0.10),
            (2.086, 20, 0.05),
            (1.960, 100000, 0.05),
        ]
        for t, df, expected in anchors:
            with self.subTest(t=t, df=df):
                got = bm.student_t_two_sided_p(t, df)
                self.assertAlmostEqual(got, expected, delta=2e-3)

    def test_symmetry(self):
        for t in (0.5, 1.5, 3.0, 9.0):
            self.assertAlmostEqual(
                bm.student_t_two_sided_p(t, 17),
                bm.student_t_two_sided_p(-t, 17),
                places=12,
            )

    def test_zero_t_is_certainly_noise(self):
        self.assertAlmostEqual(bm.student_t_two_sided_p(0.0, 5), 1.0, places=12)

    def test_p_value_falls_as_t_rises(self):
        ps = [bm.student_t_two_sided_p(t, 30) for t in (0.0, 1.0, 2.0, 3.0, 5.0)]
        self.assertEqual(ps, sorted(ps, reverse=True))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in ps))

    def test_extreme_t_underflows_cleanly(self):
        p = bm.student_t_two_sided_p(500.0, 4)
        self.assertGreaterEqual(p, 0.0)
        self.assertLess(p, 1e-6)

    def test_df_must_be_positive(self):
        with self.assertRaises(ValueError):
            bm.student_t_two_sided_p(1.0, 0)

    def test_betai_rejects_out_of_range_x(self):
        with self.assertRaises(ValueError):
            bm._betai(1.0, 1.0, 1.5)

    def test_betai_symmetric_at_half(self):
        # Beta(2,3) has pdf 12x(1-x)^2, so I_0.5(2,3) = 0.6875 exactly, and the
        # two orderings must sum to 1 — that is the symmetry being checked.
        self.assertAlmostEqual(bm._betai(2.0, 3.0, 0.5), 0.6875, places=12)
        self.assertAlmostEqual(bm._betai(3.0, 2.0, 0.5), 0.3125, places=12)
        self.assertAlmostEqual(
            bm._betai(2.0, 3.0, 0.5) + bm._betai(3.0, 2.0, 0.5), 1.0, places=12
        )


class TestTStatistic(unittest.TestCase):
    """Mean / standard error on percent returns, with honest empty answers."""

    def test_matches_hand_computation(self):
        returns = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = 3.0
        var = sum((r - mean) ** 2 for r in returns) / 4
        expected_t = mean / (math.sqrt(var) / math.sqrt(5))
        got = bm.t_statistic(returns)
        self.assertAlmostEqual(got["mean_pct"], mean, places=12)
        self.assertAlmostEqual(got["stdev_pct"], math.sqrt(var), places=12)
        self.assertAlmostEqual(got["t_stat"], expected_t, places=12)
        self.assertEqual(got["df"], 4)

    def test_no_statistic_from_one_trade(self):
        got = bm.t_statistic([2.5])
        self.assertIsNone(got["t_stat"])
        self.assertIsNone(got["p_value"])
        self.assertEqual(got["n"], 1)

    def test_empty_sample(self):
        got = bm.t_statistic([])
        self.assertEqual(got["n"], 0)
        self.assertIsNone(got["t_stat"])
        self.assertEqual(got["mean_pct"], 0.0)

    def test_zero_variance_is_not_significant(self):
        # Identical returns have no spread; the T-stat is undefined, not infinite.
        got = bm.t_statistic([1.0, 1.0, 1.0])
        self.assertIsNone(got["t_stat"])
        self.assertIsNone(got["p_value"])

    def test_mean_zero_gives_zero_t(self):
        got = bm.t_statistic([-1.0, 1.0, -1.0, 1.0])
        self.assertAlmostEqual(got["t_stat"], 0.0, places=12)


# --------------------------------------------------------------------------- #
# The accounting
# --------------------------------------------------------------------------- #

class TestReplay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.candles = synthetic_candles()
        cls.result, cls.hold = run(cls.candles)
        cls.summary = bm.summarise(cls.result, cls.hold)

    def test_it_actually_trades(self):
        self.assertGreater(self.summary["round_trips"], 5,
                           "a flat result would make every assertion below vacuous")

    def test_trade_pnl_reconciles_with_cash(self):
        total = sum(t["pnl"] for t in self.result["trades"])
        cash_change = self.result["final_cash"] - self.result["start_usdt"]
        self.assertAlmostEqual(total, cash_change, delta=1e-6)
        self.assertAlmostEqual(self.result["realized_pnl"], cash_change, delta=1e-6)

    def test_fees_are_provably_charged(self):
        self.assertGreater(self.result["fees_paid"], 0.0)
        total = sum(t["pnl"] for t in self.result["trades"])
        # Gross (pre-fee) P&L must exceed net by exactly the fees.
        self.assertAlmostEqual(total + self.result["fees_paid"],
                               self.summary["gross_pnl"], delta=1e-6)
        self.assertGreater(self.summary["gross_pnl"], total)

    def test_every_round_trip_has_two_fills(self):
        self.assertEqual(self.result["fills"], 2 * self.summary["round_trips"])

    def test_no_position_survives_the_end(self):
        self.assertFalse(self.result["equity_curve"][-1]["in_market"])
        self.assertAlmostEqual(
            self.result["equity_curve"][-1]["equity"],
            self.result["final_cash"],
            delta=1e-9,
        )

    def test_trades_do_not_overlap(self):
        for a, b in zip(self.result["trades"], self.result["trades"][1:]):
            self.assertLess(a["exit_bar"], b["entry_bar"],
                            "one position at a time, no pyramiding")

    def test_every_trade_enters_above_the_swing_high_it_broke(self):
        for trade in self.result["trades"]:
            self.assertIsNotNone(trade["swing_high"])
            self.assertGreater(trade["entry_price"], trade["swing_high"])

    def test_exit_reasons_are_known(self):
        allowed = {mss.REASON_TRAILING_STOP, mss.REASON_BEARISH_MSS, "END_OF_DATA"}
        for trade in self.result["trades"]:
            self.assertIn(trade["exit_reason"], allowed)

    def test_stop_never_fills_above_its_own_level(self):
        for trade in self.result["trades"]:
            if trade["exit_reason"] != mss.REASON_TRAILING_STOP:
                continue
            stop = trade["peak_price"] * (1.0 - self.result["trail_pct"])
            # A gap fills at the open, so the price may be below the stop, never
            # above. Compared relatively: a 1e-9 absolute delta is smaller than
            # one ulp at a price of ~91, so it would fail on rounding alone.
            self.assertLessEqual(trade["exit_price"], stop * (1 + 1e-9))

    def test_determinism(self):
        again, hold_again = run(self.candles)
        self.assertEqual(self.result["trades"], again["trades"])
        self.assertEqual(self.result["final_cash"], again["final_cash"])
        self.assertEqual(self.hold["pnl"], hold_again["pnl"])

    def test_equity_curve_covers_every_traded_bar(self):
        curve = self.result["equity_curve"]
        self.assertTrue(all(curve[i]["bar"] < curve[i + 1]["bar"]
                            for i in range(len(curve) - 1)))
        self.assertEqual(curve[-1]["bar"], self.result["bars"] - 1)

    def test_equity_is_cash_plus_holdings(self):
        for point in self.result["equity_curve"]:
            self.assertAlmostEqual(point["equity"],
                                   point["cash"] + point["market_value"], delta=1e-9)


class TestNoLookaheadInReplay(unittest.TestCase):
    """Truncating the future must not change any decision already made."""

    def test_precomputed_levels_match_per_bar_rebuild(self):
        candles = synthetic_candles(n=400, seed=9)
        with bt.sandboxed_engine():
            fast = bm.run_backtest(candles)
        # Force the naive path by hiding the levels argument from evaluate().
        original = mss.evaluate
        try:
            def rebuild(cs, **kw):
                kw.pop("levels", None)
                return original(cs, **kw)
            mss.evaluate = rebuild
            with bt.sandboxed_engine():
                slow = bm.run_backtest(candles)
        finally:
            mss.evaluate = original
        self.assertEqual(fast["trades"], slow["trades"])
        self.assertEqual(fast["final_cash"], slow["final_cash"])

    def test_appending_the_future_does_not_rewrite_the_past(self):
        early = synthetic_candles(n=300, seed=21)
        longer = synthetic_candles(n=420, seed=21)
        # Same prefix by construction, so any difference is lookahead.
        self.assertEqual(early, longer[:300])

        with bt.sandboxed_engine():
            a = bm.run_backtest(early)
        with bt.sandboxed_engine():
            b = bm.run_backtest(longer)

        def up_to(result, bar):
            return [t for t in result["trades"] if t["exit_bar"] < bar]

        self.assertEqual(up_to(a, 299), up_to(b, 299))

    def test_levels_length_mismatch_is_an_error(self):
        candles = synthetic_candles(n=60, seed=4)
        with self.assertRaises(mss.StrategyError):
            mss.evaluate(candles, levels=mss.swing_levels(candles, 2, 2)[:10])


class TestGuards(unittest.TestCase):

    def test_too_few_candles(self):
        with self.assertRaises(bm.BacktestError):
            bm.run_backtest(synthetic_candles(n=4))

    def test_non_positive_size(self):
        with self.assertRaises(bm.BacktestError):
            bm.run_backtest(synthetic_candles(n=50), size_usd=0.0)

    def test_cash_must_cover_size(self):
        with self.assertRaises(bm.BacktestError):
            bm.run_backtest(synthetic_candles(n=50), size_usd=50.0, start_usdt=10.0)

    def test_simulation_must_be_disabled(self):
        with mock.patch.object(te, "TRADING_ENABLED", True):
            self.assertEqual(bm.main(["--bars", "60"]), 2)

    def test_bad_trail_is_refused(self):
        self.assertEqual(bm.main(["--trail", "1.5"]), 2)
        self.assertEqual(bm.main(["--trail", "0"]), 2)

    def test_not_enough_bars_for_the_fractal(self):
        self.assertEqual(bm.main(["--bars", "4", "--left", "2", "--right", "2"]), 2)

    def test_live_ledger_is_never_written(self):
        before = (te.LEDGER_PATH, te.TRADES_PATH)
        run(synthetic_candles(n=120))
        self.assertEqual((te.LEDGER_PATH, te.TRADES_PATH), before)


class TestSummary(unittest.TestCase):

    def test_win_loss_split_is_consistent(self):
        result, hold = run(synthetic_candles(n=600, seed=12))
        s = bm.summarise(result, hold)
        self.assertEqual(s["wins"] + s["losses"], s["round_trips"])
        self.assertAlmostEqual(
            s["win_rate"], s["wins"] / s["round_trips"] * 100.0, delta=1e-9
        )
        self.assertAlmostEqual(s["gross_win"] - s["gross_loss"],
                               s["total_pnl"], delta=1e-6)

    def test_profit_factor_matches_the_trade_list(self):
        result, hold = run(synthetic_candles(n=600, seed=13))
        s = bm.summarise(result, hold)
        gains = sum(t["pnl"] for t in result["trades"] if t["pnl"] > 0)
        losses = abs(sum(t["pnl"] for t in result["trades"] if t["pnl"] <= 0))
        if losses > 0:
            self.assertAlmostEqual(s["profit_factor"], gains / losses, delta=1e-9)
        else:
            self.assertIsNone(s["profit_factor"])

    def test_exit_reason_counts_sum_to_the_trade_count(self):
        result, hold = run(synthetic_candles(n=600, seed=14))
        s = bm.summarise(result, hold)
        self.assertEqual(sum(s["exit_reasons"].values()), s["round_trips"])

    def test_report_renders(self):
        result, hold = run(synthetic_candles(n=300, seed=15))
        text = bm.format_report(bm.summarise(result, hold), result)
        for needle in ("win rate", "net P&L", "max drawdown", "profit factor",
                       "T-stat", "buy & hold", "verdict"):
            self.assertIn(needle, text)
        self.assertIn("TRADING_ENABLED=False", text)

    def test_verdict_does_not_overclaim_on_noise(self):
        result, hold = run(synthetic_candles(n=900, seed=3))
        s = bm.summarise(result, hold)
        verdict = bm._verdict(s)
        if s["t_stat"] is not None and abs(s["t_stat"]) < 1.0:
            self.assertIn("noise", verdict)
            self.assertNotIn("significant", verdict.replace("short of significance", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
