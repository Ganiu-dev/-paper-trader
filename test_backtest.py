#!/usr/bin/env python3
"""Tests for backtest.py — the MA-crossover replay and its execution model.

This file exists because the replay's fill model changed and nothing in the
suite was watching it. Before, a crossover seen at a bar's close was filled at
that same close: a price the strategy could only have acted on with hindsight.
Now the order is queued on the close and worked at the NEXT bar's open, and any
order still queued at the last bar is dropped rather than filled.

So the four things defended here are:

1. The fill model. Every entry and every signal-driven exit fills at the open of
   the bar after the one that produced the signal — never at the signal's own
   close.
2. The dropped order. A signal on the final bar produces no fill, and it is
   counted in ``skipped_no_next_bar`` instead of vanishing quietly.
3. The accounting. Summed trade P&L equals the change in cash, fees are real,
   and one position at a time.
4. The benchmark is untouched. ``buy_and_hold`` still enters at the first close,
   because it is a comparison point, not a strategy with a signal.

No network, no waiting: the candles are synthetic and deterministic.
"""

from __future__ import annotations

import math
import random
import unittest

import backtest as bt
import strategy


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def synthetic_candles(n: int = 400, seed: int = 7, start: float = 100.0,
                      bar_ms: int = 60 * 60 * 1000) -> list[dict]:
    """Deterministic candles with alternating drift, so crossovers happen.

    Every bar opens a gap away from the previous close on purpose. The shared
    fixtures elsewhere set ``open == close``, which makes a fill-at-close and a
    fill-at-next-open the same number and every assertion below vacuous. A gap
    is also the honest case: a bar that opens exactly where the last one closed
    cannot tell the two models apart, and real bars rarely oblige.
    """
    rng = random.Random(seed)
    t0 = 1_600_000_000_000
    price = start
    out = []
    for i in range(n):
        drift = 0.004 if (i // 40) % 2 == 0 else -0.005
        open_price = price * math.exp(rng.gauss(0, 0.002))
        price = open_price * math.exp(drift + rng.gauss(0, 0.006))
        high = max(open_price, price) * (1 + abs(rng.gauss(0, 0.003)))
        low = min(open_price, price) * (1 - abs(rng.gauss(0, 0.003)))
        out.append(
            {
                "open_time": t0 + i * bar_ms,
                "close_time": t0 + (i + 1) * bar_ms - 1,
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": 1.0,
            }
        )
    return out


def run(candles, **kw):
    with bt.sandboxed_engine():
        result = bt.run_backtest(candles, **kw)
        hold = bt.buy_and_hold(candles, start_usdt=kw.get("start_usdt", bt.START_USDT))
    return result, hold


# --------------------------------------------------------------------------- #
# The execution model
# --------------------------------------------------------------------------- #

class TestExecutionModel(unittest.TestCase):
    """The reason this file was written."""

    @classmethod
    def setUpClass(cls):
        cls.candles = synthetic_candles()
        cls.result, cls.hold = run(cls.candles)

    def test_it_actually_trades(self):
        self.assertGreater(len(self.result["trades"]), 3,
                           "a flat result would make every assertion below vacuous")
        self.assertEqual(self.result["fill_model"], "next_open")

    def test_no_trade_fills_at_the_close_of_its_own_signal_bar(self):
        for trade in self.result["trades"]:
            signal_close = float(self.candles[trade["signal_bar"]]["close"])
            self.assertNotEqual(trade["entry_price"], signal_close,
                                "filled at the close that produced the signal")

    def test_every_entry_fills_at_the_open_of_the_next_bar(self):
        for trade in self.result["trades"]:
            self.assertEqual(trade["entry_bar"], trade["signal_bar"] + 1)
            self.assertEqual(
                trade["entry_price"],
                float(self.candles[trade["entry_bar"]]["open"]),
            )
            self.assertEqual(
                trade["entry_time"], self.candles[trade["entry_bar"]]["open_time"]
            )

    def test_every_signal_exit_fills_at_the_open_of_the_next_bar(self):
        for trade in self.result["trades"]:
            if trade.get("forced_close"):
                continue
            self.assertEqual(trade["exit_bar"], trade["exit_signal_bar"] + 1)
            self.assertEqual(
                trade["exit_price"],
                float(self.candles[trade["exit_bar"]]["open"]),
            )

    def test_no_fill_happens_on_the_final_bar(self):
        for trade in self.result["trades"]:
            self.assertLess(trade["entry_bar"], len(self.candles) - 1,
                            "an entry filled on the last bar, where no open exists")
            if not trade.get("forced_close"):
                self.assertLess(trade["exit_bar"], len(self.candles) - 1)

    def test_a_signal_on_the_final_bar_is_dropped_and_counted(self):
        # Force the case instead of hoping the fixture ends on a crossover: cut
        # the series off at the bar that signalled the first real entry, so that
        # signal now arrives on the last bar. It must produce no fill and must
        # show up in the counter.
        signal_bar = self.result["trades"][0]["signal_bar"]
        truncated, _ = run(self.candles[: signal_bar + 1])
        self.assertEqual(truncated["trades"], [])
        self.assertGreaterEqual(truncated["skipped_no_next_bar"], 1)
        self.assertAlmostEqual(truncated["final_cash"], bt.START_USDT, delta=1e-9)

    def test_holding_period_is_measured_between_fill_bars(self):
        for trade in self.result["trades"]:
            self.assertEqual(
                trade["bars_held"], trade["exit_bar"] - trade["entry_bar"]
            )


# --------------------------------------------------------------------------- #
# The accounting
# --------------------------------------------------------------------------- #

class TestAccounting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.candles = synthetic_candles()
        cls.result, cls.hold = run(cls.candles)
        cls.summary = bt.summarise(cls.result, cls.hold)

    def test_pnl_reconciles_with_cash(self):
        total = sum(t["pnl"] for t in self.result["trades"])
        change = self.result["final_cash"] - self.result["start_usdt"]
        self.assertAlmostEqual(total, change, delta=1e-6)
        self.assertAlmostEqual(self.result["realized_pnl"], change, delta=1e-6)

    def test_fees_are_charged(self):
        self.assertGreater(self.result["fees_paid"], 0.0)
        total = sum(t["pnl"] for t in self.result["trades"])
        self.assertAlmostEqual(total + self.result["fees_paid"],
                               self.summary["gross_pnl"], delta=1e-6)

    def test_two_fills_per_round_trip(self):
        self.assertEqual(self.result["fills"], 2 * len(self.result["trades"]))

    def test_trades_do_not_overlap(self):
        for a, b in zip(self.result["trades"], self.result["trades"][1:]):
            self.assertLessEqual(a["exit_bar"], b["entry_bar"],
                                 "one position at a time, no pyramiding")

    def test_no_position_survives_the_end(self):
        curve = self.result["equity_curve"]
        self.assertFalse(curve[-1]["in_market"])
        self.assertAlmostEqual(curve[-1]["equity"], self.result["final_cash"], delta=1e-9)

    def test_equity_curve_covers_every_traded_bar(self):
        curve = self.result["equity_curve"]
        self.assertEqual(len(curve), self.result["bars"] - self.result["slow"])
        self.assertTrue(all(curve[i]["bar"] < curve[i + 1]["bar"]
                            for i in range(len(curve) - 1)))
        self.assertEqual(curve[-1]["bar"], self.result["bars"] - 1)

    def test_equity_is_cash_plus_holdings(self):
        for point in self.result["equity_curve"]:
            self.assertAlmostEqual(
                point["equity"], point["cash"] + point["market_value"], delta=1e-9
            )

    def test_determinism(self):
        again, hold_again = run(self.candles)
        self.assertEqual(self.result["trades"], again["trades"])
        self.assertEqual(self.result["final_cash"], again["final_cash"])
        self.assertEqual(self.hold["pnl"], hold_again["pnl"])

    def test_sizing_is_respected(self):
        for trade in self.result["trades"]:
            self.assertAlmostEqual(trade["usdt_in"], bt.SIZE_USD, delta=1e-9)


# --------------------------------------------------------------------------- #
# The benchmark and the guards
# --------------------------------------------------------------------------- #

class TestBenchmarkAndGuards(unittest.TestCase):

    def test_buy_and_hold_still_enters_at_the_first_close(self):
        # Deliberately not moved to a next-open fill: it is the comparison
        # point, and shifting it would change every difference_pct in the
        # README against a strategy whose own first entry needs slow+1 bars.
        candles = synthetic_candles()
        result, hold = run(candles)
        first = float(candles[0]["close"])
        last = float(candles[-1]["close"])
        expected = bt.START_USDT * (1 - bt.te.FEE_RATE) * (last / first) * (1 - bt.te.FEE_RATE)
        self.assertAlmostEqual(hold["pnl"], expected - bt.START_USDT, delta=0.02)

    def test_too_few_candles_is_refused(self):
        with self.assertRaises(bt.BacktestError):
            run(synthetic_candles(n=5), slow=strategy.MA_SLOW)

    def test_a_replay_with_no_bar_to_fill_in_is_refused(self):
        # slow+1 bars means the last bar is a decision bar and nothing follows
        # it: every order would be dropped and "0 trades" would be reported as
        # a finding. It is not one, so it is refused instead.
        with self.assertRaises(bt.BacktestError):
            run(synthetic_candles(n=strategy.MA_SLOW + 1), slow=strategy.MA_SLOW)

    def test_size_must_be_positive(self):
        with self.assertRaises(bt.BacktestError):
            run(synthetic_candles(), size_usd=0.0)

    def test_starting_cash_must_fund_one_entry(self):
        with self.assertRaises(bt.BacktestError):
            run(synthetic_candles(), size_usd=10.0, start_usdt=5.0)

    def test_live_ledger_is_never_written(self):
        before = bt.te.LEDGER_PATH.read_bytes() if bt.te.LEDGER_PATH.exists() else None
        run(synthetic_candles())
        after = bt.te.LEDGER_PATH.read_bytes() if bt.te.LEDGER_PATH.exists() else None
        self.assertEqual(before, after, "the replay wrote to the owner's live ledger")


if __name__ == "__main__":
    unittest.main(verbosity=2)
