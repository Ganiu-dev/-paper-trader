#!/usr/bin/env python3
"""test_run_paper_trader.py — the runner's contract with the ledger.

The bug this guards against: held_amount() looked positions up as
f"{pair}/USDT" without normalising first. Given --symbol btc it asked the
ledger for "btc/USDT", got 0.0 back, concluded it was flat, never sold the
"BTC/USDT" position it had bought, and left it orphaned on disk.

Run:  python3 test_run_paper_trader.py -v
No network: candles are injected and prices are passed as _price.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run_paper_trader as rpt
import strategy
import trade_engine as te


def flat_candles(n: int = 60) -> list[dict]:
    return [
        {
            "open_time": i * 3_600_000,
            "close_time": (i + 1) * 3_600_000 - 1,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
        for i in range(n)
    ]


class RunnerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._saved = (te.LEDGER_PATH, te.TRADES_PATH, rpt.STATE_PATH)
        te.LEDGER_PATH = self.tmp / "paper_ledger.json"
        te.TRADES_PATH = self.tmp / "paper_trades.jsonl"
        rpt.STATE_PATH = self.tmp / "runner_state.json"
        te.reset()

    def tearDown(self) -> None:
        te.LEDGER_PATH, te.TRADES_PATH, rpt.STATE_PATH = self._saved
        self._tmp.cleanup()

    def stub_signal(self, signal: str):
        """Force strategy.evaluate to return one verdict, whatever the candles."""
        real = strategy.evaluate

        def forced(candles, **kwargs):
            verdict = real(flat_candles(), fast=2, slow=3)
            verdict.update(
                signal=signal,
                candle_close_time=999,
                pair=kwargs.get("pair", "BTC/USDT"),
            )
            return verdict

        strategy.evaluate = forced
        self.addCleanup(setattr, strategy, "evaluate", real)


class TestSymbolNormalisation(RunnerTestCase):
    def test_held_amount_finds_the_position_for_every_accepted_form(self):
        te.sim_buy("BTC", 10.0, _price=75_000.0)
        self.assertEqual(list(te.load_ledger()["positions"]), ["BTC/USDT"])

        for form in ("btc", "BTC", "btcusdt", "btc/usdt", " btc-usdt "):
            with self.subTest(form=form):
                self.assertGreater(rpt.held_amount(form), 0.0, f"{form!r} read as flat")

    def test_lowercase_pair_can_sell_what_it_bought(self):
        te.sim_buy("BTC", 10.0, _price=75_000.0)
        self.stub_signal(strategy.SELL)

        record = rpt.run_once(pair="btc", candles=flat_candles(), state={"acted": []})

        self.assertEqual(record["action"], "SOLD", record["reason"])
        self.assertEqual(te.load_ledger()["positions"], {}, "position orphaned")
        self.assertEqual(record["result"]["pair"], "BTC/USDT")

    def test_lowercase_pair_does_not_pyramid(self):
        te.sim_buy("BTC", 10.0, _price=75_000.0)
        self.stub_signal(strategy.BUY)

        record = rpt.run_once(pair="btc", candles=flat_candles(), state={"acted": []})

        self.assertEqual(record["action"], "NONE")
        self.assertIn("already holding", record["reason"])
        self.assertAlmostEqual(te.load_ledger()["usdt"], 90.0, places=8)

    def test_buy_from_lowercase_pair_lands_in_the_canonical_position(self):
        self.stub_signal(strategy.BUY)
        real_buy = te.sim_buy

        def cheap_buy(symbol, amount, **kwargs):
            return real_buy(symbol, amount, _price=75_000.0)

        te.sim_buy = cheap_buy
        self.addCleanup(setattr, te, "sim_buy", real_buy)

        record = rpt.run_once(pair="btc", candles=flat_candles(), state={"acted": []})

        self.assertEqual(record["action"], "BOUGHT", record["reason"])
        self.assertEqual(list(te.load_ledger()["positions"]), ["BTC/USDT"])
        self.assertIn("BTC", record["reason"])
        self.assertNotIn("/USDT", record["reason"], "display should use the base asset")

    def test_bad_ticker_exits_before_the_loop(self):
        self.assertEqual(rpt.main(["--symbol", "BT-C", "--once"]), 2)

    def test_extra_quote_form_is_accepted(self):
        # 'btc-usdt' is a recognised input; it must not be a different position.
        te.sim_buy("BTC", 10.0, _price=75_000.0)
        self.assertGreater(rpt.held_amount("btc-usdt"), 0.0)


class TestLatchStillWorks(RunnerTestCase):
    def test_same_candle_is_not_bought_twice(self):
        self.stub_signal(strategy.BUY)
        real_buy = te.sim_buy
        te.sim_buy = lambda symbol, amount, **kw: real_buy(symbol, amount, _price=700.0)
        self.addCleanup(setattr, te, "sim_buy", real_buy)

        state = {"acted": []}
        first = rpt.run_once(pair="BNB", candles=flat_candles(), state=state)
        # Sell it off so only the latch, not the position check, can stop us.
        te.sim_sell("BNB", rpt.held_amount("BNB"), _price=700.0)
        second = rpt.run_once(pair="BNB", candles=flat_candles(), state=state)

        self.assertEqual(first["action"], "BOUGHT")
        self.assertEqual(second["action"], "NONE")
        self.assertIn("already acted on", second["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
