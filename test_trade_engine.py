#!/usr/bin/env python3
"""test_trade_engine.py — behavioural tests for the paper trading engine.

Run:  python3 test_trade_engine.py -v

Every test redirects the engine's ledger and trade log into a temp directory,
so a test run never touches paper_ledger.json or paper_trades.jsonl.
Two tests reach Binance for a live price; the rest inject one so the
arithmetic is checked exactly.

Conventions under test (see trade_engine's docstring):
  cost_basis = cash actually spent, fee included
  avg_cost   = fee-inclusive break-even price
  realized   = proceeds net of fee - basis sold  ==  change in cash
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import trade_engine as te

F = te.FEE_RATE          # 0.001 per side


class EngineTestCase(unittest.TestCase):
    """Base: isolate the ledger + trade log, start from a clean $100."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._saved = (te.LEDGER_PATH, te.TRADES_PATH)
        te.LEDGER_PATH = self.tmp / "paper_ledger.json"
        te.TRADES_PATH = self.tmp / "paper_trades.jsonl"
        te.reset()

    def tearDown(self) -> None:
        te.LEDGER_PATH, te.TRADES_PATH = self._saved
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------- #

    def ledger(self) -> dict:
        return json.loads(te.LEDGER_PATH.read_text())

    def trades(self) -> list[dict]:
        text = te.TRADES_PATH.read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestSymbolsAndFees(unittest.TestCase):
    def test_symbol_forms(self):
        for given in ("BNB", "bnb", "BNBUSDT", "BNB/USDT", "BNB-USDT", " bnb/usdt "):
            self.assertEqual(te.normalise_symbol(given), "BNB/USDT", given)
        self.assertEqual(te.binance_symbol("bnb"), "BNBUSDT")

    def test_bad_symbols_are_rejected_not_repaired(self):
        # A typo must fail loudly. Silently stripping the hyphen would turn
        # "BT-C" into a real BTC quote and mislead the ledger.
        for bad in ("", "  ", "BT-C", "BTC/USDC", "BTC/EUR", "B NB!"):
            with self.assertRaises(ValueError, msg=bad):
                te.normalise_symbol(bad)

    def test_simulation_switches(self):
        self.assertFalse(te.TRADING_ENABLED, "the engine must stay in simulation")
        self.assertEqual(te.FEE_RATE, 0.001)
        self.assertEqual(te.STARTING_USDT, 100.0)

    def test_live_trading_flag_blocks_money_functions(self):
        te.TRADING_ENABLED = True
        try:
            with self.assertRaises(RuntimeError):
                te.sim_buy("BNB", 10.0, _price=700.0)
            with self.assertRaises(RuntimeError):
                te.sim_sell("BNB", 1.0, _price=700.0)
        finally:
            te.TRADING_ENABLED = False


class TestLedgerInit(EngineTestCase):
    def test_starts_at_one_hundred_usdt(self):
        ledger = self.ledger()
        self.assertEqual(ledger["usdt"], 100.0)
        self.assertEqual(ledger["starting_usdt"], 100.0)
        self.assertEqual(ledger["mode"], "paper")
        self.assertEqual(ledger["positions"], {})
        self.assertEqual(ledger["realized_pnl"], 0.0)
        self.assertEqual(ledger["fills"], 0)

    def test_ledger_is_created_when_absent(self):
        te.LEDGER_PATH.unlink()
        self.assertEqual(te.load_ledger()["usdt"], 100.0)
        self.assertTrue(te.LEDGER_PATH.exists())

    def test_ledger_file_is_valid_json_on_disk(self):
        raw = te.LEDGER_PATH.read_text()
        self.assertTrue(raw.endswith("\n"))
        self.assertEqual(json.loads(raw)["version"], te.LEDGER_VERSION)


class TestBuySellArithmetic(EngineTestCase):
    """Deterministic: the price is injected, so every figure is exact."""

    PRICE = 700.0

    def test_buy_deducts_cash_and_opens_position(self):
        fill = te.sim_buy("BNB", 10.0, _price=self.PRICE)

        self.assertEqual(fill["side"], "buy")
        self.assertEqual(fill["symbol"], "BNB/USDT")
        self.assertEqual(fill["binance_symbol"], "BNBUSDT")
        self.assertAlmostEqual(fill["fee_usdt"], 0.01, places=12)
        self.assertAlmostEqual(fill["net_spend_usdt"], 9.99, places=12)
        self.assertAlmostEqual(fill["token_amount"], 9.99 / 700.0, places=12)

        ledger = self.ledger()
        self.assertAlmostEqual(ledger["usdt"], 90.0, places=10)
        self.assertAlmostEqual(ledger["fees_paid"], 0.01, places=12)
        self.assertEqual(ledger["fills"], 1)

        position = ledger["positions"]["BNB/USDT"]
        self.assertAlmostEqual(position["token_amount"], 9.99 / 700.0, places=12)
        self.assertAlmostEqual(position["cost_basis"], 10.0, places=12)
        # Fee-inclusive break-even, not the raw fill price.
        # avg_cost is recomputed from the 12dp-stored token amount, so compare
        # it to the ideal ratio with a tolerance a millionth wide.
        self.assertAlmostEqual(position["avg_cost"], 10.0 / (9.99 / 700.0), delta=1e-6)
        self.assertGreater(position["avg_cost"], 700.0)

    def test_round_trip_returns_100_minus_two_fees(self):
        """The spec: buy $10 of BNB, sell it straight back."""
        buy = te.sim_buy("BNB", 10.0, _price=self.PRICE)
        sell = te.sim_sell("BNB", buy["token_amount"], _price=self.PRICE)

        fee_buy, fee_sell = 0.01, 9.99 * F
        expected_cash = 100.0 - fee_buy - fee_sell

        ledger = self.ledger()
        self.assertAlmostEqual(ledger["usdt"], expected_cash, places=8)
        self.assertAlmostEqual(ledger["usdt"], 99.98001, places=8)
        self.assertAlmostEqual(ledger["realized_pnl"], -fee_buy - fee_sell, places=8)
        self.assertAlmostEqual(ledger["fees_paid"], fee_buy + fee_sell, places=8)
        self.assertEqual(ledger["fills"], 2)
        self.assertEqual(ledger["positions"], {}, "position should be closed out")

        # "Almost exactly $100": two 0.1% fees on a $10 round trip is
        # 1.999 cents, so within three cents and always below $100.
        self.assertLess(abs(ledger["usdt"] - 100.0), 0.03)
        self.assertLess(ledger["usdt"], 100.0)
        self.assertGreater(abs(ledger["usdt"] - 100.0), 0.01, "fees must actually be charged")

        # Realised P&L equals the actual change in cash — no phantom numbers.
        self.assertAlmostEqual(ledger["realized_pnl"], ledger["usdt"] - 100.0, places=8)

    def test_round_trip_profit_is_captured(self):
        buy = te.sim_buy("BNB", 50.0, _price=700.0)
        sell = te.sim_sell("BNB", buy["token_amount"], _price=770.0)

        expected = 50.0 * (1 - F) * (770.0 / 700.0) * (1 - F) - 50.0
        self.assertAlmostEqual(sell["realized_pnl"], expected, places=8)
        self.assertAlmostEqual(self.ledger()["usdt"], 100.0 + expected, places=8)
        self.assertGreater(self.ledger()["usdt"], 100.0)
        self.assertAlmostEqual(
            self.ledger()["realized_pnl"], self.ledger()["usdt"] - 100.0, places=8
        )

    def test_second_buy_averages_the_cost_basis(self):
        te.sim_buy("BNB", 10.0, _price=700.0)
        te.sim_buy("BNB", 10.0, _price=900.0)

        position = self.ledger()["positions"]["BNB/USDT"]
        tokens = 9.99 / 700.0 + 9.99 / 900.0
        self.assertAlmostEqual(position["token_amount"], tokens, places=12)
        self.assertAlmostEqual(position["cost_basis"], 20.0, places=10)
        self.assertAlmostEqual(position["avg_cost"], 20.0 / tokens, delta=1e-6)
        self.assertEqual(self.ledger()["fills"], 2)

    def test_partial_sell_keeps_position_and_avg_cost(self):
        buy = te.sim_buy("BNB", 20.0, _price=700.0)
        before = self.ledger()["positions"]["BNB/USDT"]["avg_cost"]
        te.sim_sell("BNB", buy["token_amount"] / 2, _price=700.0)

        position = self.ledger()["positions"]["BNB/USDT"]
        self.assertAlmostEqual(position["token_amount"], buy["token_amount"] / 2, places=12)
        self.assertAlmostEqual(position["avg_cost"], before, delta=1e-6)
        self.assertAlmostEqual(position["cost_basis"], 10.0, places=8)

    def test_partial_sell_basis_plus_remaining_equals_original(self):
        buy = te.sim_buy("BNB", 20.0, _price=700.0)
        sell = te.sim_sell("BNB", buy["token_amount"] / 4, _price=800.0)
        position = self.ledger()["positions"]["BNB/USDT"]
        self.assertAlmostEqual(
            sell["cost_basis_sold"] + position["cost_basis"], 20.0, places=8
        )

    def test_oversell_and_empty_positions_are_refused(self):
        with self.assertRaises(te.TradeError):
            te.sim_sell("BNB", 1.0, _price=700.0)
        te.sim_buy("BNB", 10.0, _price=700.0)
        with self.assertRaises(te.TradeError):
            te.sim_sell("BNB", 1.0, _price=700.0)   # more than held
        self.assertEqual(self.ledger()["usdt"], 90.0, "a refused sell must not move cash")
        self.assertEqual(self.ledger()["fills"], 1)

    def test_buying_more_than_cash_is_refused(self):
        with self.assertRaises(te.TradeError):
            te.sim_buy("BNB", 100.01, _price=700.0)
        self.assertEqual(self.ledger()["usdt"], 100.0)
        self.assertEqual(self.ledger()["positions"], {})

    def test_non_positive_sizes_are_refused(self):
        for bad in (0, -1, float("nan")):
            with self.assertRaises(te.TradeError, msg=bad):
                te.sim_buy("BNB", bad, _price=700.0)
            with self.assertRaises(te.TradeError, msg=bad):
                te.sim_sell("BNB", bad, _price=700.0)

    def test_buying_the_whole_balance_works(self):
        fill = te.sim_buy("BNB", 100.0, _price=700.0)
        self.assertAlmostEqual(fill["usdt_after"], 0.0, places=10)
        self.assertGreater(fill["token_amount"], 0)
        te.sim_sell("BNB", fill["token_amount"], _price=700.0)

    def test_price_change_between_buy_and_sell_shows_as_pnl(self):
        buy = te.sim_buy("BNB", 10.0, _price=700.0)
        sell = te.sim_sell("BNB", buy["token_amount"], _price=630.0)   # -10%
        self.assertLess(sell["realized_pnl"], -0.9)
        self.assertAlmostEqual(self.ledger()["usdt"], 100.0 + sell["realized_pnl"], places=8)


class TestTradeLog(EngineTestCase):
    def test_every_fill_is_appended_once(self):
        buy = te.sim_buy("BNB", 10.0, _price=700.0)
        te.sim_sell("BNB", buy["token_amount"], _price=700.0)

        rows = self.trades()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["side"] for row in rows], ["buy", "sell"])
        self.assertNotEqual(rows[0]["trade_id"], rows[1]["trade_id"])
        for row in rows:
            self.assertEqual(row["mode"], "paper")
            self.assertRegex(row["ts"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
            self.assertEqual(row["symbol"], "BNB/USDT")
            self.assertEqual(row["price"], 700.0)

        self.assertIsNone(rows[0]["realized_pnl"], "a buy realises nothing")
        self.assertAlmostEqual(rows[1]["realized_pnl"], -0.01999, places=8)
        self.assertAlmostEqual(rows[1]["proceeds_usdt"], 9.98001, places=8)

    def test_log_is_append_only_across_many_fills(self):
        for _ in range(3):
            buy = te.sim_buy("BNB", 5.0, _price=700.0)
            te.sim_sell("BNB", buy["token_amount"], _price=700.0)
        rows = self.trades()
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["side"] for r in rows], ["buy", "sell"] * 3)
        self.assertEqual(self.ledger()["fills"], 6)

    def test_reset_clears_log_and_balance(self):
        te.sim_buy("BNB", 10.0, _price=700.0)
        self.assertEqual(len(self.trades()), 1)
        te.reset()
        self.assertEqual(self.trades(), [])
        self.assertEqual(self.ledger()["usdt"], 100.0)
        self.assertEqual(self.ledger()["positions"], {})

    def test_read_trades_reports_missing_file_as_empty(self):
        te.TRADES_PATH.unlink()
        self.assertEqual(te.read_trades(), [])


class TestLivePrice(EngineTestCase):
    """Hits Binance. Skipped automatically when the network says no."""

    def test_binance_price_is_plausible(self):
        try:
            price = te.fetch_price("BNB")
        except te.PriceUnavailable as exc:
            self.skipTest(f"Binance unreachable: {exc}")
        self.assertGreater(price, 1.0)
        self.assertLess(price, 100_000.0)

    def test_live_round_trip_loses_only_fees_and_drift(self):
        """The spec's test against the real market, not an injected price."""
        try:
            te.fetch_price("BNB")
        except te.PriceUnavailable as exc:
            self.skipTest(f"Binance unreachable: {exc}")

        buy = te.sim_buy("BNB", 10.0)
        sell = te.sim_sell("BNB", buy["token_amount"])

        ledger = self.ledger()
        drift = abs(sell["price"] - buy["price"]) / buy["price"]
        expected = 100.0 - buy["fee_usdt"] - sell["fee_usdt"] - buy["net_spend_usdt"] * drift

        self.assertLess(abs(ledger["usdt"] - expected), 0.02)
        self.assertLess(abs(ledger["usdt"] - 100.0), 0.05, "balance should be back near $100")
        self.assertLess(ledger["usdt"], 100.0, "fees mean it can never come back above $100")
        self.assertEqual(ledger["positions"], {})

    def test_price_failure_is_raised_not_guessed(self):
        def boom(request, timeout=None):
            raise OSError("network down")

        with self.assertRaises(te.PriceUnavailable):
            te.fetch_price("BNB", _opener=boom)

    def test_binance_rejection_is_not_silently_zero(self):
        import urllib.error

        def not_found(request, timeout=None):
            raise urllib.error.HTTPError(
                getattr(request, "full_url", "url"), 400, "Bad Request", {}, None
            )

        with self.assertRaises(te.PriceUnavailable):
            te.fetch_price("BNB", _opener=not_found)


class TestStatus(EngineTestCase):
    def test_status_marks_to_last_fill_offline(self):
        te.sim_buy("BNB", 10.0, _price=700.0)
        snapshot = te.status(mark_to_market=False)

        self.assertEqual(snapshot["mode"], "paper")
        self.assertFalse(snapshot["trading_enabled"])
        self.assertEqual(snapshot["usdt"], 90.0)
        self.assertAlmostEqual(snapshot["equity"], 99.99, places=8)
        self.assertEqual(snapshot["positions"][0]["price_source"], "last_fill")
        self.assertAlmostEqual(snapshot["realized_pnl"], 0.0, places=12)

    def test_status_with_no_positions_is_all_cash(self):
        snapshot = te.status(mark_to_market=False)
        self.assertEqual(snapshot["equity"], 100.0)
        self.assertEqual(snapshot["positions"], [])
        self.assertEqual(snapshot["total_return_pct"], 0.0)


class TestAtomicCommit(EngineTestCase):
    """A refused or crashed trade must leave the disk exactly as it was.

    The old order was save_ledger() then record_trade(): a failing append left
    cash already spent on disk with no matching fill in the log.
    """

    def _fail_save(self):
        real = te.save_ledger

        def boom(ledger):
            raise OSError("disk full")

        te.save_ledger = boom
        self.addCleanup(setattr, te, "save_ledger", real)

    def test_buy_write_failure_rolls_everything_back(self):
        before = self.ledger()
        log_before = te.TRADES_PATH.stat().st_size if te.TRADES_PATH.exists() else 0

        self._fail_save()
        with self.assertRaises(OSError):
            te.sim_buy("BNB", 10.0, _price=700.0)

        after = self.ledger()
        self.assertEqual(after["usdt"], before["usdt"], "cash moved on a failed commit")
        self.assertEqual(after["positions"], {}, "position written on a failed commit")
        self.assertEqual(after["fills"], before["fills"])
        size = te.TRADES_PATH.stat().st_size if te.TRADES_PATH.exists() else 0
        self.assertEqual(size, log_before, "orphan trade-log line left behind")

    def test_sell_write_failure_rolls_everything_back(self):
        fill = te.sim_buy("BNB", 10.0, _price=700.0)
        before = self.ledger()
        log_before = te.TRADES_PATH.stat().st_size

        self._fail_save()
        with self.assertRaises(OSError):
            te.sim_sell("BNB", fill["token_amount"], _price=700.0)

        after = self.ledger()
        self.assertEqual(after["usdt"], before["usdt"], "cash changed on a failed sell")
        self.assertEqual(after["positions"], before["positions"], "position changed")
        self.assertEqual(te.TRADES_PATH.stat().st_size, log_before)

    def test_insufficient_funds_writes_nothing(self):
        with self.assertRaises(te.TradeError):
            te.sim_buy("BNB", 10_000.0, _price=700.0)
        self.assertEqual(self.ledger()["usdt"], 100.0)
        self.assertEqual(te.TRADES_PATH.stat().st_size, 0)

    def test_price_failure_writes_nothing(self):
        real = te.fetch_price
        te.fetch_price = lambda symbol, **kw: (_ for _ in ()).throw(
            te.PriceUnavailable("network down")
        )
        self.addCleanup(setattr, te, "fetch_price", real)

        with self.assertRaises(te.PriceUnavailable):
            te.sim_buy("BNB", 10.0)
        self.assertEqual(self.ledger()["usdt"], 100.0)
        self.assertEqual(te.TRADES_PATH.stat().st_size, 0)

    def test_dust_buy_is_a_trade_error_not_a_crash(self):
        # 1e-9 USDT at 1e12 price rounds to zero quantity; dividing cost_basis
        # by it used to raise ZeroDivisionError from inside the engine.
        with self.assertRaises(te.TradeError):
            te.sim_buy("BNB", 1e-9, _price=1e12)
        self.assertEqual(self.ledger()["usdt"], 100.0)

    def test_successful_buy_keeps_ledger_and_log_in_step(self):
        fill = te.sim_buy("BNB", 10.0, _price=700.0)
        ledger = self.ledger()
        lines = [json.loads(l) for l in te.TRADES_PATH.read_text().splitlines() if l.strip()]

        self.assertEqual(ledger["usdt"], 90.0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["trade_id"], fill["trade_id"])
        self.assertEqual(
            ledger["positions"]["BNB/USDT"]["token_amount"], fill["token_amount"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
