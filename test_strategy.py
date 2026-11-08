#!/usr/bin/env python3
"""test_strategy.py — behavioural tests for T3 (strategy + runner).

Run:  python3 test_strategy.py -v

No network except one opt-out live test. Candles are built by hand so every
crossover happens at a bar I chose, and the runner writes to a temporary
ledger, so a test run never touches paper_ledger.json, paper_trades.jsonl or
the runner's latch file.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

import run_paper_trader as rp
import strategy as st
import trade_engine as te

HOUR_MS = 3_600_000
START_MS = 1_700_000_000_000


def candles_from(closes: list[float], *, start_ms: int = START_MS) -> list[dict]:
    """Closed candles from a list of closes, one hour apart."""
    out = []
    for i, close in enumerate(closes):
        open_time = start_ms + i * HOUR_MS
        out.append(
            {
                "open_time": open_time,
                "open": float(close),
                "high": float(close),
                "low": float(close),
                "close": float(close),
                "volume": 1.0,
                "close_time": open_time + HOUR_MS - 1,
            }
        )
    return out


FLAT = [700.0] * 40
# A flat stretch, then one big candle: the 10-bar average jumps over the
# 30-bar average on the final bar and on no other.
CROSS_UP_CLOSES = FLAT + [760.0]
CROSS_DOWN_CLOSES = FLAT + [640.0]
UP_TREND_CLOSES = [700.0 + i for i in range(50)]   # fast stays above slow throughout
FLAT_CLOSES = [700.0] * 50


class _Response:
    """Stands in for the object urlopen returns."""

    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


def opener_for(rows):
    return lambda request, timeout=None: _Response(json.dumps(rows).encode())


def opener_raising(exc):
    def boom(request, timeout=None):
        raise exc

    return boom


def klines_rows(closes: list[float], *, first_open_ms: int = START_MS) -> list[list]:
    """Binance-shaped kline rows (12 fields) for a list of closes."""
    rows = []
    for i, close in enumerate(closes):
        open_time = first_open_ms + i * HOUR_MS
        rows.append([
            open_time, str(close), str(close), str(close), str(close), "1.0",
            open_time + HOUR_MS - 1, "0", 1, "0", "0", "0",
        ])
    return rows


# --------------------------------------------------------------------------- #
# The indicator
# --------------------------------------------------------------------------- #

class TestSMA(unittest.TestCase):
    def test_values_are_aligned_and_correct(self):
        series = st.sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(series[:2], [None, None])
        self.assertAlmostEqual(series[2], 2.0)
        self.assertAlmostEqual(series[3], 3.0)
        self.assertAlmostEqual(series[4], 4.0)

    def test_short_series_gives_nones_not_an_error(self):
        self.assertEqual(st.sma([1, 2], 5), [None, None])

    def test_zero_period_is_refused(self):
        with self.assertRaises(ValueError):
            st.sma([1, 2, 3], 0)


class TestEvaluate(unittest.TestCase):
    def test_cross_above_is_a_buy(self):
        result = st.evaluate(candles_from(CROSS_UP_CLOSES), fast=10, slow=30, pair="BNB/USDT")
        self.assertEqual(result["signal"], st.BUY)
        self.assertTrue(result["crossed_up"])
        self.assertFalse(result["crossed_down"])
        self.assertEqual(result["trend"], "UP")
        self.assertLessEqual(result["prev_fast_ma"], result["prev_slow_ma"])
        self.assertGreater(result["fast_ma"], result["slow_ma"])
        self.assertEqual(result["price"], 760.0)

    def test_cross_below_is_a_sell(self):
        result = st.evaluate(candles_from(CROSS_DOWN_CLOSES), fast=10, slow=30, pair="BNB/USDT")
        self.assertEqual(result["signal"], st.SELL)
        self.assertTrue(result["crossed_down"])
        self.assertFalse(result["crossed_up"])
        self.assertEqual(result["trend"], "DOWN")
        self.assertGreaterEqual(result["prev_fast_ma"], result["prev_slow_ma"])
        self.assertLess(result["fast_ma"], result["slow_ma"])

    def test_a_trend_without_a_crossing_is_hold(self):
        """Fast above slow is a state that lasts hours; it must not re-fire."""
        result = st.evaluate(candles_from(UP_TREND_CLOSES), fast=10, slow=30)
        self.assertEqual(result["signal"], st.HOLD)
        self.assertEqual(result["trend"], "UP")
        self.assertFalse(result["crossed_up"])

    def test_the_crossing_bar_is_the_only_bar_that_fires(self):
        closes = FLAT + [760.0, 761.0, 762.0]
        signals = [
            st.evaluate(candles_from(closes[:end]), fast=10, slow=30)["signal"]
            for end in range(41, len(closes) + 1)
        ]
        self.assertEqual(signals, [st.BUY, st.HOLD, st.HOLD])

    def test_flat_market_is_hold(self):
        result = st.evaluate(candles_from(FLAT_CLOSES), fast=10, slow=30)
        self.assertEqual(result["signal"], st.HOLD)
        self.assertEqual(result["trend"], "FLAT")
        self.assertEqual(result["gap"], 0.0)

    def test_candle_metadata_is_reported(self):
        result = st.evaluate(candles_from(CROSS_UP_CLOSES), fast=10, slow=30, pair="BNB/USDT")
        self.assertEqual(result["candles"], len(CROSS_UP_CLOSES))
        last_open = START_MS + (len(CROSS_UP_CLOSES) - 1) * HOUR_MS
        self.assertEqual(result["candle_close_time"], last_open + HOUR_MS - 1)
        self.assertRegex(result["candle_close_iso"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        self.assertRegex(result["candle_close_clock"], r"^\d{2}:\d{2}$")
        self.assertEqual(result["candle_open_time"], START_MS + (len(CROSS_UP_CLOSES) - 1) * HOUR_MS)

    def test_minimum_candle_count_is_slow_plus_one(self):
        with self.assertRaises(st.StrategyError):
            st.evaluate(candles_from(FLAT_CLOSES[:30]), fast=10, slow=30)
        st.evaluate(candles_from(FLAT_CLOSES[:31]), fast=10, slow=30)   # exactly enough

    def test_fast_must_be_shorter_than_slow(self):
        for fast, slow in ((30, 30), (40, 30), (0, 30), (10, 0)):
            with self.assertRaises(st.StrategyError, msg=f"fast={fast} slow={slow}"):
                st.evaluate(candles_from(FLAT_CLOSES), fast=fast, slow=slow)

    def test_describe_is_one_line_with_the_verdict(self):
        result = st.evaluate(candles_from(CROSS_UP_CLOSES), fast=10, slow=30, pair="BNB/USDT")
        line = st.describe(result)
        for expected in ("BNB/USDT", "fast(10)", "slow(30)", "BUY"):
            self.assertIn(expected, line)
        self.assertNotIn("\n", line)


# --------------------------------------------------------------------------- #
# The HTTP boundary
# --------------------------------------------------------------------------- #

class TestKlines(unittest.TestCase):
    def test_rows_become_candles(self):
        rows = klines_rows([700.0, 701.0, 702.0])
        got = st.fetch_klines("BNB", limit=3, _opener=opener_for(rows))
        self.assertEqual([c["close"] for c in got], [700.0, 701.0, 702.0])
        self.assertEqual(got[0]["open_time"], START_MS)
        self.assertEqual(got[0]["close_time"], START_MS + HOUR_MS - 1)

    def test_the_forming_candle_is_dropped(self):
        """The open candle's close is just the live price and flips a crossing."""
        now_ms = st._now_ms()
        closed = klines_rows([700.0, 701.0], first_open_ms=now_ms - 5 * HOUR_MS)
        still_open = [[
            now_ms - 1000, "999", "999", "999", "999", "1",
            now_ms + HOUR_MS - 1, "0", 1, "0", "0", "0",
        ]]
        got = st.fetch_klines("BNB", limit=2, _opener=opener_for(closed + still_open))
        self.assertEqual([c["close"] for c in got], [700.0, 701.0])
        self.assertNotIn(999.0, [c["close"] for c in got])

    def test_open_candle_can_be_kept_on_request(self):
        now_ms = st._now_ms()
        rows = [[
            now_ms - 1000, "999", "999", "999", "999", "1",
            now_ms + HOUR_MS - 1, "0", 1, "0", "0", "0",
        ]]
        got = st.fetch_klines("BNB", limit=1, closed_only=False, _opener=opener_for(rows))
        self.assertEqual(got[-1]["close"], 999.0)

    def test_rows_are_sorted_by_open_time(self):
        rows = klines_rows([700.0, 701.0, 702.0])
        got = st.fetch_klines("BNB", limit=3, _opener=opener_for(list(reversed(rows))))
        self.assertEqual([c["close"] for c in got], [700.0, 701.0, 702.0])

    def test_duplicate_open_times_are_dropped(self):
        """A repeated bar would weight one candle twice inside every average."""
        rows = klines_rows([700.0, 701.0])
        got = st.fetch_klines("BNB", limit=5, _opener=opener_for(rows + rows[1:]))
        self.assertEqual(len(got), 2, "the repeated bar must be dropped, not counted twice")
        self.assertEqual([c["open_time"] for c in got], [rows[0][0], rows[1][0]])
        self.assertEqual([c["close"] for c in got], [700.0, 701.0])

    def test_http_error_is_raised_with_the_body(self):
        err = urllib.error.HTTPError("url", 400, "Bad Request", {}, None)
        err.read = lambda: b'{"code":-1121,"msg":"Invalid symbol."}'   # type: ignore[method-assign]
        with self.assertRaises(st.MarketDataUnavailable) as ctx:
            st.fetch_klines("BNB", _opener=opener_raising(err))
        self.assertIn("Invalid symbol", str(ctx.exception))

    def test_network_failure_is_raised(self):
        with self.assertRaises(st.MarketDataUnavailable):
            st.fetch_klines("BNB", _opener=opener_raising(OSError("network down")))

    def test_non_json_payload_is_refused(self):
        with self.assertRaises(st.MarketDataUnavailable):
            st.fetch_klines("BNB", _opener=lambda request, timeout=None: _Response(b"<html>"))

    def test_empty_payload_is_refused_not_treated_as_no_candles(self):
        with self.assertRaises(st.MarketDataUnavailable):
            st.fetch_klines("BNB", _opener=opener_for([]))

    def test_malformed_row_is_refused(self):
        with self.assertRaises(st.MarketDataUnavailable):
            st.fetch_klines("BNB", _opener=lambda request, timeout=None: _Response(b'[["123","700"]]'))

    def test_non_positive_close_is_refused(self):
        rows = klines_rows([700.0])
        rows[0][4] = "0"
        with self.assertRaises(st.MarketDataUnavailable):
            st.fetch_klines("BNB", _opener=opener_for(rows))

    def test_bad_limit_is_refused(self):
        with self.assertRaises(ValueError):
            st.fetch_klines("BNB", limit=0, _opener=opener_for(klines_rows([700.0])))

    def test_symbol_is_built_for_the_request(self):
        self.assertEqual(st.binance_symbol("BNB"), "BNBUSDT")
        self.assertEqual(st.binance_symbol("bnb/usdt"), "BNBUSDT")
        self.assertEqual(st.binance_symbol("BNB-USDT"), "BNBUSDT")
        with self.assertRaises(ValueError):
            st.binance_symbol("BT-C")


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #

class RunnerTestCase(unittest.TestCase):
    """Isolates the paper ledger, the trade log and the runner's latch."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._saved = (te.LEDGER_PATH, te.TRADES_PATH, rp.STATE_PATH)
        te.LEDGER_PATH = self.tmp / "paper_ledger.json"
        te.TRADES_PATH = self.tmp / "paper_trades.jsonl"
        rp.STATE_PATH = self.tmp / "paper_runner_state.json"
        te.reset()

    def tearDown(self) -> None:
        te.LEDGER_PATH, te.TRADES_PATH, rp.STATE_PATH = self._saved
        self._tmp.cleanup()

    def ledger(self) -> dict:
        return json.loads(te.LEDGER_PATH.read_text())

    def trade_rows(self) -> list[dict]:
        text = te.TRADES_PATH.read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestRunnerDecisions(RunnerTestCase):
    def test_buy_signal_opens_a_ten_dollar_position(self):
        record = rp.run_once(candles=candles_from(CROSS_UP_CLOSES))
        self.assertEqual(record["signal"], st.BUY)
        self.assertEqual(record["action"], "BOUGHT")

        ledger = self.ledger()
        self.assertEqual(ledger["fills"], 1)
        self.assertAlmostEqual(ledger["usdt"], 90.0, places=8)
        self.assertAlmostEqual(ledger["positions"]["BNB/USDT"]["cost_basis"], 10.0, places=8)
        self.assertEqual(len(self.trade_rows()), 1)

    def test_sell_signal_closes_the_whole_position(self):
        rp.run_once(candles=candles_from(CROSS_UP_CLOSES))
        record = rp.run_once(candles=candles_from(CROSS_DOWN_CLOSES))
        self.assertEqual(record["signal"], st.SELL)
        self.assertEqual(record["action"], "SOLD")

        ledger = self.ledger()
        self.assertEqual(ledger["positions"], {}, "a sell signal must fully close")
        self.assertEqual(ledger["fills"], 2)
        self.assertEqual(len(self.trade_rows()), 2)

    def test_the_same_candle_is_not_bought_twice(self):
        """The crossing candle stays 'the last closed one' for up to an hour."""
        candles = candles_from(CROSS_UP_CLOSES)
        first = rp.run_once(candles=candles)
        second = rp.run_once(candles=candles)

        self.assertEqual(first["action"], "BOUGHT")
        self.assertEqual(second["action"], "NONE")
        self.assertEqual(self.ledger()["fills"], 1)
        self.assertAlmostEqual(self.ledger()["usdt"], 90.0, places=8)

    def test_no_pyramiding_while_holding(self):
        te.sim_buy("BNB", 10.0, _price=700.0)
        record = rp.run_once(candles=candles_from(CROSS_UP_CLOSES))
        self.assertEqual(record["signal"], st.BUY)
        self.assertEqual(record["action"], "NONE")
        self.assertIn("holding", record["reason"])
        self.assertEqual(self.ledger()["fills"], 1)

    def test_sell_while_flat_does_nothing(self):
        record = rp.run_once(candles=candles_from(CROSS_DOWN_CLOSES))
        self.assertEqual(record["signal"], st.SELL)
        self.assertEqual(record["action"], "NONE")
        self.assertIn("flat", record["reason"])
        self.assertEqual(self.ledger()["fills"], 0)
        self.assertAlmostEqual(self.ledger()["usdt"], 100.0, places=10)

    def test_hold_signal_never_trades(self):
        record = rp.run_once(candles=candles_from(UP_TREND_CLOSES))
        self.assertEqual(record["signal"], st.HOLD)
        self.assertEqual(record["action"], "NONE")
        self.assertEqual(self.ledger()["fills"], 0)
        self.assertEqual(self.trade_rows(), [])

    def test_dry_run_buys_nothing(self):
        record = rp.run_once(candles=candles_from(CROSS_UP_CLOSES), dry_run=True)
        self.assertEqual(record["action"], "WOULD_BUY")
        self.assertEqual(self.ledger()["fills"], 0)
        self.assertEqual(self.ledger()["usdt"], 100.0)
        self.assertEqual(self.trade_rows(), [])

    def test_dry_run_sells_nothing(self):
        te.sim_buy("BNB", 10.0, _price=700.0)
        record = rp.run_once(candles=candles_from(CROSS_DOWN_CLOSES), dry_run=True)
        self.assertEqual(record["action"], "WOULD_SELL")
        self.assertEqual(self.ledger()["fills"], 1)
        self.assertIn("BNB/USDT", self.ledger()["positions"])

    def test_a_custom_size_is_respected(self):
        rp.run_once(candles=candles_from(CROSS_UP_CLOSES), size_usd=25.0)
        self.assertAlmostEqual(self.ledger()["usdt"], 75.0, places=8)

    def test_the_latch_survives_a_reload(self):
        candles = candles_from(CROSS_UP_CLOSES)
        rp.run_once(candles=candles)
        record = rp.run_once(candles=candles, state=rp.load_state())
        self.assertEqual(record["action"], "NONE")
        self.assertEqual(self.ledger()["fills"], 1)

    def test_an_unreadable_latch_does_not_crash(self):
        rp.STATE_PATH.write_text("{not json")
        self.assertEqual(rp.load_state(), {"acted": []})

    def test_report_shows_the_market_and_the_decision(self):
        text = rp.report(rp.run_once(candles=candles_from(CROSS_UP_CLOSES)))
        self.assertIn("BNB/USDT", text)
        self.assertIn("BOUGHT", text)
        self.assertEqual(len(text.splitlines()), 2)

    def test_a_flipped_flag_blocks_the_trade(self):
        """The engine's guard, reached through the runner, not around it."""
        te.TRADING_ENABLED = True
        try:
            record = rp.run_once(candles=candles_from(CROSS_UP_CLOSES))
            self.assertEqual(record["action"], "REFUSED")
            self.assertEqual(self.ledger()["fills"], 0)
            self.assertAlmostEqual(self.ledger()["usdt"], 100.0, places=10)
            self.assertEqual(self.trade_rows(), [])
        finally:
            te.TRADING_ENABLED = False

    def test_the_flag_is_false_on_import(self):
        self.assertFalse(te.TRADING_ENABLED)
        self.assertFalse(rp.te.TRADING_ENABLED)


# --------------------------------------------------------------------------- #
# Live
# --------------------------------------------------------------------------- #

class TestLive(unittest.TestCase):
    def test_real_candles_produce_a_verdict(self):
        try:
            candles = st.fetch_klines("BNB", "1h", 50)
        except st.MarketDataUnavailable as exc:
            self.skipTest(f"Binance unreachable: {exc}")

        self.assertEqual(len(candles), 50)
        self.assertTrue(
            all(c["close_time"] < st._now_ms() for c in candles),
            "every candle used must already be closed",
        )
        self.assertEqual(
            [c["open_time"] for c in candles],
            sorted(c["open_time"] for c in candles),
            "candles must run oldest to newest",
        )

        result = st.evaluate(candles, fast=10, slow=30, pair="BNB/USDT")
        self.assertIn(result["signal"], (st.BUY, st.SELL, st.HOLD))
        self.assertGreater(result["price"], 1.0)
        self.assertEqual(result["candles"], 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
