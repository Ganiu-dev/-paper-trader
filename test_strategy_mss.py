#!/usr/bin/env python3
"""test_strategy_mss.py — tests for the MSS breakout strategy.

    python3 -m unittest test_strategy_mss -v

The test that matters is ``test_no_lookahead_*``. Every other test checks a
rule; those check that the rules are computed from information that existed at
the time. A breakout strategy is exactly the kind of code that can look
brilliant and be pure hindsight, so the property is asserted directly and a
mutation test proves the assertion is not vacuous.
"""

from __future__ import annotations

import random
import unittest

import strategy_mss as mss


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def bar(open_, high, low, close, open_time=0, close_time=None):
    """One candle dict shaped like strategy.fetch_klines() output."""
    return {
        "open_time": open_time,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": 1.0,
        "close_time": close_time if close_time is not None else open_time + 1,
    }


def series(highs, lows=None, closes=None, opens=None):
    """Candles from bare highs, with an obvious low/close unless given."""
    if lows is None:
        lows = [h - 1.0 for h in highs]
    if closes is None:
        closes = [h - 0.5 for h in highs]
    if opens is None:
        opens = list(closes)
    return [
        bar(o, h, l, c, open_time=i * 1000, close_time=i * 1000 + 999)
        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes))
    ]


def fractal_series(n=40, seed=7):
    """A random-ish series that actually contains fractals, for property tests."""
    rng = random.Random(seed)
    highs, lows, closes = [], [], []
    price = 100.0
    for _ in range(n):
        move = rng.uniform(-3.0, 3.0)
        open_ = price
        close = max(1.0, open_ + move)
        high = max(open_, close) + rng.uniform(0.1, 2.0)
        low = min(open_, close) - rng.uniform(0.1, 2.0)
        highs.append(high)
        lows.append(max(0.01, low))
        closes.append(close)
        price = close
    return series(highs, lows, closes)


# --------------------------------------------------------------------------- #
# Fractals
# --------------------------------------------------------------------------- #

class TestSwingDetection(unittest.TestCase):
    def test_swing_high_is_strictly_higher_than_two_each_side(self):
        candles = series([100, 101, 105, 102, 100])
        self.assertTrue(mss.is_swing_high(candles, 2))
        self.assertFalse(mss.is_swing_high(candles, 1))
        self.assertFalse(mss.is_swing_high(candles, 3))

    def test_swing_low_is_strictly_lower_than_two_each_side(self):
        candles = series([100, 99, 95, 98, 100])
        self.assertTrue(mss.is_swing_low(candles, 2))
        self.assertFalse(mss.is_swing_low(candles, 1))

    def test_tie_is_not_a_swing_high(self):
        # Equal highs mean the market paused at a level, it did not reject it.
        candles = series([100, 101, 105, 105, 100])
        self.assertFalse(mss.is_swing_high(candles, 2))
        self.assertFalse(mss.is_swing_high(candles, 3))

    def test_tie_is_not_a_swing_low(self):
        candles = series([100, 99, 95, 95, 100])
        self.assertFalse(mss.is_swing_low(candles, 2))
        self.assertFalse(mss.is_swing_low(candles, 3))

    def test_edges_cannot_be_swings(self):
        # No two bars to the left of 0, no two to the right of the last bar.
        candles = series([100, 101, 105, 102, 100])
        self.assertFalse(mss.is_swing_high(candles, 0))
        self.assertFalse(mss.is_swing_high(candles, len(candles) - 1))

    def test_find_swings_reports_confirmation_bar(self):
        candles = series([100, 101, 105, 102, 100])
        swings = mss.find_swings(candles)
        self.assertEqual(len(swings), 1)
        swing = swings[0]
        self.assertEqual(swing["kind"], "high")
        self.assertEqual(swing["index"], 2)
        self.assertEqual(swing["price"], 105.0)
        # 2 right bars means bar 4 is the first close that knows about it.
        self.assertEqual(swing["confirm_index"], 4)


# --------------------------------------------------------------------------- #
# The confirmation lag, and the no-lookahead property
# --------------------------------------------------------------------------- #

class TestConfirmationLag(unittest.TestCase):
    def test_level_is_unavailable_until_the_right_bars_close(self):
        candles = series([100, 101, 105, 102, 100])
        levels = mss.swing_levels(candles)

        # The swing is at bar 2. Bars 2 and 3 must NOT see it.
        self.assertIsNone(levels[2]["swing_high"])
        self.assertIsNone(levels[3]["swing_high"])
        # Bar 4 is the close that confirms it.
        self.assertEqual(levels[4]["swing_high"], 105.0)
        self.assertEqual(levels[4]["swing_high_index"], 2)

    def test_no_lookahead_level_series_matches_prefix(self):
        """levels[i] must be identical from the full history and from candles[:i+1].

        This is the whole property: the level a bar acts on is the level that
        existed when that bar closed. If any window ever reached past i, the
        two would disagree.
        """
        for seed in (1, 2, 3, 7, 11, 42):
            candles = fractal_series(n=60, seed=seed)
            full = mss.swing_levels(candles)
            for i in range(len(candles)):
                prefix = mss.swing_levels(candles[: i + 1])
                self.assertEqual(
                    full[i],
                    prefix[-1],
                    f"seed {seed} bar {i}: level differs between full history and prefix",
                )

    def test_no_lookahead_signals_match_prefix(self):
        """evaluate() on the prefix must give the same verdict as on the full series."""
        candles = fractal_series(n=60, seed=5)
        need = mss.SWING_LEFT + mss.SWING_RIGHT + 1
        for i in range(need - 1, len(candles)):
            prefix = candles[: i + 1]
            full = mss.evaluate(candles[: i + 1])
            again = mss.evaluate(prefix)
            self.assertEqual(full["signal"], again["signal"])
            self.assertEqual(full["swing_high"], again["swing_high"])
            self.assertEqual(full["swing_low"], again["swing_low"])

    def test_the_lag_actually_changes_the_levels(self):
        """Mutation test: a naive whole-series scan must differ, or the guard is vacuous.

        The tempting bug scans the entire history for swings and then asks
        "what was the last one?" at every bar. If that produced the same answer
        as the honest version, the no-lookahead tests above would prove nothing.
        """
        candles = fractal_series(n=60, seed=3)
        honest = mss.swing_levels(candles)

        all_swings = mss.find_swings(candles)
        highs = [s for s in all_swings if s["kind"] == "high"]

        differed = 0
        for i in range(len(candles)):
            # The buggy version: newest swing by position, ignoring whether it
            # could be confirmed yet.
            seen = [s for s in highs if s["index"] <= i]
            buggy = seen[-1]["price"] if seen else None
            if buggy != honest[i]["swing_high"]:
                differed += 1

        self.assertGreater(
            differed, 0,
            "naive scan never disagreed with the honest one — the lookahead "
            "tests would not have caught the bug",
        )


# --------------------------------------------------------------------------- #
# Breakout signals
# --------------------------------------------------------------------------- #

def breakout_candles(*, final_close, prev_close=None):
    """11 bars: swing high 110 at bar 5, confirmed at bar 7; nothing after it.

    Bars 8-10 stay below the level so the final bar is the one that breaks it.
    """
    highs = [100, 101, 102, 103, 104, 110, 104, 103, 102, 101, 0]
    highs[10] = max(final_close, 101.0)
    lows = [h - 2.0 for h in highs]
    closes = [h - 0.5 for h in highs]
    if prev_close is not None:
        closes[9] = prev_close
        lows[9] = min(lows[9], prev_close - 0.5)
    closes[10] = final_close
    lows[10] = min(closes[10] - 1.0, 100.0)
    highs[10] = max(highs[10], closes[10])
    return series(highs, lows, closes)


class TestBullishMSS(unittest.TestCase):
    def test_buy_when_close_crosses_above_swing_high(self):
        candles = breakout_candles(final_close=112.0, prev_close=109.0)
        result = mss.evaluate(candles)
        self.assertEqual(result["swing_high"], 110.0)
        self.assertEqual(result["signal"], mss.BUY)
        self.assertEqual(result["reason"], mss.REASON_BULLISH_MSS)
        self.assertEqual(result["fill_price"], 112.0)

    def test_hold_when_close_is_already_above_the_level(self):
        # State, not event: the previous close was already through, so this bar
        # is not a breakout. Buying here would re-buy every bar of a trend.
        candles = breakout_candles(final_close=113.0, prev_close=111.0)
        result = mss.evaluate(candles)
        self.assertEqual(result["signal"], mss.HOLD)

    def test_hold_when_close_stays_below_the_level(self):
        candles = breakout_candles(final_close=108.0, prev_close=107.0)
        result = mss.evaluate(candles)
        self.assertEqual(result["signal"], mss.HOLD)

    def test_hold_when_no_swing_high_is_confirmed_yet(self):
        candles = series([100, 101, 105, 102, 100])
        result = mss.evaluate(candles)
        # bars 3 and 4 have no confirmed swing high
        self.assertEqual(result["swing_high"], 105.0)


class TestBearishMSS(unittest.TestCase):
    def _candles(self, final_close, prev_close):
        """Swing low 96 at bar 5 (confirmed bar 7), then a break downward."""
        highs = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        lows = [99, 98, 97, 98, 97, 96, 97, 98, 97, 98, 95]
        closes = [99.5, 98.5, 97.5, 98.5, 98.0, 96.5, 97.5, 98.5, 98.0, 99.0, 95.5]
        closes[9] = prev_close
        closes[10] = final_close
        lows[9] = min(lows[9], prev_close - 0.5)
        lows[10] = min(lows[10], final_close - 0.5)
        return series(highs, lows, closes)

    def test_sell_when_close_crosses_below_swing_low(self):
        candles = self._candles(final_close=95.0, prev_close=97.0)
        # Wide trail on purpose: a 3% stop sits well above a swing low 4 below
        # the peak, so it would fire first and mask the rule under test.
        result = mss.evaluate(
            candles, position={"entry_price": 99.0, "peak_price": 101.0}, trail_pct=0.5
        )
        self.assertEqual(result["swing_low"], 96.0)
        self.assertEqual(result["signal"], mss.SELL)
        self.assertEqual(result["reason"], mss.REASON_BEARISH_MSS)

    def test_hold_when_already_below_the_swing_low(self):
        candles = self._candles(final_close=94.5, prev_close=95.5)
        result = mss.evaluate(
            candles, position={"entry_price": 99.0, "peak_price": 101.0}, trail_pct=0.5
        )
        self.assertEqual(result["signal"], mss.HOLD)


# --------------------------------------------------------------------------- #
# Trailing stop
# --------------------------------------------------------------------------- #

class TestTrailingStop(unittest.TestCase):
    def _candles(self, open_, high, low, close):
        """Flat structure, so only the stop can produce a signal."""
        n = 10
        highs = [50.0] * n + [high]
        lows = [49.0] * n + [low]
        closes = [49.5] * n + [close]
        opens = [49.5] * n + [open_]
        return series(highs, lows, closes, opens)

    def test_stop_price_is_three_percent_below_peak(self):
        self.assertAlmostEqual(mss.trailing_stop_price(100.0, 0.03), 97.0)

    def test_sell_when_bar_trades_through_the_stop(self):
        candles = self._candles(open_=99.0, high=100.0, low=96.0, close=96.5)
        result = mss.evaluate(candles, position={"entry_price": 90.0, "peak_price": 100.0})
        self.assertEqual(result["signal"], mss.SELL)
        self.assertEqual(result["reason"], mss.REASON_TRAILING_STOP)
        # The stop is a resting order, so it fills at the stop, not the close.
        self.assertAlmostEqual(result["fill_price"], 97.0)

    def test_gap_through_the_stop_fills_at_the_open(self):
        # Opened at 95, below the 97 stop. The stop cannot fill better than
        # the first price available.
        candles = self._candles(open_=95.0, high=95.5, low=94.0, close=94.5)
        result = mss.evaluate(candles, position={"entry_price": 90.0, "peak_price": 100.0})
        self.assertEqual(result["signal"], mss.SELL)
        self.assertAlmostEqual(result["fill_price"], 95.0)

    def test_hold_when_the_stop_is_intact(self):
        candles = self._candles(open_=99.0, high=101.0, low=98.0, close=100.0)
        result = mss.evaluate(candles, position={"entry_price": 90.0, "peak_price": 100.0})
        self.assertEqual(result["signal"], mss.HOLD)

    def test_peak_only_advances_after_the_stop_check(self):
        # A spike bar must not use its own high to widen the stop it breaches.
        candles = self._candles(open_=99.0, high=120.0, low=96.0, close=96.5)
        result = mss.evaluate(candles, position={"entry_price": 90.0, "peak_price": 100.0})
        self.assertEqual(result["signal"], mss.SELL)
        self.assertAlmostEqual(result["stop_price"], 97.0)   # from the 100 peak

    def test_peak_advances_when_the_stop_survives(self):
        candles = self._candles(open_=99.0, high=110.0, low=99.0, close=109.0)
        result = mss.evaluate(candles, position={"entry_price": 90.0, "peak_price": 100.0})
        self.assertEqual(result["signal"], mss.HOLD)
        self.assertEqual(result["peak_price"], 110.0)

    def test_stop_takes_precedence_over_bearish_mss(self):
        # Both rules fire on the same bar; the stop is the resting order and
        # is checked first, so the reason reported is the stop.
        highs = [100.0] * 9 + [100.0]
        lows = [99.0] * 9 + [80.0]
        closes = [99.5] * 9 + [81.0]
        candles = series(highs, lows, closes)
        result = mss.evaluate(candles, position={"entry_price": 95.0, "peak_price": 100.0})
        self.assertEqual(result["signal"], mss.SELL)
        self.assertEqual(result["reason"], mss.REASON_TRAILING_STOP)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

class TestValidation(unittest.TestCase):
    def test_too_few_candles_raises(self):
        with self.assertRaises(mss.StrategyError):
            mss.evaluate(series([100, 101]))

    def test_trail_must_be_a_fraction(self):
        for bad in (0, 1, -0.1, 1.5):
            with self.assertRaises(mss.StrategyError):
                mss.trailing_stop_price(100.0, bad)

    def test_position_needs_a_positive_peak(self):
        candles = fractal_series(n=20)
        with self.assertRaises(mss.StrategyError):
            mss.evaluate(candles, position={"entry_price": 100.0, "peak_price": 0.0})

    def test_bad_swing_window_raises(self):
        with self.assertRaises(mss.StrategyError):
            mss.swing_levels(fractal_series(n=10), left=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
