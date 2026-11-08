#!/usr/bin/env python3
"""Tests for strategy_rsi.py — the RSI, the crossings, the stop, the guards.

    python3 -m unittest test_strategy_rsi -v

The RSI itself is checked against values that can be worked out with a pencil,
not against whatever this implementation happens to produce. The crossing tests
are built from candle series engineered so a specific crossing occurs on a
specific bar, so an assertion failing means the strategy moved, not the data.
"""

from __future__ import annotations

import unittest

import strategy_rsi as rsi


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def bar(open_, high, low, close, open_time=0, close_time=None):
    """One candle in the shape strategy.fetch_klines() returns."""
    return {
        "open_time": open_time,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": 1.0,
        "close_time": close_time if close_time is not None else open_time + 1,
    }


def series_from_closes(closes):
    """Candles whose open/high/low hug the close, so only closes matter."""
    out = []
    for i, close in enumerate(closes):
        out.append(bar(close, close, close, close, open_time=i * 1000))
    return out


def closes_from_rsi_walk(start, steps):
    """Closes that are the running sum of ``steps`` (a list of price changes)."""
    closes = [float(start)]
    for step in steps:
        closes.append(closes[-1] + float(step))
    return closes


def crosses_on_last(closes, level, downward=True, period=rsi.PERIOD):
    """True when the final bar is the crossing bar for ``level``."""
    values = rsi.rsi_series(closes, period)
    i = len(closes) - 1
    if i < 1 or values[i] is None or values[i - 1] is None:
        return False
    prev, now = values[i - 1], values[i]
    if downward:
        return prev >= level > now
    return prev <= level < now


def extend_until_cross(closes, level, *, downward=True, step=1.0,
                       period=rsi.PERIOD, max_steps=400):
    """Append ``step`` until the last bar is the crossing of ``level``.

    Checking the last bar rather than the first crossing matters: a series that
    already crossed once would otherwise stop growing on the old crossing and
    the test would assert against the wrong bar entirely.
    """
    closes = list(closes)
    for _ in range(max_steps):
        closes.append(closes[-1] + step)
        if crosses_on_last(closes, level, downward, period):
            return closes
    raise AssertionError(
        f"stepping {step:+g} never crossed {level} — the test data is wrong"
    )


def walk_to_oversold(start=100.0, rises=20, rise=1.0, fall=-4.0,
                     period=rsi.PERIOD, max_falls=80):
    """A steady rise, then falls — truncated exactly on the bar RSI crosses 30."""
    closes = closes_from_rsi_walk(start, [rise] * rises)
    return extend_until_cross(closes, rsi.OVERSOLD, downward=True,
                              step=fall, period=period)


# --------------------------------------------------------------------------- #
# RSI values
# --------------------------------------------------------------------------- #

class TestRSIValues(unittest.TestCase):
    def test_monotonic_rise_is_100(self):
        closes = list(range(1, 40))
        values = rsi.rsi_series(closes, 14)
        self.assertEqual(values[14], 100.0)
        self.assertEqual(values[-1], 100.0)

    def test_monotonic_fall_is_0(self):
        closes = list(range(40, 1, -1))
        values = rsi.rsi_series(closes, 14)
        self.assertEqual(values[14], 0.0)
        self.assertEqual(values[-1], 0.0)

    def test_alternating_is_about_50(self):
        # Equal gains and losses: RS is 1, so the seed is exactly 50. After
        # that Wilder's recursion oscillates slightly around 50 rather than
        # sitting on it, because the last change dominates one average.
        closes = [100.0 + (1.0 if i % 2 else 0.0) for i in range(40)]
        values = rsi.rsi_series(closes, 14)
        self.assertAlmostEqual(values[14], 50.0, places=9)
        self.assertAlmostEqual(values[-1], 50.0, delta=3.0)
        for value in values[14:]:
            self.assertAlmostEqual(value, 50.0, delta=5.0)

    def test_flat_window_is_neutral_not_extreme(self):
        # No gain and no loss must not be read as "overbought" or "oversold".
        values = rsi.rsi_series([100.0] * 30, 14)
        self.assertEqual(values[14], 50.0)

    def test_warmup_is_none(self):
        values = rsi.rsi_series(list(range(1, 30)), 14)
        for i in range(14):
            self.assertIsNone(values[i], f"bar {i} must be warm-up")
        self.assertIsNotNone(values[14])

    def test_hand_computed_wilder_value(self):
        """A pencil check of the recursive average.

        period=3, closes 10, 11, 12, 11.
        changes: +1, +1, -1
        seed avg_gain = 2/3, avg_loss = 1/3 -> RS = 2 -> RSI = 66.6667
        then change -2 (11 -> 9): avg_gain = (2/3 * 2 + 0)/3 = 0.444444
                                  avg_loss = (1/3 * 2 + 2)/3 = 0.888889
        RS = 0.5 -> RSI = 33.3333
        """
        closes = [10.0, 11.0, 12.0, 11.0, 9.0]
        values = rsi.rsi_series(closes, 3)
        self.assertAlmostEqual(values[3], 200.0 / 3.0, places=9)
        self.assertAlmostEqual(values[4], 100.0 / 3.0, places=9)

    def test_wilder_smoothing_is_not_simple_mean(self):
        """The recursive step must actually smooth, not re-average the window."""
        closes = [100, 101, 102, 103, 104, 90, 91, 92, 93, 94, 95, 96, 97]
        values = rsi.rsi_series(closes, 5)
        self.assertIsNotNone(values[-1])
        # A simple mean over the last 5 changes (1,1,1,1,1) would be 100.
        self.assertLess(values[-1], 100.0)

    def test_bounds_hold(self):
        for closes in (list(range(1, 60)), list(range(60, 1, -1)),
                       [50.0 + 3 * ((-1) ** i) for i in range(60)]):
            for value in rsi.rsi_series(closes, 14):
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)

    def test_period_none_returns_too_short(self):
        self.assertEqual(rsi.rsi_series([1.0, 2.0], 14), [None, None])

    def test_bad_period_refused(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.rsi_series([1.0, 2.0, 3.0], 0)


# --------------------------------------------------------------------------- #
# No lookahead
# --------------------------------------------------------------------------- #

class TestNoLookahead(unittest.TestCase):
    def test_prefix_identical_to_full_series(self):
        """rsi[i] must be the same whether you pass everything or only the prefix.

        This is what licenses backtest_rsi.py to compute the series once and
        slice it, instead of rebuilding it every bar.
        """
        import random

        random.seed(11)
        closes = [100.0]
        for _ in range(300):
            closes.append(max(1.0, closes[-1] * (1 + random.uniform(-0.03, 0.03))))
        full = rsi.rsi_series(closes, 14)
        for i in (20, 50, 137, 299):
            prefix = rsi.rsi_series(closes[: i + 1], 14)
            self.assertAlmostEqual(prefix[i], full[i], places=12,
                                   msg=f"bar {i} depends on future data")

    def test_future_does_not_rewrite_past(self):
        import random

        random.seed(12)
        closes = [100.0]
        for _ in range(200):
            closes.append(max(1.0, closes[-1] * (1 + random.uniform(-0.02, 0.02))))
        before = rsi.rsi_series(closes, 14)
        after = rsi.rsi_series(closes + [closes[-1] * 3.0], 14)
        self.assertEqual(before, after[: len(before)])

    def test_mutation_proves_the_test_bites(self):
        """A deliberately leaky RSI must fail the prefix comparison.

        Without this, test_prefix_identical_to_full_series could be passing
        because it cannot fail.
        """
        def leaky(closes, period=14):
            # Centre the window on bar i: reads bars after i.
            out = [None] * len(closes)
            centred = [closes[min(len(closes) - 1, i + 5)] for i in range(len(closes))]
            honest = rsi.rsi_series(centred, period)
            for i in range(len(closes)):
                out[i] = honest[i]
            return out

        import random

        random.seed(13)
        closes = [100.0]
        for _ in range(120):
            closes.append(max(1.0, closes[-1] * (1 + random.uniform(-0.05, 0.05))))

        full = leaky(closes, 14)
        mismatch = False
        for i in (30, 60, 100):
            prefix = leaky(closes[: i + 1], 14)
            if abs((prefix[i] or 0.0) - (full[i] or 0.0)) > 1e-9:
                mismatch = True
        self.assertTrue(mismatch, "the leak detector failed to detect a leak")


# --------------------------------------------------------------------------- #
# BUY on the oversold crossing
# --------------------------------------------------------------------------- #

class TestOversoldBuy(unittest.TestCase):
    def test_crossing_below_30_buys(self):
        closes = walk_to_oversold()
        result = rsi.evaluate(series_from_closes(closes))
        self.assertEqual(result["signal"], rsi.BUY)
        self.assertEqual(result["reason"], rsi.REASON_OVERSOLD)
        self.assertLess(result["rsi"], 30.0)
        self.assertGreaterEqual(result["prev_rsi"], 30.0)

    def test_state_below_30_does_not_repeat_buy(self):
        """The whole point of event-not-state: a waterfall buys once."""
        # Crossing bar, then a long drift well below 30.
        closes = walk_to_oversold()
        closes += [closes[-1] - 0.05] * 25
        candles = series_from_closes(closes)
        signals = []
        for i in range(14 + 2, len(candles)):
            signals.append(rsi.evaluate(candles[: i + 1])["signal"])
        buys = signals.count(rsi.BUY)
        self.assertEqual(buys, 1, f"expected exactly one BUY, got {buys}")
        # And confirm the state really was oversold for a long stretch.
        values = rsi.rsi_series(closes, 14)
        below = sum(1 for v in values if v is not None and v < 30.0)
        self.assertGreater(below, 10, "test data must sit below 30 for many bars")

    def test_no_buy_while_in_a_position(self):
        closes = walk_to_oversold()
        result = rsi.evaluate(
            series_from_closes(closes), position={"entry_price": 90.0}
        )
        self.assertNotEqual(result["signal"], rsi.BUY)

    def test_no_buy_on_first_valid_bar(self):
        """A crossing needs a previous reading; the warm-up bar is not a signal."""
        closes = closes_from_rsi_walk(100.0, [-5.0] * 20)
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(series_from_closes(closes[: 15]))


# --------------------------------------------------------------------------- #
# SELL side: stop, overbought, midline
# --------------------------------------------------------------------------- #

class TestExits(unittest.TestCase):
    def test_hard_stop_fills_at_stop_price(self):
        entry = 100.0
        stop = rsi.stop_price(entry, 0.05)
        self.assertAlmostEqual(stop, 95.0, places=9)
        # Bar trades through the stop but opens above it.
        candle = [bar(99.0, 99.5, 93.0, 94.0)]
        closes = [100.0] * 20 + [99.5, 94.0]
        candles = series_from_closes(closes)
        candles[-1] = candle[0]
        result = rsi.evaluate(candles, position={"entry_price": entry})
        self.assertEqual(result["signal"], rsi.SELL)
        self.assertEqual(result["reason"], rsi.REASON_STOP_LOSS)
        self.assertAlmostEqual(result["fill_price"], 95.0, places=9)

    def test_gap_through_stop_fills_at_open(self):
        """A bar that opens below the stop fills at the open, not the stop."""
        closes = [100.0] * 20 + [99.5, 92.0]
        candles = series_from_closes(closes)
        candles[-1] = bar(92.0, 92.5, 91.0, 92.0)
        result = rsi.evaluate(candles, position={"entry_price": 100.0})
        self.assertEqual(result["reason"], rsi.REASON_STOP_LOSS)
        self.assertAlmostEqual(result["fill_price"], 92.0, places=9)
        self.assertLess(result["fill_price"], 95.0)

    def test_stop_beats_every_other_rule_on_the_same_bar(self):
        """If a bar both stops out and closes overbought, the stop wins.

        The stop is an order in the book and fires intrabar; a close-based
        exit cannot be assumed to have been available first.
        """
        closes = [100.0] * 20 + [99.5, 101.0]
        candles = series_from_closes(closes)
        candles[-1] = bar(99.0, 101.5, 90.0, 101.0)
        result = rsi.evaluate(candles, position={"entry_price": 100.0})
        self.assertEqual(result["reason"], rsi.REASON_STOP_LOSS)

    def test_no_stop_when_still_above_it(self):
        closes = [100.0] * 20 + [99.5, 97.0]
        candles = series_from_closes(closes)
        candles[-1] = bar(99.0, 99.5, 96.5, 97.0)
        result = rsi.evaluate(candles, position={"entry_price": 100.0})
        self.assertNotEqual(result["reason"], rsi.REASON_STOP_LOSS)

    def test_midline_down_exit(self):
        """In a position, RSI falling back through 50 exits at the midline."""
        # A zigzag that ends mid-range, then fades until the bar that crosses 50.
        closes = extend_until_cross(
            closes_from_rsi_walk(100.0, [3.0, 0.0] * 20),  # ends high, RSI ~52
            rsi.MIDLINE, downward=True, step=-0.8,
        )
        result = rsi.evaluate(
            series_from_closes(closes), position={"entry_price": 95.0}
        )
        self.assertEqual(result["signal"], rsi.SELL)
        self.assertEqual(result["reason"], rsi.REASON_MIDLINE)
        self.assertLess(result["rsi"], 50.0)
        self.assertGreaterEqual(result["prev_rsi"], 50.0)

    def test_midline_down_does_not_fire_when_flat(self):
        """Flat, a downward 50 crossing is not a signal — there is nothing to sell."""
        closes = extend_until_cross(
            closes_from_rsi_walk(100.0, [3.0, 0.0] * 20),
            rsi.MIDLINE, downward=True, step=-0.8,
        )
        result = rsi.evaluate(series_from_closes(closes))
        self.assertEqual(result["signal"], rsi.HOLD)

    def test_midline_up_mode_exits_on_recovery(self):
        closes = extend_until_cross(
            walk_to_oversold(), rsi.MIDLINE, downward=False, step=3.0
        )
        result = rsi.evaluate(
            series_from_closes(closes),
            position={"entry_price": 92.0},   # the oversold entry, well below
            midline_mode="up",
        )
        self.assertEqual(result["signal"], rsi.SELL)
        self.assertEqual(result["reason"], rsi.REASON_MIDLINE)
        self.assertGreater(result["rsi"], 50.0)
        self.assertLessEqual(result["prev_rsi"], 50.0)

    def test_up_mode_ignores_a_downward_crossing(self):
        """The two modes are genuinely different, not the same rule twice."""
        closes = extend_until_cross(
            closes_from_rsi_walk(100.0, [3.0, 0.0] * 20),
            rsi.MIDLINE, downward=True, step=-0.8,
        )
        result = rsi.evaluate(
            series_from_closes(closes),
            position={"entry_price": 95.0},
            midline_mode="up",
        )
        self.assertNotEqual(result["reason"], rsi.REASON_MIDLINE)

    def test_overbought_exit_reachable_only_when_midline_down(self):
        """The claim made in the module docstring, checked rather than asserted.

        Under ``midline_mode="up"`` every long exits at 50 on the way up, so the
        70 rule can never fire. Under ``"down"`` it can. Driven bar by bar from
        the oversold entry with the 5% stop switched off, so the stop cannot cut
        the recovery short and muddy the result.
        """
        closes = walk_to_oversold()      # the last bar is the RSI<30 BUY
        entry_price = closes[-1]

        def run(mode):
            candles = series_from_closes(closes)
            price = closes[-1]
            seen = set()
            in_pos = True                # the BUY filled on the last bar
            for _ in range(120):
                price += 2.5
                candles.append(bar(price, price, price, price,
                                   open_time=len(candles) * 1000))
                result = rsi.evaluate(
                    candles,
                    position=({"entry_price": entry_price} if in_pos else None),
                    midline_mode=mode,
                    stop_pct=0.9,        # keep the stop out of this test
                )
                if result["signal"] == rsi.SELL and in_pos:
                    seen.add(result["reason"])
                    in_pos = False
            return seen

        down, up = run("down"), run("up")
        self.assertIn(rsi.REASON_OVERBOUGHT, down)
        self.assertNotIn(rsi.REASON_OVERBOUGHT, up)
        # And up-mode does exit — at the midline, on the way up.
        self.assertIn(rsi.REASON_MIDLINE, up)

    def test_overbought_fires_only_on_the_upward_crossing(self):
        """An upward crossing into 70 has prev_rsi *below* 70 by definition.

        And the bars that follow, where RSI is above 70 on both sides, must stay
        quiet — otherwise the rule is a state, not an event, and it would fire on
        every bar of an extended rally.
        """
        closes = extend_until_cross(
            walk_to_oversold(), rsi.OVERBOUGHT, downward=False, step=3.0
        )
        cross_index = len(closes) - 1

        # Keep rallying: RSI stays above 70 for several bars after the crossing.
        price = closes[-1]
        for _ in range(8):
            price += 2.0
            closes.append(price)

        fired = []
        for i in range(cross_index, len(closes)):
            result = rsi.evaluate(
                series_from_closes(closes[: i + 1]),
                position={"entry_price": 92.0},
                stop_pct=0.9,
            )
            if result["reason"] == rsi.REASON_OVERBOUGHT:
                fired.append((i, result))

        self.assertEqual(len(fired), 1, f"fired on {[i for i, _ in fired]} — expected once")
        idx, result = fired[0]
        self.assertEqual(idx, cross_index)
        self.assertGreaterEqual(result["rsi"], 70.0)
        self.assertLess(result["prev_rsi"], 70.0)

        # Proof the quiet bars really were above 70 on both sides.
        values = rsi.rsi_series(closes, rsi.PERIOD)
        after = values[cross_index + 1 :]
        self.assertTrue(after)
        self.assertTrue(all(v > 70.0 for v in after),
                        f"post-crossing RSI dipped below 70: {[round(v, 2) for v in after]}")


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.candles = series_from_closes(closes_from_rsi_walk(100.0, [1.0] * 30))

    def test_too_few_candles(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles[:15])

    def test_oversold_above_overbought(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles, oversold=80.0, overbought=20.0)

    def test_midline_must_sit_between_bands(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles, midline=85.0)

    def test_bad_stop(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles, position={"entry_price": 100.0}, stop_pct=0.0)
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles, position={"entry_price": 100.0}, stop_pct=1.5)

    def test_bad_midline_mode(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles, midline_mode="sideways")

    def test_position_needs_entry_price(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles, position={})
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles, position={"entry_price": 0})

    def test_mismatched_precomputed_series_refused(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.evaluate(self.candles, rsi=[50.0] * 5)

    def test_precomputed_series_equals_internal(self):
        a = rsi.evaluate(self.candles)
        b = rsi.evaluate(
            self.candles,
            rsi=rsi.rsi_series([c["close"] for c in self.candles], rsi.PERIOD),
        )
        self.assertEqual(a, b)

    def test_stop_price_helper_rejects_bad_input(self):
        with self.assertRaises(rsi.StrategyError):
            rsi.stop_price(0.0)
        with self.assertRaises(rsi.StrategyError):
            rsi.stop_price(100.0, 0.0)
        with self.assertRaises(rsi.StrategyError):
            rsi.stop_price(100.0, 1.0)

    def test_default_config_is_the_brief(self):
        self.assertEqual(rsi.PERIOD, 14)
        self.assertEqual(rsi.INTERVAL, "4h")
        self.assertEqual(rsi.OVERSOLD, 30.0)
        self.assertEqual(rsi.OVERBOUGHT, 70.0)
        self.assertEqual(rsi.MIDLINE, 50.0)
        self.assertEqual(rsi.STOP_PCT, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
