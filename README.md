# paper-trader

A paper-trading research engine. It fetches public Binance candles, runs a
strategy over them, and books simulated fills against a JSON ledger with a
0.1% taker fee charged per side. No keys, no signing, no orders.

The point of the repository is the tests and the rejections. Six experiments
were built and measured; each is documented below with the numbers that
decided it. One of them (T3/T4) was not disproved — it simply cannot be shown
to work on the sample available.

> This is an educational research engine to test statistical significance, not
> a money printer. The funding module is a yield scanner, not a full
> delta-neutral portfolio simulator.

## Safety

`TRADING_ENABLED` is `False` in `trade_engine.py` and every money function
calls `_assert_simulation()` first, so flipping the flag raises rather than
sends anything. The only outbound requests are public `GET` calls to
`api.binance.com` for prices and klines (`strategy.py`) and funding rates
(`strategy_funding.py`). There is no private endpoint, no API key and no
wallet code anywhere in the tree.

## Files

Engine and runner:

| File | Lines | What it is |
| --- | ---: | --- |
| `trade_engine.py` | 671 | Ledger, `sim_buy` / `sim_sell`, fee accounting, status |
| `run_paper_trader.py` | 283 | 60-second loop: fetch candles, evaluate, act, latch |
| `strategy.py` | 363 | MA crossover on closed 1h candles |

Strategies:

| File | Lines | What it is |
| --- | ---: | --- |
| `strategy_mss.py` | 402 | Market-structure-shift breakout + trailing stop |
| `strategy_rsi.py` | 546 | RSI(14) mean reversion, optional 200-MA trend gate |
| `strategy_funding.py` | 545 | Perp funding-rate scanner (read-only, no positions) |

Backtests and research runs:

| File | Lines | What it is |
| --- | ---: | --- |
| `backtest.py` | 794 | Generic MA-crossover backtest over any interval |
| `backtest_mss.py` | 753 | MSS breakout backtest with composable exits |
| `backtest_rsi.py` | 753 | RSI mean-reversion backtest |
| `backtest_multi.py` | 517 | Pools trade statistics across BTC/ETH/SOL |
| `split_half.py` | 244 | Fit/test split with frozen parameters |

Tests — 261 in total, all passing:

| File | Tests |
| --- | ---: |
| `test_trade_engine.py` | 34 |
| `test_strategy.py` | 40 |
| `test_strategy_mss.py` | 27 |
| `test_strategy_rsi.py` | 51 |
| `test_backtest.py` | 22 |
| `test_backtest_mss.py` | 42 |
| `test_backtest_rsi.py` | 38 |
| `test_run_paper_trader.py` | 7 |

Data files: `paper_ledger.json`, `paper_trades.jsonl`,
`split_half_results.json`, `split_half_results_trend.json`,
`backtest_multi_results.json`.

## Running it

```bash
python3 run_paper_trader.py --symbol bnb --once --dry-run
python3 run_paper_trader.py --symbol bnb --interval 300
python3 backtest.py --symbol BNB --days 365 --interval 1h
python3 backtest_mss.py --symbol BNB --days 1095 --interval 4h
python3 backtest_multi.py
python3 split_half.py
