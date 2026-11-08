# 📈 Paper Trader: Algorithmic Crypto Engine & Backtester

A production-grade Python framework for simulating crypto trading strategies, evaluating statistical significance, and executing paper trades with strict fee-inclusive accounting.

This repository contains a complete quantitative research pipeline: from data fetching and strategy evaluation to historical backtesting, split-half out-of-sample validation, and live paper trading.

## 🏗️ Architecture

The system is divided into modular, testable components:
- **`trade_engine.py`**: The core simulator. Manages the paper ledger and enforces strict 0.1% fees per side.
- **`strategy.py`**: Moving Average (MA) Crossover strategy.
- **`strategy_mss.py`**: Market Structure Shift (MSS) fractal breakout strategy.
- **`strategy_rsi.py`**: RSI Mean Reversion strategy.
- **`strategy_funding.py`**: Funding Rate Arbitrage (Delta-Neutral yield capture).
- **`backtest_*.py`**: Replays historical data bar-by-bar, modeling strict fee drag.
- **`split_half.py`**: Out-of-sample validation to prevent curve-fitting.

## 📊 Strategies Evaluated & Rejected

This repo doesn't just claim an edge; it proves (or disproves) it using statistical significance testing (T-stat, p-value). 

1. **MA Crossover (1h/4h)**: *Result: No exploitable edge. Fees ate 83% of gross profit.*
2. **MSS Breakout (4h)**: *Result: A churn harness. The trailing stop bled out 190 breakouts.*
3. **RSI Mean Reversion (4h)**: *Result: Rejected. Tested on BNB (T+1.33), but pooled across BTC/ETH/SOL it failed completely (T-stat −2.57). The BNB edge was pure curve-fitting.*
4. **Funding Rate Gating**: *Result: Rejected. A threshold-gated entry throws away 98% of the yield. Funding capture is a continuous yield play, not an event-driven trade.*

## 🛡️ Safety & Anti-Curve-Fitting Features

- **`TRADING_ENABLED = False`**: The engine physically raises an exception if any script attempts to execute a live order.
- **Sandboxing (`@sandboxes_engine`)**: Backtests run in a temporary directory. Running a backtest will **never** overwrite your live `paper_ledger.json`.
- **No Lookahead**: Strategies are strictly causal. All indicators only use data up to bar `i`.
- **Statistical Rigor**: T-stats and p-values are computed via a pure-Python Student-t distribution, and trades are pooled across timeframes to prevent p-hacking.

## 🚀 Quick Start

You don't need any external dependencies. This runs on the Python Standard Library (Python 3.10+).

**1. Run a Backtest**
```bash
python3 backtest_rsi.py --symbol BNB --days 1095
