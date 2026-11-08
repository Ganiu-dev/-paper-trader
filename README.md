# Tree T — paper trading engine

A simulated BNB/USDT trader. It reads the real Binance spot price and keeps a
fake portfolio on disk. Nothing here can move money: there is no wallet, no
key, no signing, and no endpoint in the code but one public price GET.

Starting balance is **$100 of simulated USDT**.

## Files

| File | What it is |
|---|---|
| `trade_engine.py` | the engine (stdlib only — no pip install needed) |
| `test_trade_engine.py` | 28 tests, including the spec's own round trip |
| `paper_ledger.json` | created on first run: cash, positions, P&L, fees |
| `paper_trades.jsonl` | append-only log, one JSON object per simulated fill |

You do not need to create the two data files — the engine writes them.

## Run it

```bash
python3 trade_engine.py price BNB          # live BNB/USDT quote
python3 trade_engine.py buy BNB 10         # spend $10 of simulated USDT
python3 trade_engine.py status             # cash, positions, mark-to-market P&L
python3 trade_engine.py sell BNB 0.0143    # sell a token quantity
python3 trade_engine.py reset              # wipe back to $100, clear the log
```

`buy` takes USDT, `sell` takes tokens. That asymmetry is deliberate: you can
size a buy in dollars but you can only sell what you actually hold, so the
engine makes you name the quantity.

Tickers are forgiving about form and strict about mistakes. `BNB`, `bnbusdt`,
`BNB/USDT` and `BNB-USDT` all mean the same market; `BT-C`, `BTC/EUR` and
`B NB!` are rejected rather than quietly repaired, because a typo that gets
silently fixed into a real quote is a lie with a price next to it.

## The test the spec asked for

```bash
python3 test_trade_engine.py -v
```

28 tests, all passing, in about half a second. The one that matters:

> buy $10 of BNB, sell it immediately, and the balance comes back to almost
> exactly $100 minus fees.

It runs twice — once against a pinned price of 700 so the arithmetic is exact,
and once against the live Binance API so the real path is exercised. A live run
just now:

```
buy   $10.00  @ 699.16  ->  0.01428857 BNB   cash 90.0000
sell  all     @ 699.17  ->  9.98015275 USDT  cash 99.98015275
```

That is 99.9802, i.e. **1.98 cents short of $100** on a price that ticked up
between the two fills. The shortfall is fees, not error.

## Fees: 0.1% per side, so 0.2% a round trip

One deliberate reading of the spec, worth confirming. "Minus a simulated 0.1%
fee" can mean 0.1% once per round trip or 0.1% on each fill. This engine
charges **each side**, because that is what a real spot venue does and because
the point of a paper ledger is to show what churn costs. On $10 that is
$0.01999 — two cents — not one.

If you want the single-charge reading, change `FEE_RATE` to `0.0005` and
nothing else: every fee in the engine is derived from that one constant.

## Accounting, so the numbers reconcile

A position's `cost_basis` is the **cash actually spent, fee included**. Two
consequences worth knowing before you read a statement:

- `avg_cost` is the fee-inclusive break-even price. Buy at 699.16 and it reads
  699.8599 — that is the price you need to sell at to get your ten dollars back.
- `realized_pnl` is sell proceeds (after the sell fee) minus that basis, so it
  equals the change in cash. It cannot drift away from the balance.

Token quantities are stored to 12 decimals. Rounding to 8 leaves dust, and dust
turns into phantom profit once you buy and sell a few times.

## Safety rails

- `TRADING_ENABLED = False`. Every money function calls `_assert_simulation()`
  first and raises if the flag is ever flipped — the flag does not enable
  trading, it only trips an alarm. Verified: with it set to `True`, both
  `sim_buy` and `sim_sell` refuse to run.
- Only one outbound call exists, a GET to
  `https://api.binance.com/api/v3/ticker/price`. No auth, no keys, no orders.
- If the price cannot be fetched, the engine raises `PriceUnavailable` instead
  of guessing. `status` then values the position at its last known fill price
  and labels the source `last_fill`, so an outage is never silently reported as
  a loss.
- Buys larger than cash and sells larger than the position are refused, not
  clamped.

## Limitations

- **Spot only, one pair.** BNB/USDT is hard-coded as the quoted market; the
  symbol handling generalises but nothing else has been exercised.
- **No slippage, no spread, no partial fills.** You trade at the ticker price
  for any size. A real $10 order is fine; a real $10,000 order is not, and this
  engine would not tell you.
- **Fills are instant and free of market impact,** which makes fast strategies
  look better here than they are. The fee is the only friction modelled.
- **No persistence beyond the two JSON files.** Runs are not concurrent-safe:
  two processes trading at once can lose a write.
- **Binance geo-blocks some regions** (HTTP 451). The price call then fails
  loudly; set `BINANCE_PRICE_URL` to another venue's ticker endpoint if you
  deploy somewhere blocked.
