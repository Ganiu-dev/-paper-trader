#!/usr/bin/env python3
"""trade_engine.py — Tree T: a paper trading engine.

Simulation only. Nothing here touches a wallet, a key or a chain: the single
outbound call is a GET to Binance's public ticker endpoint, and everything
else is arithmetic against a JSON ledger on disk.

State
-----
paper_ledger.json    cash, open positions, realised P&L, fees paid
paper_trades.jsonl   one JSON object per simulated fill, append-only

Usage
-----
    python3 trade_engine.py price BNB
    python3 trade_engine.py init
    python3 trade_engine.py buy BNB 10
    python3 trade_engine.py sell BNB 0.0143
    python3 trade_engine.py status
    python3 trade_engine.py reset        # wipe ledger + trade log, back to $100

Every trade is charged a 0.1% taker fee on both sides, so a buy followed
immediately by a sell costs about 0.2% of the notional. That is the point:
the ledger should never flatter a strategy that just churns.

Accounting conventions
----------------------
* A position's ``cost_basis`` is the **cash actually spent**, fee included.
  So ``avg_cost`` is the fee-inclusive break-even price: buy at 700 and the
  position's average cost reads 700.70, because that is what you must sell at
  to get your money back.
* ``realized_pnl`` on a sell is proceeds (net of the sell fee) minus that
  basis, so it equals the real change in cash. Buy $10 and sell it straight
  back and realised P&L is -$0.02, matching the balance.
* Token quantities are stored rounded to 12 decimals. Rounding them harder
  leaves dust that shows up as phantom profit later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: The whole engine is a simulator. There is no live execution path in this
#: module, and this flag must stay False. It exists so that a future version
#: cannot quietly grow one: every money function calls _assert_simulation().
TRADING_ENABLED = False

BASE_DIR = Path(__file__).resolve().parent
LEDGER_PATH = BASE_DIR / "paper_ledger.json"
TRADES_PATH = BASE_DIR / "paper_trades.jsonl"

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
QUOTE = "USDT"
FEE_RATE = 0.001        # 0.1% per side — Binance spot taker, roughly
HTTP_TIMEOUT = 10       # seconds
STARTING_USDT = 100.0
LEDGER_VERSION = 1

_EPS = 1e-12


class PriceUnavailable(RuntimeError):
    """The exchange did not give us a usable price."""


class TradeError(RuntimeError):
    """A simulated trade was refused (bad size, insufficient funds, ...)."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _now() -> str:
    """UTC timestamp, ISO 8601, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _r(value: float, places: int = 8) -> float:
    """Round and kill -0.0, so ledgers stay readable."""
    return round(value, places) + 0.0


def _assert_simulation() -> None:
    if TRADING_ENABLED:
        raise RuntimeError(
            "TRADING_ENABLED is True but trade_engine.py has no live execution "
            "path. Refusing to run. Set it back to False."
        )


def normalise_symbol(symbol: str) -> str:
    """'BNB' / 'bnbusdt' / 'BNB/USDT' / 'BNB-USDT' -> 'BNB/USDT'.

    Only USDT pairs are accepted, and a symbol is never silently repaired:
    a typo like 'BT-C' is a bad ticker, not a request for BTC.
    """
    raw = str(symbol).strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("empty symbol")

    if "/" in raw or "-" in raw:
        sep = "/" if "/" in raw else "-"
        base, _, quote = raw.partition(sep)
        if not base or quote != QUOTE:
            raise ValueError(
                f"unsupported pair {symbol!r}: this engine trades <BASE>/{QUOTE} only"
            )
    elif raw.endswith(QUOTE) and len(raw) > len(QUOTE):
        base = raw[: -len(QUOTE)]
    else:
        base = raw

    if not base.isalnum():
        raise ValueError(f"bad ticker {symbol!r}")
    return f"{base}/{QUOTE}"


def binance_symbol(symbol: str) -> str:
    """'BNB/USDT' -> 'BNBUSDT'."""
    return normalise_symbol(symbol).replace("/", "")


# --------------------------------------------------------------------------- #
# Price feed — Binance public REST, read-only
# --------------------------------------------------------------------------- #

def fetch_price(symbol: str, *, _opener=None) -> float:
    """Live spot price of <symbol> in USDT, straight from Binance.

    Raises PriceUnavailable on any transport or payload problem rather than
    returning a stale or guessed number.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    pair = binance_symbol(symbol)
    url = f"{BINANCE_PRICE_URL}?symbol={urllib.parse.quote(pair)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "paper-trader/1.0", "Accept": "application/json"},
        method="GET",
    )

    open_url = _opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=HTTP_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise PriceUnavailable(f"Binance HTTP {exc.code} for {pair}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PriceUnavailable(f"could not reach Binance for {pair}: {exc}") from exc

    try:
        price = float(payload["price"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PriceUnavailable(f"unexpected Binance payload for {pair}: {payload!r}") from exc

    if not price > 0:
        raise PriceUnavailable(f"Binance returned a non-positive price for {pair}: {price!r}")
    return price


# --------------------------------------------------------------------------- #
# Ledger I/O
# --------------------------------------------------------------------------- #

def _blank_ledger(starting_usdt: float = STARTING_USDT) -> dict[str, Any]:
    return {
        "version": LEDGER_VERSION,
        "mode": "paper",
        "starting_usdt": _r(starting_usdt),
        "usdt": _r(starting_usdt),
        "realized_pnl": 0.0,
        "fees_paid": 0.0,
        "positions": {},
        "fills": 0,
        "created_at": _now(),
        "updated_at": _now(),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write via a temp file in the same directory, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_ledger(*, create: bool = True) -> dict[str, Any]:
    """Read paper_ledger.json, creating it with $100 USDT if it is missing."""
    path = Path(LEDGER_PATH)
    if not path.exists():
        if not create:
            raise FileNotFoundError(path)
        ledger = _blank_ledger()
        _write_json_atomic(path, ledger)
        return ledger

    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        ledger = _blank_ledger()
        _write_json_atomic(path, ledger)
        return ledger

    ledger = json.loads(text)
    ledger.setdefault("positions", {})
    ledger.setdefault("realized_pnl", 0.0)
    ledger.setdefault("fees_paid", 0.0)
    ledger.setdefault("fills", 0)
    return ledger


def save_ledger(ledger: dict[str, Any]) -> None:
    ledger["updated_at"] = _now()
    _write_json_atomic(Path(LEDGER_PATH), ledger)


def reset(*, starting_usdt: float = STARTING_USDT) -> dict[str, Any]:
    """Wipe the ledger and the trade log. Back to a clean $100."""
    ledger = _blank_ledger(starting_usdt)
    save_ledger(ledger)
    Path(TRADES_PATH).write_text("", encoding="utf-8")
    return ledger


# --------------------------------------------------------------------------- #
# Trade log
# --------------------------------------------------------------------------- #

def record_trade(entry: dict[str, Any]) -> dict[str, Any]:
    """Append one fill to paper_trades.jsonl. Never rewrites earlier lines."""
    record = {"trade_id": uuid.uuid4().hex[:12], "mode": "paper", "ts": _now(), **entry}
    path = Path(TRADES_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def read_trades() -> list[dict[str, Any]]:
    """Every recorded fill, oldest first. Skipped lines are reported, not hidden."""
    path = Path(TRADES_PATH)
    if not path.exists():
        return []
    trades, bad = [], 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"warning: skipped {bad} unreadable line(s) in {path.name}", file=sys.stderr)
    return trades


def _truncate_to(path: Path, size: int) -> None:
    """Cut a file back to `size` bytes. Used to roll an append back off."""
    try:
        with path.open("r+", encoding="utf-8") as handle:
            handle.truncate(size)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:            # rollback failed; say so, never hide it
        print(f"warning: could not roll back {path.name}: {exc}", file=sys.stderr)


def _commit(ledger: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Persist the trade-log line and the ledger, or neither of them.

    The append happens first and is truncated back off if the ledger write
    fails. The old order (ledger, then log) meant a failed append left cash
    already spent on disk with no matching fill record -- a ledger you cannot
    reconcile against the log.
    """
    path = Path(TRADES_PATH)
    before = path.stat().st_size if path.exists() else 0
    try:
        record = record_trade(entry)
        save_ledger(ledger)
    except BaseException:
        _truncate_to(path, before)
        raise
    return record


# --------------------------------------------------------------------------- #
# Simulated trading
# --------------------------------------------------------------------------- #

def sim_buy(
    symbol: str,
    usdt_amount: float,
    *,
    fee_rate: float = FEE_RATE,
    _price: float | None = None,
) -> dict[str, Any]:
    """Spend `usdt_amount` of simulated USDT on `symbol` at the live price.

    The 0.1% fee comes out of the amount spent, so $10 buys $9.99 of exposure.
    Positions are averaged: buying twice raises the cost basis rather than
    opening a second lot.
    """
    _assert_simulation()

    pair = normalise_symbol(symbol)
    amount = float(usdt_amount)
    if not amount > 0:
        raise TradeError(f"buy size must be positive, got {usdt_amount!r}")
    if not 0 <= fee_rate < 1:
        raise TradeError(f"fee_rate must be in [0, 1), got {fee_rate!r}")

    ledger = load_ledger()
    cash = float(ledger["usdt"])
    if amount > cash + _EPS:
        raise TradeError(f"insufficient paper cash: need {amount:.8f}, have {cash:.8f} USDT")

    price = float(_price) if _price is not None else fetch_price(pair)
    if not price > 0:
        raise TradeError(f"refusing to buy at a non-positive price: {price!r}")

    fee = _r(amount * fee_rate)
    net_spend = _r(amount - fee)
    quantity = _r(net_spend / price, 12)
    if not quantity > 0:
        # At 12 decimals a tiny size against a huge price rounds to nothing.
        # Left alone that divides by zero when avg_cost is recomputed.
        raise TradeError(
            f"buy size too small for the price: {amount:.8f} USDT at {price:.8f} "
            f"rounds to a zero quantity"
        )

    position = ledger["positions"].get(pair) or {
        "symbol": pair,
        "token_amount": 0.0,
        "cost_basis": 0.0,
        "avg_cost": 0.0,
    }
    position["token_amount"] = _r(float(position["token_amount"]) + quantity, 12)
    # Cost basis is cash out of pocket, fee included. avg_cost is therefore the
    # fee-inclusive break-even price, not the raw fill price.
    position["cost_basis"] = _r(float(position["cost_basis"]) + amount)
    position["avg_cost"] = _r(float(position["cost_basis"]) / float(position["token_amount"]))
    position["last_price"] = price
    position["opened_at"] = position.get("opened_at") or _now()
    position["updated_at"] = _now()
    ledger["positions"][pair] = position

    ledger["usdt"] = _r(cash - amount)
    ledger["fees_paid"] = _r(float(ledger["fees_paid"]) + fee)
    ledger["fills"] = int(ledger["fills"]) + 1

    return _commit(
        ledger,
        {
            "side": "buy",
            "symbol": pair,
            "binance_symbol": binance_symbol(pair),
            "price": price,
            "usdt_amount": _r(amount),
            "fee_usdt": fee,
            "net_spend_usdt": net_spend,
            "token_amount": quantity,
            "usdt_after": ledger["usdt"],
            "position_after": position["token_amount"],
            "avg_cost_after": position["avg_cost"],
            "realized_pnl": None,
        }
    )


def sim_sell(
    symbol: str,
    token_amount: float,
    *,
    fee_rate: float = FEE_RATE,
    _price: float | None = None,
) -> dict[str, Any]:
    """Sell `token_amount` of a held position at the live price.

    Proceeds are credited net of the 0.1% fee. Realised P&L is proceeds minus
    the fee-inclusive cost basis of the tokens sold, so an instant round trip
    realises exactly the two fees and nothing else.
    """
    _assert_simulation()

    pair = normalise_symbol(symbol)
    quantity = float(token_amount)
    if not quantity > 0:
        raise TradeError(f"sell size must be positive, got {token_amount!r}")
    if not 0 <= fee_rate < 1:
        raise TradeError(f"fee_rate must be in [0, 1), got {fee_rate!r}")

    ledger = load_ledger()
    position = ledger["positions"].get(pair)
    if not position:
        raise TradeError(f"no open position in {pair}")

    held = float(position["token_amount"])
    if quantity > held * (1 + 1e-9) + _EPS:
        raise TradeError(f"cannot sell {quantity:.8f} {pair}: only {held:.8f} held")
    quantity = min(quantity, held)          # clamp float dust from a "sell all"

    price = float(_price) if _price is not None else fetch_price(pair)
    if not price > 0:
        raise TradeError(f"refusing to sell at a non-positive price: {price!r}")

    gross = _r(quantity * price)
    fee = _r(gross * fee_rate)
    proceeds = _r(gross - fee)

    # Basis sold uses the fee-inclusive average cost, so a full round trip
    # realises exactly the sum of the two fees.
    avg_cost = float(position["avg_cost"])
    held_basis = float(position["cost_basis"])
    basis_sold = _r(quantity * avg_cost)
    if quantity >= held * (1 - 1e-12):        # selling everything: no dust
        basis_sold = _r(held_basis)
    realized = _r(proceeds - basis_sold)

    remaining = _r(held - quantity, 12)
    if remaining <= _EPS:
        remaining = 0.0
        ledger["positions"].pop(pair, None)
    else:
        position["token_amount"] = remaining
        position["cost_basis"] = _r(max(0.0, held_basis - basis_sold))
        position["avg_cost"] = _r(float(position["cost_basis"]) / remaining)
        position["last_price"] = price
        position["updated_at"] = _now()

    ledger["usdt"] = _r(float(ledger["usdt"]) + proceeds)
    ledger["realized_pnl"] = _r(float(ledger["realized_pnl"]) + realized)
    ledger["fees_paid"] = _r(float(ledger["fees_paid"]) + fee)
    ledger["fills"] = int(ledger["fills"]) + 1

    return _commit(
        ledger,
        {
            "side": "sell",
            "symbol": pair,
            "binance_symbol": binance_symbol(pair),
            "price": price,
            "token_amount": quantity,
            "gross_usdt": gross,
            "fee_usdt": fee,
            "proceeds_usdt": proceeds,
            "cost_basis_sold": basis_sold,
            "realized_pnl": realized,
            "usdt_after": ledger["usdt"],
            "position_after": remaining,
            "avg_cost_after": avg_cost if remaining else None,
        }
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def status(*, mark_to_market: bool = True) -> dict[str, Any]:
    """Cash, open positions marked at the live price, and equity.

    Equity is cash plus what the positions are worth. If a live price cannot
    be fetched the position keeps its last fill price and says so via
    ``price_source`` — a failed read is never silently counted as a loss.
    """
    ledger = load_ledger()
    positions: list[dict[str, Any]] = []
    market_value = 0.0
    unrealized = 0.0

    for pair, position in sorted(ledger["positions"].items()):
        quantity = float(position["token_amount"])
        basis = float(position["cost_basis"])
        price: float | None = None
        source = "cost_basis"

        if mark_to_market:
            try:
                price, source = fetch_price(pair), "live"
            except PriceUnavailable as exc:
                entry_error = str(exc)
                price = float(position.get("last_price") or 0.0) or None
                source = "last_fill" if price else "cost_basis"
            else:
                entry_error = None
        else:
            entry_error = None
            price = float(position.get("last_price") or 0.0) or None
            source = "last_fill" if price else "cost_basis"

        if price is None:
            value = basis
            position_pnl = 0.0
        else:
            value = _r(quantity * price)
            position_pnl = _r(value - basis)

        entry: dict[str, Any] = {
            "symbol": pair,
            "token_amount": quantity,
            "avg_cost": float(position["avg_cost"]),
            "cost_basis": basis,
            "price": price,
            "price_source": source,
            "market_value": value,
            "unrealized_pnl": position_pnl,
        }
        if entry_error:
            entry["error"] = entry_error

        market_value = _r(market_value + value)
        unrealized = _r(unrealized + position_pnl)
        positions.append(entry)

    cash = float(ledger["usdt"])
    equity = _r(cash + market_value)
    starting = float(ledger["starting_usdt"])
    return {
        "mode": "paper",
        "trading_enabled": TRADING_ENABLED,
        "starting_usdt": starting,
        "usdt": cash,
        "positions": positions,
        "market_value": market_value,
        "equity": equity,
        "realized_pnl": float(ledger["realized_pnl"]),
        "unrealized_pnl": unrealized,
        "fees_paid": float(ledger["fees_paid"]),
        "fills": int(ledger["fills"]),
        "total_return_pct": _r((equity / starting - 1) * 100, 6),
        "updated_at": ledger.get("updated_at"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _fmt(value: float, places: int = 4) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.{places}f}"
    return f"{value:.8f}".rstrip("0").rstrip(".")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper trading engine (simulation only)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create paper_ledger.json with $100 USDT if missing")

    p_price = sub.add_parser("price", help="print the live price")
    p_price.add_argument("symbol")

    p_buy = sub.add_parser("buy", help="simulated buy")
    p_buy.add_argument("symbol")
    p_buy.add_argument("usdt_amount", type=float)

    p_sell = sub.add_parser("sell", help="simulated sell")
    p_sell.add_argument("symbol")
    p_sell.add_argument("token_amount", type=float)

    p_status = sub.add_parser("status", help="cash, positions and equity")
    p_status.add_argument("--offline", action="store_true", help="skip live pricing")

    p_reset = sub.add_parser("reset", help="wipe ledger and trade log")
    p_reset.add_argument("--usdt", type=float, default=STARTING_USDT)

    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            ledger = load_ledger()
            print(f"ledger ready: {_fmt(ledger['usdt'])} USDT at {LEDGER_PATH}")
            return 0

        if args.command == "price":
            pair = normalise_symbol(args.symbol)
            price = fetch_price(pair)
            print(f"{pair} {_fmt(price)}")
            return 0

        if args.command == "buy":
            fill = sim_buy(args.symbol, args.usdt_amount)
            print(
                f"BOUGHT {_fmt(fill['token_amount'], 8)} {fill['symbol']} "
                f"@ {_fmt(fill['price'])} = {_fmt(fill['net_spend_usdt'])} USDT "
                f"(fee {_fmt(fill['fee_usdt'])}) | cash {_fmt(fill['usdt_after'])} USDT"
            )
            return 0

        if args.command == "sell":
            fill = sim_sell(args.symbol, args.token_amount)
            sign = "+" if fill["realized_pnl"] >= 0 else ""
            print(
                f"SOLD {_fmt(fill['token_amount'], 8)} {fill['symbol']} "
                f"@ {_fmt(fill['price'])} = {_fmt(fill['proceeds_usdt'])} USDT net "
                f"(fee {_fmt(fill['fee_usdt'])}) | P&L {sign}{_fmt(fill['realized_pnl'])} "
                f"| cash {_fmt(fill['usdt_after'])} USDT"
            )
            return 0

        if args.command == "status":
            snapshot = status(mark_to_market=not args.offline)
            print(f"mode: paper   TRADING_ENABLED={TRADING_ENABLED}")
            print(f"cash: {_fmt(snapshot['usdt'])} USDT")
            for position in snapshot["positions"]:
                price = _fmt(position["price"]) if position["price"] else "n/a"
                pnl = position["unrealized_pnl"]
                pnl_text = f"{pnl:+.4f}" if pnl is not None else "n/a"
                print(
                    f"  {position['symbol']}: {_fmt(position['token_amount'], 8)} "
                    f"@ {_fmt(position['avg_cost'])} (now {price}) | P&L {pnl_text}"
                )
            if not snapshot["positions"]:
                print("  no open positions")
            print(
                f"equity: {_fmt(snapshot['equity'])} USDT "
                f"({snapshot['total_return_pct']:+.4f}%) | "
                f"realised {snapshot['realized_pnl']:+.4f} | "
                f"fees {_fmt(snapshot['fees_paid'])} | fills {snapshot['fills']}"
            )
            return 0

        if args.command == "reset":
            ledger = reset(starting_usdt=args.usdt)
            print(f"reset: {_fmt(ledger['usdt'])} USDT, trade log cleared")
            return 0

    except (PriceUnavailable, TradeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
