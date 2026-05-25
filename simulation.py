from __future__ import annotations
import random
import itertools
import csv
from typing import Optional
from main import OrderBook, Order, Trade
import matplotlib.pyplot as plt

SEED = 34
STEPS = 100_000
FAIR_VALUE = 100.0
TICK = 0.01 # min price increment

MM_HALF_SPREAD = 0.10 # MM quotes mid ± $0.10
MM_QTY = 5
LOG_FILE = "trades.log"
SNAPSHOT_EVERY = 1_000 # tell us progress every N steps

_oid = itertools.count(1)

def _new_id(prefix: str = "O") -> str:
    return f"{prefix}{next(_oid)}"


def random_order(book: OrderBook, rng: random.Random) -> Order:
    """Random order generator"""
    bb = book.best_bid()
    ba = book.best_ask()

    if bb and ba:
        mid = (bb + ba) / 2.0
    elif bb:
        mid = bb + 0.50
    elif ba:
        mid = ba - 0.50
    else:
        mid = FAIR_VALUE

    side = rng.choice(["bid", "ask"])
    aggressive = rng.random() < 0.30 # 30% chance of crossing the spread

    if aggressive:
        if side == "bid":
            ref = ba if ba else mid
            price = round(ref + rng.uniform(0.01, 0.25), 2)
        elif side == "ask":
            ref = bb if bb else mid
            price = round(ref - rng.uniform(0.01, 0.25), 2)
    else:
        raw = mid + rng.uniform(-1.50, 1.50)
        price = round(round(raw / TICK) * TICK, 2)

    qty = int(rng.lognormvariate(1.5, 1.0)) + 1 # median=6, and has spikes
    return Order(_new_id(), side, price, qty)


# Agent

class MarketMaker:
    def __init__(
        self,
        half_spread: float = MM_HALF_SPREAD,
        qty: int = MM_QTY,
    ) -> None:
        self.half_spread = half_spread
        self.qty = qty
        self.position: int = 0 # + long / - short
        self.cash: float = 0.0
        self.bid_id: Optional[str] = None
        self.ask_id: Optional[str] = None
        self.position_history: list[tuple[int, int]] = []
        self.fill_count: int = 0

    def refresh_quotes(self, book: OrderBook, step: int) -> list[Order]:
        """Cancel stale quotes and return new bid and ask to add to the book."""
        if self.bid_id:
            book.cancel(self.bid_id, quiet=True)
            self.bid_id = None
        if self.ask_id:
            book.cancel(self.ask_id, quiet=True)
            self.ask_id = None

        bb = book.best_bid()
        ba = book.best_ask()
        if bb and ba:
            mid = (bb + ba) / 2.0
        elif bb:
            mid = bb + self.half_spread
        elif ba:
            mid = ba - self.half_spread
        else:
            mid = FAIR_VALUE

        bid_price = round(mid - self.half_spread, 2)
        ask_price = round(mid + self.half_spread, 2)

        self.position_history.append((step, self.position))

        if bid_price >= ask_price:
            return []

        bid = Order(_new_id("MM"), "bid", bid_price, self.qty)
        ask = Order(_new_id("MM"), "ask", ask_price, self.qty)
        self.bid_id = bid.order_id
        self.ask_id = ask.order_id
        return [bid, ask]

    def on_trade(self, trade: Trade) -> None:
        """Update inventory and cash whenever MM is a party to a trade."""
        mm_ids = {id_ for id_ in (self.bid_id, self.ask_id) if id_ is not None}
        involved_id = None
        if trade.resting_id in mm_ids:
            involved_id = trade.resting_id
        elif trade.aggressor_id in mm_ids:
            involved_id = trade.aggressor_id
        else:
            return

        self.fill_count += 1
        # Determine direction: bid fill = bought, ask fill = sold
        if involved_id == self.bid_id:
            self.position += trade.qty
            self.cash -= trade.price * trade.qty
        else:
            self.position -= trade.qty
            self.cash += trade.price * trade.qty


# Simulation
def run(steps: int = STEPS, seed: int = SEED) -> None:
    rng = random.Random(seed)
    book = OrderBook()
    mm = MarketMaker()
    total_trades = 0

    print(f"Running {steps:,}-step simulation  (seed={seed})")
    print(f"Log to {LOG_FILE}\n")

    with open(LOG_FILE, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "aggressor_id", "resting_id", "price", "qty"])

        for step in range(1, steps + 1):
            # 1. MM refreshes quotes
            for order in mm.refresh_quotes(book, step):
                for t in book.add(order):
                    mm.on_trade(t)
                    writer.writerow([step, t.aggressor_id, t.resting_id, t.price, t.qty])
                    total_trades += 1

            # 2. Random agent submits one order
            for t in book.add(random_order(book, rng)):
                mm.on_trade(t)
                writer.writerow([step, t.aggressor_id, t.resting_id, t.price, t.qty])
                total_trades += 1

            # 3. Periodic stdout snapshot
            if step % SNAPSHOT_EVERY == 0:
                bb = book.best_bid() or 0.0
                ba = book.best_ask() or 0.0
                mid = (bb + ba) / 2.0 if bb and ba else FAIR_VALUE
                pnl = mm.cash + mm.position * mid
                print(
                    f"  step {step:>6,}  |  pos={mm.position:>+5}  "
                    f"cash={mm.cash:>+10.2f}  pnl≈{pnl:>+8.2f}  "
                    f"mid={mid:.2f}  MM-fills={mm.fill_count}"
                )

    # summary
    bb = book.best_bid() or 0.0
    ba = book.best_ask() or 0.0
    mid = (bb + ba) / 2.0 if bb and ba else FAIR_VALUE
    pnl = mm.cash + mm.position * mid

    print(f"  Simulation finished  ({steps:,} steps)")
    print(f"  Total trades logged : {total_trades:,}  →  {LOG_FILE}")
    print(f"  MM fills            : {mm.fill_count:,}")
    print(f"  MM final position   : {mm.position:>+,}")
    print(f"  MM cash             : {mm.cash:>+,.2f}")
    print(f"  MM P&L   : {pnl:>+,.2f}  (mid={mid:.2f})")
    print(f"{'═'*60}")

    _plot_position_chart(mm.position_history)


def _plot_position_chart(history: list[tuple[int, int]]) -> None:
    if not history:
        return

    steps = [s for s, _ in history]
    positions = [p for _, p in history]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, positions, linewidth=0.8, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(steps, positions, 0, where=[p > 0 for p in positions], alpha=0.25, color="green", label="long")
    ax.fill_between(steps, positions, 0, where=[p < 0 for p in positions], alpha=0.25, color="red", label="short")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position")
    ax.set_title("Market-maker inventory over time")
    ax.legend()
    fig.tight_layout()
    plt.savefig("mm_position.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run()
