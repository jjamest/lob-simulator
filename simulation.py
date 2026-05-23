from __future__ import annotations
import random
import itertools
import csv
from typing import Optional
from main import OrderBook, Order, Trade
import matplotlib.pyplot as plt

SEED = 35
STEPS = 100_000
FAIR_VALUE = 100.0
TICK = 0.01 # min price increment

# Ornstein-Uhlenbeck fair value parameters
OU_THETA = 0.005   # mean-reversion speed (per step)
OU_MU    = 100.0   # long-run mean
OU_SIGMA = 0.05    # per-step volatility

MM_HALF_SPREAD = 0.10 # MM quotes mid ± $0.10
MM_QTY = 10
MM_SKEW_K = 0.01   # price skew per unit of inventory
MM_QTY_K  = 0.10   # qty adjustment per unit of inventory (fraction of base qty)
LOG_FILE = "trades.log"
SNAPSHOT_EVERY = 1_000 # tell us progress every N steps

_oid = itertools.count(1)

def _new_id(prefix: str = "O") -> str:
    return f"{prefix}{next(_oid)}"


def ou_step(fv: float, rng: random.Random) -> float:
    """One step of the Ornstein-Uhlenbeck mean-reverting process."""
    return fv + OU_THETA * (OU_MU - fv) + OU_SIGMA * rng.gauss(0, 1)


def random_order(book: OrderBook, rng: random.Random, fv: float) -> Order:
    """Random order generator anchored to the current fair value."""
    bb = book.best_bid()
    ba = book.best_ask()

    if bb and ba:
        mid = (bb + ba) / 2.0
    elif bb:
        mid = bb + 0.50
    elif ba:
        mid = ba - 0.50
    else:
        mid = fv

    side = rng.choice(["bid", "ask"])

    # 10% chance of market order
    if rng.random() < 0.10:
        qty = int(rng.lognormvariate(1.5, 1.0)) + 1
        return Order(_new_id(), side, None, qty)

    aggressive = rng.random() < 0.30 # 30% chance of crossing the spread

    if aggressive:
        if side == "bid":
            ref = ba if ba else mid
            price = round(ref + rng.uniform(0.01, 0.25), 2)
        else:
            ref = bb if bb else mid
            price = round(ref - rng.uniform(0.01, 0.25), 2)
    else:
        # anchor to fair value rather than just mid to follow the OU process
        raw = fv + rng.uniform(-1.50, 1.50)
        price = round(round(raw / TICK) * TICK, 2)

    qty = int(rng.lognormvariate(1.5, 1.0)) + 1 # median=6, and has spikes
    return Order(_new_id(), side, price, qty)


# Agent

class MarketMaker:
    def __init__(
        self,
        half_spread: float = MM_HALF_SPREAD,
        qty: int = MM_QTY,
        skew_k: float = MM_SKEW_K,
        qty_k: float = MM_QTY_K,
    ) -> None:
        self.half_spread = half_spread
        self.qty = qty
        self.skew_k = skew_k
        self.qty_k = qty_k
        self.position: int = 0 # + long / - short
        self.cash: float = 0.0
        self.bid_id: Optional[str] = None
        self.ask_id: Optional[str] = None
        self.position_history: list[tuple[int, int]] = []
        self.fill_count: int = 0

    def refresh_quotes(self, book: OrderBook, step: int) -> list[Order]:
        """Cancel stale quotes and return fresh bid + ask to add to the book.
        Quotes are skewed away from inventory: long → lower both prices to sell,
        short → raise both to buy."""
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

        # skew: positive position shifts mid down so we sell cheaper to unwind
        skew = self.skew_k * self.position
        skewed_mid = mid - skew

        bid_price = round(skewed_mid - self.half_spread, 2)
        ask_price = round(skewed_mid + self.half_spread, 2)

        self.position_history.append((step, self.position))

        if bid_price >= ask_price:
            return []

        # When long: shrink bid qty (avoid buying more), grow ask qty (sell faster).
        # When short: mirror. Clamp both sides to at least 1.
        adj = self.qty_k * self.position
        bid_qty = max(1, round(self.qty - adj))
        ask_qty = max(1, round(self.qty + adj))

        bid = Order(_new_id("MM"), "bid", bid_price, bid_qty)
        ask = Order(_new_id("MM"), "ask", ask_price, ask_qty)
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
    fv = FAIR_VALUE  # current Ornstein-Uhlenbeck fair value

    # track resting order ids submitted by random agents so they can cancel them
    resting_order_ids: list[str] = []

    print(f"Running {steps:,}-step simulation  (seed={seed})")
    print(f"Log to {LOG_FILE}\n")

    with open(LOG_FILE, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "aggressor_id", "resting_id", "price", "qty", "fair_value"])

        for step in range(1, steps + 1):
            # 0. Evolve fair value via OU process
            fv = ou_step(fv, rng)

            # 1. MM refreshes quotes
            for order in mm.refresh_quotes(book, step):
                for t in book.add(order):
                    mm.on_trade(t)
                    writer.writerow([step, t.aggressor_id, t.resting_id, t.price, t.qty, round(fv, 4)])
                    total_trades += 1

            # 2. Random agent: ~20% chance cancels one of its own stale resting orders
            if resting_order_ids and rng.random() < 0.20:
                cancel_id = rng.choice(resting_order_ids)
                if book.cancel(cancel_id, quiet=True):
                    resting_order_ids.remove(cancel_id)

            # 3. Random agent submits one order
            new_order = random_order(book, rng, fv)
            trades = book.add(new_order)
            # if the order rested (no trades and it was a limit order), track it
            if not trades and new_order.price is not None:
                resting_order_ids.append(new_order.order_id)
            # trim the tracking list to avoid unbounded growth
            if len(resting_order_ids) > 200:
                resting_order_ids = resting_order_ids[-200:]
            for t in trades:
                mm.on_trade(t)
                writer.writerow([step, t.aggressor_id, t.resting_id, t.price, t.qty, round(fv, 4)])
                total_trades += 1

            # 4. Periodic stdout snapshot
            if step % SNAPSHOT_EVERY == 0:
                bb = book.best_bid() or 0.0
                ba = book.best_ask() or 0.0
                mid = (bb + ba) / 2.0 if bb and ba else fv
                pnl = mm.cash + mm.position * mid
                print(
                    f"  step {step:>6,}  |  pos={mm.position:>+5}  "
                    f"cash={mm.cash:>+10.2f}  pnl≈{pnl:>+8.2f}  "
                    f"mid={mid:.2f}  fv={fv:.2f}  MM-fills={mm.fill_count}"
                )

    # summary
    bb = book.best_bid() or 0.0
    ba = book.best_ask() or 0.0
    mid = (bb + ba) / 2.0 if bb and ba else fv
    pnl = mm.cash + mm.position * mid

    print(f"\n  Simulation finished  ({steps:,} steps)")
    print(f"  Total trades logged    : {total_trades:,}  →  {LOG_FILE}")
    print(f"  MM fills               : {mm.fill_count:,}")
    print(f"  MM final position      : {mm.position:>+,}")
    print(f"  MM cash                : {mm.cash:>+,.2f}")
    print(f"  MM net liquidation P&L : {pnl:>+,.2f}  (mid={mid:.2f}  fv={fv:.2f})")

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
