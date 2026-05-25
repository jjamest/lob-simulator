from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional
import itertools

_seq = itertools.count(1)

@dataclass
class Order:
    order_id: str
    side: str # bid or ask
    price: Optional[float]  # None = market order
    qty: int
    seq: int = field(default_factory=lambda: next(_seq))


@dataclass
class Trade:
    aggressor_id: str # trade coming in
    resting_id: str # trade sitting
    price: float
    qty: int

    def __str__(self) -> str:
        return (
            f"  TRADE  {self.aggressor_id} x {self.resting_id}"
            f"  @ ${self.price:.2f}  qty={self.qty}"
        )


class OrderBook:
    """
    Price-level dict of first in first out deques. We use deques because of popleft() is O(1) 
    while a pop(0) with a list if O(n), and not a queue because that doesn't support middle deletion like deques does with del q[i]

    bids[price] -> deque[Order]  (highest price = best bid)
    asks[price] -> deque[Order]  (lowest  price = best ask)
    _index maps order_id -> (side, price) for O(1) cancel lookup.
    """

    def __init__(self) -> None:
        self.bids: dict[float, deque[Order]] = defaultdict(deque)
        self.asks: dict[float, deque[Order]] = defaultdict(deque)
        self._index: dict[str, tuple[str, float]] = {}

    # Accessors
    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    # Events
    def add(self, order: Order) -> list[Trade]:
        """
        Add an order; match immediately if a trade is possible
        Returns a list of trades that resulted from given order being added
        """
        trades: list[Trade] = []
        is_market = order.price is None

        if order.side == "bid":
            while order.qty > 0 and self.asks:
                best = self.best_ask()
                if order.price < best: # current bid price is lower than the lowest ask, meaning we cannot fill
                    break
                trades += self._fill(order, self.asks, best) # fill the order
            if order.qty > 0: # order is filled, but may be remaining quantities left to wait
                self.bids[order.price].append(order)
                self._index[order.order_id] = ("bid", order.price)

        elif order.side == "ask":
            while order.qty > 0 and self.bids:
                best = self.best_bid()
                if order.price > best: # current ask price is higher than the highest bid, meaning we cannot fill
                    break
                trades += self._fill(order, self.bids, best) # fill the order
            if order.qty > 0:
                self.asks[order.price].append(order)
                self._index[order.order_id] = ("ask", order.price)

        return trades

    def cancel(self, order_id: str, quiet: bool = False) -> bool:
        """Remove an order by ID. Returns True if found, False if not found."""
        if order_id not in self._index:
            if not quiet:
                print(f"  CANCEL {order_id} not found")
            return False

        side, price = self._index.pop(order_id)
        book_side = self.bids if side == "bid" else self.asks
        q = book_side[price] # deque
        for i, o in enumerate(q):
            if o.order_id == order_id:
                del q[i]   # deque supports O(n) index delete. only need to delete once
                break
        if not q: # if the cancelled order was the last order, delete that price level
            del book_side[price]
        if not quiet:
            print(f"  CANCEL {order_id}  (side={side}, price={price})")
        return True
    

    def _fill(
        self,
        aggressor: Order,
        resting_side: dict[float, deque[Order]],
        price: float,
    ) -> list[Trade]:
        """Execute fills at a single price level"""

        trades: list[Trade] = []
        q = resting_side[price]
        while q and aggressor.qty > 0: # while remaininng agressor quantity left to continue filling...
            resting = q[0]
            fill = min(aggressor.qty, resting.qty) # amount to fill
            trades.append(Trade(aggressor.order_id, resting.order_id, price, fill))
            aggressor.qty -= fill
            resting.qty -= fill
            if resting.qty == 0:
                q.popleft()
                self._index.pop(resting.order_id, None)
        if not q:
            del resting_side[price]
        return trades

    # Display
    def print_state(self) -> None:
        W = 52
        print("\n" + "─" * W)

        print("  ASKS")
        if not self.asks:
            print("    (empty)")
        for price in sorted(self.asks):
            orders = list(self.asks[price])
            total = sum(o.qty for o in orders)
            detail = "  ".join(
                f"{o.order_id}[q={o.qty} s#{o.seq}]" for o in orders
            )
            print(f"    ${price:>7.2f}  total={total:<4}  {detail}")

        spread_str = "no spread"
        if self.best_bid() and self.best_ask():
            spread = self.best_ask() - self.best_bid()
            spread_str = f"spread = ${spread:.2f}"
        print(f"\n  {'─── ' + spread_str + ' ───':^{W - 2}}\n")

        print("  BIDS")
        if not self.bids:
            print("    (empty)")
        for price in sorted(self.bids, reverse=True):
            orders = list(self.bids[price])
            total = sum(o.qty for o in orders)
            detail = "  ".join(
                f"{o.order_id}[q={o.qty} s#{o.seq}]" for o in orders
            )
            print(f"    ${price:>7.2f}  total={total:<4}  {detail}")

        print("─" * W + "\n")


# Scenarios

def event(label: str, book: OrderBook, fn, *args) -> None:
    print(f"\n{'═' * 52}")
    print(f"  EVENT: {label}")
    result = fn(*args)
    if isinstance(result, list):
        for t in result:
            print(str(t))
    book.print_state()


def main() -> None:
    book = OrderBook()

    event("Add bid $99 qty=10  [B1]", book,
          book.add, Order("B1", "bid", 99.0, 10))
    event("Add bid $98 qty=5   [B2]", book,
          book.add, Order("B2", "bid", 98.0, 5))
    event("Add bid $99 qty=7   [B3]  ← same price as B1, seq# is higher → behind B1", book,
          book.add, Order("B3", "bid", 99.0, 7))
    event("Add ask $101 qty=8  [A1]", book,
          book.add, Order("A1", "ask", 101.0, 8))
    event("Add ask $102 qty=4  [A2]", book,
          book.add, Order("A2", "ask", 102.0, 4))
    event("Add ask $101 qty=3  [A3]  ← same price as A1, behind A1 in queue", book,
          book.add, Order("A3", "ask", 101.0, 3))
    event("Cancel B2", book, book.cancel, "B2")
    event("Cancel ZZZZ (non-existent)", book, book.cancel, "ZZZZ")
    event(
        "Aggressive ask $99 qty=12  [A4] → should fill B1(10) then B3(2)",
        book, book.add, Order("A4", "ask", 99.0, 12),
    )
    event(
        "Aggressive bid $105 qty=6  [B4] → partial fill on A1",
        book, book.add, Order("B4", "bid", 105.0, 6),
    )
    event(
        "Bid at ask $101 qty=2  [B5] → exactly clears remaining A1",
        book, book.add, Order("B5", "bid", 101.0, 2),
    )
    event(
        "Aggressive bid $103 qty=5  [B6] → fills all of A3(3), rests 2 at $103",
        book, book.add, Order("B6", "bid", 103.0, 5),
    )
    event(
        "Bid $100 qty=5  [B7] → no match, rests in book",
        book, book.add, Order("B7", "bid", 100.0, 5),
    )

if __name__ == "__main__":
    main()
