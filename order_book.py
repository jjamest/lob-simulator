from __future__ import annotations
from collections import defaultdict, deque
from typing import Optional

from order import Order, OrderSide, Trade

class OrderBook:
    def __init__(self) -> None:
        self.bids: dict[float, deque[Order]] = defaultdict(deque)
        self.asks: dict[float, deque[Order]] = defaultdict(deque)
        self._index: dict[str, Order] = {}

    @property
    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def depth(self) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        """
        Aggregate resting qty per price level, excluding canceled orders.
        Returns (bids, asks) sorted best-to-worst.
        """

        bids = [(p, q) for p, q in
                ((p, sum(o.qty for o in dq if not o.is_canceled)) for p, dq in self.bids.items())
                if q > 0]
        asks = [(p, q) for p, q in
                ((p, sum(o.qty for o in dq if not o.is_canceled)) for p, dq in self.asks.items())
                if q > 0]
        bids.sort(key=lambda level: -level[0])
        asks.sort(key=lambda level: level[0])
        return bids, asks

    def add(self, order: Order) -> list[Trade]:
        """
        Add an order.
        Match if trade is possible.
        Returns a list of trades that resulted if any
        """

        trades: list[Trade] = []
        is_market = order.price is None

        if order.side == OrderSide.BID:
            # order is buying

            # match if possible
            while order.qty > 0 and self.asks:
                best = self.best_ask
                if best is None:
                    break
                if not is_market and order.price is not None and order.price < best: # check if a limit buy isn't offering enough for the best ask
                    break
                trades += self._fill(order, self.asks, best)

            # leftover limit orders
            if order.qty > 0 and not is_market:
                assert order.price is not None
                self.bids[order.price].append(order)
                self._index[order.order_id] = order

            # for leftover market orders that don't have a price we just won't do anything as they can't be executed
        elif order.side == OrderSide.ASK:
            # order is selling

            while order.qty > 0 and self.bids:
                best = self.best_bid
                if best is None:
                    break
                if not is_market and order.price is not None and order.price > best: # do not process if ask is too high compared to highest bid
                    # we do not process and those limit orders will be added to the order book
                    break
                trades += self._fill(order, self.bids, best)

            if order.qty > 0 and not is_market:
                assert order.price is not None
                self.asks[order.price].append(order)
                self._index[order.order_id] = order

        return trades

    def cancel(self, order_id: str) -> bool:
        """
        Remove an order in O(1).
        Return true if found, false if not.
        """

        if order_id not in self._index:
            return False

        order = self._index.pop(order_id)
        order.is_canceled = True

        return True

    def print_state(self) -> None:
        print("\nASKS")
        if not self.asks:
            print(" (empty)")
        else:
            for price in sorted(self.asks, reverse=True):
                orders = " ".join(f"{o.order_id}({o.qty})" for o in self.asks[price] if not o.is_canceled)
                print(f" ${price:>6.2f} : {orders}")

        bb, ba = self.best_bid, self.best_ask
        spread = f"Spread: ${ba - bb:.2f}" if bb is not None and ba is not None else "No Spread"
        print(f"-- {spread} --")

        print("BIDS")
        if not self.bids:
            print(" (empty)")
        else:
            for price in sorted(self.bids, reverse=True):
                orders = " ".join(f"{o.order_id}({o.qty})" for o in self.bids[price] if not o.is_canceled)
                print(f" ${price:>6.2f} : {orders}")
        print()
        
    def _fill(self, aggressor: Order, resting_side: dict[float, deque[Order]], price: float) -> list[Trade]:
        """
        Execute fills at a price.
        """

        trades: list[Trade] = []
        q = resting_side[price]

        while q and aggressor.qty > 0: # keep filling until aggressor qty runs out
            # lazy delete canceled orders
            if q[0].is_canceled:
                q.popleft()
                continue # must continue as next could also be canceled

            resting = q[0] # oldest item in deque
            fill_amount = min(aggressor.qty, resting.qty)
            trades.append(Trade(aggressor.order_id, resting.order_id, price, fill_amount))
            aggressor.qty -= fill_amount
            resting.qty -= fill_amount
            if resting.qty == 0:
                q.popleft()
                self._index.pop(resting.order_id, None)

        if not q:
            del resting_side[price]

        return trades


if __name__ == "__main__":
    book = OrderBook()

    def event(label: str, book: "OrderBook", fn, *args) -> None:
        print(f"\n{label}")
        result = fn(*args)
        if isinstance(result, list):
            for t in result:
                print(f"  TRADED: {t}")
        book.print_state()

    event("Add bid $99 qty=10  [B1]", book,
          book.add, Order("B1", OrderSide.BID, 99.0, 10))
    event("Add bid $98 qty=5   [B2]", book,
          book.add, Order("B2", OrderSide.BID, 98.0, 5))
    event("Add bid $99 qty=7   [B3]", book,
          book.add, Order("B3", OrderSide.BID, 99.0, 7))
    event("Add ask     qty=4   [M1]", book,
          book.add, Order("M1", OrderSide.ASK, None, 4))
    event("Add ask $101 qty=8  [A1]", book,
          book.add, Order("A1", OrderSide.ASK, 101.0, 8))
    event("Add ask $102 qty=4  [A2]", book,
          book.add, Order("A2", OrderSide.ASK, 102.0, 4))
    event("Add ask $101 qty=3  [A3]", book,
          book.add, Order("A3", OrderSide.ASK, 101.0, 3))
    event("Cancel B2", book, book.cancel, "B2")
    event("Cancel ZZZZ (non-existent)", book, book.cancel, "ZZZZ")
    event(
        "Aggressive ask $99 qty=12",
        book, book.add, Order("A4", OrderSide.ASK, 99.0, 12),
    )
    event(
        "Aggressive bid $105 qty=6",
        book, book.add, Order("B4", OrderSide.BID, 105.0, 6),
    )
    event(
        "Bid at ask $101 qty=2",
        book, book.add, Order("B5", OrderSide.BID, 101.0, 2),
    )
    event(
        "Aggressive bid $103 qty=5",
        book, book.add, Order("B6", OrderSide.BID, 103.0, 5),
    )
    event(
        "Bid $100 qty=5",
        book, book.add, Order("B7", OrderSide.BID, 100.0, 5),
    )

