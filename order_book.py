from __future__ import annotations
from collections import defaultdict, deque
from typing import Optional

from order import Order, OrderSide, Trade

class OrderBook:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
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

        bids = []
        for price, orders in self.bids.items():
            total_qty = sum(o.qty for o in orders if not o.is_canceled)
            if total_qty > 0:
                bids.append((price, total_qty))

        asks = []
        for price, orders in self.asks.items():
            total_qty = sum(o.qty for o in orders if not o.is_canceled)
            if total_qty > 0:
                asks.append((price, total_qty))

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

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
            trades.append(Trade(aggressor.order_id, resting.order_id, price, fill_amount, self.symbol))
            aggressor.qty -= fill_amount
            resting.qty -= fill_amount
            if resting.qty == 0:
                q.popleft()
                self._index.pop(resting.order_id, None)

        if not q:
            del resting_side[price]

        return trades
