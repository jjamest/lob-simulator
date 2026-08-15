from collections import defaultdict
import itertools
from shutil import ExecError
from typing import Optional
import uuid

from order import Order, Trade, OrderRequest, MarketSnapshot
from order_book import OrderBook

class Exchange:
    def __init__(self) -> None:
        self._step = itertools.count(0)
        self.books: dict[str, OrderBook] = {} # key=symbol

    def submit(self, order_request: OrderRequest) -> tuple[str, list[Trade]]:
        if order_request.symbol not in self.books:
            raise Exception(f"{order_request.symbol} doesn't exist")

        order = Order.from_request(order_request, self._generate_id())
        return order.order_id, self.books[order_request.symbol].add(order)

    def cancel(self, order_id: str, symbol: str) -> bool:
        return self.books[symbol].cancel(order_id)

    def _generate_id(self) -> str:
        return str(uuid.uuid4())

    def best_ask(self, symbol: str) -> Optional[float]:
        return self.books[symbol].best_ask

    def best_bid(self, symbol: str) -> Optional[float]:
        return self.books[symbol].best_bid

    def snapshot(self, symbol: str) -> MarketSnapshot:
        bids, asks = self.books[symbol].depth()
        return MarketSnapshot(
            best_bid=self.books[symbol].best_bid,
            best_ask=self.books[symbol].best_ask,
            bids=bids,
            asks=asks,
            symbol=symbol
        )

    def register_symbol(self, symbol: str) -> bool:
        """Register a symbol on the exchange, and returning success"""
        if symbol in self.books:
            return False

        self.books[symbol] = OrderBook(symbol)
        return True 