import itertools
from typing import Optional
import uuid

from order import Order, Trade, OrderRequest, MarketSnapshot
from order_book import OrderBook

class Exchange:
    def __init__(self) -> None:
        self._step = itertools.count(0)
        self.book = OrderBook()

    def submit(self, order_request: OrderRequest) -> tuple[str, list[Trade]]:
        order = Order.from_request(order_request, self._generate_id())
        return order.order_id, self.book.add(order)

    def cancel(self, order_id: str) -> bool:
        return self.book.cancel(order_id)

    def _generate_id(self) -> str:
        return str(uuid.uuid4())

    @property
    def best_ask(self) -> Optional[float]:
        return self.book.best_ask

    @property
    def best_bid(self) -> Optional[float]:
        return self.book.best_bid

    def snapshot(self) -> MarketSnapshot:
        bids, asks = self.book.depth()
        return MarketSnapshot(
            best_bid=self.book.best_bid,
            best_ask=self.book.best_ask,
            bids=bids,
            asks=asks,
        )
