from __future__ import annotations
from typing import Optional

from agent import Agent, AgentAction
from order import MarketSnapshot, OrderRequest, OrderSide, Trade


class MarketMaker(Agent):
    def __init__(
        self,
        half_spread: float = 0.10,
        qty: int = 10,
        skew_k: float = 0.01,
        qty_k: float = 0.10,
        fallback_mid: float = 100.0,
    ) -> None:
        self.half_spread = half_spread
        self.qty = qty
        self.skew_k = skew_k
        self.qty_k = qty_k
        self.fallback_mid = fallback_mid

        self.position = 0
        self.cash = 0.0
        self.bid_id: Optional[str] = None
        self.ask_id: Optional[str] = None

    def act(self, snapshot: MarketSnapshot) -> AgentAction:
        cancels = [order_id for order_id in (self.bid_id, self.ask_id) if order_id is not None]
        self.bid_id = None
        self.ask_id = None

        if snapshot.best_bid is not None and snapshot.best_ask is not None:
            mid = (snapshot.best_bid + snapshot.best_ask) / 2.0
        elif snapshot.best_bid is not None:
            mid = snapshot.best_bid + self.half_spread
        elif snapshot.best_ask is not None:
            mid = snapshot.best_ask - self.half_spread
        else:
            mid = self.fallback_mid

        # skew: positive position shifts mid down so we sell cheaper to unwind
        skewed_mid = mid - self.skew_k * self.position
        bid_price = round(skewed_mid - self.half_spread, 2)
        ask_price = round(skewed_mid + self.half_spread, 2)

        if bid_price >= ask_price:
            return AgentAction(cancels=cancels)

        # long -> shrink bid qty, grow ask qty; short -> mirror. clamp to at least 1.
        adj = self.qty_k * self.position
        bid_qty = max(1, round(self.qty - adj))
        ask_qty = max(1, round(self.qty + adj))

        return AgentAction(
            cancels=cancels,
            submits=[
                OrderRequest(OrderSide.BID, bid_price, bid_qty),
                OrderRequest(OrderSide.ASK, ask_price, ask_qty),
            ],
        )

    def on_order_accepted(self, order_id: str, request: OrderRequest, trades: list[Trade]) -> None:
        if request.side == OrderSide.BID:
            self.bid_id = order_id
        else:
            self.ask_id = order_id

    def on_fill(self, trade: Trade) -> None:
        mm_ids = {order_id for order_id in (self.bid_id, self.ask_id) if order_id is not None}
        if trade.resting_id in mm_ids:
            involved_id = trade.resting_id
        elif trade.aggressor_id in mm_ids:
            involved_id = trade.aggressor_id
        else:
            return

        if involved_id == self.bid_id:
            self.position += trade.qty
            self.cash -= trade.price * trade.qty
        else:
            self.position -= trade.qty
            self.cash += trade.price * trade.qty
