from typing import Optional

from agent import Agent, AgentAction
from order import MarketSnapshot, OrderRequest, OrderSide, Trade


class MarketMaker(Agent):
    def __init__(
            self, 
            half_spread: float = 0.1, 
            qty: int = 10,
            skew_k: float = 0.01,
            qty_k: float = 0.10,
            fallback_mid: float = 100.0
            ) -> None:
        self.half_spread = half_spread
        self.qty = qty
        self.skew_k = skew_k # sensitivity controlling how much position skews midpoint price
        self.qty_k = qty_k # sensitivity controlling how much position skews quote sizes
        self.fallback_mid = fallback_mid

        self.position = 0
        self.cash = 0.0

        # we only hold a pair of bid and ask every step
        self.bid_id: Optional[str] = None
        self.ask_id: Optional[str] = None

        # for every order submitted
        self._order_side: dict[str, OrderSide] = {}

    def act(self, snapshot: MarketSnapshot) -> AgentAction:
        cancels = [order_id for order_id in (self.bid_id, self.ask_id) if order_id is not None]

        # cancel previous resting orders
        self.bid_id = None
        self.ask_id = None

        has_bid = snapshot.best_bid is not None
        has_ask = snapshot.best_ask is not None

        mid = None
        if has_bid and has_ask:
            assert snapshot.best_bid
            assert snapshot.best_ask 
            mid = (snapshot.best_bid + snapshot.best_ask) / 2.0
        elif has_bid: # when there are no asks, only buys
            assert snapshot.best_bid
            mid = snapshot.best_bid + self.half_spread # fair value must be at least greater than best bid because then it would have been filled
        elif has_ask: # when there are no bids, only asks
            assert snapshot.best_ask
            mid = snapshot.best_ask - self.half_spread # fair value must be at least smaller than best ask
        else: # book is empty
            mid = self.fallback_mid

        # skew: positive position shifts mid down so we sell cheaper to unwind. vise versa for negative position
        skewed_mid = mid - self.skew_k * self.position
        bid_price = round(skewed_mid - self.half_spread, 2)
        ask_price = round(skewed_mid + self.half_spread, 2)
        # TODO: implement dynamic half market spreads instead of fixed value

        # guard against our own bid from meeting ask
        if bid_price >= ask_price:
            return AgentAction(cancels=cancels)

        # adjust qty based on position
        adj = self.qty_k * self.position
        bid_qty = max(1, round(self.qty - adj))
        ask_qty = max(1, round(self.qty + adj))

        return AgentAction(
            cancels=cancels,
            submits=[
                OrderRequest(OrderSide.BID, bid_price, bid_qty),
                OrderRequest(OrderSide.ASK, ask_price, ask_qty)
            ]
        )

    def on_order_accepted(self, order_id: str, request: OrderRequest, trades: list[Trade]) -> None:
        # update lifetime trades
        self._order_side[order_id] = request.side

        filled_qty = sum(trade.qty for trade in trades) # immediate fill
        resting_id = order_id if filled_qty < request.qty else None

        if request.side == OrderSide.BID:
            self.bid_id = resting_id
        else:
            self.ask_id = resting_id

    def on_fill(self, trade: Trade) -> None:
        side = self._order_side.get(trade.resting_id, None) or self._order_side.get(trade.aggressor_id, None)
        if side is None:
            return
        elif side == OrderSide.BID:
            self.position += trade.qty
            self.cash -= trade.price * trade.qty
        else:
            self.position -= trade.qty
            self.cash += trade.price * trade.qty
