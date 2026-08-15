from typing import Optional

from agent import Agent, AgentAction
from order import MarketSnapshot, OrderRequest, OrderSide, Trade


class MarketMaker(Agent):
    def __init__(
            self, 
            symbols: list[str],
            half_spread: float = 0.1, 
            qty: int = 10,
            skew_k: float = 0.01,
            qty_k: float = 0.10,
            fallback_mid: float = 100.0,
            max_position: int = 20
            ) -> None:
        self.symbols = symbols
        self.half_spread = half_spread
        self.qty = qty
        self.skew_k = skew_k # sensitivity controlling how much position skews midpoint price
        self.qty_k = qty_k # sensitivity controlling how much position skews quote sizes
        self.fallback_mid = fallback_mid

        # maximum position that could be held by this agent. This mirrors real life risk asw as competition limits
        self.max_position = max_position # will be symmetric to both +position and -position

        self.positions: dict[str, int] = {symbol: 0 for symbol in self.symbols}
        self.cash: dict[str, float] = {symbol: 0.0 for symbol in self.symbols}

        # we only hold a pair of bid and ask for each symbol
        self.bid_ids: dict[str, Optional[str]] = {sym: None for sym in symbols}
        self.ask_ids: dict[str, Optional[str]] = {sym: None for sym in symbols}

        # for every order submitted
        self._order_side: dict[str, OrderSide] = {}

    def act(self, snapshot: MarketSnapshot) -> AgentAction:
        cancels = [order_id for order_id in (self.bid_ids[snapshot.symbol], self.ask_ids[snapshot.symbol]) if order_id is not None]

        # cancel previous resting orders
        self.bid_ids[snapshot.symbol] = None
        self.ask_ids[snapshot.symbol] = None

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
        skewed_mid = mid - self.skew_k * self.positions[snapshot.symbol]
        bid_price = round(skewed_mid - self.half_spread, 2)
        ask_price = round(skewed_mid + self.half_spread, 2)
        # TODO: implement dynamic half market spreads instead of fixed value

        # guard against our own bid from meeting ask
        if bid_price >= ask_price:
            return AgentAction(cancels=cancels)

        # adjust qty based on position
        adj = self.qty_k * self.positions[snapshot.symbol]
        bid_qty = max(1, round(self.qty - adj))
        ask_qty = max(1, round(self.qty + adj))

        # headroom relative to max_position
        # cancels happen before submits in sim, so no double counting from resting orders
        room_to_buy = self.max_position - self.positions[snapshot.symbol]
        room_to_sell = self.max_position + self.positions[snapshot.symbol]

        # clamp qty for headroom. no problem if either is 0
        bid_qty = min(bid_qty, room_to_buy)
        ask_qty = min(ask_qty, room_to_sell)

        return AgentAction(
            cancels=cancels,
            submits=[
                OrderRequest(OrderSide.BID, bid_price, bid_qty, snapshot.symbol),
                OrderRequest(OrderSide.ASK, ask_price, ask_qty, snapshot.symbol)
            ]
        )

    def get_symbols(self) -> list[str]:
        return self.symbols

    def on_order_accepted(self, order_id: str, request: OrderRequest, trades: list[Trade]) -> None:
        # update lifetime trades
        self._order_side[order_id] = request.side

        filled_qty = sum(trade.qty for trade in trades) # immediate fill
        resting_id = order_id if filled_qty < request.qty else None

        if request.side == OrderSide.BID:
            self.bid_ids[request.symbol] = resting_id
        else:
            self.ask_ids[request.symbol] = resting_id

    def on_fill(self, trade: Trade) -> None:
        side = self._order_side.get(trade.resting_id, None) or self._order_side.get(trade.aggressor_id, None)
        if side is None:
            return
        elif side == OrderSide.BID:
            self.positions[trade.symbol] += trade.qty
            self.cash[trade.symbol] -= trade.price * trade.qty
        else:
            self.positions[trade.symbol] -= trade.qty
            self.cash[trade.symbol] += trade.price * trade.qty
