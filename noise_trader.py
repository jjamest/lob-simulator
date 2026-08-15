from __future__ import annotations
import random
from typing import Optional

from agent import Agent, AgentAction
from order import MarketSnapshot, OrderRequest, OrderSide, Trade


class NoiseTrader(Agent):
    """Random flow anchored to an internal OU fair-value process."""

    def __init__(
        self,
        seed: Optional[int] = None,
        fair_value: float = 100.0,
        ou_theta: float = 0.005,
        ou_sigma: float = 0.05,
        tick: float = 0.01,
        cancel_prob: float = 0.20,
        market_order_prob: float = 0.10,
        aggressive_prob: float = 0.30,
        max_resting: int = 200,
    ) -> None:
        self.rng = random.Random(seed)
        self.fair_value = fair_value
        self.ou_theta = ou_theta
        self.ou_mu = fair_value
        self.ou_sigma = ou_sigma
        self.tick = tick
        self.cancel_prob = cancel_prob
        self.market_order_prob = market_order_prob
        self.aggressive_prob = aggressive_prob
        self.max_resting = max_resting

        self._resting_ids: list[str] = []

    def act(self, snapshot: MarketSnapshot) -> AgentAction:
        self.fair_value += (
            self.ou_theta * (self.ou_mu - self.fair_value) + self.ou_sigma * self.rng.gauss(0, 1)
        )

        cancels = []
        if self._resting_ids and self.rng.random() < self.cancel_prob:
            cancel_id = self.rng.choice(self._resting_ids)
            self._resting_ids.remove(cancel_id)
            cancels.append(cancel_id)

        return AgentAction(cancels=cancels, submits=[self._random_request(snapshot)])

    def on_order_accepted(self, order_id: str, request: OrderRequest, trades: list[Trade]) -> None:
        # only track it if it actually rested (untouched limit order)
        if not trades and request.price is not None:
            self._resting_ids.append(order_id)
            if len(self._resting_ids) > self.max_resting:
                self._resting_ids = self._resting_ids[-self.max_resting:]

    def _random_request(self, snapshot: MarketSnapshot) -> OrderRequest:
        bb, ba = snapshot.best_bid, snapshot.best_ask
        if bb is not None and ba is not None:
            mid = (bb + ba) / 2.0
        elif bb is not None:
            mid = bb + 0.50
        elif ba is not None:
            mid = ba - 0.50
        else:
            mid = self.fair_value

        side = self.rng.choice([OrderSide.BID, OrderSide.ASK])

        if self.rng.random() < self.market_order_prob:
            qty = int(self.rng.lognormvariate(1.5, 1.0)) + 1
            return OrderRequest(side, None, qty)

        aggressive = self.rng.random() < self.aggressive_prob
        if aggressive:
            if side == OrderSide.BID:
                ref = ba if ba is not None else mid
                price = round(ref + self.rng.uniform(0.01, 0.25), 2)
            else:
                ref = bb if bb is not None else mid
                price = round(ref - self.rng.uniform(0.01, 0.25), 2)
        else:
            # anchor to fair value rather than just mid, to follow the OU process
            raw = self.fair_value + self.rng.uniform(-1.50, 1.50)
            price = round(round(raw / self.tick) * self.tick, 2)

        qty = int(self.rng.lognormvariate(1.5, 1.0)) + 1
        return OrderRequest(side, price, qty)
