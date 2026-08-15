import random
from typing import Optional

from agent import Agent, AgentAction
from order import MarketSnapshot, OrderRequest, OrderSide


class NoiseTrader(Agent):
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
        self.fair_value += self.ou_theta * (self.ou_mu - self.fair_value) + self.ou_sigma * self.rng.gauss(0, 1)

        cancels = []
        if self._resting_ids and self.rng.random() < self.cancel_prob:
            cancel_id = self.rng.choice(self._resting_ids)
            self._resting_ids.remove(cancel_id)
            cancels.append(cancel_id)

        submits = []
        best_bid, best_ask = snapshot.best_bid, snapshot.best_ask
        has_bid = best_bid is not None
        has_ask = best_ask is not None
        if has_bid and has_ask:
            assert best_bid
            assert best_ask
            mid = (best_bid + best_ask) / 2.0
        elif has_bid:
            assert best_bid
            mid = best_bid + 0.50
        elif has_ask:
            assert best_ask
            mid = best_ask - 0.50
        else:
            mid = self.fair_value

        side = self.rng.choice([OrderSide.BID, OrderSide.ASK])

        if self.rng.random() < self.market_order_prob:
            # expected value from this log normal dist = 7.38906
            qty = int(self.rng.lognormvariate(1.5, 1.0)) + 1
            order_request = OrderRequest(side, None, qty)
            return AgentAction(cancels=cancels, submits=[order_request])

        aggresive = self.rng.random() < self.aggressive_prob
        price = None
        if aggresive: # will cross spread
            if side == OrderSide.BID:
                ref = best_ask if has_ask else mid
                price = round(ref + self.rng.uniform(0.01, 0.25), 2) # E[x] = 0.13
            else:
                ref = best_bid if has_bid else mid
                price = round(ref - self.rng.uniform(0.01, 0.25), 2)
        else:
            # passive limit order. doesn't cross spread
            # anchor to fair value rather than just mid, to follow the OU process
            raw = self.fair_value + self.rng.uniform(-1.50, 1.50) # E[x] = 0
            price = round(round(raw / self.tick) * self.tick, 2) 

        qty = int(self.rng.lognormvariate(1.5, 1.0)) + 1 # E[x] = 7.38906

        return AgentAction(cancels=cancels, submits=[OrderRequest(side, price, qty)])
