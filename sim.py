from __future__ import annotations

from agent import Agent
from exchange import Exchange


class Simulation:
    def __init__(self, exchange: Exchange, agents: list[Agent]) -> None:
        self.exchange = exchange
        self.agents = agents

    def step(self) -> None:
        for agent in self.agents:
            snapshot = self.exchange.snapshot()
            action = agent.act(snapshot)

            for order_id in action.cancels:
                self.exchange.cancel(order_id)

            for request in action.submits:
                order_id, trades = self.exchange.submit(request)
                agent.on_order_accepted(order_id, request, trades)
                for trade in trades:
                    for other in self.agents:
                        other.on_fill(trade)

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()


if __name__ == "__main__":
    import pandas as pd

    from market_maker import MarketMaker
    from noise_trader import NoiseTrader

    exchange = Exchange()
    mm = MarketMaker()
    noise = NoiseTrader(seed=111)
    sim = Simulation(exchange, [mm, noise])

    SNAPSHOT_EVERY = 1_000
    STEPS = 10_000

    rows = []
    for i in range(1, STEPS + 1):
        sim.step()
        if i % SNAPSHOT_EVERY == 0:
            snap = exchange.snapshot()
            mid = (snap.best_bid + snap.best_ask) / 2.0 if snap.best_bid and snap.best_ask else noise.fair_value
            pnl = mm.cash + mm.position * mid
            rows.append({
                "step": i,
                "position": mm.position,
                "cash": round(mm.cash, 2),
                "pnl": round(pnl, 2),
                "mid": round(mid, 2),
                "fair_value": round(noise.fair_value, 2),
            })

    df = pd.DataFrame(rows).set_index("step")
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(df)
