import pandas as pd

from agent import Agent
from exchange import Exchange
from market_maker import MarketMaker
from noise_trader import NoiseTrader


class Simulation:
    def __init__(self, exchange: Exchange, agents: list[Agent]) -> None:
        self.exchange = exchange
        self.agents = agents

    def step(self) -> None:
        for agent in self.agents:
            snapshot = self.exchange.snapshot()
            action = agent.act(snapshot)

            # perform all specified cancels
            for order_id in action.cancels:
                self.exchange.cancel(order_id)

            # do submits
            for request in action.submits:
                order_id, trades = self.exchange.submit((request))
                agent.on_order_accepted(order_id, request, trades)
                for trade in trades:
                    for other in self.agents:
                        other.on_fill(trade) # includes all, so each needs to have their own checks to see if a trade is their own

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()


exchange = Exchange()
mm = MarketMaker(qty=100, skew_k=0.1)
mm2 = MarketMaker(qty=1000)
noise = NoiseTrader(seed=11)
sim = Simulation(exchange, [mm2, mm, noise])

SNAPSHOT_EVERY = 100
STEPS = 1_000

mm_rows = []
mm2_rows = []
for i in range(1, STEPS + 1):
    sim.step()
    if i % SNAPSHOT_EVERY == 0:
        print("On", i)
        snap = exchange.snapshot()
        mid = (snap.best_bid + snap.best_ask) / 2.0 if snap.best_bid and snap.best_ask else noise.fair_value
        pnl = mm.cash + mm.position * mid

        pnl2 = mm2.cash + mm2.position * mid

        mm_rows.append({
            "step": i,
            "position": mm.position,
            "cash": round(mm.cash, 2),
            "pnl": round(pnl, 2),
            "mid": round(mid, 2),
            "fair_value": round(noise.fair_value, 2),
        })

        mm2_rows.append({
                    "step": i,
                    "position": mm2.position,
                    "cash": round(mm2.cash, 2),
                    "pnl": round(pnl, 2),
                    "mid": round(mid, 2),
                    "fair_value": round(noise.fair_value, 2),
                })

df = pd.DataFrame(mm_rows).set_index("step")
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
print(df)

# df2 = pd.DataFrame(mm2_rows).set_index("step")
# pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
# print(df2)