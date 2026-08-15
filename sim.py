from collections import defaultdict

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
            for symbol in agent.get_symbols():
                snapshot = self.exchange.snapshot(symbol)
                action = agent.act(snapshot)

                # perform all specified cancels
                for order_id in action.cancels:
                    self.exchange.cancel(order_id, symbol)

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
exchange.register_symbol("apple")
exchange.register_symbol("banana")
mm = MarketMaker(symbols=["apple", "banana"])
noise = NoiseTrader(seed=10, symbols=["apple", "banana"])
sim = Simulation(exchange, [mm, noise])

SNAPSHOT_EVERY = 100
STEPS = 1_000

symbols_to_snapshot = ["apple", "banana"]
results = {symbol: [] for symbol in symbols_to_snapshot}
for i in range(1, STEPS + 1):
    sim.step()
    if i % SNAPSHOT_EVERY == 0:
        for symbol in symbols_to_snapshot:
            snap = exchange.snapshot(symbol)
            mid = (snap.best_bid + snap.best_ask) / 2.0 if snap.best_bid and snap.best_ask else noise.fair_value
            pnl = mm.cash[symbol] + mm.positions[symbol] * mid

            results[symbol].append({
                "step": i,
                "position": mm.positions[symbol],
                "cash": round(mm.cash[symbol], 2),
                "pnl": round(pnl, 2),
                "mid": round(mid, 2),
                "fair_value": round(noise.fair_value, 2),
            })

for symbol in symbols_to_snapshot:
    print("\n")
    print(symbol.upper())
    df = pd.DataFrame(results[symbol]).set_index("step")
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(df)
