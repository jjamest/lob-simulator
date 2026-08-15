from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from order import MarketSnapshot, OrderRequest, Trade


@dataclass
class AgentAction:
    cancels: list[str] = field(default_factory=list)
    submits: list[OrderRequest] = field(default_factory=list)


class Agent(ABC):
    @abstractmethod
    def act(self, snapshot: MarketSnapshot) -> AgentAction:
        """Decide what to cancel and what to submit this step."""

    def on_order_accepted(self, order_id: str, request: OrderRequest, trades: list[Trade]) -> None:
        """Called once per submitted order with the id the exchange assigned it,
        and any trades that resulted immediately."""

    def on_fill(self, trade: Trade) -> None:
        """Called for every trade in the book; agents ignore trades that aren't theirs."""
