from enum import Enum
from dataclasses import dataclass, field
import itertools
from typing import Optional

_seq = itertools.count(1)

class OrderSide(Enum):
    BID = 0
    ASK = 1
    
@dataclass
class Order:
    order_id: str
    side: OrderSide
    price: Optional[float] # None = market order
    qty: int
    symbol: str

    seq: int = field(default_factory=lambda: next(_seq))
    is_canceled: bool = False

    @classmethod
    def from_request(cls, request: OrderRequest, order_id: str) -> Order:
        return cls(
            order_id=order_id,
            side=request.side,
            price=request.price,
            qty=request.qty,
            symbol=request.symbol
        )


@dataclass
class OrderRequest:
    side: OrderSide
    price: Optional[float]
    qty: int
    symbol: str


@dataclass
class Trade:
    aggressor_id: str # trade coming in
    resting_id: str # trade sitting
    price: float
    qty: int
    symbol: str


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    bids: list[tuple[float, int]] # (price, qty), best to worst
    asks: list[tuple[float, int]] # (price, qty), best to worst
    