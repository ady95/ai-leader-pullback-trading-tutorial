"""15-1 위험관리·주문 변환: 엔진의 주문 의도(단위)를 증권사에 낼 수 있는 주문(주식 수·호가)으로 바꾼다.

B0에는 채택된 위험관리 구성요소가 없다(12~14장). 그래서 이 모듈은 전략을 바꾸지 않고,
'낼 수 없는 주문'을 막는 일만 한다.
  - 호가 단위: 매수 지정가는 내림, 매도 지정가는 올림 (백테스트는 수정주가라 반올림하지 않았다)
  - 상한가 잔여 매도: 전일 종가 기준 상한가에 지정가 매도
  - 주식 수: 1차 = floor(1차 금액 / 지정가), 2차 = 1차 체결 수량 x second_units,
             1차 익절 = floor(보유 / 2) (보유가 1주면 1주)
  - 현금: 매수 주문 합계가 주문 가능 현금을 넘으면 넘는 주문을 거절
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# KRX 유가증권·코스닥 공통 호가 단위 (2023-01-25 이후)
TICKS = [(2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500)]


def tick(price: float) -> int:
    for bound, t in TICKS:
        if price < bound:
            return t
    return 1_000


def floor_tick(price: float) -> float:
    t = tick(price)
    return math.floor(price / t + 1e-9) * t


def ceil_tick(price: float) -> float:
    t = tick(price)
    v = math.ceil(price / t - 1e-9) * t
    return v if tick(v) == t else ceil_tick(v)   # 경계를 넘으면 위 단위로 다시 맞춘다


def upper_limit(prev_close: float) -> float:
    """상한가 근사: 전일 종가 x 1.3 을 호가 단위로 내림."""
    return floor_tick(prev_close * 1.3)


@dataclass
class BrokerOrder:
    code: str
    side: str
    kind: str
    order_type: str      # limit | market
    price: float         # 지정가(원). 시장가는 0
    qty: int
    key: str


def convert(order, prev_close: float, held_qty: int = 0, first_qty: int = 0,
            first_money: float = 1e6) -> tuple[BrokerOrder | None, str]:
    """주문 의도 하나를 증권사 주문으로 바꾼다. (주문, 거절 사유)."""
    k = order.kind
    if k == "buy1":
        px = floor_tick(order.price)
        qty = int(first_money // px) if px > 0 else 0
        typ = "limit"
    elif k == "buy2":
        px = floor_tick(order.price)
        qty = int(round(first_qty * order.units))
        typ = "limit"
    elif k == "tp1":
        px = ceil_tick(order.price)
        qty = max(1, held_qty // 2) if held_qty > 0 else 0
        typ = "limit"
    elif k == "tp2_limit_up":
        px = upper_limit(prev_close)
        qty = held_qty
        typ = "limit"
    elif k == "exit_open":
        px, qty, typ = 0.0, held_qty, "market"
    else:
        return None, f"알 수 없는 주문 종류 {k}"
    if qty <= 0:
        return None, "주식 수 0"
    if typ == "limit" and order.side == "buy" and px < prev_close * 0.7:
        return None, "하한가 아래 지정가 (체결 불가)"
    return BrokerOrder(order.code, order.side, k, typ, px, qty, order.key), ""


def cash_check(orders: list[BrokerOrder], cash: float) -> tuple[list[BrokerOrder], list[BrokerOrder]]:
    """매수 주문을 순서대로 담다가 현금을 넘으면 그 뒤는 거절한다. (통과, 거절)."""
    ok, rejected, need = [], [], 0.0
    for o in orders:
        if o.side == "buy":
            if need + o.price * o.qty > cash:
                rejected.append(o)
                continue
            need += o.price * o.qty
        ok.append(o)
    return ok, rejected
