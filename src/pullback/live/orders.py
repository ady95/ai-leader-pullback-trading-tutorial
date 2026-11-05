"""15-1 주문 의도(Order)와 체결(Fill).

운영 엔진은 증권사 주문을 직접 만들지 않는다. "무엇을 얼마에 몇 단위"라는 주문 의도만 만들고,
실제 주식 수·호가 단위·잔고 확인은 위험관리 모듈(risk.py)이, 전송은 증권사 어댑터(16장)가 맡는다.

kind
  buy1          1차 매수 지정가
  buy2          2차 매수 지정가
  tp1           1차 익절 지정가
  tp2_limit_up  상한가 잔여 매도 (전일 종가 대비 +29.5% 이상에 닿으면)
  exit_open     시가 청산 (reason: stop | time)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Order:
    code: str
    kind: str
    price: float          # 지정가. exit_open 은 nan, tp2_limit_up 은 기준 전일 종가
    units: float
    decided_on: str       # 이 주문을 정한 장 마감일 (YYYY-MM-DD)
    reason: str = ""
    wave: str = ""        # buy1: 파동 고점일 (같은 파동 재진입 금지)

    @property
    def side(self) -> str:
        return "buy" if self.kind.startswith("buy") else "sell"

    @property
    def key(self) -> str:
        """중복 주문 방지 키 (16-2). 같은 날 같은 종목의 같은 종류 주문은 하나뿐이다."""
        return f"{self.decided_on}:{self.code}:{self.kind}"

    def as_dict(self) -> dict:
        return asdict(self) | {"key": self.key}


@dataclass
class Fill:
    code: str
    date: str
    kind: str             # buy1 | buy2 | sell | defer
    price: float
    units: float
    reason: str = ""      # sell: tp1 | tp2_limit_up | stop | time
    order_price: float = float("nan")   # 체결된 주문의 지정가 (기록용)

    def as_dict(self) -> dict:
        return asdict(self)
