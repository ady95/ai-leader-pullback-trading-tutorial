"""15-1 운영 엔진: B0 규칙을 '하루씩, 상태 파일과 함께' 실행한다.

백테스트(backtest.simulate_stock)는 한 종목의 전체 일봉을 한 번에 훑는다. 운영에서는 내일의 일봉이
없으므로 같은 규칙을 둘로 나눈다.
  end_of_day(): 장 마감 뒤. 손절·시간 청산 판정, 다음 거래일 주문 의도 생성
  on_fill():    장중 체결 통보. 상태 갱신, 필요하면 같은 날 주문을 고친다
                (2차 매수 체결 -> 1차 익절가 재계산, 1차 익절 체결 -> 상한가 잔여 매도 주문)
상태는 JSON 파일로 저장해 프로그램을 다시 켜도 이어진다. 지원하는 규칙은 B0 기본값뿐이다
(07~14장의 대안 옵션은 채택되지 않았으므로 운영 코드에 넣지 않는다).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pullback.backtest import Rules
from pullback.live.orders import Fill, Order


def _plain(o):
    """numpy 숫자를 JSON 에 쓸 수 있는 파이썬 숫자로."""
    return o.item() if hasattr(o, "item") else str(o)


@dataclass
class Position:
    code: str
    entry_date: str
    wave: str
    buys: list = field(default_factory=list)      # [가격, 단위, 날짜]
    sells: list = field(default_factory=list)
    units: float = 0.0
    second_price: float = 0.0
    second_done: bool = False
    tp1_done: bool = False
    exit_next_open: str = ""
    days_held: int = 0                             # 진입일 이후 지난 일봉 수 (거래정지일 포함)

    def avg(self) -> float:
        return sum(p * u for p, u, _ in self.buys) / sum(u for _, u, _ in self.buys)


@dataclass
class State:
    last_date: str = ""
    positions: dict = field(default_factory=dict)  # code -> Position
    used_waves: dict = field(default_factory=dict) # code -> [파동 고점일]

    def save(self, path: Path) -> None:
        d = {"last_date": self.last_date, "used_waves": self.used_waves,
             "positions": {c: asdict(p) for c, p in self.positions.items()}}
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1, default=_plain), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "State":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["last_date"], {c: Position(**p) for c, p in d["positions"].items()}, d["used_waves"])


class LiveEngine:
    def __init__(self, rules: Rules, state: State | None = None):
        self.r = rules
        self.s = state or State()
        self.book: dict[str, list[Order]] = {}     # 오늘 살아 있는 주문 (종목별)
        self.carried: list[Order] = []

    # ---------- 장중 ----------
    def on_fill(self, f: Fill) -> list[Order]:
        """체결 통보를 반영하고, 같은 날 새로 내거나 고칠 주문을 돌려준다."""
        r, s = self.r, self.s
        out: list[Order] = []
        if f.kind == "defer":
            return out
        if f.kind == "buy1":
            o = next(o for o in self.book.get(f.code, []) if o.kind == "buy1")
            s.used_waves.setdefault(f.code, []).append(o.wave)
            s.positions[f.code] = Position(code=f.code, entry_date=f.date, wave=o.wave,
                                           buys=[[f.price, f.units, f.date]], units=f.units,
                                           second_price=f.price * (1 - r.second_drop))
            return out                              # 1차 체결일에는 팔지 않고, 2차 매수도 다음 날부터
        p = s.positions[f.code]
        if f.kind == "buy2":
            p.buys.append([f.price, f.units, f.date])
            p.units += f.units
            p.second_done = True
            if not p.tp1_done:                      # 평균 단가가 바뀌었으므로 1차 익절 주문을 고친다
                out.append(self._tp1(p, f.date))
        elif f.kind == "sell":
            p.sells.append([f.price, f.units, f.date])
            p.units -= f.units
            if f.reason == "tp1":
                p.tp1_done = True
                if p.units > 1e-12:
                    out.append(Order(f.code, "tp2_limit_up", math.nan, p.units, f.date))
            if p.units <= 1e-12:
                del s.positions[f.code]
        self._replace(f.code, out)
        return out

    def _tp1(self, p: Position, day: str) -> Order:
        return Order(p.code, "tp1", p.avg() * (1 + self.r.take_profit_1),
                     p.units * self.r.take_profit_1_fraction, day)

    def _replace(self, code: str, new: list[Order]) -> None:
        kinds = {o.kind for o in new}
        self.book[code] = [o for o in self.book.get(code, []) if o.kind not in kinds] + new

    def save(self, path: Path) -> None:
        """상태와 '내일 낼 주문'을 함께 저장한다. 밤에 프로그램을 꺼도 다음 날 아침 이어서 쓴다."""
        self.s.save(path)
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["book"] = [o.as_dict() for os in self.book.values() for o in os]
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1, default=_plain), encoding="utf-8")

    @classmethod
    def load(cls, rules: Rules, path: Path) -> "LiveEngine":
        eng = cls(rules, State.load(path))
        for o in json.loads(Path(path).read_text(encoding="utf-8")).get("book", []):
            o.pop("key", None)
            eng.book.setdefault(o["code"], []).append(Order(**o))
        return eng

    def cancel(self, code: str, kind: str) -> None:
        self.book[code] = [o for o in self.book.get(code, []) if o.kind != kind]

    # ---------- 장 마감 ----------
    def end_of_day(self, day: str, bars: dict, setups: list[tuple[str, float, str]]) -> list[Order]:
        """bars: 보유 종목과 주문 중인 종목 가운데 오늘 일봉이 있는 것 {code: {"close", "halted"}}.
        setups: 오늘 B0 후보 [(code, 1차 매수선, 파동 고점일)]. 다음 거래일 주문 의도를 돌려준다.
        오늘 일봉이 없는 종목(데이터 누락·폐지)은 판단하지 않고 기존 주문을 그대로 넘긴다."""
        r, s = self.r, self.s
        orders: list[Order] = []
        self.carried = [o for c, os in self.book.items() if c not in bars for o in os]
        orders += self.carried
        for code, p in s.positions.items():
            b = bars.get(code)
            if b is None:
                continue
            if p.entry_date != day:
                p.days_held += 1
            if not b["halted"]:
                if b["close"] < p.second_price * (1 - r.stop_zone_buffer):
                    p.exit_next_open = "stop"
                elif p.days_held >= r.max_hold_days:
                    p.exit_next_open = "time"
            if p.exit_next_open:
                orders.append(Order(code, "exit_open", math.nan, p.units, day, p.exit_next_open))
                continue
            if not p.second_done and r.second_units > 0:
                orders.append(Order(code, "buy2", p.second_price, r.second_units, day))
            if not p.tp1_done:
                orders.append(self._tp1(p, day))
            elif p.units > 0:
                orders.append(Order(code, "tp2_limit_up", math.nan, p.units, day))
        for code, line, wave in setups:
            if (code in s.positions or wave in s.used_waves.get(code, []) or not line == line
                    or any(o.code == code for o in self.carried)):
                continue
            orders.append(Order(code, "buy1", line, 1.0, day, wave=wave))
        s.last_date = day
        self.book = {}
        for o in orders:
            self.book.setdefault(o.code, []).append(o)
        return orders
