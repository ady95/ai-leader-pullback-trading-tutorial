"""16-1 증권사 어댑터: 주문 관리(oms.py)가 증권사를 몰라도 되게 하는 얇은 층.

Broker 프로토콜을 지키면 어떤 증권사든 바꿔 끼울 수 있다.
  KiwoomBroker  키움 REST (모의투자 서버). 주문 kt10000~kt10003, 조회 ka10075·ka10076·kt00018
  FakeBroker    테스트용 가짜 증권사. 체결·부분 체결·거부·응답 유실을 마음대로 만든다

모든 조회 결과는 같은 모양의 dict 로 바꿔 돌려준다 (증권사별 필드 이름은 이 파일 밖으로 나가지 않는다).
  open_orders(): {ord_no, orig_ord_no, code, side, qty, price, remaining}
  executions():  {ord_no, orig_ord_no, code, side, qty, price, filled, fill_price, remaining, status}
  holdings():    {code: 보유 수량}
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Protocol

from pullback.kiwoom import KiwoomClient, KiwoomError


@dataclass
class Ack:
    ok: bool
    ord_no: str = ""
    code: str = ""          # 증권사 응답 코드 (0 이면 정상)
    msg: str = ""
    ms: float = 0.0         # 응답까지 걸린 시간


class Broker(Protocol):
    def submit(self, code: str, side: str, qty: int, price: float, order_type: str) -> Ack: ...
    def amend(self, ord_no: str, code: str, qty: int, price: float) -> Ack: ...
    def cancel(self, ord_no: str, code: str, qty: int = 0) -> Ack: ...
    def open_orders(self) -> list[dict]: ...
    def executions(self) -> list[dict]: ...
    def holdings(self) -> dict[str, int]: ...


def _num(s) -> float:
    """키움 숫자 문자열('+12,345', '-000123', '')을 숫자로."""
    s = str(s or "0").replace(",", "").replace("+", "").replace("-", "").strip()
    return float(s) if s else 0.0


def _code(s: str) -> str:
    s = str(s).strip()
    return s[1:] if s[:1] == "A" else s


def _side(s: str) -> str:
    return "buy" if "매수" in str(s) else "sell"


class KiwoomBroker:
    """키움 REST 주문·조회. 계좌는 앱 키에 묶여 있어 요청에 계좌번호를 넣지 않는다."""
    ORDER = "/api/dostk/ordr"
    ACNT = "/api/dostk/acnt"
    TRDE_TP = {"limit": "0", "market": "3"}     # 0 보통(지정가), 3 시장가

    def __init__(self, kw: KiwoomClient | None = None):
        self.kw = kw or KiwoomClient()

    def _order(self, api_id: str, body: dict) -> Ack:
        t = time.perf_counter()
        try:
            j, _ = self.kw.request(api_id, self.ORDER, body, max_retry=0)
        except KiwoomError as e:
            return Ack(False, "", str(e.code), e.msg, (time.perf_counter() - t) * 1000)
        return Ack(True, str(j.get("ord_no", "")), "0", j.get("return_msg", ""), (time.perf_counter() - t) * 1000)

    def submit(self, code, side, qty, price, order_type="limit") -> Ack:
        body = {"dmst_stex_tp": "KRX", "stk_cd": code, "ord_qty": str(int(qty)),
                "ord_uv": "" if order_type == "market" else str(int(price)),
                "trde_tp": self.TRDE_TP[order_type], "cond_uv": ""}
        return self._order("kt10000" if side == "buy" else "kt10001", body)

    def amend(self, ord_no, code, qty, price) -> Ack:
        body = {"dmst_stex_tp": "KRX", "orig_ord_no": ord_no, "stk_cd": code,
                "mdfy_qty": str(int(qty)), "mdfy_uv": str(int(price)), "mdfy_cond_uv": ""}
        return self._order("kt10002", body)

    def cancel(self, ord_no, code, qty=0) -> Ack:
        body = {"dmst_stex_tp": "KRX", "orig_ord_no": ord_no, "stk_cd": code, "cncl_qty": str(int(qty))}
        return self._order("kt10003", body)       # cncl_qty 0 = 잔량 전부

    def _rows(self, api_id, body, key):
        out = []
        for rows in self.kw.paginate(api_id, self.ACNT, body, key):
            out += rows
        return out

    def open_orders(self) -> list[dict]:
        rows = self._rows("ka10075", {"all_stk_tp": "0", "trde_tp": "0", "stk_cd": "", "stex_tp": "0"}, "oso")
        return [{"ord_no": r["ord_no"], "orig_ord_no": r.get("orig_ord_no", ""), "code": _code(r["stk_cd"]),
                 "side": _side(r.get("io_tp_nm", "")), "qty": int(_num(r.get("ord_qty"))),
                 "price": _num(r.get("ord_pric")), "remaining": int(_num(r.get("oso_qty")))} for r in rows]

    def executions(self) -> list[dict]:
        rows = self._rows("ka10076", {"stk_cd": "", "qry_tp": "0", "sell_tp": "0", "ord_no": "", "stex_tp": "0"},
                          "cntr")
        return [{"ord_no": r["ord_no"], "orig_ord_no": r.get("orig_ord_no", ""), "code": _code(r["stk_cd"]),
                 "side": _side(r.get("io_tp_nm", "")), "qty": int(_num(r.get("ord_qty"))),
                 "price": _num(r.get("ord_pric")), "filled": int(_num(r.get("cntr_qty"))),
                 "fill_price": _num(r.get("cntr_pric")), "remaining": int(_num(r.get("oso_qty"))),
                 "status": r.get("ord_stt", "")} for r in rows]

    def holdings(self) -> dict[str, int]:
        rows = self._rows("kt00018", {"qry_tp": "1", "dmst_stex_tp": "KRX"}, "acnt_evlt_remn_indv_tot")
        return {_code(r["stk_cd"]): int(_num(r["rmnd_qty"])) for r in rows if int(_num(r["rmnd_qty"])) > 0}

    def quote(self, code: str) -> dict:
        """ka10001 주식기본정보: 현재가·상한가·하한가 (시험 주문 가격을 정할 때만 쓴다)."""
        j, _ = self.kw.request("ka10001", "/api/dostk/stkinfo", {"stk_cd": code})
        return {"price": _num(j.get("cur_prc")), "upper": _num(j.get("upl_pric")),
                "lower": _num(j.get("lst_pric")), "base": _num(j.get("base_pric"))}


def ws_event(values: dict) -> dict:
    """키움 WebSocket 주문체결(00) 값을 executions() 와 같은 모양으로. 계좌번호(9201)는 버린다."""
    return {"ord_no": str(values.get("9203", "")).strip(), "orig_ord_no": str(values.get("904", "")).strip(),
            "code": _code(values.get("9001", "")), "side": _side(values.get("905", "")),
            "order_kind": str(values.get("905", "")).strip(), "status": str(values.get("913", "")).strip(),
            "qty": int(_num(values.get("900"))), "price": _num(values.get("901")),
            "remaining": int(_num(values.get("902"))), "fill_qty": int(_num(values.get("911"))),
            "fill_price": _num(values.get("910")), "time": str(values.get("908", "")).strip()}


# ---------------------------------------------------------------- 테스트용 가짜 증권사
@dataclass
class _FO:
    ord_no: str
    code: str
    side: str
    qty: int
    price: float
    order_type: str
    orig_ord_no: str = ""
    filled: int = 0
    value: float = 0.0
    live: bool = True


@dataclass
class FakeBroker:
    """메모리 안의 증권사. events 에 WebSocket 주문체결(00) 통보를 흉내 낸 dict 가 쌓인다."""
    pos: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    lose_next_ack: bool = False         # True 면 다음 주문은 접수되지만 응답이 사라진다 (시간 초과)
    reject_codes: set = field(default_factory=set)
    calls: list = field(default_factory=list)

    def __post_init__(self):
        self._no = itertools.count(1)
        self.orders: dict[str, _FO] = {}

    def _ev(self, o: _FO, status, fill_qty=0, fill_price=0.0, kind=""):
        self.events.append({"ord_no": o.ord_no, "orig_ord_no": o.orig_ord_no, "code": o.code, "side": o.side,
                            "order_kind": kind or ("+매수" if o.side == "buy" else "-매도"), "status": status,
                            "qty": o.qty, "price": o.price, "remaining": o.qty - o.filled if o.live else 0,
                            "fill_qty": fill_qty, "fill_price": fill_price, "time": ""})

    def submit(self, code, side, qty, price, order_type="limit") -> Ack:
        self.calls.append(("submit", code, side, qty, price))
        if code in self.reject_codes or qty <= 0:
            return Ack(False, "", "9", "주문 거부")
        o = _FO(f"{next(self._no):07d}", code, side, int(qty), float(price), order_type)
        self.orders[o.ord_no] = o
        self._ev(o, "접수")
        if self.lose_next_ack:
            self.lose_next_ack = False
            raise TimeoutError("응답 없음 (주문은 접수됨)")
        return Ack(True, o.ord_no, "0")

    def amend(self, ord_no, code, qty, price) -> Ack:
        self.calls.append(("amend", ord_no, qty, price))
        o = self.orders.get(ord_no)
        if not o or not o.live:
            return Ack(False, "", "9", "정정 가능 수량 없음")
        rest = o.qty - o.filled
        o.live = False
        n = _FO(f"{next(self._no):07d}", code, o.side, min(int(qty), rest), float(price), o.order_type, ord_no)
        self.orders[n.ord_no] = n
        self._ev(n, "확인", kind="정정")
        return Ack(True, n.ord_no, "0")

    def cancel(self, ord_no, code, qty=0) -> Ack:
        self.calls.append(("cancel", ord_no))
        o = self.orders.get(ord_no)
        if not o or not o.live:
            return Ack(False, "", "9", "취소 가능 수량 없음")
        o.live = False
        self._ev(o, "확인", kind="취소")
        return Ack(True, f"{next(self._no):07d}", "0")

    def fill(self, ord_no: str, qty: int, price: float) -> None:
        """시험에서 체결을 만든다."""
        o = self.orders[ord_no]
        qty = min(qty, o.qty - o.filled)
        o.filled += qty
        o.value += qty * price
        sign = 1 if o.side == "buy" else -1
        self.pos[o.code] = self.pos.get(o.code, 0) + sign * qty
        if o.filled == o.qty:
            o.live = False
        self._ev(o, "체결", qty, price)

    def open_orders(self) -> list[dict]:
        return [{"ord_no": o.ord_no, "orig_ord_no": o.orig_ord_no, "code": o.code, "side": o.side, "qty": o.qty,
                 "price": o.price, "remaining": o.qty - o.filled} for o in self.orders.values() if o.live]

    def executions(self) -> list[dict]:
        return [{"ord_no": o.ord_no, "orig_ord_no": o.orig_ord_no, "code": o.code, "side": o.side, "qty": o.qty,
                 "price": o.price, "filled": o.filled, "fill_price": o.value / o.filled if o.filled else 0.0,
                 "remaining": o.qty - o.filled if o.live else 0, "status": "체결" if o.filled else "접수"}
                for o in self.orders.values()]

    def holdings(self) -> dict[str, int]:
        return {c: q for c, q in self.pos.items() if q}
