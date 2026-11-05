"""16-1~16-3 주문 관리(OMS): 주문 상태 머신, 주문 일지, 중복 방지, 복구, 긴급 중지.

상태 전이 (이 표 밖의 전이는 예외)
  NEW      -> SENT                      일지에 쓰고 보내기 직전 (긴급 중지·차단 종목이면 장부에 넣지 않고 막는다)
  SENT     -> ACCEPTED, REJECTED, UNKNOWN, PARTIAL, FILLED
                                        응답 / 거부 / 응답 없음 / 응답보다 체결 통보가 먼저 온 경우
  UNKNOWN  -> ACCEPTED, PARTIAL, FILLED, CANCELLED, REJECTED
                                        복구 조회로 확인 (증권사에 없으면 REJECTED)
  ACCEPTED -> ACCEPTED(정정), PARTIAL, FILLED, CANCELLED
  PARTIAL  -> PARTIAL, FILLED, CANCELLED
  FILLED, CANCELLED, REJECTED           끝

주문 일지(journal)는 15-3의 해시 체인 감사 기록을 그대로 쓴다. 상태가 바뀔 때마다 주문 전체를 한 줄로
남기므로, 프로그램이 죽어도 일지의 마지막 줄들로 주문 장부를 다시 만든다. 일지는 증권사에 보내기
'전에' 쓴다. 그래서 "보냈는데 기록이 없는" 주문은 생기지 않고, "기록은 있는데 응답이 없는" 주문(SENT,
UNKNOWN)만 생긴다. 그런 주문은 recover() 가 증권사 조회로 다시 찾는다.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pullback.live.audit import AuditLog
from pullback.live.broker import Broker

TERMINAL = {"FILLED", "CANCELLED", "REJECTED"}
TRANSITIONS = {
    "NEW": {"SENT"},
    "SENT": {"ACCEPTED", "REJECTED", "UNKNOWN", "PARTIAL", "FILLED"},
    "UNKNOWN": {"ACCEPTED", "PARTIAL", "FILLED", "CANCELLED", "REJECTED"},
    "ACCEPTED": {"ACCEPTED", "PARTIAL", "FILLED", "CANCELLED"},
    "PARTIAL": {"PARTIAL", "FILLED", "CANCELLED"},
    "FILLED": set(), "CANCELLED": set(), "REJECTED": set(),
}


class IllegalTransition(RuntimeError):
    pass


class Blocked(RuntimeError):
    """긴급 중지·차단 종목·모호한 복구로 주문을 내지 않았다."""


@dataclass
class ManagedOrder:
    key: str                     # 중복 방지 키 (15-1 Order.key)
    code: str
    side: str
    qty: int
    price: float
    order_type: str = "limit"
    kind: str = ""
    state: str = "NEW"
    ord_no: str = ""             # 현재 살아 있는 증권사 주문번호 (정정하면 바뀐다)
    ord_nos: list = field(default_factory=list)    # 이 주문이 거친 모든 주문번호
    filled_by_no: dict = field(default_factory=dict)   # 주문번호 -> 누적 체결 수량
    value_by_no: dict = field(default_factory=dict)    # 주문번호 -> 누적 체결 금액
    cancel_requested: bool = False
    sent_at: float = 0.0
    note: str = ""
    history: list = field(default_factory=list)    # [상태, 시각, 사유]

    @property
    def filled(self) -> int:
        return sum(self.filled_by_no.values())

    @property
    def avg_price(self) -> float:
        return sum(self.value_by_no.values()) / self.filled if self.filled else 0.0

    def go(self, new: str, why: str = "") -> None:
        if new not in TRANSITIONS[self.state]:
            raise IllegalTransition(f"{self.key}: {self.state} -> {new} ({why})")
        self.state = new
        self.history.append([new, round(time.time(), 3), why])


class OMS:
    def __init__(self, broker: Broker, journal: str | Path, kill_file: str | Path = "STOP",
                 crash_after_send: bool = False):
        self.b = broker
        self.kill_file = Path(kill_file)
        self.crash_after_send = crash_after_send   # 시험용: 전송 직후, 응답 기록 전에 죽는다
        self.orders: dict[str, ManagedOrder] = {}
        self.by_no: dict[str, str] = {}
        self.orphans: list[dict] = []              # 주문번호를 아직 모르는 통보 (응답보다 먼저 온 통보)
        self.blocked_codes: dict[str, str] = {}
        self.blocked_keys: dict[str, str] = {}
        self.stats = {"dup_suppressed": 0, "blocked": 0, "sent": 0}
        self._replay(Path(journal))
        self.j = AuditLog(journal)

    # ---------- 일지 ----------
    def _replay(self, path: Path) -> None:
        if not path.exists():
            return
        import json
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["kind"] == "order":
                mo = ManagedOrder(**rec["data"])
                self.orders[mo.key] = mo
                for n in mo.ord_nos:
                    self.by_no[n] = mo.key
            elif rec["kind"] == "block_key":
                self.blocked_keys[rec["data"]["key"]] = rec["data"]["why"]

    def _log(self, mo: ManagedOrder) -> None:
        self.j.append("order", asdict(mo))

    def _link(self, mo: ManagedOrder, ord_no: str) -> None:
        if ord_no and ord_no not in mo.ord_nos:
            mo.ord_nos.append(ord_no)
        mo.ord_no = ord_no
        self.by_no[ord_no] = mo.key

    # ---------- 긴급 중지 ----------
    @property
    def killed(self) -> bool:
        return self.kill_file.exists()

    # ---------- 주문 ----------
    def submit(self, key: str, code: str, side: str, qty: int, price: float,
               order_type: str = "limit", kind: str = "") -> ManagedOrder:
        if key in self.orders:                     # 같은 키는 두 번 보내지 않는다
            self.stats["dup_suppressed"] += 1
            return self.orders[key]
        mo = ManagedOrder(key, code, side, int(qty), float(price), order_type, kind)
        why = ("긴급 중지" if self.killed else self.blocked_keys.get(key) or
               (f"차단 종목: {self.blocked_codes[code]}" if code in self.blocked_codes else ""))
        if why:                                    # 막은 주문은 장부에 넣지 않는다 (중지를 풀면 같은 키로 낼 수 있게)
            self.stats["blocked"] += 1
            self.j.append("blocked", {"key": key, "code": code, "why": why})
            raise Blocked(f"{key}: {why}")
        self.orders[key] = mo
        mo.sent_at = time.time()
        mo.go("SENT")
        self._log(mo)                              # 보내기 전에 기록한다
        self.stats["sent"] += 1
        try:
            ack = self.b.submit(code, side, mo.qty, mo.price, order_type)
        except Exception as e:                     # 시간 초과 등: 접수됐는지 모른다
            mo.go("UNKNOWN", type(e).__name__)
            self._log(mo)
            return mo
        if self.crash_after_send:
            raise SystemExit("시험: 전송 직후 종료")
        if ack.ok:
            self._link(mo, ack.ord_no)
            if mo.state == "SENT":
                mo.go("ACCEPTED", f"{ack.ms:.0f}ms")
            self._log(mo)
            self._drain_orphans()
        else:
            mo.go("REJECTED", f"{ack.code} {ack.msg}")
            mo.note = ack.msg
            self._log(mo)
        return mo

    def amend(self, key: str, price: float, qty: int | None = None) -> ManagedOrder:
        mo = self.orders[key]
        if self.killed:
            raise Blocked(f"{key}: 긴급 중지")
        if mo.state not in ("ACCEPTED", "PARTIAL"):
            raise IllegalTransition(f"{key}: {mo.state} 에서는 정정할 수 없음")
        rest = mo.qty - mo.filled if qty is None else qty
        ack = self.b.amend(mo.ord_no, mo.code, rest, price)
        if not ack.ok:
            mo.note = f"정정 실패 {ack.code} {ack.msg}"
            self._log(mo)
            return mo
        mo.price = float(price)
        self._link(mo, ack.ord_no)
        mo.go(mo.state, f"정정 {price:g}")
        self._log(mo)
        return mo

    def cancel(self, key: str) -> ManagedOrder:
        mo = self.orders[key]
        if mo.state in TERMINAL:
            return mo
        mo.cancel_requested = True
        ack = self.b.cancel(mo.ord_no, mo.code, 0)
        mo.note = "" if ack.ok else f"취소 실패 {ack.code} {ack.msg}"
        self._log(mo)                              # CANCELLED 는 통보나 조회로 확인한 뒤에
        return mo

    def live_orders(self) -> list[ManagedOrder]:
        return [m for m in self.orders.values() if m.state not in TERMINAL]

    # ---------- 체결 통보 (WebSocket 00) ----------
    def on_event(self, ev: dict) -> None:
        key = self.by_no.get(ev["ord_no"]) or self.by_no.get(ev.get("orig_ord_no", ""))
        if key is None:
            self.orphans.append(ev)                # 응답보다 먼저 왔거나, 이 시스템의 주문이 아니다
            return
        mo = self.orders[key]
        no = ev["ord_no"]
        if ev["status"] == "체결" and no in mo.ord_nos:
            cum = ev["qty"] - ev["remaining"]       # 누적 체결 = 주문 수량 - 미체결 (같은 통보가 두 번 와도 같다)
            prev = mo.filled_by_no.get(no, 0)
            if cum > prev:
                mo.filled_by_no[no] = cum
                mo.value_by_no[no] = mo.value_by_no.get(no, 0.0) + (cum - prev) * ev["fill_price"]
        elif "취소" in ev.get("order_kind", "") and ev["status"] == "확인":
            mo.cancel_requested = True
        self._settle(mo, f"통보 {ev['status']}", open_nos=None if ev["status"] != "확인" else set())
        self._log(mo)

    def _drain_orphans(self) -> None:
        pending, self.orphans = self.orphans, []
        for ev in pending:
            self.on_event(ev)

    def _settle(self, mo: ManagedOrder, why: str, open_nos: set | None) -> None:
        """체결 수량과 (알면) 미체결 여부로 상태를 정한다. open_nos: 증권사에 살아 있는 주문번호 집합."""
        if mo.state in TERMINAL:
            return
        if mo.filled >= mo.qty:
            mo.go("FILLED", why)
        elif open_nos is not None and mo.ord_no not in open_nos and mo.cancel_requested:
            mo.go("CANCELLED", why)
        elif mo.filled > 0:
            mo.go("PARTIAL", why)
        elif mo.state in ("SENT", "UNKNOWN"):
            mo.go("ACCEPTED", why)

    # ---------- REST 대사 (재접속 뒤, 장 마감 뒤) ----------
    def sync(self) -> dict:
        """체결 조회·미체결 조회로 모든 미완료 주문을 증권사 기준으로 맞춘다. 통보를 놓쳐도 여기서 따라잡는다."""
        ex = {e["ord_no"]: e for e in self.b.executions()}
        open_nos = {o["ord_no"] for o in self.b.open_orders()}
        changed = 0
        for mo in self.live_orders():
            if mo.state in ("SENT", "UNKNOWN"):
                continue                           # 주문번호를 모르는 주문은 recover() 가 맡는다
            before = (mo.state, mo.filled)
            for no in mo.ord_nos:
                e = ex.get(no)
                if e and e["filled"] > mo.filled_by_no.get(no, 0):
                    mo.value_by_no[no] = e["filled"] * e["fill_price"]
                    mo.filled_by_no[no] = e["filled"]
            e = ex.get(mo.ord_no)
            if mo.ord_no not in open_nos and mo.filled < mo.qty and e and e["remaining"] == 0:
                mo.cancel_requested = True         # 미체결에 없고 체결 조회의 잔량도 0: 취소·만료된 주문
            self._settle(mo, "조회 대사", open_nos)
            if (mo.state, mo.filled) != before:
                changed += 1
                self._log(mo)
        return {"changed": changed}

    def recover(self) -> dict:
        """응답을 못 받은 주문(SENT·UNKNOWN)을 증권사 조회에서 찾아 주문번호를 다시 붙인다.

        같은 (종목, 방향, 수량, 가격) 이면서 아직 어느 주문에도 붙지 않은 증권사 주문을 찾는다.
        하나면 붙이고, 없으면 증권사에 닿지 않은 것으로 보고 REJECTED, 둘 이상이면 어느 것인지 모르므로
        그 키를 막는다 (사람이 확인할 때까지 같은 키로 다시 내지 않는다).
        """
        rows = {e["ord_no"]: e for e in self.b.executions()}
        for o in self.b.open_orders():
            rows.setdefault(o["ord_no"], o | {"filled": 0, "fill_price": 0.0})
        out = {"linked": 0, "not_found": 0, "ambiguous": 0}
        for mo in [m for m in self.orders.values() if m.state in ("SENT", "UNKNOWN")]:
            cand = [r for n, r in rows.items() if n not in self.by_no and not r.get("orig_ord_no", "").strip("0")
                    and r["code"] == mo.code and r["side"] == mo.side and r["qty"] == mo.qty
                    and (mo.order_type == "market" or abs(r["price"] - mo.price) < 1e-6)]
            if len(cand) == 1:
                r = cand[0]
                self._link(mo, r["ord_no"])
                if r["filled"]:
                    mo.filled_by_no[r["ord_no"]] = r["filled"]
                    mo.value_by_no[r["ord_no"]] = r["filled"] * r["fill_price"]
                open_nos = {o for o, x in rows.items() if x.get("remaining", 0) > 0}
                if mo.state == "SENT":
                    mo.go("UNKNOWN", "응답 기록 없음")
                self._settle(mo, "복구: 주문번호 찾음", open_nos)
                out["linked"] += 1
            elif not cand:
                if mo.state == "SENT":
                    mo.go("UNKNOWN", "응답 기록 없음")
                mo.go("REJECTED", "복구: 증권사에 없음")
                out["not_found"] += 1
            else:
                why = f"복구 모호: 후보 {len(cand)}건"
                self.blocked_keys[mo.key] = why
                self.j.append("block_key", {"key": mo.key, "why": why})
                mo.note = why
                out["ambiguous"] += 1
            self._log(mo)
        return out

    # ---------- 긴급 중지 ----------
    def kill_all(self, reason: str, timeout: float = 10.0, poll: float = 1.0) -> dict:
        """중지 파일을 만들고(이후 신규 주문 거부) 자기 미체결 주문만 모두 취소한 뒤 사라질 때까지 확인한다."""
        t0 = time.time()
        if any(m.state in ("SENT", "UNKNOWN") for m in self.orders.values()):
            self.recover()                         # 주문번호를 모르는 주문도 찾아서 취소한다
        self.kill_file.write_text(f"{reason}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        mine = [m for m in self.live_orders() if m.ord_no]
        cancel_calls = []
        for mo in mine:
            self.cancel(mo.key)
            cancel_calls.append(mo.ord_no)
        left = set(cancel_calls)
        while left and time.time() - t0 < timeout:
            left &= {o["ord_no"] for o in self.b.open_orders()}
            if left:
                time.sleep(poll)
        self.sync()
        return {"cancel_calls": cancel_calls, "left": sorted(left), "seconds": round(time.time() - t0, 2)}


def own_positions(oms: OMS) -> dict[str, int]:
    """주문 장부의 체결로 계산한 이 시스템의 보유 수량."""
    pos: dict[str, int] = {}
    for mo in oms.orders.values():
        if mo.filled:
            pos[mo.code] = pos.get(mo.code, 0) + (mo.filled if mo.side == "buy" else -mo.filled)
    return {c: q for c, q in pos.items() if q}


def reconcile(expected: dict[str, int], broker: dict[str, int], owned: set[str]) -> list[dict]:
    """16-3 잔고 대사. expected: 시스템이 아는 보유, broker: 증권사 잔고, owned: 시스템이 거래한 적 있는 종목."""
    out = []
    for c in sorted(set(expected) | set(broker)):
        e, b = expected.get(c, 0), broker.get(c, 0)
        if e == b:
            continue
        if e == 0 and c not in owned:
            kind = "외부 보유"                      # 이 시스템이 거래한 적 없는 종목: 보고만 한다
        elif e == 0:
            kind = "증권사에만 있음"
        elif b == 0:
            kind = "엔진에만 있음"
        else:
            kind = "수량 차이"
        out.append({"code": c, "kind": kind, "expected": e, "broker": b})
    return out


def block_mismatches(oms: OMS, diffs: list[dict]) -> list[str]:
    """대사 차이가 난 종목의 신규 주문을 막는다 (외부 보유는 막지 않는다 — 원래 주문하지 않으므로)."""
    codes = []
    for d in diffs:
        if d["kind"] != "외부 보유":
            oms.blocked_codes[d["code"]] = f"잔고 불일치 {d['kind']} (기대 {d['expected']}, 증권사 {d['broker']})"
            codes.append(d["code"])
    return codes
