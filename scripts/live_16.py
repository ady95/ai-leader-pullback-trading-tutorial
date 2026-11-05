"""16장 모의투자 주문 시험: 주문 수명주기, 체결 통보, 중복 방지, 정정, 재접속, 잔고 대사, 긴급 중지.

키움 모의투자 서버에 시험용 주문(1~2주)만 낸다. 15장의 B0 주문 계획은 내지 않는다.
장중(09:00~15:30)에 실행해야 한다. 끝나면 시험 종목 보유를 시장가로 정리해 시작 상태로 돌려놓는다.

사용:
    uv run python scripts/live_16.py --code 034220            # 전체 시험 (약 5분)
    uv run python scripts/live_16.py --code 034220 --preopen  # 장 시작 전 주문 접수 여부만 (접수되면 바로 취소)
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from pullback.live.broker import KiwoomBroker
from pullback.live.oms import OMS, Blocked, block_mismatches, own_positions, reconcile
from pullback.live.risk import ceil_tick, floor_tick, tick
from pullback.live.ws import FillStream

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("data/live16")


def version() -> str:
    """git 커밋 (+dirty: 커밋하지 않은 변경이 있음). 15-3절의 교훈대로 기록에 남긴다."""
    h = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return h + ("+dirty" if dirty else "")


class Run:
    def __init__(self, code: str):
        self.b = KiwoomBroker()
        self.code = code
        self.day = time.strftime("%Y%m%d")
        self.dir = OUT / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kill = self.dir / "STOP"
        self.oms = OMS(self.b, self.dir / "journal.jsonl", kill_file=self.kill)
        self.others: list[OMS] = []
        self.events: list[dict] = []
        self.res: dict = {"code": code, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "code_version": version()}
        self.ws_view: dict = {}               # 체결 통보만으로 본 (체결 수량, 평균가) — 조회 대사 전 값
        self.stream: FillStream | None = None

    # ---------- 공통 ----------
    def pump(self) -> None:
        if not self.stream:
            return
        for ev in self.stream.drain():
            self.events.append(ev)
            for o in [self.oms, *self.others]:
                o.on_event(ev)
        for mo in self.oms.orders.values():
            if mo.state == "FILLED" and mo.key not in self.ws_view:
                self.ws_view[mo.key] = (mo.filled, mo.avg_price, time.time())

    def wait(self, cond, timeout: float, step: float = 0.3) -> float | None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.pump()
            if cond():
                return round(time.time() - t0, 2)
            time.sleep(step)
        self.pump()
        return round(time.time() - t0, 2) if cond() else None

    def open_map(self) -> dict:
        return {o["ord_no"]: o for o in self.b.open_orders()}

    def first_event(self, ord_no: str, status: str):
        return next((e for e in self.events if e["ord_no"] == ord_no and e["status"] == status), None)

    def save(self) -> None:
        (self.dir / "result.json").write_text(json.dumps(self.res, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
        with (self.dir / "ws_events.jsonl").open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        if self.stream:
            (self.dir / "ws_log.json").write_text(json.dumps(self.stream.log, indent=1), encoding="utf-8")

    # ---------- 16-1 수명주기 ----------
    def lifecycle(self, q) -> None:
        p1 = floor_tick(q["price"] * 0.85)
        p2 = p1 - tick(p1)
        r = {"price": p1, "amend_to": p2}
        t = time.time()
        mo = self.oms.submit(f"T{self.day}:{self.code}:life", self.code, "buy", 1, p1)
        r["submit"] = [mo.state, mo.history[-1][2]]
        first_no = mo.ord_no
        r["visible_s"] = self.wait(lambda: first_no in self.open_map(), 5, 0.5)
        ev = self.first_event(first_no, "접수")
        r["ws_accept_s"] = round(ev["recv_t"] - t, 2) if ev else None
        self.oms.amend(mo.key, p2)
        r["amend"] = [mo.ord_no != first_no, mo.history[-1][2], mo.note]
        new_no = mo.ord_no
        r["amend_visible_s"] = self.wait(lambda: self.open_map().get(new_no, {}).get("price") == p2, 5, 0.5)
        r["old_gone"] = first_no not in self.open_map()
        self.oms.cancel(mo.key)
        r["cancel_s"] = self.wait(lambda: mo.state == "CANCELLED", 10)
        if mo.state != "CANCELLED":
            self.oms.sync()
        r["final"] = mo.state
        r["gone"] = new_no not in self.open_map()
        r["history"] = [h[0] for h in mo.history]
        self.res["lifecycle"] = r
        print("lifecycle", r)

    # ---------- 16-1 체결 ----------
    def fill(self, base_qty: int) -> None:
        t = time.time()
        mo = self.oms.submit(f"T{self.day}:{self.code}:buy", self.code, "buy", 2, 0, order_type="market")
        r = {"submit": [mo.state, mo.history[-1][2]]}
        r["filled_s"] = self.wait(lambda: mo.state == "FILLED", 30)
        ev = [e for e in self.events if e["ord_no"] == mo.ord_no and e["status"] == "체결"]
        r["ws_fill_events"] = len(ev)
        r["ws_first_fill_s"] = round(ev[0]["recv_t"] - t, 2) if ev else None
        r["ws_view"] = self.ws_view.get(mo.key, [None, None])[:2]
        if mo.state != "FILLED":
            self.oms.sync()
        r["final"] = [mo.state, mo.filled, mo.avg_price]
        h = self.wait(lambda: self.b.holdings().get(self.code, 0) == base_qty + 2, 10, 1.0)
        r["holding_reflected_s"] = h
        self.res["fill_buy"] = r
        print("fill", r)

    # ---------- 16-2 정정 (1차 익절가 고치기) ----------
    def amend_tp(self, q) -> None:
        p1 = ceil_tick(q["price"] * 1.15)
        p2 = ceil_tick(q["price"] * 1.12)
        mo = self.oms.submit(f"T{self.day}:{self.code}:tp1", self.code, "sell", 1, p1)
        r = {"price": p1, "amend_to": p2, "submit": [mo.state, mo.history[-1][2]]}
        self.wait(lambda: mo.ord_no in self.open_map(), 5, 0.5)
        self.oms.amend(mo.key, p2)
        no = mo.ord_no
        r["amend_note"] = mo.note
        r["amend_visible_s"] = self.wait(lambda: self.open_map().get(no, {}).get("price") == p2, 5, 0.5)
        r["open_price_after"] = self.open_map().get(no, {}).get("price")
        self.oms.cancel(mo.key)
        r["cancel_s"] = self.wait(lambda: mo.state == "CANCELLED", 10)
        if mo.state != "CANCELLED":
            self.oms.sync()
        r["final"] = mo.state
        self.res["amend_tp1"] = r
        print("amend", r)

    # ---------- 16-2 중복 방지 ----------
    def idempotency(self, q) -> None:
        base = floor_tick(q["price"] * 0.80)
        t = tick(base)
        r = {"repeat": [], "crash": []}
        for i in range(5):
            key = f"T{self.day}:{self.code}:rep{i}"
            px = base - i * t
            a = self.oms.submit(key, self.code, "buy", 1, px)
            b = self.oms.submit(key, self.code, "buy", 1, px)
            r["repeat"].append({"key": key, "same": a is b, "ord_no_set": bool(a.ord_no)})
        r["dup_suppressed"] = self.oms.stats["dup_suppressed"]
        opens = self.open_map()
        r["repeat_open_at_broker"] = sum(1 for o in opens.values()
                                         if o["code"] == self.code and o["price"] in {base - i * t for i in range(5)})
        for i in range(5):
            key = f"T{self.day}:{self.code}:crash{i}"
            px = base - (5 + i) * t
            jp = self.dir / f"crash{i}.jsonl"
            try:
                OMS(self.b, jp, kill_file=self.kill, crash_after_send=True).submit(key, self.code, "buy", 1, px)
            except SystemExit:
                pass
            o = OMS(self.b, jp, kill_file=self.kill)                    # 재시작
            state_after_restart = o.orders[key].state
            rec = o.recover()
            o.submit(key, self.code, "buy", 1, px)                      # 재시작한 프로그램이 다시 내려 해도
            n = sum(1 for x in self.open_map().values() if x["code"] == self.code and x["price"] == px)
            self.others.append(o)
            r["crash"].append({"key": key, "restart_state": state_after_restart, "recover": rec,
                               "state": o.orders[key].state, "at_broker": n})
        # 모호한 경우 (사전 등록 밖의 서술): 같은 조건의 주문 두 개가 모두 응답 기록 전에 죽었다
        px = base - 10 * t
        jp = self.dir / "crash_amb.jsonl"
        for k in ("amb0", "amb1"):
            try:
                OMS(self.b, jp, kill_file=self.kill, crash_after_send=True).submit(
                    f"T{self.day}:{self.code}:{k}", self.code, "buy", 1, px)
            except SystemExit:
                pass
        o = OMS(self.b, jp, kill_file=self.kill)
        r["ambiguous"] = o.recover()
        r["ambiguous_blocked"] = sorted(o.blocked_keys)
        self.others.append(o)
        # 정리: 시험 주문 모두 취소 (모호한 두 건은 주문 관리가 주문번호를 모르므로 사람이 하듯 직접 취소)
        for o2 in [self.oms, *self.others]:
            for mo in o2.live_orders():
                if mo.ord_no:
                    o2.cancel(mo.key)
        for x in self.open_map().values():
            if x["code"] == self.code and x["price"] == px:
                self.b.cancel(x["ord_no"], self.code, 0)
        time.sleep(2)
        for o2 in [self.oms, *self.others]:
            o2.sync()
        r["left_open"] = sum(1 for x in self.open_map().values() if x["code"] == self.code)
        self.res["idempotency"] = r
        print("idempotency", json.dumps(r, ensure_ascii=False))

    # ---------- 16-3 재접속 ----------
    def reconnect(self) -> None:
        s = self.stream
        r = {}
        t0 = time.time()
        s.drop(hold_down=5)                                  # 5초 동안 끊긴 채로 둔다 (장애 흉내)
        time.sleep(0.5)
        mo = self.oms.submit(f"T{self.day}:{self.code}:sell1", self.code, "sell", 1, 0, order_type="market")
        r["submit_while_down"] = [mo.state, round(time.time() - t0, 2)]
        got = s.reconnects.get(timeout=60)
        while got < t0:                                      # 앞서 쌓인 접속 기록은 건너뛴다
            got = s.reconnects.get(timeout=60)
        r["down_total_s"] = round(got - t0, 2)
        r["reconnect_after_hold_s"] = round(got - t0 - 5 - 0.0, 2)
        time.sleep(1)
        self.pump()
        r["ws_fill_events_for_order"] = len([e for e in self.events if e["ord_no"] == mo.ord_no])
        r["state_before_sync"] = [mo.state, mo.filled]
        r["sync"] = self.oms.sync()
        r["state_after_sync"] = [mo.state, mo.filled, mo.avg_price]
        ex = {e["ord_no"]: e for e in self.b.executions()}.get(mo.ord_no, {})
        r["broker_execution"] = [ex.get("filled"), ex.get("fill_price")]
        self.res["reconnect"] = r
        print("reconnect", r)

    # ---------- 16-3 잔고 대사 ----------
    def reconcile_test(self, baseline: dict) -> None:
        brk = self.b.holdings()
        own = own_positions(self.oms)
        owned = {mo.code for mo in self.oms.orders.values() if mo.filled}
        expected = {c: q for c, q in baseline.items() if c not in owned} | own   # 외부 보유는 시작 때 값
        real = reconcile(own, brk, owned)
        r = {"own": own, "broker_codes": len(brk), "real_diffs": real}
        inj = {
            "수량 차이": ({**own, self.code: own.get(self.code, 0) + 1}, owned),
            "엔진에만 있음": ({**own, "000020": 1}, owned | {"000020"}),
            "증권사에만 있음": ({c: q for c, q in own.items() if c != self.code}, owned),
        }
        r["injected"] = {}
        for name, (exp, ow) in inj.items():
            d = [x for x in reconcile(exp, brk, ow) if x["kind"] != "외부 보유"]
            r["injected"][name] = [x["kind"] for x in d]
        d = reconcile({**own, self.code: own.get(self.code, 0) + 1}, brk, owned)
        r["blocked"] = block_mismatches(self.oms, d)
        try:
            self.oms.submit(f"T{self.day}:{self.code}:blocked", self.code, "buy", 1, 0, order_type="market")
            r["submit_on_blocked"] = "sent"
        except Blocked as e:
            r["submit_on_blocked"] = str(e)
        self.oms.blocked_codes.clear()
        r["expected_incl_foreign"] = expected == brk
        self.res["reconcile"] = r
        print("reconcile", json.dumps(r, ensure_ascii=False))

    # ---------- 16-3 긴급 중지 ----------
    def kill_test(self, q) -> None:
        base = floor_tick(q["price"] * 0.83)
        for i in range(3):
            self.oms.submit(f"T{self.day}:{self.code}:kill{i}", self.code, "buy", 1, base - i * tick(base))
        mine = {mo.ord_no for mo in self.oms.live_orders()}
        self.wait(lambda: mine <= set(self.open_map()), 5, 0.5)
        foreign_before = {n for n in self.open_map() if n not in self.oms.by_no}
        r = self.oms.kill_all("시험", timeout=10, poll=0.5)
        try:
            self.oms.submit(f"T{self.day}:{self.code}:after_kill", self.code, "buy", 1, base)
            r["after_kill"] = "sent"
        except Blocked as e:
            r["after_kill"] = str(e)
        foreign_after = {n for n in self.open_map() if n not in self.oms.by_no}
        r["foreign_before"], r["foreign_after"] = len(foreign_before), len(foreign_after)
        r["foreign_cancel_calls"] = sum(1 for n in r["cancel_calls"] if n in foreign_before)
        r["states"] = [self.oms.orders[f"T{self.day}:{self.code}:kill{i}"].state for i in range(3)]
        self.kill.unlink()
        self.res["kill"] = r
        print("kill", r)

    def cleanup(self, base_qty: int) -> None:
        held = self.b.holdings().get(self.code, 0) - base_qty
        r = {"test_shares_left": held}
        if held > 0:
            mo = self.oms.submit(f"T{self.day}:{self.code}:cleanup", self.code, "sell", held, 0, order_type="market")
            self.wait(lambda: mo.state == "FILLED", 30)
            r["cleanup"] = mo.state
        r["final_holding"] = self.wait(lambda: self.b.holdings().get(self.code, 0) == base_qty, 10, 1.0)
        self.res["cleanup"] = r
        print("cleanup", r)

    def ws_match(self) -> None:
        """체결 통보로 본 체결 == 체결 조회 (주문번호별 수량·가격)."""
        ex = {e["ord_no"]: e for e in self.b.executions()}
        rows = []
        for mo in self.oms.orders.values():
            fills = [e for e in self.events if e["status"] == "체결" and e["ord_no"] in mo.ord_nos]
            if not fills:
                continue
            for no in mo.ord_nos:
                f = [e for e in fills if e["ord_no"] == no]
                if not f:
                    continue
                q = sum(e["fill_qty"] for e in f)
                v = sum(e["fill_qty"] * e["fill_price"] for e in f)
                e = ex.get(no, {})
                rows.append({"key": mo.key.split(":")[-1], "ord_no": no, "ws_qty": q, "ws_avg": v / q,
                             "rest_qty": e.get("filled"), "rest_price": e.get("fill_price"),
                             "match": q == e.get("filled") and abs(v / q - e.get("fill_price", -1)) < 0.5})
        self.res["ws_match"] = rows
        print("ws_match", rows)


def preopen(code: str) -> None:
    b = KiwoomBroker()
    q = b.quote(code)
    px = floor_tick(q["price"] * 0.85)
    ack = b.submit(code, "buy", 1, px, "limit")
    line = {"t": time.strftime("%H:%M:%S"), "ok": ack.ok, "code": ack.code, "msg": ack.msg, "ms": round(ack.ms)}
    if ack.ok:
        time.sleep(1)
        line["visible"] = ack.ord_no in {o["ord_no"] for o in b.open_orders()}
        c = b.cancel(ack.ord_no, code, 0)
        line["cancel"] = [c.ok, c.code, c.msg]
    print("preopen", json.dumps(line, ensure_ascii=False))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "preopen.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="034220")
    ap.add_argument("--preopen", action="store_true")
    a = ap.parse_args()
    if a.preopen:
        preopen(a.code)
        return
    run = Run(a.code)
    baseline = run.b.holdings()
    run.res["baseline_codes"] = len(baseline)
    base_qty = baseline.get(a.code, 0)
    assert base_qty == 0, "시험 종목은 시작할 때 보유가 없어야 한다"
    q = run.b.quote(a.code)
    run.res["quote"] = q
    assert 1_000 < q["price"] <= 30_000
    run.stream = FillStream(run.b.kw).start()
    run.res["ws_ready_s"] = run.wait(run.stream.ready.is_set, 20)
    t0 = time.time()
    try:
        run.lifecycle(q)
        run.fill(base_qty)
        run.amend_tp(q)
        run.idempotency(q)
        run.reconnect()
        run.reconcile_test(baseline)
        run.kill_test(q)
        run.cleanup(base_qty)
        run.ws_match()
    finally:
        if run.kill.exists():
            run.kill.unlink()
        run.res["seconds"] = round(time.time() - t0)
        run.res["ping"] = run.stream.ping
        run.res["rest_calls"] = run.b.kw.calls
        lat = [float(h[2][:-2]) for mo in run.oms.orders.values() for h in mo.history
               if h[0] == "ACCEPTED" and str(h[2]).endswith("ms")]
        run.res["ack_ms"] = {"n": len(lat), "median": statistics.median(lat) if lat else None,
                             "max": max(lat) if lat else None}
        run.stream.stop()
        run.save()
        print("saved", run.dir, "seconds", run.res["seconds"], "ack_ms", run.res["ack_ms"])


if __name__ == "__main__":
    main()
