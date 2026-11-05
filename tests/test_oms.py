"""16장 주문 관리 테스트: 가짜 증권사(FakeBroker)로 상태 머신·중복 방지·복구·대사·긴급 중지를 확인한다."""
import itertools

import pytest

from pullback.live.broker import FakeBroker
from pullback.live.oms import (OMS, TRANSITIONS, Blocked, IllegalTransition, ManagedOrder, block_mismatches,
                               own_positions, reconcile)


@pytest.fixture
def env(tmp_path):
    b = FakeBroker()
    return b, OMS(b, tmp_path / "j.jsonl", kill_file=tmp_path / "STOP"), tmp_path


def pump(b, oms):
    """가짜 증권사의 통보를 주문 관리로 흘려보낸다 (WebSocket 00 수신과 같은 역할)."""
    evs, b.events = b.events, []
    for ev in evs:
        oms.on_event(ev)


def test_every_illegal_transition_raises():
    states = list(TRANSITIONS)
    bad = 0
    for a, z in itertools.product(states, states):
        mo = ManagedOrder("k", "000010", "buy", 1, 100.0, state=a)
        if z in TRANSITIONS[a]:
            mo.go(z)
        else:
            with pytest.raises(IllegalTransition):
                mo.go(z)
            bad += 1
    assert bad == 64 - sum(len(v) for v in TRANSITIONS.values())


def test_immediate_fill(env):
    b, oms, _ = env
    mo = oms.submit("d:000010:buy1", "000010", "buy", 10, 1000)
    b.fill(mo.ord_no, 10, 1000)
    pump(b, oms)
    assert mo.state == "FILLED" and mo.filled == 10 and mo.avg_price == 1000


def test_partial_then_cancel_rest(env):
    b, oms, _ = env
    mo = oms.submit("k", "000010", "buy", 10, 1000)
    b.fill(mo.ord_no, 4, 990)
    pump(b, oms)
    assert mo.state == "PARTIAL" and mo.filled == 4
    oms.cancel("k")
    pump(b, oms)
    assert mo.state == "CANCELLED" and mo.filled == 4


def test_duplicate_event_is_idempotent(env):
    b, oms, _ = env
    mo = oms.submit("k", "000010", "buy", 10, 1000)
    b.fill(mo.ord_no, 4, 990)
    evs = list(b.events)
    pump(b, oms)
    for ev in evs:                       # 같은 통보가 한 번 더 와도
        oms.on_event(ev)
    assert mo.filled == 4


def test_amend_then_fill_counts_both_order_numbers(env):
    b, oms, _ = env
    mo = oms.submit("k", "000010", "sell", 10, 1200)
    first = mo.ord_no
    b.fill(first, 3, 1200)
    pump(b, oms)
    oms.amend("k", 1150)                 # 잔량 7주를 새 주문번호로
    assert mo.ord_no != first and mo.qty == 10
    b.fill(mo.ord_no, 7, 1150)
    pump(b, oms)
    assert mo.state == "FILLED" and mo.filled == 10
    assert mo.avg_price == pytest.approx((3 * 1200 + 7 * 1150) / 10)


def test_reject(env):
    b, oms, _ = env
    b.reject_codes.add("000020")
    assert oms.submit("k", "000020", "buy", 1, 1000).state == "REJECTED"


def test_event_before_ack_is_buffered(env):
    b, oms, _ = env
    mo = oms.submit("k", "000010", "buy", 1, 1000)
    # 응답보다 통보가 먼저 온 상황: 주문번호를 모르는 상태에서 체결 통보를 받는다
    oms.by_no.pop(mo.ord_no)
    b.fill(mo.ord_no, 1, 1000)
    pump(b, oms)
    assert mo.state == "ACCEPTED" and oms.orphans
    oms._link(mo, mo.ord_no)
    oms._drain_orphans()
    assert mo.state == "FILLED"


def test_same_key_twice_sends_once(env):
    b, oms, _ = env
    oms.submit("k", "000010", "buy", 1, 1000)
    oms.submit("k", "000010", "buy", 1, 1000)
    assert sum(c[0] == "submit" for c in b.calls) == 1 and oms.stats["dup_suppressed"] == 1


def test_lost_ack_then_recover(env):
    b, oms, _ = env
    b.lose_next_ack = True
    mo = oms.submit("k", "000010", "buy", 1, 1000)
    assert mo.state == "UNKNOWN" and not mo.ord_no
    assert oms.recover() == {"linked": 1, "not_found": 0, "ambiguous": 0}
    assert mo.state == "ACCEPTED" and mo.ord_no


def test_crash_after_send_restart_does_not_duplicate(env):
    b, _, tmp = env
    crashing = OMS(b, tmp / "c.jsonl", kill_file=tmp / "STOP", crash_after_send=True)
    with pytest.raises(SystemExit):
        crashing.submit("k", "000010", "buy", 1, 1000)
    again = OMS(b, tmp / "c.jsonl", kill_file=tmp / "STOP")      # 재시작: 일지에서 장부를 다시 만든다
    assert again.orders["k"].state == "SENT"
    assert again.recover()["linked"] == 1
    again.submit("k", "000010", "buy", 1, 1000)                  # 재시작한 프로그램이 같은 주문을 다시 내려 해도
    assert sum(c[0] == "submit" for c in b.calls) == 1


def test_crash_ambiguous_blocks_key(env):
    b, _, tmp = env
    b.submit("000010", "buy", 1, 1000)                           # 같은 조건의 주문이 이미 하나 있다 (다른 주체)
    crashing = OMS(b, tmp / "c.jsonl", kill_file=tmp / "STOP", crash_after_send=True)
    with pytest.raises(SystemExit):
        crashing.submit("k", "000010", "buy", 1, 1000)
    again = OMS(b, tmp / "c.jsonl", kill_file=tmp / "STOP")
    assert again.recover()["ambiguous"] == 1 and "k" in again.blocked_keys
    assert OMS(b, tmp / "c.jsonl", kill_file=tmp / "STOP").blocked_keys["k"]   # 다시 켜도 막혀 있다


def test_not_sent_is_rejected_on_recover(env):
    b, _, tmp = env
    o = OMS(b, tmp / "n.jsonl", kill_file=tmp / "STOP")
    mo = ManagedOrder("k", "000010", "buy", 1, 1000)
    mo.go("SENT")
    o.orders["k"] = mo                                           # 일지에는 썼지만 증권사에 닿지 않은 주문
    assert o.recover()["not_found"] == 1 and mo.state == "REJECTED"


def test_sync_catches_missed_fill(env):
    b, oms, _ = env
    mo = oms.submit("k", "000010", "buy", 2, 1000)
    b.events.clear()
    b.fill(mo.ord_no, 2, 995)
    b.events.clear()                     # 연결이 끊겨 통보를 못 받았다
    assert mo.state == "ACCEPTED"
    oms.sync()
    assert mo.state == "FILLED" and mo.avg_price == 995


def test_reconcile_kinds_and_block(env):
    b, oms, _ = env
    exp = {"000010": 2, "000030": 1}
    brk = {"000010": 3, "000040": 5, "005930": 1}
    d = {x["code"]: x["kind"] for x in reconcile(exp, brk, owned={"000010", "000030", "000040"})}
    assert d == {"000010": "수량 차이", "000030": "엔진에만 있음", "000040": "증권사에만 있음", "005930": "외부 보유"}
    blocked = block_mismatches(oms, reconcile(exp, brk, owned={"000010", "000030", "000040"}))
    assert sorted(blocked) == ["000010", "000030", "000040"]
    with pytest.raises(Blocked):
        oms.submit("k2", "000010", "buy", 1, 1000)
    oms.submit("k3", "005930", "buy", 1, 1000)                   # 외부 보유 종목은 막지 않는다


def test_kill_switch_cancels_only_own_and_blocks(env):
    b, oms, tmp = env
    foreign = b.submit("000010", "buy", 1, 900).ord_no          # 이 시스템이 내지 않은 주문
    for i in range(3):
        oms.submit(f"k{i}", "000010", "buy", 1, 950 - i)
    r = oms.kill_all("test", poll=0)
    assert r["left"] == [] and len(r["cancel_calls"]) == 3 and foreign not in r["cancel_calls"]
    assert [o["ord_no"] for o in b.open_orders()] == [foreign]
    sent = oms.stats["sent"]
    with pytest.raises(Blocked):
        oms.submit("k9", "000010", "buy", 1, 940)
    assert oms.stats["sent"] == sent and (tmp / "STOP").exists()
    (tmp / "STOP").unlink()
    oms.submit("k9", "000010", "buy", 1, 940)                    # 중지를 풀면 같은 키로 낼 수 있다
    assert oms.orders["k9"].state == "ACCEPTED"


def test_own_positions(env):
    b, oms, _ = env
    a = oms.submit("a", "000010", "buy", 2, 1000)
    b.fill(a.ord_no, 2, 1000)
    s = oms.submit("s", "000010", "sell", 1, 1100)
    b.fill(s.ord_no, 1, 1100)
    pump(b, oms)
    assert own_positions(oms) == {"000010": 1}
