"""15-1 모의 체결기: 주문 의도를 그날 일봉에 맞춰 백테스트와 같은 규칙으로 체결시킨다.

16장의 증권사 어댑터가 생기기 전까지 운영 엔진의 짝이 되고, 그 뒤에도 '백테스트라면 어떻게
체결됐을까'를 실제 체결과 나란히 기록하는 데 쓴다(17-3). 하루 안의 순서는 06-1절과 같다.
  1) 시가 청산  2) 지정가 매수(2차)  3) 1차 익절  4) 상한가 잔여 매도  5) 1차 매수
"""
from __future__ import annotations

from pullback.backtest import Rules
from pullback.live.engine import LiveEngine
from pullback.live.orders import Fill


def _touch(low: float, price: float, strict: bool) -> bool:
    return low < price if strict else low <= price


class SimBroker:
    def __init__(self, rules: Rules):
        self.r = rules

    def run_day(self, eng: LiveEngine, day: str, bars: dict) -> list[Fill]:
        """bars: {code: {"open","high","low","close","prev_close","halted"}}. 체결 목록을 돌려준다."""
        r = self.r
        fills: list[Fill] = []

        def fill(f: Fill):
            fills.append(f)
            eng.on_fill(f)

        for code in sorted(eng.book):
            b = bars.get(code)
            if b is None:
                continue
            o, h, lo, c, pc, halted = (b[k] for k in ("open", "high", "low", "close", "prev_close", "halted"))
            get = lambda kind: next((x for x in eng.book.get(code, []) if x.kind == kind), None)  # noqa: E731
            if halted:
                continue
            ex = get("exit_open")
            if ex is not None:
                locked_down = h == lo and pc > 0 and c / pc - 1 <= -r.limit_up_ret
                if locked_down:
                    fill(Fill(code, day, "defer", c, ex.units, "locked_limit_down"))
                else:
                    fill(Fill(code, day, "sell", o, ex.units, ex.reason))
                continue
            b2 = get("buy2")
            if b2 is not None and _touch(lo, b2.price, r.strict_touch):
                eng.cancel(code, "buy2")
                fill(Fill(code, day, "buy2", min(o, b2.price), b2.units, "", b2.price))
            t1 = get("tp1")
            if t1 is not None and h >= t1.price:
                eng.cancel(code, "tp1")
                fill(Fill(code, day, "sell", max(o, t1.price), t1.units, "tp1", t1.price))
            lu = get("tp2_limit_up")
            if lu is not None and pc > 0 and h / pc - 1 >= r.limit_up_ret:
                eng.cancel(code, "tp2_limit_up")
                fill(Fill(code, day, "sell", h, lu.units, "tp2_limit_up", pc))
            b1 = get("buy1")
            if b1 is not None and _touch(lo, b1.price, r.strict_touch):
                fill(Fill(code, day, "buy1", min(o, b1.price), b1.units, "", b1.price))
                eng.cancel(code, "buy1")
        return fills
