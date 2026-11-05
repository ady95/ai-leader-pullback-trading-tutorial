"""15-2 장중 감시 채널: 키움 WebSocket 접속·로그인·주문체결(00) 구독·PING 응답을 시험한다.

B0는 장 전에 지정가·시가 주문을 걸어 두는 일봉 전략이다. 장중에 시세를 보며 판단하지 않으므로
필요한 실시간 정보는 '내 주문이 체결됐다'는 통보(00)뿐이다. 체결 통보를 받으면 운영 엔진의
on_fill() 이 1차 익절가를 고치거나 상한가 잔여 매도를 건다(15-1).

사용:
    uv run python scripts/ws_15.py --minutes 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import websockets

from pullback.kiwoom import KiwoomClient

sys.stdout.reconfigure(encoding="utf-8")
URLS = {"paper": "wss://mockapi.kiwoom.com:10000/api/dostk/websocket",
        "real": "wss://api.kiwoom.com:10000/api/dostk/websocket"}


async def watch(kw: KiwoomClient, minutes: float, on_message=print) -> dict:
    stat = {"login": None, "reg": None, "ping": 0, "real": 0, "gaps": []}
    t0 = time.time()
    last_ping = t0
    async with websockets.connect(URLS[kw.mode]) as ws:
        await ws.send(json.dumps({"trnm": "LOGIN", "token": kw.token()}))
        while time.time() - t0 < minutes * 60:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            except asyncio.TimeoutError:
                continue
            tr = m.get("trnm")
            if tr == "PING":                          # 서버가 보낸 PING 을 그대로 돌려줘야 연결이 유지된다
                await ws.send(json.dumps(m))
                stat["ping"] += 1
                stat["gaps"].append(round(time.time() - last_ping, 1))
                last_ping = time.time()
            elif tr == "LOGIN":
                stat["login"] = m.get("return_code")
                await ws.send(json.dumps({"trnm": "REG", "grp_no": "1", "refresh": "1",
                                          "data": [{"item": [""], "type": ["00"]}]}))
            elif tr == "REG":
                stat["reg"] = m.get("return_code")
            elif tr == "REAL":
                stat["real"] += 1
                on_message(m)
        stat["seconds"] = round(time.time() - t0)
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    a = ap.parse_args()
    kw = KiwoomClient()
    st = asyncio.run(watch(kw, a.minutes))
    g = st.pop("gaps")
    print(f"mode {kw.mode}, LOGIN {st['login']}, REG(00) {st['reg']}, 유지 {st['seconds']}s, PING {st['ping']}회 "
          f"(간격 {min(g[1:], default=0)}~{max(g[1:], default=0)}s), 실시간 메시지 {st['real']}건")


if __name__ == "__main__":
    main()
