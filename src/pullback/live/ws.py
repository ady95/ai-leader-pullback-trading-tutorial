"""16-3 끊겨도 다시 붙는 체결 통보 채널.

15-2의 ws_15.py 는 한 번 붙어서 버티는지만 봤다. 운영에서는 연결이 끊긴다고 가정해야 한다.
  - 끊기면 1, 2, 4, ... 최대 30초 간격으로 다시 접속해 LOGIN·REG(00)를 다시 한다
  - 끊긴 동안 온 체결 통보는 다시 오지 않는다. 그래서 다시 붙을 때마다 on_reconnect() 로
    REST 체결 조회 대사(OMS.sync)를 돌린다
  - 통보는 별도 스레드에서 받아 큐에 넣고, 주문 관리는 메인 스레드에서 큐를 비우며 처리한다
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time

import websockets

from pullback.kiwoom import KiwoomClient
from pullback.live.broker import ws_event

URLS = {"paper": "wss://mockapi.kiwoom.com:10000/api/dostk/websocket",
        "real": "wss://api.kiwoom.com:10000/api/dostk/websocket"}


class FillStream:
    def __init__(self, kw: KiwoomClient, types: tuple[str, ...] = ("00",)):
        self.kw, self.types = kw, list(types)
        self.q: queue.Queue = queue.Queue()
        self.log: list[dict] = []          # 연결 사건: connect, login, reg, drop (시각 포함)
        self.reconnects: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._drop = threading.Event()
        self.hold_down = 0.0               # 시험용: 끊긴 뒤 다시 붙기 전에 일부러 기다리는 시간
        self.ready = threading.Event()     # LOGIN·REG 가 끝나 통보를 받을 수 있는 상태
        self.ping = 0
        self._th: threading.Thread | None = None

    def _mark(self, what: str, **kw) -> None:
        self.log.append({"t": round(time.time(), 3), "event": what} | kw)

    async def _session(self) -> None:
        async with websockets.connect(URLS[self.kw.mode]) as ws:
            self._mark("connect")
            await ws.send(json.dumps({"trnm": "LOGIN", "token": self.kw.token()}))
            while not self._stop.is_set():
                if self._drop.is_set():
                    self._drop.clear()
                    await ws.close()
                    self._mark("drop", by="test")
                    return
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                except asyncio.TimeoutError:
                    continue
                tr = m.get("trnm")
                if tr == "PING":
                    await ws.send(json.dumps(m))
                    self.ping += 1
                elif tr == "LOGIN":
                    self._mark("login", rc=m.get("return_code"))
                    await ws.send(json.dumps({"trnm": "REG", "grp_no": "1", "refresh": "1",
                                              "data": [{"item": [""], "type": self.types}]}))
                elif tr == "REG":
                    self._mark("reg", rc=m.get("return_code"))
                    if m.get("return_code") in (0, "0"):
                        self.reconnects.put(time.time())
                        self.ready.set()
                elif tr == "REAL":
                    for d in m.get("data", []):
                        if d.get("type") == "00":
                            ev = ws_event(d.get("values", {}))
                            ev["recv_t"] = time.time()
                            self.q.put(ev)

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            self.ready.clear()
            try:
                await self._session()
                backoff = 1.0
            except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                self._mark("drop", by=type(e).__name__)
            if self._stop.is_set():
                break
            if self.hold_down:
                await asyncio.sleep(self.hold_down)
                self.hold_down = 0.0
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def start(self) -> "FillStream":
        self._th = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
        self._th.start()
        return self

    def drop(self, hold_down: float = 0.0) -> None:
        """시험용: 지금 연결을 끊는다. hold_down 초 동안 다시 붙지 않는다(장애가 이어지는 상황)."""
        self.hold_down = hold_down
        while not self.reconnects.empty():          # 앞선 접속 기록을 비운다
            self.reconnects.get_nowait()
        self.ready.clear()
        self._drop.set()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)

    def drain(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out
