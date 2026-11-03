"""키움 REST API 최소 클라이언트 (04-1).

- 접근 토큰 발급과 파일 캐시(.cache/)
- 요청 간격 조절(throttle)과 호출 제한 초과 시 재시도
- cont-yn / next-key 헤더를 이용한 연속 조회

사용 예:
    from pullback.kiwoom import KiwoomClient
    kw = KiwoomClient()                      # TRADING_MODE (기본 paper)
    rows = kw.daily_chart("005930", "20261103", max_pages=1)
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

BASE_URLS = {
    "real": "https://api.kiwoom.com",
    "paper": "https://mockapi.kiwoom.com",
}
CACHE_DIR = Path(".cache")
DEFAULT_INTERVAL = {"paper": 1.0, "real": 0.25}


class KiwoomError(RuntimeError):
    """return_code 가 0 이 아닌 응답."""

    def __init__(self, api_id: str, code, msg: str):
        super().__init__(f"[{api_id}] return_code={code} {msg}")
        self.api_id, self.code, self.msg = api_id, code, msg


class KiwoomClient:
    def __init__(self, mode: str | None = None, min_interval: float | None = None, timeout: float = 20):
        self.mode = (mode or os.getenv("TRADING_MODE", "paper")).lower()
        if self.mode not in BASE_URLS:
            raise ValueError("mode 는 paper 또는 real 이어야 합니다")
        prefix = f"KIWOOM_{self.mode.upper()}_"
        self.appkey = os.environ[prefix + "APP_KEY"]
        self.secret = os.environ[prefix + "APP_SECRET"]
        self.base = BASE_URLS[self.mode]
        # 요청 간격: 모의투자 서버는 1초 1건이 안전했다 (04-1 실측). 실전 서버 값은 미실측
        self.min_interval = min_interval if min_interval is not None else DEFAULT_INTERVAL[self.mode]
        self.timeout = timeout
        self.session = requests.Session()
        self._last_call = 0.0
        self._token: str | None = None
        self.calls = 0          # 실제 보낸 요청 수 (실측용)
        self.retries = 0        # 호출 제한으로 재시도한 횟수 (실측용)

    # ---------- 토큰 ----------
    def token(self) -> str:
        if self._token:
            return self._token
        cache = CACHE_DIR / f"kiwoom_token_{self.mode}.json"
        if cache.exists():
            c = json.loads(cache.read_text(encoding="utf-8"))
            # 만료 10분 전까지만 재사용
            if datetime.strptime(c["expires_dt"], "%Y%m%d%H%M%S").timestamp() - time.time() > 600:
                self._token = c["token"]
                return self._token
        r = self.session.post(
            self.base + "/oauth2/token",
            json={"grant_type": "client_credentials", "appkey": self.appkey, "secretkey": self.secret},
            headers={"Content-Type": "application/json;charset=UTF-8"},
            timeout=self.timeout,
        )
        j = r.json()
        if not j.get("token"):
            raise KiwoomError("au10001", j.get("return_code"), j.get("return_msg", ""))
        CACHE_DIR.mkdir(exist_ok=True)
        cache.write_text(json.dumps({"token": j["token"], "expires_dt": j["expires_dt"]}), encoding="utf-8")
        self._token = j["token"]
        return self._token

    # ---------- 요청 ----------
    def _wait(self):
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.monotonic()

    def request(self, api_id: str, path: str, body: dict, next_key: str | None = None,
                max_retry: int = 5) -> tuple[dict, dict]:
        """한 번 호출하고 (응답 JSON, 연속조회 헤더)를 돌려준다."""
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": "Bearer " + self.token(),
            "api-id": api_id,
        }
        if next_key:
            headers.update({"cont-yn": "Y", "next-key": next_key})
        for attempt in range(max_retry + 1):
            self._wait()
            self.calls += 1
            r = self.session.post(self.base + path, json=body, headers=headers, timeout=self.timeout)
            try:
                j = r.json()
            except ValueError:
                j = {"return_code": r.status_code, "return_msg": r.text[:200]}
            code = j.get("return_code")
            # 호출 제한 초과: HTTP 429 또는 return_code 5 (메시지에 '허용된 요청 개수' 포함)
            limited = r.status_code == 429 or "허용된 요청" in str(j.get("return_msg", ""))
            if limited and attempt < max_retry:
                self.retries += 1
                time.sleep(1.0 * (attempt + 1))
                continue
            if code not in (0, "0"):
                raise KiwoomError(api_id, code, j.get("return_msg", ""))
            cont = {"cont_yn": r.headers.get("cont-yn", "N"), "next_key": r.headers.get("next-key", "")}
            return j, cont
        raise KiwoomError(api_id, "retry", "호출 제한 재시도 초과")

    def paginate(self, api_id: str, path: str, body: dict, list_key: str, max_pages: int = 1000):
        """연속 조회를 끝까지(또는 max_pages까지) 따라가며 행 묶음을 차례로 돌려준다."""
        next_key = None
        for _ in range(max_pages):
            j, cont = self.request(api_id, path, body, next_key)
            rows = j.get(list_key) or []
            yield rows
            if cont["cont_yn"] != "Y" or not cont["next_key"] or not rows:
                return
            next_key = cont["next_key"]

    # ---------- 자주 쓰는 조회 ----------
    def daily_chart(self, code: str, base_dt: str, adjusted: bool = True, max_pages: int = 1000) -> list[dict]:
        """ka10081 주식일봉차트조회. base_dt 부터 과거 방향으로 조회한다."""
        body = {"stk_cd": code, "base_dt": base_dt, "upd_stkpc_tp": "1" if adjusted else "0"}
        out: list[dict] = []
        for rows in self.paginate("ka10081", "/api/dostk/chart", body, "stk_dt_pole_chart_qry", max_pages):
            out.extend(rows)
        return out

    def minute_chart(self, code: str, tic_scope: str = "1", adjusted: bool = True, max_pages: int = 1000) -> list[dict]:
        """ka10080 주식분봉차트조회. 가장 최근 분봉부터 과거 방향으로 조회한다."""
        body = {"stk_cd": code, "tic_scope": tic_scope, "upd_stkpc_tp": "1" if adjusted else "0"}
        out: list[dict] = []
        for rows in self.paginate("ka10080", "/api/dostk/chart", body, "stk_min_pole_chart_qry", max_pages):
            out.extend(rows)
        return out

    def stock_list(self, market: str) -> list[dict]:
        """ka10099 종목정보 리스트. market: '0' 코스피, '10' 코스닥."""
        out: list[dict] = []
        for rows in self.paginate("ka10099", "/api/dostk/stkinfo", {"mrkt_tp": market}, "list"):
            out.extend(rows)
        return out
