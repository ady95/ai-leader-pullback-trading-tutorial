"""15-3 운영 모니터링 화면 (Streamlit).

    uv run streamlit run app/dashboard.py

data/live/ 의 상태 파일·주문 계획·감사 기록만 읽는다. 화면에서 주문을 내거나 고치는 기능은 두지 않는다
(긴급 중지는 16-3절에서 별도 경로로 만든다).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from pullback.live.audit import verify

LIVE = Path(os.getenv("LIVE_DIR", "data/live"))

st.set_page_config(page_title="B0 운영 모니터", layout="wide")
st.title("B0 운영 모니터 (모의)")

state_path = LIVE / "state_live.json"
if not state_path.exists():
    state_path = LIVE / "state.json"
if not state_path.exists():
    st.warning(f"상태 파일이 없습니다: {state_path}")
    st.stop()
state = json.loads(state_path.read_text(encoding="utf-8"))

# 1. 감사 기록 무결성
cols = st.columns(3)
for col, name in zip(cols, ["audit.jsonl", "replay_audit.jsonl"]):
    path = LIVE / name
    if path.exists():
        ok, line = verify(path)
        n = sum(1 for _ in path.open(encoding="utf-8"))
        col.metric(name, f"{n:,}줄", "해시 체인 정상" if ok else f"{line}번째 줄 불일치",
                   delta_color="normal" if ok else "inverse")
cols[2].metric("마지막 장 마감 처리", state["last_date"])

# 2. 보유
st.subheader("보유")
pos = pd.DataFrame([{"종목": c, "진입일": p["entry_date"], "단위": p["units"], "2차 매수": p["second_done"],
                     "1차 익절": p["tp1_done"], "보유 일수": p["days_held"], "다음 날 시가 청산": p["exit_next_open"]}
                    for c, p in state["positions"].items()])
st.dataframe(pos, width="stretch", hide_index=True)

# 3. 다음 거래일 주문 계획
plans = sorted(LIVE.glob("plan_*.json"))
if plans:
    st.subheader(f"주문 계획 {plans[-1].stem.removeprefix('plan_')}")
    st.dataframe(pd.DataFrame(json.loads(plans[-1].read_text(encoding="utf-8"))), width="stretch",
                 hide_index=True)

# 4. 일별 결정 추이 (재생 기록)
days_path = LIVE / "replay_days.parquet"
if days_path.exists():
    st.subheader("일별 후보·주문·보유 (재생)")
    d = pd.read_parquet(days_path).set_index("date")[["setups", "orders", "positions"]]
    st.line_chart(d.tail(250))

# 5. 최근 감사 기록
audit = LIVE / "audit.jsonl"
if audit.exists():
    st.subheader("최근 감사 기록")
    recs = [json.loads(x) for x in audit.read_text(encoding="utf-8").splitlines()[-5:]]
    for r in reversed(recs):
        with st.expander(f"#{r['seq']} {r['kind']} {r['data'].get('date', r['data'].get('for', ''))}  {r['hash'][:12]}"):
            st.json(r["data"])
