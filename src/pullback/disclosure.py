"""10-1 공시 제목 구조화 (LLM) 와 키워드 기준 분류.

LLM 에는 '정규화한 공시 제목' 만 보낸다. 회사명·날짜·종목코드는 보내지 않는다.
LLM 이 어느 회사의 어느 시점인지 알 수 없으므로, 학습 데이터에 들어 있을지 모르는
사후 정보(그 뒤 주가가 어떻게 됐는지)가 분류에 섞일 경로가 없다 (10-2).

사용:
    uv run python -m pullback.disclosure            # 제목 분류 (캐시가 있으면 새 제목만)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pullback.paths import DART_DIR, DISCLOSURES

sys.stdout.reconfigure(encoding="utf-8")
LABELS = DART_DIR / "title_labels.parquet"
CATEGORIES = ["정기보고", "지분변동", "자금조달", "계약수주", "실적", "지배구조", "관리감독",
              "소송사고", "투자합병", "주주환원", "주총기타", "기타"]
IMPACTS = ["호재", "악재", "중립"]
BATCH = 40
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

PROMPT = """당신은 한국 상장사 공시를 분류하는 애널리스트입니다.
아래는 회사명과 날짜를 지운 DART 공시 제목 목록입니다. 각 제목에 대해 다음을 판단하세요.

- category: {categories} 중 하나
- impact: 이 종류의 공시가 나왔을 때 기존 주주에게 일반적으로 '호재', '악재', '중립' 중 무엇인지
- material: 주가를 움직일 만한 새로운 기업 이벤트이면 true, 정기·의례적 공시면 false

제목만으로 판단하고, 특정 회사나 시점을 추측하지 마세요.
반드시 {{"items": [{{"id": 번호, "category": "...", "impact": "...", "material": true}}, ...]}} 형식의 JSON 으로만 답하세요.

{lines}"""

# 키워드 기준 분류 (LLM 과 비교하는 단순 기준, 10-1)
BAD_WORDS = ["감사의견", "거절", "부적정", "한정", "관리종목", "상장폐지", "상장적격성", "불성실공시", "매매거래정지",
             "횡령", "배임", "회생절차", "파산", "부도", "영업정지", "자본잠식", "소송등의제기", "벌금", "과징금",
             "유상증자결정", "전환사채권발행결정", "신주인수권부사채권발행결정", "교환사채권발행결정", "감자결정",
             "최대주주변경", "공급계약해지", "손익구조30%", "적자전환"]
GOOD_WORDS = ["공급계약체결", "단일판매", "무상증자결정", "자기주식취득결정", "자기주식취득신탁계약체결", "현금ㆍ현물배당",
              "특허권취득", "흑자전환", "기술이전", "주식소각결정", "타법인주식및출자증권취득결정"]


def normalize(title: str) -> str:
    """정정 표시·기간 표시·공백을 지운 제목."""
    t = re.sub(r"^(\[[^\]]*\]\s*)+", "", str(title)).strip()
    t = re.sub(r"\(\s*\d{4}\.\d{2}\s*\)", "", t)
    return re.sub(r"\s+", "", t)


def is_amendment(title: str) -> bool:
    """[기재정정], [첨부정정], [발행조건확정] 등 앞선 공시를 고치거나 덧붙이는 공시."""
    return str(title).lstrip().startswith("[")


def keyword_impact(norm: str) -> str:
    if any(w in norm for w in BAD_WORDS):
        return "악재"
    if any(w in norm for w in GOOD_WORDS):
        return "호재"
    return "중립"


def prompt_hash() -> str:
    return hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:12]


def _client():
    from openai import OpenAI
    load_dotenv()
    load_dotenv("../.env")
    return OpenAI(), os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _valid(items, n):
    if not isinstance(items, list) or len(items) != n:
        return False
    for i, x in enumerate(items):
        if (not isinstance(x, dict) or x.get("id") != i + 1 or x.get("category") not in CATEGORIES
                or x.get("impact") not in IMPACTS or not isinstance(x.get("material"), bool)):
            return False
    return True


def classify(titles: list[str], client, model: str, tries: int = 4) -> list[dict]:
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    content = PROMPT.format(categories=", ".join(CATEGORIES), lines=lines)
    for attempt in range(tries):
        try:
            r = client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                               messages=[{"role": "user", "content": content}])
            USAGE["calls"] += 1
            if r.usage:
                USAGE["prompt_tokens"] += r.usage.prompt_tokens
                USAGE["completion_tokens"] += r.usage.completion_tokens
            items = json.loads(r.choices[0].message.content).get("items")
            if _valid(items, len(titles)):
                return items
        except Exception as e:  # 네트워크·형식 오류는 다시 시도
            print("  재시도:", type(e).__name__, flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError("분류 실패: 형식이 맞는 응답을 받지 못함")


def label_titles() -> pd.DataFrame:
    d = pd.read_parquet(DISCLOSURES)
    norm = d["report_nm"].map(normalize)
    counts = norm.value_counts()
    done = pd.read_parquet(LABELS) if LABELS.exists() else pd.DataFrame(columns=["title"])
    todo = [t for t in counts.index if t not in set(done["title"])]
    client, model = _client()
    rows, t0 = [], time.perf_counter()
    for s in range(0, len(todo), BATCH):
        chunk = todo[s:s + BATCH]
        for t, x in zip(chunk, classify(chunk, client, model)):
            rows.append({"title": t, "category": x["category"], "impact": x["impact"], "material": x["material"],
                         "model": model, "prompt": prompt_hash()})
        if (s // BATCH) % 10 == 0:
            print(f"  {s + len(chunk)}/{len(todo)} 제목, {time.perf_counter() - t0:.0f}s", flush=True)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if rows else done
    out["n_filings"] = out["title"].map(counts).fillna(0).astype(int)
    out["keyword_impact"] = out["title"].map(keyword_impact)
    out.to_parquet(LABELS, index=False)
    return out


def events() -> pd.DataFrame:
    """공시 한 건 = 한 행. 정정 공시는 새 사건이 아니다(is_event=False)."""
    d = pd.read_parquet(DISCLOSURES)
    lab = pd.read_parquet(LABELS).set_index("title")
    d["title"] = d["report_nm"].map(normalize)
    d["is_event"] = ~d["report_nm"].map(is_amendment)
    for c in ["category", "impact", "material", "keyword_impact"]:
        d[c] = d["title"].map(lab[c])
    d["date"] = pd.to_datetime(d["rcept_dt"])
    return d.rename(columns={"stock_code": "code"})


def main() -> None:
    out = label_titles()
    print(f"제목 {len(out):,}개 분류 -> {LABELS}")
    print(out["impact"].value_counts().to_dict(), USAGE)


if __name__ == "__main__":
    main()


DISC_FEATURES = ["bad_since_peak", "n_events_since_peak", "has_material_rise", "fresh_material", "days_since_good"]


def _count_between(dates: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """정렬된 사건 날짜 중 [lo, hi] 안에 든 개수."""
    return np.searchsorted(dates, hi, side="right") - np.searchsorted(dates, lo, side="left")


def order_day_features(df: pd.DataFrame, mask: pd.Series, ev: pd.DataFrame, impact_col: str = "impact") -> pd.DataFrame:
    """mask 인 날(t일 장 마감)마다 t일까지 접수된 공시로 만든 특징량.

    bad_since_peak      파동 고점일 ~ t일에 '악재' 공시가 있는가 (A2 필터)
    n_events_since_peak 같은 구간의 공시 사건 수
    has_material_rise   상승 구간(저점일 ~ 고점일)에 '호재·material' 공시가 있는가
    fresh_material      그 재료가 새로운가: 저점일 전 365일 동안 같은 범주의 호재·material 공시가 없었음
    days_since_good     t일 기준 가장 최근 호재·material 공시 이후 경과일 (없으면 NaN)
    """
    ev = ev[ev["is_event"]]
    rows = df.loc[mask, ["code", "date", "peak_date", "trough_date"]]
    out = pd.DataFrame(index=rows.index, columns=DISC_FEATURES, dtype=float)
    for code, r in rows.groupby("code", sort=False):
        e = ev[ev["code"] == code].sort_values("date")
        t = r["date"].to_numpy("datetime64[ns]")
        pk = pd.to_datetime(r["peak_date"]).to_numpy("datetime64[ns]")
        tr = pd.to_datetime(r["trough_date"]).to_numpy("datetime64[ns]")
        all_d = e["date"].to_numpy("datetime64[ns]")
        bad = e.loc[e[impact_col] == "악재", "date"].to_numpy("datetime64[ns]")
        good = e[(e["impact"] == "호재") & e["material"].astype(bool)]
        gd = good["date"].to_numpy("datetime64[ns]")
        out.loc[r.index, "bad_since_peak"] = (_count_between(bad, pk, t) > 0).astype(float)
        out.loc[r.index, "n_events_since_peak"] = _count_between(all_d, pk, t)
        out.loc[r.index, "has_material_rise"] = (_count_between(gd, tr, pk) > 0).astype(float)
        fresh = np.zeros(len(r))
        for cat, g in good.groupby("category"):
            d = g["date"].to_numpy("datetime64[ns]")
            in_rise = _count_between(d, tr, pk) > 0
            before = _count_between(d, tr - np.timedelta64(365, "D"), tr - np.timedelta64(1, "D")) > 0
            fresh = np.maximum(fresh, (in_rise & ~before).astype(float))
        out.loc[r.index, "fresh_material"] = fresh
        k = np.searchsorted(gd, t, side="right") - 1
        last = pd.Series(gd[np.clip(k, 0, None)] if len(gd) else pd.NaT, index=r.index).where(k >= 0)
        out.loc[r.index, "days_since_good"] = (pd.Series(t, index=r.index) - pd.to_datetime(last)).dt.days
    return out
