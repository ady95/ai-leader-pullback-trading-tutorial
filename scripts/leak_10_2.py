"""10-2 LLM 사후 지식 유입 실험.

B0 신호(2022~2024) 표본에서 파동마다 하나씩 뽑아, LLM 에 '그 뒤 60거래일 안에 +20% 와 -6.9% 중
무엇이 먼저 나왔는지' 를 묻는다. 회사명·날짜를 알려 준 조건과 지운 조건을 비교한다.
알려 준 조건에서만 정답률이 높다면, LLM 은 미래를 '예측' 한 것이 아니라 '기억' 한 것이다.

사용:
    uv run python scripts/leak_10_2.py --n 120
    uv run python scripts/leak_10_2.py --n 120 --forced
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from pullback.disclosure import _client
from pullback.paths import DATA, UNIVERSE

sys.stdout.reconfigure(encoding="utf-8")
ASK = """다음은 한국 상장 종목과 날짜입니다. 각 항목에 대해, 그 날짜 다음 거래일부터 60거래일 안에
주가가 '+20% 상승'과 '-6.9% 하락' 중 어느 쪽에 먼저 닿았는지 답하세요.
{rule}
{{"items": [{{"id": 번호, "answer": "{choices}"}}, ...]}} 형식의 JSON 으로만 답하세요.

{lines}"""


RULES = {False: ('알고 있으면 "상승" 또는 "하락", 모르면 "모름"이라고 답하세요.', "상승|하락|모름"),
         True: ('확실하지 않더라도 반드시 "상승" 또는 "하락" 중 더 가능성이 높은 쪽을 고르세요.', "상승|하락")}


def ask(rows: list[str], client, model, forced: bool = False) -> list[str]:
    lines = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rows))
    rule, choices = RULES[forced]
    for _ in range(4):
        r = client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                           messages=[{"role": "user", "content": ASK.format(lines=lines, rule=rule, choices=choices)}])
        try:
            items = json.loads(r.choices[0].message.content)["items"]
            ans = [x["answer"] for x in sorted(items, key=lambda x: x["id"])]
            if len(ans) == len(rows) and set(ans) <= set(choices.split("|")):
                return ans
        except (KeyError, ValueError, TypeError):
            pass
    raise RuntimeError("응답 형식 오류")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--forced", action="store_true", help="'모름' 없이 둘 중 하나를 고르게 한다")
    a = ap.parse_args()
    ds = pd.read_parquet(DATA / "dataset_a1.parquet")
    ds = ds[pd.to_datetime(ds["date"]).dt.year.between(2022, 2024) & ds["y"].isin([1, -1])]
    one = ds.groupby("wave_id").head(1).sample(a.n, random_state=0)
    names = pd.read_parquet(UNIVERSE).drop_duplicates("code").set_index("code")["name"]
    one["name"] = one["code"].map(names)
    client, model = _client()
    res = {}
    for cond in ["실명·날짜", "익명"]:
        ans = []
        for s in range(0, len(one), 20):
            ch = one.iloc[s:s + 20]
            rows = ([f"{n}({c}), {pd.Timestamp(d):%Y-%m-%d}" for n, c, d in zip(ch["name"], ch["code"], ch["date"])]
                    if cond == "실명·날짜" else [f"종목 {i + 1}, 날짜 비공개" for i in range(len(ch))])
            ans.extend(ask(rows, client, model, a.forced))
        one[cond] = ans
        truth = np.where(one["y"] == 1, "상승", "하락")
        answered = one[cond] != "모름"
        acc = (one.loc[answered, cond] == truth[answered.to_numpy()]).mean() if answered.any() else float("nan")
        res[cond] = (answered.mean(), acc)
        up = one[cond] == "상승"
        prec = (one.loc[up, "y"] == 1).mean() if up.any() else float("nan")
        print(f"[{cond}] 답한 비율 {answered.mean():.3f}, 답한 것 중 정답률 {acc:.3f}, 분포 {one[cond].value_counts().to_dict()},"
              f" '상승'이라 답한 {int(up.sum())}개 중 실제 상승 {prec:.3f}")
    base = (one["y"] == 1).mean()
    print(f"표본 {len(one)}개, 실제 '상승' 비율 {base:.3f} (항상 '하락'이라고 답하면 정답률 {1 - base:.3f})")
    print("모델", model)


if __name__ == "__main__":
    main()
