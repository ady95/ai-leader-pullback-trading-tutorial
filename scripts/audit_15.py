"""15-3 감사 기록 검증과 변조 시험.

    uv run python scripts/audit_15.py

1) data/live/ 의 감사 기록 해시 체인을 검증한다.
2) 재생 기록을 임시 파일로 복사해 (a) 한 줄의 체결가를 고치고 (b) 한 줄을 지우고 (c) 마지막 줄을 지워 본다.
   (a)(b)는 그 줄에서 검증이 실패해야 한다. (c)처럼 끝을 잘라 내는 것은 해시 체인만으로는 알 수 없으므로,
   마지막 해시를 따로 보관(예: 매일 밤 메신저로 전송)해야 한다는 것을 보인다.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from pullback.live.audit import verify
from pullback.paths import DATA

sys.stdout.reconfigure(encoding="utf-8")
LIVE = DATA / "live"


def main() -> None:
    for name in ("replay_audit.jsonl", "audit.jsonl"):
        p = LIVE / name
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
            print(f"{name}: {len(lines):,}줄, 검증 {verify(p)}, 마지막 해시 {json.loads(lines[-1])['hash'][:16]}")
    src = LIVE / "replay_audit.jsonl"
    lines = src.read_text(encoding="utf-8").splitlines()
    target = next(i for i, x in enumerate(lines) if json.loads(x)["data"]["fills"] and i > len(lines) // 2)
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "a.jsonl"
        shutil.copy(src, t)
        rec = json.loads(lines[target])
        f0 = rec["data"]["fills"][0]
        f0["price"] = round(f0["price"] * 1.01, 2)            # (a) 체결가 1% 조작
        t.write_text("\n".join(lines[:target] + [json.dumps(rec, ensure_ascii=False)] + lines[target + 1:]) + "\n",
                     encoding="utf-8")
        print(f"(a) {target + 1}번째 줄 체결가 수정 ({rec['data']['date']} {f0['code']}): 검증 {verify(t)}")
        t.write_text("\n".join(lines[:target] + lines[target + 1:]) + "\n", encoding="utf-8")
        print(f"(b) {target + 1}번째 줄 삭제: 검증 {verify(t)}")
        t.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        print(f"(c) 마지막 줄 삭제: 검증 {verify(t)}  <- 체인만으로는 못 잡는다. 보관해 둔 마지막 해시와 비교해야 한다")


if __name__ == "__main__":
    main()
