"""15-3 감사 기록: 한 줄에 한 기록(JSON Lines), 앞 줄의 해시를 이어 붙이는 해시 체인.

각 줄의 hash = sha256(앞 줄 hash + 이 줄의 내용). 중간의 한 줄을 고치거나 지우면
그 줄부터 해시가 맞지 않아 verify() 가 위치를 알려 준다. 기록은 덧붙이기만 한다.

사용:
    log = AuditLog("data/live/audit.jsonl")
    log.append("batch", {"date": "2026-11-05", "orders": 3})
    ok, bad_line = verify("data/live/audit.jsonl")
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + raw).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prev, self.seq = GENESIS, 0
        if self.path.exists():                     # 이어 쓰기: 마지막 줄의 해시에서 계속
            for line in self.path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.prev, self.seq = rec["hash"], rec["seq"]

    def append(self, kind: str, data: dict) -> str:
        self.seq += 1
        body = {"seq": self.seq, "kind": kind, "data": data, "prev": self.prev}
        h = _digest(self.prev, body)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(body | {"hash": h}, ensure_ascii=False, default=str) + "\n")
        self.prev = h
        return h


def verify(path: str | Path) -> tuple[bool, int]:
    """(통과 여부, 처음 틀린 줄 번호 또는 0)."""
    prev = GENESIS
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        rec = json.loads(line)
        h = rec.pop("hash")
        if rec.get("prev") != prev or _digest(prev, rec) != h:
            return False, n
        prev = h
    return True, 0
