"""03-3 전략 명세서 읽기와 검증.

명세서(TOML)는 사전 등록 문서다. 읽을 때 두 가지를 확인한다.
  1. 02장 정의 코드(Params)와 명세서의 값이 같은가
  2. 파일 내용의 SHA-256 지문 (결과 보고서에 함께 기록해 명세서가 바뀌지 않았음을 보인다)

사용:
    uv run python -m pullback.spec configs/b0.toml
"""
from __future__ import annotations

import hashlib
import sys
import tomllib
from pathlib import Path

from pullback.definitions import Params

sys.stdout.reconfigure(encoding="utf-8")

# 명세서 키 -> Params 필드
PARAM_MAP = {
    ("leader", "limit_up_ret"): "limit_up_ret",
    ("leader", "wave_amount_sum_min"): "amount_min",
    ("wave", "window"): "wave_window",
    ("wave", "rise_min"): "wave_rise_min",
    ("wave", "pullback_min"): "pullback_min",
    ("wave", "pullback_max"): "pullback_max",
    ("wave", "pullback_days_min"): "pullback_days_min",
    ("support", "env_period"): "env_period",
    ("support", "env_pct"): "env_pct",
    ("support", "overlap_pct"): "overlap_pct",
}


def load(path: str | Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes().replace(bytes([13, 10]), bytes([10]))   # 운영체제별 줄바꿈(CRLF) 차이를 지운다
    return tomllib.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def to_params(spec: dict) -> Params:
    kw = {field: spec[sec][key] for (sec, key), field in PARAM_MAP.items()}
    return Params(**kw)


def check(spec: dict) -> list[str]:
    """명세서와 코드 기본값이 다르면 그 항목을 돌려준다."""
    default = Params()
    errors = []
    for (sec, key), field in PARAM_MAP.items():
        if spec[sec][key] != getattr(default, field):
            errors.append(f"{sec}.{key}={spec[sec][key]} != Params.{field}={getattr(default, field)}")
    p = spec["periods"]
    order = [p["warmup_start"], p["dev_start"], p["dev_end"], p["validation_start"],
             p["validation_end"], p["holdout_start"], p["holdout_end"]]
    if order != sorted(order):
        errors.append("periods 순서가 맞지 않음")
    return errors


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/b0.toml"
    spec, digest = load(path)
    errors = check(spec)
    print(f"{spec['meta']['id']} v{spec['meta']['version']} ({spec['meta']['registered']})")
    print(f"sha256 {digest}")
    print("검증 통과" if not errors else "불일치:\n  " + "\n  ".join(errors))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
