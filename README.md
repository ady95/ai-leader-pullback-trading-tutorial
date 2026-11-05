# AI로 검증하는 주도주 눌림매매 따라하기 — 예제 코드

위키독스 책 「[AI로 검증하는 주도주 눌림매매 따라하기](https://wikidocs.net/book/21991)」의 실습 코드 저장소입니다.

> 이 저장소는 교육·연구 목적의 예제 코드이며 투자 권유가 아닙니다. 주문 실습은 모의투자 환경에서만 진행합니다.

## 빠른 시작

```bash
git clone https://github.com/ady95/ai-leader-pullback-trading-tutorial.git
cd ai-leader-pullback-trading-tutorial
uv sync                      # Python 3.11+ 가상환경과 패키지 설치
cp .env.example .env         # 앱키 등을 채운다 (.env 는 커밋 금지)
```

## 장별 실행 순서

책의 연재에 맞춰 장별 실행 명령을 이 표에 추가합니다.

| 장 | 명령 | 결과물 |
|---|---|---|
| 2 | `uv run python scripts/describe_02.py` | 주도주·파동·지지선 정의의 기술 통계 (수익률 미계산, 약 30초) |
| 3-1 | `uv run python scripts/verify_03_1.py` | 지표 구현 검증 (빠른 구현 vs 단순 구현, 손 계산) |
| 3-2 | `uv run python scripts/describe_03.py` | 규칙 구조 통계 (예열 결측, 기준선 고정성, 2차 매수 위치) |
| 3-3 | `uv run python -m pullback.spec configs/b0.toml` | B0 명세서 검증과 SHA-256 지문 |
| 4-1 | `uv run python scripts/measure_04_1.py` | 키움 API 조회 범위·호출 제한 측정 결과 (화면 출력) |
| 4-2 | `uv run python -m pullback.collect.universe` | `data/raw/universe.parquet` (상장 + 2021년 이후 폐지 종목) |
| 4-2 | `uv run python -m pullback.collect.daily --start 2020-01-01 --end 2026-11-03` | `data/raw/daily/*.parquet` (종목별 일봉, 약 2시간) |
| 4-3 | `uv run python -m pullback.storage` | `data/pullback.duckdb` (`daily`, `universe` 뷰) |
| 4-3 | `uv run python -m pullback.quality --start 2021-01-01` | 품질 검사 리포트 (출처별 Q1~Q6) |
| 5-1 | `uv run python scripts/describe_05.py` | B0 후보일 특징량 분포·중복도, `data/features_b0.parquet` |
| 5-2 | `uv run python -m pullback.collect.index` | `data/raw/index.parquet` (코스피·코스닥 지수 일봉) |
| 5-2 | `uv run python -m pullback.collect.theme_snapshot` | `data/raw/theme_snapshots/날짜.parquet` (키움 테마 스냅샷, 장 마감 뒤 매일) |
| 5-2 | `uv run python scripts/describe_05_2.py` | 상대강도·동반 급등 테마 특징량, 키움 테마 비교, 절단 검증 |
| 5-3 | `uv run python scripts/verify_05_3.py` | 절단 검증(미래 데이터 누수 검사), 수정주가 누수 규모 (약 4분) |
| 6 | `uv run pytest -q` | B0 엔진 규칙 테스트 (손 계산 시나리오 10개) |
| 6 | `uv run python scripts/run_b0.py` | B0 백테스트 실행 통계, `data/results/B0_v1.2_*.parquet` (수익률은 07장에서 `--show-returns`) |
| 7-1 | `uv run python scripts/eval_07_1.py` | B0 개발·검증 구간 성과, 신뢰구간, 연도별, MAE·MFE, 계좌 MDD, 비용 민감도 |
| 7-2 | `uv run python scripts/eval_07_2.py` | 국면·규모별 B0 성과, B1(`configs/b1.toml`) |
| 7-3 | `uv run python scripts/eval_07_3.py` | H2·H3·H4와 명세서 대안 비교 (약 10분) |
| 7-3 | `uv run python scripts/eval_07_3_account.py` | 주요 대안의 계좌 손익·MDD |
| 8 | `uv run python -m pullback.dataset` | `data/dataset_a1.parquet` (신호일 표본, Triple Barrier 라벨, 26개 특징량) |
| 8 | `uv run python scripts/describe_08.py` | 표본·라벨 분포(2026 비공개), 폴드별 퍼지 |
| 9 | `uv run python scripts/train_09.py` | Walk-forward 학습, 기준 모델 비교, 보정, SHAP, `data/results/a1_scores.parquet` |
| 9 | `uv run python scripts/eval_09_3.py` | A1 필터 매매 평가와 채택 판정 |
| 10 | `uv run python -m pullback.collect.dart` | DART 공시 목록 수집 (`.env` 에 `DART_API_KEY`) |
| 10 | `uv run python -m pullback.disclosure` | 공시 제목 LLM 분류 (`.env` 에 `OPENAI_API_KEY`, `OPENAI_MODEL`) |
| 10 | `uv run python scripts/leak_10_2.py --n 120 --forced` | LLM 사후 지식 유입 실험 |
| 10 | `uv run python scripts/eval_10.py` | 공시 구조화 통계, 필터 적용 건수 (`--show-returns` 매매 성과·채택 판정, `--lgbm` AUC 진단) |
| 11 | `uv run python scripts/eval_11.py` | B0·B1·A1·A2 동일 조건 비교, 제외 거래, 무작위 필터 기준, 비용 민감도 |
| 11 | `uv run python scripts/eval_11.py --ledger` | 시행 대장 (`configs/trials_ledger.csv`), 개발-검증 순위 상관 |
| 12 | `uv run python scripts/eval_12.py` | A3 위험관리 구성요소(P1 변동성 수량, C1 조건부 2차, E1·E2 노출 한도, L1 월간 손실 한도)와 채택 판정 |
| 12 | `uv run python scripts/eval_12.py --skipped` | 노출 한도가 건너뛴 거래의 연도별 성과 |
| 13 | `uv run python scripts/eval_13.py` | A3 청산 구성요소(X1 고정 목표, X2 낮은 1차 익절, X3 ATR 손절, X4 트레일링, T1 시간 청산, T2 무진전 청산), MFE·회전율 진단, 채택 판정 |
| 14 | `uv run python scripts/eval_14.py` | 국면 규칙(R1 급락 직후, R2 깊은 하락장) 판정, 스트레스 시험, Walk-forward 선택 (2026 성과는 계산하지 않음) |
| 14 | `uv run python scripts/eval_14.py --holdout` | 최종 홀드아웃 2026-01-01 ~ 2026-11-03 (한 번만) |
| 15-1 | `uv run python scripts/replay_15.py` | 운영 엔진(`src/pullback/live/`)을 하루씩 재생해 백테스트와 이벤트 대조, 주문 변환 차이, `data/live/state.json` (약 1분) |
| 15-1 | `uv run python scripts/replay_15.py --mutate tp1` | 일부러 틀린 엔진으로 대조가 차이를 잡는지 확인 (`tp1`, `hold`) |
| 15-2 | `uv run python scripts/batch_15.py --coverage` | 감시 목록(최근 60거래일 상한가)이 B0 후보일을 모두 덮는지 |
| 15-2 | `uv run python scripts/batch_15.py --truncation` | 마지막 20거래일 절단 배치: t일까지의 데이터만으로 같은 후보가 나오는지 |
| 15-2 | `uv run python scripts/batch_15.py --live` | 실제 장 마감 데이터로 배치, 다음 거래일 주문 계획 `data/live/plan_*.json` (주문은 내지 않음, 약 5분) |
| 15-2 | `uv run python scripts/ws_15.py --minutes 10` | 키움 WebSocket 로그인·주문체결(00) 구독·PING 응답 시험 |
| 15-3 | `uv run python scripts/audit_15.py` | 감사 기록 해시 체인 검증과 변조 시험 |
| 15-3 | `uv run streamlit run app/dashboard.py` | 운영 모니터링 화면 (읽기 전용) |

## 데이터 이용 안내

원시 시세·뉴스 원문은 제공사 이용 조건 때문에 저장소에 넣지 않습니다. `data/` 폴더는 각자 수집 스크립트로 채웁니다.

## 라이선스

MIT
