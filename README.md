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

## 데이터 이용 안내

원시 시세·뉴스 원문은 제공사 이용 조건 때문에 저장소에 넣지 않습니다. `data/` 폴더는 각자 수집 스크립트로 채웁니다.

## 라이선스

MIT
