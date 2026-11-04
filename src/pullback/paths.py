"""데이터 경로. 모든 수집·가공 결과는 data/ 아래에 둔다 (git 제외)."""
from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw"
DAILY_DIR = RAW / "daily"
UNIVERSE = RAW / "universe.parquet"
DB = DATA / "pullback.duckdb"
INDEX = RAW / "index.parquet"
THEME_DIR = RAW / "theme_snapshots"
DART_DIR = RAW / "dart"
DISCLOSURES = DART_DIR / "disclosures.parquet"
