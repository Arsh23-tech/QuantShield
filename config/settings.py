from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"


# ============================================================
# PORTFOLIO CONFIGURATION
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "JPM",
    "JNJ",
    "XOM",
    "SPY",
    "QQQ",
    "IWM",
]

BENCHMARK = "SPY"


# ============================================================
# DATA CONFIGURATION
# ============================================================

START_DATE = "2016-10-01"
END_DATE = "2026-10-11"

DATA_FREQUENCY = "daily"


# ============================================================
# MARKET DATA FIELDS
# ============================================================

MARKET_DATA_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]