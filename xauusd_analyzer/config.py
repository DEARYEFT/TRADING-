"""
Configuration file for XAU/USD Analyzer
All settings, constants, and parameters
"""

import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# SYMBOL & DATA SETTINGS
# ═══════════════════════════════════════════════════════════════
SYMBOL = "XAUUSD=X"
PERIOD = "2y"
INTERVAL = "1d"

# ═══════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.65
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35

# ═══════════════════════════════════════════════════════════════
# ENSEMBLE WEIGHTS
# ═══════════════════════════════════════════════════════════════
WEIGHT_TECHNICAL = 0.40
WEIGHT_ML = 0.40
WEIGHT_NEWS = 0.20

# ═══════════════════════════════════════════════════════════════
# DIRECTORY PATHS
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")

# Create directories if they don't exist
for directory in [LOGS_DIR, EXPORTS_DIR, CACHE_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
NEWS_CACHE_FILE = os.path.join(CACHE_DIR, "news_cache.json")
MODEL_FILE = os.path.join(MODELS_DIR, "xauusd_model.joblib")
SIGNALS_LOG = os.path.join(LOGS_DIR, "signals.log")
SIGNALS_HISTORY = os.path.join(EXPORTS_DIR, "signals_history.csv")
ANALYSIS_PLOT = os.path.join(EXPORTS_DIR, "analysis.png")

# ═══════════════════════════════════════════════════════════════
# TECHNICAL INDICATOR PARAMETERS
# ═══════════════════════════════════════════════════════════════
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 20
EMA_MEDIUM = 50
EMA_LONG = 200
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
MOMENTUM_PERIOD = 10
STOCH_K = 14
STOCH_D = 3
ADX_PERIOD = 14

# ═══════════════════════════════════════════════════════════════
# ML MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════
ML_TRAIN_RATIO = 0.80
ML_CV_SPLITS = 5
ML_RANDOM_STATE = 42
ML_N_ESTIMATORS = 100
ML_MAX_DEPTH = 10
ML_LEARNING_RATE = 0.1

# ═══════════════════════════════════════════════════════════════
# BACKTEST PARAMETERS
# ═══════════════════════════════════════════════════════════════
INITIAL_CAPITAL = 10000.0
STOP_LOSS_ATR_MULT = 2.0
TAKE_PROFIT_ATR_MULT = 3.0
MAX_POSITION_SIZE = 0.20  # Kelly criterion max

# ═══════════════════════════════════════════════════════════════
# NEWS SENTIMENT KEYWORDS
# ═══════════════════════════════════════════════════════════════
BULLISH_KEYWORDS = [
    "crisis", "war", "inflation", "recession", "geopolitical",
    "uncertainty", "safe haven", "federal reserve cut", "dollar fall",
    "risk off", "conflict", "fear", "tension", "sanctions", "debt",
    "bankruptcy", "collapse", "crash", "panic", "emergency"
]

BEARISH_KEYWORDS = [
    "growth", "strong economy", "rate hike", "hawkish", "dollar rally",
    "risk on", "boom", "recovery", "employment rise", "optimism",
    "bull market", "rally", "surge", "gain", "profit", "success",
    "expansion", "upturn", "breakthrough", "positive"
]

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = "gold OR XAU OR precious metals"
GDELT_MAX_RECORDS = 25
GDELT_TIMEOUT = 10

# ═══════════════════════════════════════════════════════════════
# LIVE MODE SETTINGS
# ═══════════════════════════════════════════════════════════════
LIVE_UPDATE_INTERVAL = 60  # seconds

# ═══════════════════════════════════════════════════════════════
# LOGGING SETTINGS
# ═══════════════════════════════════════════════════════════════
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
