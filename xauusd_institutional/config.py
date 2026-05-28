"""
═══════════════════════════════════════════════════════════════════
XAU/USD INSTITUTIONAL AI TRADING SYSTEM - CONFIGURATION
═══════════════════════════════════════════════════════════════════
Production-grade configuration for institutional gold trading.
All parameters calibrated for real-money futures trading.
═══════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CORE SYMBOL & TIMEFRAMES
# ═══════════════════════════════════════════════════════════════
SYMBOL = "XAUUSD=X"
BASE_CURRENCY = "USD"

TIMEFRAMES = {
    "8H": {"period": "60d", "interval": "1h", "candles_forward": 8},
    "1D": {"period": "2y", "interval": "1d", "candles_forward": 1},
    "3D": {"period": "3y", "interval": "1d", "candles_forward": 3},
}

# ═══════════════════════════════════════════════════════════════
# ENSEMBLE WEIGHTS - CALIBRATED FOR MAX SHARPE
# ═══════════════════════════════════════════════════════════════
ENSEMBLE_WEIGHTS = {
    "technical": 0.25,
    "ml": 0.30,
    "news": 0.10,
    "structure": 0.15,
    "macro": 0.10,
    "orderflow": 0.10,
}

# ═══════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS - CONSERVATIVE INSTITUTIONAL GRADE
# ═══════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.68
MIN_ADX_FOR_TRADE = 25
MAX_ATR_PERCENTILE = 90
MIN_VOLUME_RATIO = 0.8
MAX_SPREAD_BPS = 50

# ═══════════════════════════════════════════════════════════════
# RISK MANAGEMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════
INITIAL_CAPITAL = 100000.0
KELLY_MAX_FRACTION = 0.15
KELLY_MULTIPLIER = 0.5  # Half-Kelly for safety
MAX_POSITION_SIZE_PCT = 0.20
MAX_DRAWDOWN_LIMIT = 0.15
MAX_CONSECUTIVE_LOSSES = 5
STOP_LOSS_ATR_MULT = 2.5
TAKE_PROFIT_ATR_MULT = 4.0
TRAILING_STOP_ATR_MULT = 1.5
BREAKEVEN_TRIGGER_ATR_MULT = 1.5

# ═══════════════════════════════════════════════════════════════
# BACKTEST REALISM
# ═══════════════════════════════════════════════════════════════
COMMISSION_PER_LOT = 5.0  # USD per lot
SLIPPAGE_BPS = 2  # Basis points
SPREAD_BPS = 10  # Average XAUUSD spread
EXECUTION_LATENCY_MS = 50
MAX_HOLD_BARS = 15

# ═══════════════════════════════════════════════════════════════
# ML MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════
ML_CONFIG = {
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "catboost": {
        "iterations": 500,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "random_state": 42,
        "verbose": False,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "random_state": 42,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "random_state": 42,
        "n_jobs": -1,
    },
    "hist_gradient_boost": {
        "max_iter": 500,
        "max_depth": 8,
        "learning_rate": 0.03,
        "random_state": 42,
    },
}

CV_SPLITS = 5
CALIBRATION_METHOD = "isotonic"
RETRAIN_DAYS = 7

# ═══════════════════════════════════════════════════════════════
# SMART MONEY CONCEPTS (SMC) PARAMETERS
# ═══════════════════════════════════════════════════════════════
SMC_CONFIG = {
    "bos_lookback": 20,
    "choch_lookback": 10,
    "fvg_min_size_bps": 15,
    "order_block_lookback": 15,
    "liquidity_sweep_threshold": 1.5,
    "premium_discount_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
}

# ═══════════════════════════════════════════════════════════════
# MACRO DATA SOURCES
# ═══════════════════════════════════════════════════════════════
MACRO_SOURCES = {
    "dxy": "^DX-Y.NYB",
    "us10y": "^TNX",
    "vix": "^VIX",
    "sp500": "^GSPC",
    "gold": "GC=F",
}

FRED_ENDPOINTS = {
    "real_yield": "DFII10",
    "inflation": "T10YIE",
    "fed_funds": "DFF",
}

# ═══════════════════════════════════════════════════════════════
# NEWS & SENTIMENT
# ═══════════════════════════════════════════════════════════════
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = "gold OR XAU OR \"safe haven\" OR inflation OR \"federal reserve\""
NEWS_CACHE_MINUTES = 30
MAX_NEWS_ARTICLES = 50

BULLISH_GOLD_KEYWORDS = [
    "war", "conflict", "crisis", "geopolitical", "uncertainty",
    "inflation", "recession", "safe haven", "risk off",
    "dollar weakens", "rate cut", "fed dovish", "debt crisis",
    "banking crisis", "sanctions", "gold rally", "central bank buying",
    "fear", "panic", "market crash", "stimulus", "money printing",
    "currency debasement", "hyperinflation", "default", "collapse"
]

BEARISH_GOLD_KEYWORDS = [
    "rate hike", "hawkish", "strong economy", "dollar rally",
    "risk on", "employment growth", "gdp growth", "recovery",
    "tightening", "yield rise", "bond rally", "tech rally",
    "optimism", "stable", "rate hold", "fed hawkish", "bull market",
    "productivity", "innovation", "earnings beat", "economic data"
]

# ═══════════════════════════════════════════════════════════════
# MARKET REGIME DETECTION
# ═══════════════════════════════════════════════════════════════
REGIME_CONFIG = {
    "hurst_window": 252,
    "atr_lookback": 100,
    "volatility_clustering_lag": 5,
    "entropy_window": 50,
}

REGIME_THRESHOLDS = {
    "trend_adx_min": 25,
    "range_adx_max": 20,
    "high_vol_atr_percentile": 75,
    "low_vol_atr_percentile": 25,
    "panic_vix_level": 30,
}

# ═══════════════════════════════════════════════════════════════
# FILE PATHS
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"
EXPORT_DIR = BASE_DIR / "exports"

MODEL_PATH = CACHE_DIR / "ml_ensemble_model.pkl"
CALIBRATOR_PATH = CACHE_DIR / "probability_calibrator.pkl"
NEWS_CACHE_PATH = CACHE_DIR / "news_cache.json"
MACRO_CACHE_PATH = CACHE_DIR / "macro_cache.json"
SIGNAL_HISTORY_PATH = EXPORT_DIR / "signals_history.csv"
BACKTEST_TRADES_PATH = EXPORT_DIR / "backtest_trades.csv"
DASHBOARD_PATH = EXPORT_DIR / "dashboard.png"
REPORT_PATH = EXPORT_DIR / "market_report.txt"

# ═══════════════════════════════════════════════════════════════
# LIVE MODE SETTINGS
# ═══════════════════════════════════════════════════════════════
LIVE_UPDATE_SECONDS = 60
MAX_SIGNALS_PER_DAY = 10
NO_TRADE_NEWS_WINDOWS = ["NFP", "FOMC", "CPI", "PCE"]  # Economic events

# ═══════════════════════════════════════════════════════════════
# LOGGING CONFIG
# ═══════════════════════════════════════════════════════════════
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
ERROR_LOG_PATH = LOGS_DIR / "errors.log"
SIGNAL_LOG_PATH = LOGS_DIR / "signals.log"
SYSTEM_LOG_PATH = LOGS_DIR / "system.log"

# ═══════════════════════════════════════════════════════════════
# QUALITY GRADES
# ═══════════════════════════════════════════════════════════════
SIGNAL_QUALITY_THRESHOLDS = {
    "A+": 0.85,
    "A": 0.78,
    "B": 0.70,
    "C": 0.62,
    "D": 0.55,
}

# ═══════════════════════════════════════════════════════════════
# VALIDATION FLAGS
# ═══════════════════════════════════════════════════════════════
ENABLE_MACRO_FILTER = True
ENABLE_SMC_FILTER = True
ENABLE_ORDERFLOW_FILTER = True
ENABLE_NEWS_FILTER = True
ENABLE_REGIME_FILTER = True
ENABLE_SELF_EVALUATION = True
