# XAU/USD Analyzer

**AI-Powered Gold Trading Analysis System**

A comprehensive Python-based trading analysis system for XAU/USD (Gold) that combines:
- 📈 Technical Analysis (RSI, MACD, EMA, Bollinger Bands, ADX, ATR, Stochastic)
- 🤖 Machine Learning (XGBoost/RandomForest with time-series cross-validation)
- 📰 News Sentiment Analysis (GDELT API with keyword-based NLP)
- ⚖️ Ensemble Signal Generation (weighted combination of all signals)
- 🛡️ Risk Management (Kelly Criterion, volatility filtering, position sizing)
- 📊 Backtesting Engine (with realistic entry/exit simulation)
- 📉 Interactive Visualizations (dark theme charts)

## 📁 Project Structure

```
xauusd_analyzer/
├── main.py                # Entry point, orchestrates the entire pipeline
├── config.py              # All configuration settings and constants
├── requirements.txt       # Python dependencies
├── data/
│   └── data_engine.py     # Data loading from yfinance, cleaning, normalization
├── features/
│   └── feature_engine.py  # Technical indicators and market structure features
├── models/
│   └── ml_model.py        # ML model training and prediction (XGBoost/RF)
├── strategies/
│   └── ensemble.py        # Advanced ensemble strategies
├── news_engine.py         # GDELT news fetching and sentiment analysis
├── signal_engine.py       # Signal generation and ensemble combination
├── backtest.py            # Historical backtesting engine
├── risk_manager.py        # Risk assessment and position sizing
├── visualizer.py          # Matplotlib chart generation
├── logs/                  # Log files directory
├── exports/               # Exported signals and charts
├── cache/                 # Cached news data
└── README.md              # This file
```

## 🚀 Quick Start

### Installation

```bash
cd xauusd_analyzer
pip install -r requirements.txt
```

### Run Analysis

```bash
# Single analysis run
python main.py

# Live mode (continuous updates every 60 seconds)
python main.py --live

# Verbose output
python main.py --verbose
```

## 📊 Features

### Technical Indicators
- **RSI (14)** - Overbought/Oversold detection
- **MACD (12,26,9)** - Trend momentum
- **EMA (20,50,200)** - Multi-timeframe trend
- **Bollinger Bands (20,2)** - Volatility bands
- **ATR (14)** - Volatility measurement
- **ADX (14)** - Trend strength
- **Stochastic (14,3)** - Momentum oscillator
- **Momentum (10)** - Rate of change

### Machine Learning
- **XGBoost Classifier** (with RandomForest fallback)
- **TimeSeriesSplit Cross-Validation** (5 folds)
- **Feature Importance Analysis**
- **Model Persistence** (joblib)

### News Sentiment
- **GDELT API Integration** - Real-time news fetching
- **Keyword-Based NLP** - Bullish/Bearish classification
- **Sentiment Aggregation** - Multiple article analysis
- **Caching** - Fallback for API failures

### Signal Generation
- **Weighted Ensemble**: 40% Technical + 40% ML + 20% News
- **Confidence Scoring**: Based on signal agreement
- **Threshold-Based Actions**: BUY (>0.65), SELL (<0.35), HOLD

### Risk Management
- **Kelly Criterion** - Optimal position sizing
- **Volatility Filtering** - ATR-based risk assessment
- **Sideways Market Detection** - ADX < 20 filter
- **Stop Loss / Take Profit** - ATR-multiple based

### Backtesting
- **Historical Simulation** - Realistic entry/exit
- **Performance Metrics**: Win rate, Sharpe ratio, Max drawdown
- **Equity Curve** - Visual performance tracking

## 📈 Output

### Console Output
```
╔═══════════════════════════════════╗
║     XAU/USD ANALYZER v1.0         ║
╚═══════════════════════════════════╝

[1/8] Loading market data...
    ✓ Loaded 504 bars from 2022-01-01 to 2024-01-15

[2/8] Computing technical indicators...
    ✓ Computed 27 features

[3/8] Preparing ML model...
    ✓ Model trained (CV Accuracy: 54.32%)
    ✓ ML Prediction: 0.623 (BULLISH)

[4/8] Analyzing news sentiment...
    ✓ Sentiment: BULLISH (score=12.45)
    ✓ Articles analyzed: 25

[5/8] Generating ensemble signal...
══════════════════════════════════
⚡ XAU/USD SIGNAL: 🟢 BUY
📊 Confidence: 78.4%
══════════════════════════════════
...
```

### Generated Files
- `exports/analysis.png` - Comprehensive 2x2 chart
- `exports/signals_history.csv` - Signal history (live mode)
- `logs/signals.log` - Signal log
- `models/saved/xauusd_model.joblib` - Trained ML model

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Symbol and data
SYMBOL = "XAUUSD=X"
PERIOD = "2y"
INTERVAL = "1d"

# Signal thresholds
CONFIDENCE_THRESHOLD = 0.65
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35

# Ensemble weights
WEIGHT_TECHNICAL = 0.40
WEIGHT_ML = 0.40
WEIGHT_NEWS = 0.20

# Backtest settings
INITIAL_CAPITAL = 10000.0
STOP_LOSS_ATR_MULT = 2.0
TAKE_PROFIT_ATR_MULT = 3.0
```

## 🔧 API Keys

No API keys required! The system uses:
- **yfinance** - Free Yahoo Finance data
- **GDELT** - Free news API (no auth required)

## 📝 Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. 

- Not financial advice
- Past performance does not guarantee future results
- Always do your own research
- Trading involves substantial risk of loss

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

---

**Built with ❤️ by Quantitative Trading Systems**
