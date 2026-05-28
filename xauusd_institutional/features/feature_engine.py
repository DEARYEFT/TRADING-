"""
═══════════════════════════════════════════════════════════════════
FEATURE ENGINE - 25+ INSTITUTIONAL TECHNICAL INDICATORS
═══════════════════════════════════════════════════════════════════
Comprehensive feature engineering with trend, momentum, volatility,
volume, and market structure indicators.
═══════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List
import ta
from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator, AroonIndicator, CCIIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator
from ta.volume import money_flow_index

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Institutional feature engineering engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FeatureEngine")
        
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Computing technical indicators...")
        df = df.copy()
        df = self._compute_trend_indicators(df)
        df = self._compute_momentum_indicators(df)
        df = self._compute_volatility_indicators(df)
        df = self._compute_volume_indicators(df)
        df = self._compute_market_structure(df)
        df = self._compute_composite_scores(df)
        self.logger.info(f"Computed {len([c for c in df.columns if c not in ['Open','High','Low','Close','Volume']])} features")
        return df
    
    def _compute_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df["ema_50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
        df["ema_200"] = EMAIndicator(close=df["Close"], window=200).ema_indicator()
        
        df["ema_cross_signal"] = np.where(
            (df["ema_20"] > df["ema_50"]) & (df["ema_20"].shift(1) <= df["ema_50"].shift(1)), 1,
            np.where((df["ema_20"] < df["ema_50"]) & (df["ema_20"].shift(1) >= df["ema_50"].shift(1)), -1, 0))
        
        adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
        df["adx"] = adx.adx()
        df["plus_di"] = adx.adx_pos()
        df["minus_di"] = adx.adx_neg()
        
        aroon = AroonIndicator(high=df["High"], low=df["Low"], window=25)
        df["aroon_up"] = aroon.aroon_up()
        df["aroon_down"] = aroon.aroon_down()
        df["aroon_oscillator"] = df["aroon_up"] - df["aroon_down"]
        
        period9_high = df["High"].rolling(window=9).max()
        period9_low = df["Low"].rolling(window=9).min()
        df["ichimoku_conversion"] = (period9_high + period9_low) / 2
        
        period26_high = df["High"].rolling(window=26).max()
        period26_low = df["Low"].rolling(window=26).min()
        df["ichimoku_base"] = (period26_high + period26_low) / 2
        
        df["ichimoku_a"] = ((df["ichimoku_conversion"] + df["ichimoku_base"]) / 2).shift(26)
        period52_high = df["High"].rolling(window=52).max()
        period52_low = df["Low"].rolling(window=52).min()
        df["ichimoku_b"] = ((period52_high + period52_low) / 2).shift(26)
        
        return df
    
    def _compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = RSIIndicator(close=df["Close"], window=14).rsi()
        
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        
        macd = MACD(close=df["Close"], window_fast=12, window_slow=26, window_sign=9)
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        
        cci = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20)
        df["cci"] = cci.cci()
        
        df["roc"] = ROCIndicator(close=df["Close"], window=10).roc()
        df["williams_r"] = WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], lbp=14).wr()
        df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
        
        return df
    
    def _compute_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = df["atr"] / df["Close"] * 100
        
        bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        kc = KeltnerChannel(close=df["Close"], high=df["High"], low=df["Low"], window=20, multiplier=1.5)
        df["keltner_upper"] = kc.keltner_channel_hband()
        df["keltner_lower"] = kc.keltner_channel_lband()
        df["keltner_mid"] = kc.keltner_channel_mband()
        
        atr_percentile = df["atr_pct"].rank(pct=True) * 100
        df["volatility_regime"] = np.where(atr_percentile > 75, "HIGH", np.where(atr_percentile < 25, "LOW", "NORMAL"))
        
        return df
    
    def _compute_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
        df["obv"] = obv.on_balance_volume()
        df["obv_change"] = df["obv"].pct_change(10)
        
        df["volume_sma20"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma20"]
        
        mfi_vals = money_flow_index(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14)
        df["mfi"] = mfi_vals
        
        return df
    
    def _compute_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        df["trend_direction"] = np.where(
            (df["Close"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]), 1,
            np.where((df["Close"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]), -1, 0))
        
        df["support_level"] = df["Low"].rolling(window=20).min()
        df["resistance_level"] = df["High"].rolling(window=20).max()
        
        df["price_vs_support_pct"] = (df["Close"] - df["support_level"]) / df["support_level"] * 100
        df["price_vs_resistance_pct"] = (df["resistance_level"] - df["Close"]) / df["resistance_level"] * 100
        
        df["breakout_up"] = ((df["Close"] > df["bb_upper"]) & (df["volume_ratio"] > 1.5)).astype(int)
        df["breakout_down"] = ((df["Close"] < df["bb_lower"]) & (df["volume_ratio"] > 1.5)).astype(int)
        
        df["higher_highs"] = ((df["High"] > df["High"].shift(1)) & (df["High"].shift(1) > df["High"].shift(2))).astype(int)
        df["lower_lows"] = ((df["Low"] < df["Low"].shift(1)) & (df["Low"].shift(1) < df["Low"].shift(2))).astype(int)
        
        return df
    
    def _compute_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        bull_score = np.zeros(len(df))
        bull_score += np.where(df["rsi"] < 40, 15, 0)
        bull_score += np.where(df["macd_hist"] > 0, 10, 0)
        bull_score += np.where(df["Close"] > df["ema_20"], 10, 0)
        bull_score += np.where(df["Close"] > df["ema_50"], 15, 0)
        bull_score += np.where(df["Close"] > df["ema_200"], 20, 0)
        bull_score += np.where(df["adx"] > 25, 10, 0)
        bull_score += np.where(df["breakout_up"] == 1, 20, 0)
        df["bull_score"] = np.clip(bull_score, 0, 100)
        
        bear_score = np.zeros(len(df))
        bear_score += np.where(df["rsi"] > 60, 15, 0)
        bear_score += np.where(df["macd_hist"] < 0, 10, 0)
        bear_score += np.where(df["Close"] < df["ema_20"], 10, 0)
        bear_score += np.where(df["Close"] < df["ema_50"], 15, 0)
        bear_score += np.where(df["Close"] < df["ema_200"], 20, 0)
        bear_score += np.where(df["adx"] > 25, 10, 0)
        bear_score += np.where(df["breakout_down"] == 1, 20, 0)
        df["bear_score"] = np.clip(bear_score, 0, 100)
        
        df["technical_signal"] = (df["bull_score"] - df["bear_score"]) / 100
        df["technical_signal_norm"] = (df["technical_signal"] + 1) / 2
        
        return df
