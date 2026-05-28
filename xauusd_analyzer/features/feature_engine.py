"""
Feature Engine Module
Computes technical indicators and market structure features for XAU/USD
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_SHORT, EMA_MEDIUM, EMA_LONG, BB_PERIOD, BB_STD,
    ATR_PERIOD, MOMENTUM_PERIOD, STOCH_K, STOCH_D, ADX_PERIOD,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Computes all technical indicators and market structure features.
    Uses the 'ta' library for standard indicators and custom calculations
    for market structure analysis.
    """
    
    def __init__(self):
        logger.info("FeatureEngine initialized")
        
        # Store indicator parameters
        self.rsi_period = RSI_PERIOD
        self.macd_fast = MACD_FAST
        self.macd_slow = MACD_SLOW
        self.macd_signal = MACD_SIGNAL
        self.ema_short = EMA_SHORT
        self.ema_medium = EMA_MEDIUM
        self.ema_long = EMA_LONG
        self.bb_period = BB_PERIOD
        self.bb_std = BB_STD
        self.atr_period = ATR_PERIOD
        self.momentum_period = MOMENTUM_PERIOD
        self.stoch_k = STOCH_K
        self.stoch_d = STOCH_D
        self.adx_period = ADX_PERIOD
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with all features added
        """
        logger.info("Computing all technical indicators...")
        
        df_features = df.copy()
        
        # Momentum indicators
        df_features = self._compute_rsi(df_features)
        df_features = self._compute_macd(df_features)
        df_features = self._compute_stochastic(df_features)
        df_features = self._compute_momentum(df_features)
        
        # Trend indicators
        df_features = self._compute_ema(df_features)
        df_features = self._compute_adx(df_features)
        
        # Volatility indicators
        df_features = self._compute_bollinger_bands(df_features)
        df_features = self._compute_atr(df_features)
        
        # Market structure features
        df_features = self._compute_market_structure(df_features)
        
        # Drop any remaining NaN rows from indicator calculations
        df_features = df_features.dropna()
        
        logger.info(f"Feature computation complete. Total features: {len(df_features.columns)}")
        
        return df_features
    
    def _compute_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Relative Strength Index (RSI).
        
        RSI measures the speed and magnitude of recent price changes
        to evaluate overbought or oversold conditions.
        """
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        logger.debug(f"Computed RSI({self.rsi_period})")
        
        return df
    
    def _compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MACD (Moving Average Convergence Divergence).
        
        MACD shows the relationship between two EMAs of price.
        """
        ema_fast = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        logger.debug(f"Computed MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})")
        
        return df
    
    def _compute_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Exponential Moving Averages (EMA).
        
        EMAs give more weight to recent prices.
        """
        df[f'EMA_{self.ema_short}'] = df['Close'].ewm(span=self.ema_short, adjust=False).mean()
        df[f'EMA_{self.ema_medium}'] = df['Close'].ewm(span=self.ema_medium, adjust=False).mean()
        df[f'EMA_{self.ema_long}'] = df['Close'].ewm(span=self.ema_long, adjust=False).mean()
        
        logger.debug(f"Computed EMA({self.ema_short}, {self.ema_medium}, {self.ema_long})")
        
        return df
    
    def _compute_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Bands.
        
        Bollinger Bands consist of a middle band (SMA) and two outer bands
        at standard deviation levels.
        """
        sma = df['Close'].rolling(window=self.bb_period).mean()
        std = df['Close'].rolling(window=self.bb_period).std()
        
        df['BB_Upper'] = sma + (self.bb_std * std)
        df['BB_Lower'] = sma - (self.bb_std * std)
        df['BB_Middle'] = sma
        
        # Bandwidth: (Upper - Lower) / Middle
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # %B: Where price is relative to bands
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        logger.debug(f"Computed Bollinger Bands({self.bb_period},{self.bb_std})")
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Average True Range (ATR).
        
        ATR measures market volatility.
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        df['ATR'] = true_range.rolling(self.atr_period).mean()
        
        # ATR as percentage of price
        df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
        
        logger.debug(f"Computed ATR({self.atr_period})")
        
        return df
    
    def _compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Momentum indicator.
        
        Momentum measures the rate of rise or fall in security prices.
        """
        df['Momentum'] = df['Close'] - df['Close'].shift(self.momentum_period)
        df['Momentum_Rate'] = (df['Close'] / df['Close'].shift(self.momentum_period) - 1) * 100
        
        logger.debug(f"Computed Momentum({self.momentum_period})")
        
        return df
    
    def _compute_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Stochastic Oscillator.
        
        Stochastic compares a particular closing price to a range of prices
        over a certain period.
        """
        low_min = df['Low'].rolling(window=self.stoch_k).min()
        high_max = df['High'].rolling(window=self.stoch_k).max()
        
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=self.stoch_d).mean()
        
        logger.debug(f"Computed Stochastic({self.stoch_k},{self.stoch_d})")
        
        return df
    
    def _compute_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Average Directional Index (ADX).
        
        ADX quantifies trend strength regardless of direction.
        """
        # Calculate +DM and -DM
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Smooth with EMA
        tr_smooth = pd.Series(tr).ewm(span=self.adx_period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=self.adx_period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=self.adx_period, adjust=False).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # Calculate DX
        di_sum = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
        
        # Calculate ADX
        df['ADX'] = dx.ewm(span=self.adx_period, adjust=False).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        logger.debug(f"Computed ADX({self.adx_period})")
        
        return df
    
    def _compute_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market structure features.
        
        These are derived features that describe the current market state.
        """
        # Trend strength based on ADX
        df['trend_strength'] = df['ADX'].apply(self._classify_trend_strength)
        
        # Volatility regime based on ATR percentile
        df['volatility_regime'] = df['ATR_Percent'].apply(self._classify_volatility_regime)
        
        # Breakout detection
        df['breakout_detected'] = self._detect_breakout(df)
        
        # Price distance from EMA200
        df['price_vs_ema200'] = ((df['Close'] - df[f'EMA_{self.ema_long}']) / 
                                  df[f'EMA_{self.ema_long}']) * 100
        
        # EMA alignment (bullish/bearish stack)
        df['ema_alignment'] = df.apply(self._check_ema_alignment, axis=1)
        
        # Volume relative to average
        avg_volume = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / avg_volume
        
        logger.debug("Computed market structure features")
        
        return df
    
    def _classify_trend_strength(self, adx: float) -> str:
        """Classify trend strength based on ADX value."""
        if pd.isna(adx):
            return 'UNKNOWN'
        elif adx >= 40:
            return 'VERY_STRONG'
        elif adx >= 25:
            return 'STRONG'
        elif adx >= 20:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def _classify_volatility_regime(self, atr_pct: float) -> str:
        """Classify volatility regime based on ATR percentage."""
        if pd.isna(atr_pct):
            return 'UNKNOWN'
        elif atr_pct >= 2.0:
            return 'HIGH'
        elif atr_pct >= 1.0:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _detect_breakout(self, df: pd.DataFrame) -> str:
        """
        Detect breakouts above/below Bollinger Bands with volume confirmation.
        """
        breakout = []
        
        for i in range(len(df)):
            if i == 0:
                breakout.append('NONE')
                continue
            
            close = df['Close'].iloc[i]
            upper = df['BB_Upper'].iloc[i] if not pd.isna(df['BB_Upper'].iloc[i]) else np.inf
            lower = df['BB_Lower'].iloc[i] if not pd.isna(df['BB_Lower'].iloc[i]) else -np.inf
            vol_ratio = df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1.0
            
            if close > upper and vol_ratio > 1.2:
                breakout.append('BULLISH')
            elif close < lower and vol_ratio > 1.2:
                breakout.append('BEARISH')
            else:
                breakout.append('NONE')
        
        return breakout
    
    def _check_ema_alignment(self, row: pd.Series) -> str:
        """Check if EMAs are aligned bullishly or bearishly."""
        ema_short = row.get(f'EMA_{self.ema_short}', np.nan)
        ema_medium = row.get(f'EMA_{self.ema_medium}', np.nan)
        ema_long = row.get(f'EMA_{self.ema_long}', np.nan)
        close = row['Close']
        
        if pd.isna(ema_short) or pd.isna(ema_medium) or pd.isna(ema_long):
            return 'NEUTRAL'
        
        if close > ema_short > ema_medium > ema_long:
            return 'BULLISH_STACK'
        elif close < ema_short < ema_medium < ema_long:
            return 'BEARISH_STACK'
        else:
            return 'MIXED'
    
    def get_feature_names(self) -> list:
        """Return list of all computed feature names."""
        return [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            f'EMA_{self.ema_short}', f'EMA_{self.ema_medium}', f'EMA_{self.ema_long}',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'BB_Percent',
            'ATR', 'ATR_Percent', 'Momentum', 'Momentum_Rate',
            'Stoch_K', 'Stoch_D', 'ADX', 'Plus_DI', 'Minus_DI',
            'trend_strength', 'volatility_regime', 'breakout_detected',
            'price_vs_ema200', 'ema_alignment', 'volume_ratio'
        ]
