"""
═══════════════════════════════════════════════════════════════════
DATA ENGINE - MULTI-TIMEFRAME INSTITUTIONAL DATA LOADER
═══════════════════════════════════════════════════════════════════
Handles data acquisition, cleaning, validation, and session tagging.
Production-grade with outlier detection and data quality checks.
═══════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import json

from config import (
    SYMBOL, TIMEFRAMES, CACHE_DIR, LOGS_DIR,
    MAX_ATR_PERCENTILE, MIN_VOLUME_RATIO
)

logger = logging.getLogger(__name__)


class DataError(Exception):
    """Custom exception for data-related errors."""
    pass


class DataEngine:
    """
    Institutional-grade data engine for loading and preprocessing
    multi-timeframe market data with comprehensive validation.
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.DataEngine")
        
    def load_multi_timeframe(
        self, 
        timeframes: Dict = TIMEFRAMES
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple timeframes simultaneously.
        
        Args:
            timeframes: Dict with timeframe configs
            
        Returns:
            Dict mapping timeframe label to DataFrame
        """
        self.logger.info("Loading multi-timeframe data...")
        results = {}
        
        for label, config in timeframes.items():
            try:
                df = self.load_data(
                    symbol=SYMBOL,
                    period=config["period"],
                    interval=config["interval"]
                )
                df = self.clean_data(df)
                df = self.add_session_labels(df)
                df["timeframe"] = label
                results[label] = df
                
                self.logger.info(
                    f"Loaded {label}: {len(df)} candles, "
                    f"{config['period']} period"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to load {label}: {e}")
                raise DataError(f"Multi-timeframe load failed for {label}: {e}")
        
        if not results:
            raise DataError("No timeframes loaded successfully")
            
        return results
    
    def load_data(
        self, 
        symbol: str = SYMBOL,
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download historical data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol
            period: Time period (e.g., "2y", "60d")
            interval: Candle interval (e.g., "1d", "1h")
            
        Returns:
            Clean DataFrame with OHLCV data
        """
        self.logger.info(
            f"Downloading {symbol} [{interval}] for {period}..."
        )
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.download(
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            
            if df.empty:
                raise DataError(f"No data returned for {symbol}")
            
            # Standardize column names
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Ensure required columns exist
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing columns: {missing_cols}")
                for col in missing_cols:
                    if col == "Volume":
                        df[col] = 0
                    else:
                        df[col] = df["Close"]
            
            # Rename to standard format
            df = df.rename(columns={
                "Open": "Open",
                "High": "High", 
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            })
            
            # Validate minimum data requirements
            if len(df) < 50:
                raise DataError(
                    f"Insufficient data: {len(df)} rows (min: 50)"
                )
            
            self.logger.info(
                f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            raise DataError(f"Data download error: {e}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline.
        
        Steps:
        1. Remove duplicate indices
        2. Forward-fill then drop NaN
        3. Remove outliers using IQR method
        4. Ensure chronological order
        5. Validate price relationships
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning data...")
        df = df.copy()
        initial_rows = len(df)
        
        # 1. Remove duplicate indices
        df = df[~df.index.duplicated(keep="first")]
        self.logger.debug(f"After dedup: {len(df)} rows")
        
        # 2. Handle NaN values
        df = df.ffill()
        df = df.bfill()
        df = df.dropna()
        self.logger.debug(f"After NaN handling: {len(df)} rows")
        
        # 3. Remove outliers using IQR method on Close
        df = self._remove_outliers_iqr(df, column="Close", multiplier=3.0)
        self.logger.debug(f"After outlier removal: {len(df)} rows")
        
        # 4. Ensure chronological order
        df = df.sort_index()
        
        # 5. Validate price relationships
        df = self._validate_price_relationships(df)
        
        # 6. Reset volume outliers
        df = self._cap_volume_outliers(df)
        
        removed_rows = initial_rows - len(df)
        self.logger.info(
            f"Cleaning complete. Removed {removed_rows} rows "
            f"({removed_rows/initial_rows*100:.2f}%)"
        )
        
        if len(df) < 100:
            raise DataError(
                f"Too few rows after cleaning: {len(df)} (min: 100)"
            )
        
        return df
    
    def _remove_outliers_iqr(
        self, 
        df: pd.DataFrame, 
        column: str, 
        multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using Interquartile Range method.
        
        Args:
            df: DataFrame
            column: Column to check for outliers
            multiplier: IQR multiplier for fence calculation
            
        Returns:
            DataFrame with outliers removed
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_fence = Q1 - multiplier * IQR
        upper_fence = Q3 + multiplier * IQR
        
        mask = (df[column] >= lower_fence) & (df[column] <= upper_fence)
        
        return df[mask]
    
    def _validate_price_relationships(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Validate and fix invalid price relationships.
        Ensures: High >= Low, High >= Open, High >= Close,
                 Low <= Open, Low <= Close
        """
        # Fix High < Low
        invalid_hl = df["High"] < df["Low"]
        if invalid_hl.any():
            self.logger.warning(f"Fixing {invalid_hl.sum()} High<Low violations")
            max_price = df.loc[invalid_hl, ["Open", "Close"]].max(axis=1)
            min_price = df.loc[invalid_hl, ["Open", "Close"]].min(axis=1)
            df.loc[invalid_hl, "High"] = max_price
            df.loc[invalid_hl, "Low"] = min_price
        
        # Fix High < Open or High < Close
        df["High"] = df[["High", "Open", "Close"]].max(axis=1)
        
        # Fix Low > Open or Low > Close
        df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
        
        return df
    
    def _cap_volume_outliers(
        self, 
        df: pd.DataFrame, 
        percentile: float = 99.5
    ) -> pd.DataFrame:
        """
        Cap extreme volume spikes to prevent distortion.
        
        Args:
            df: DataFrame
            percentile: Percentile to cap volume at
            
        Returns:
            DataFrame with capped volume
        """
        volume_cap = df["Volume"].quantile(percentile / 100)
        df["Volume"] = df["Volume"].clip(upper=volume_cap)
        return df
    
    def add_session_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trading session labels based on timestamp.
        
        Sessions:
        - Asian: 00:00-08:00 UTC
        - London: 08:00-16:00 UTC  
        - NY: 13:00-22:00 UTC
        - Overlap: London/NY overlap 13:00-16:00 UTC
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with 'session' column
        """
        df = df.copy()
        
        # Get hour in UTC
        hours = df.index.hour
        
        # Define sessions
        conditions = [
            (hours >= 0) & (hours < 8),      # Asian
            (hours >= 8) & (hours < 13),     # London only
            (hours >= 13) & (hours < 16),    # London/NY overlap
            (hours >= 16) & (hours < 22),    # NY only
            (hours >= 22) | (hours < 0),     # Afternoon/Early
        ]
        
        choices = ["Asian", "London", "Overlap", "NY", "Off-Hours"]
        
        df["session"] = np.select(conditions, choices, default="Unknown")
        
        return df
    
    def get_market_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alias for add_session_labels for backward compatibility.
        """
        return self.add_session_labels(df)
    
    def calculate_returns(
        self, 
        df: pd.DataFrame, 
        periods: list = [1, 5, 21]
    ) -> pd.DataFrame:
        """
        Calculate log returns for multiple periods.
        
        Args:
            df: DataFrame with Close column
            periods: List of lookback periods
            
        Returns:
            DataFrame with added return columns
        """
        df = df.copy()
        
        for period in periods:
            df[f"return_{period}d"] = np.log(
                df["Close"] / df["Close"].shift(period)
            )
        
        return df
    
    def save_to_cache(
        self, 
        df: pd.DataFrame, 
        filename: str
    ) -> None:
        """
        Save DataFrame to parquet cache.
        
        Args:
            df: DataFrame to save
            filename: Filename without extension
        """
        filepath = self.cache_dir / f"{filename}.parquet"
        df.to_parquet(filepath)
        self.logger.info(f"Saved cache: {filepath}")
    
    def load_from_cache(
        self, 
        filename: str
    ) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from parquet cache.
        
        Args:
            filename: Filename without extension
            
        Returns:
            DataFrame or None if not found
        """
        filepath = self.cache_dir / f"{filename}.parquet"
        
        if filepath.exists():
            df = pd.read_parquet(filepath)
            self.logger.info(f"Loaded cache: {filepath}")
            return df
        
        self.logger.debug(f"Cache not found: {filepath}")
        return None
    
    def validate_data_quality(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Comprehensive data quality validation report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with quality metrics
        """
        quality_report = {
            "total_rows": len(df),
            "date_range": (str(df.index[0]), str(df.index[-1])),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "negative_volume": int((df["Volume"] < 0).sum()),
            "zero_volume": int((df["Volume"] == 0).sum()),
            "price_spikes": self._detect_price_spikes(df),
            "gap_percentage": self._calculate_gap_percentage(df),
        }
        
        # Quality score (0-100)
        score = 100
        if quality_report["duplicate_rows"] > 0:
            score -= 10
        if quality_report["missing_values"]["Close"] > 0:
            score -= 20
        if quality_report["negative_volume"] > 0:
            score -= 15
        if quality_report["price_spikes"] > 5:
            score -= 10
            
        quality_report["quality_score"] = max(0, score)
        
        return quality_report
    
    def _detect_price_spikes(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.05
    ) -> int:
        """Detect abnormal price moves (>threshold% in one candle)."""
        daily_returns = df["Close"].pct_change().abs()
        return int((daily_returns > threshold).sum())
    
    def _calculate_gap_percentage(
        self, 
        df: pd.DataFrame
    ) -> float:
        """Calculate percentage of gap days (High-Low)/Close > 2%."""
        ranges = (df["High"] - df["Low"]) / df["Close"]
        return float((ranges > 0.02).mean() * 100)
