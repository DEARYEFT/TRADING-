"""
Data Engine Module
Handles data loading, cleaning, and preprocessing for XAU/USD
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional
import yfinance as yf

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SYMBOL, PERIOD, INTERVAL, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class DataEngine:
    """
    Handles all data-related operations for XAU/USD analysis.
    Loads data from yfinance, cleans it, and prepares for analysis.
    """
    
    def __init__(self):
        self.symbol = SYMBOL
        self.period = PERIOD
        self.interval = INTERVAL
        logger.info(f"DataEngine initialized for {self.symbol}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Download historical data from yfinance.
        
        Returns:
            pd.DataFrame: Raw OHLCV data
        """
        try:
            logger.info(f"Downloading {self.period} of {self.symbol} data...")
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=self.period, interval=self.interval)
            
            if df.empty:
                logger.error("No data downloaded from yfinance")
                raise ValueError("Empty dataset received from yfinance")
            
            logger.info(f"Downloaded {len(df)} rows of data")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling NaN values, duplicates, and outliers.
        
        Args:
            df: Raw DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Starting data cleaning process...")
        df_clean = df.copy()
        
        initial_rows = len(df_clean)
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        dupes_removed = initial_rows - len(df_clean)
        if dupes_removed > 0:
            logger.info(f"Removed {dupes_removed} duplicate rows")
        
        # Handle missing values
        null_counts_before = df_clean.isnull().sum().sum()
        
        # Forward fill then backward fill for any remaining NaNs
        df_clean = df_clean.ffill()
        df_clean = df_clean.bfill()
        
        null_counts_after = df_clean.isnull().sum().sum()
        if null_counts_before > null_counts_after:
            logger.info(f"Filled {null_counts_before - null_counts_after} missing values")
        
        # Remove outliers using IQR method
        df_clean = self._remove_outliers_iqr(df_clean)
        
        # Validate minimum data requirements
        if len(df_clean) < 100:
            logger.warning(f"Only {len(df_clean)} rows after cleaning. Minimum 100 required.")
            raise ValueError("Insufficient data after cleaning")
        
        # Ensure proper column names (yfinance may use different casing)
        df_clean = self._standardize_columns(df_clean)
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        
        return df_clean
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using Interquartile Range (IQR) method.
        
        Args:
            df: DataFrame to clean
            multiplier: IQR multiplier (default 3.0 for extreme outliers)
            
        Returns:
            pd.DataFrame: DataFrame with outliers capped
        """
        df_out = df.copy()
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in numeric_cols if col in df_out.columns]
        
        for col in available_cols:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Cap outliers instead of removing to maintain time series continuity
            outliers_below = (df_out[col] < lower_bound).sum()
            outliers_above = (df_out[col] > upper_bound).sum()
            
            df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            
            if outliers_below > 0 or outliers_above > 0:
                logger.info(f"Capped {outliers_below + outliers_above} outliers in {col}")
        
        return df_out
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to expected format.
        
        Args:
            df: DataFrame with potentially inconsistent column names
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        # Map common yfinance column variations to standard names
        column_mapping = {
            'Adj Close': 'Adj_Close',
            'Dividends': 'Dividends',
            'Stock Splits': 'Stock_Splits'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Normalize specified columns using MinMax scaling.
        
        Args:
            df: DataFrame to normalize
            columns: List of columns to normalize (default: OHLCV)
            
        Returns:
            pd.DataFrame: DataFrame with normalized columns
        """
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        df_norm = df.copy()
        
        for col in columns:
            if col in df_norm.columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                
                if max_val - min_val > 0:
                    df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[f'{col}_norm'] = 0.5
        
        logger.info(f"Normalized columns: {columns}")
        
        return df_norm
    
    def get_atr_percentile(self, df: pd.DataFrame, atr_col: str = 'ATR', 
                           window: int = 252) -> float:
        """
        Calculate current ATR percentile over rolling window.
        
        Args:
            df: DataFrame with ATR column
            atr_col: Name of ATR column
            window: Lookback window in days
            
        Returns:
            float: Current ATR percentile (0-100)
        """
        if atr_col not in df.columns:
            logger.warning(f"Column {atr_col} not found in DataFrame")
            return 50.0
        
        recent_atr = df[atr_col].iloc[-1]
        historical_atr = df[atr_col].iloc[-window:]
        
        percentile = (historical_atr < recent_atr).sum() / len(historical_atr) * 100
        
        return percentile
