"""
ML Model Module
Machine learning model for XAU/USD price direction prediction
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import joblib
import os

# Try to import XGBoost, fallback to sklearn
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, using RandomForestClassifier fallback")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ML_TRAIN_RATIO, ML_CV_SPLITS, ML_RANDOM_STATE, 
    ML_N_ESTIMATORS, ML_MAX_DEPTH, ML_LEARNING_RATE,
    MODEL_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class MLModel:
    """
    Machine Learning model for predicting XAU/USD price direction.
    Uses XGBoost if available, otherwise falls back to RandomForest.
    """
    
    def __init__(self):
        logger.info(f"Initializing MLModel (XGBoost available: {XGBOOST_AVAILABLE})")
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
        # Define feature columns for ML
        self.feature_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'EMA_20', 'EMA_50', 'EMA_200',
            'BB_Width', 'BB_Percent',
            'ATR', 'ATR_Percent', 'Momentum', 'Momentum_Rate',
            'Stoch_K', 'Stoch_D', 'ADX', 'Plus_DI', 'Minus_DI',
            'price_vs_ema200', 'volume_ratio'
        ]
        
        # Encode categorical features
        self.trend_strength_map = {
            'UNKNOWN': 0, 'WEAK': 1, 'MODERATE': 2, 
            'STRONG': 3, 'VERY_STRONG': 4
        }
        
        self.volatility_regime_map = {
            'UNKNOWN': 0, 'LOW': 1, 'MODERATE': 2, 'HIGH': 3
        }
        
        self.breakout_map = {
            'NONE': 0, 'BULLISH': 1, 'BEARISH': -1
        }
        
        self.ema_alignment_map = {
            'NEUTRAL': 0, 'MIXED': 1, 'BEARISH_STACK': -1, 'BULLISH_STACK': 2
        }
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model.
        
        Args:
            df: DataFrame with all computed features
            
        Returns:
            pd.DataFrame: Prepared feature matrix
        """
        logger.info("Preparing features for ML model...")
        
        df_ml = df.copy()
        
        # Encode categorical features
        df_ml['trend_strength_encoded'] = df_ml['trend_strength'].map(self.trend_strength_map).fillna(0)
        df_ml['volatility_regime_encoded'] = df_ml['volatility_regime'].map(self.volatility_regime_map).fillna(0)
        df_ml['breakout_encoded'] = df_ml['breakout_detected'].map(self.breakout_map).fillna(0)
        df_ml['ema_alignment_encoded'] = df_ml['ema_alignment'].map(self.ema_alignment_map).fillna(0)
        
        # Build feature set
        ml_features = self.feature_columns + [
            'trend_strength_encoded', 'volatility_regime_encoded',
            'breakout_encoded', 'ema_alignment_encoded'
        ]
        
        # Check which features are available
        available_features = [f for f in ml_features if f in df_ml.columns]
        missing_features = set(ml_features) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Extract and clean features
        X = df_ml[available_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"Prepared {X.shape[1]} features from {X.shape[0]} samples")
        
        return X
    
    def create_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        Create target variable: 1 if price goes up, 0 otherwise.
        
        Args:
            df: DataFrame with Close prices
            horizon: Prediction horizon in days
            
        Returns:
            pd.Series: Target variable
        """
        logger.info(f"Creating target variable (horizon={horizon} day(s))")
        
        # Shift close price to get future returns
        future_close = df['Close'].shift(-horizon)
        y = (future_close > df['Close']).astype(int)
        
        # Remove last row (no future data)
        y = y.iloc[:-horizon]
        
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        logger.info(f"Target distribution: UP={pos_count}, DOWN={neg_count}, "
                   f"ratio={pos_count/len(y)*100:.1f}%")
        
        return y
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the ML model using time series cross-validation.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            dict: Training metrics
        """
        logger.info("Starting model training...")
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = self.create_target(df)
        
        # Align X and y (remove NaN from target)
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Ensure minimum samples
        if len(X) < 100:
            logger.error(f"Insufficient samples for training: {len(X)}")
            raise ValueError("Insufficient samples for training")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=ML_N_ESTIMATORS,
                max_depth=ML_MAX_DEPTH,
                learning_rate=ML_LEARNING_RATE,
                random_state=ML_RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            logger.info("Using XGBoost classifier")
        else:
            self.model = RandomForestClassifier(
                n_estimators=ML_N_ESTIMATORS,
                max_depth=ML_MAX_DEPTH,
                random_state=ML_RANDOM_STATE,
                n_jobs=-1
            )
            logger.info("Using RandomForestClassifier (XGBoost not available)")
        
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=ML_CV_SPLITS)
        
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        logger.info(f"Running {ML_CV_SPLITS}-fold time series cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Predict
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            cv_metrics['accuracy'].append(accuracy)
            cv_metrics['precision'].append(precision)
            cv_metrics['recall'].append(recall)
            cv_metrics['f1'].append(f1)
            
            logger.info(f"Fold {fold}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
                       f"Recall={recall:.4f}, F1={f1:.4f}")
        
        # Train final model on full dataset
        logger.info("Training final model on full dataset...")
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        # Compute average metrics
        avg_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
        logger.info(f"Average CV Metrics: {avg_metrics}")
        
        return avg_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of upward price movement.
        
        Args:
            df: DataFrame with features
            
        Returns:
            np.ndarray: Probability predictions (0.0 - 1.0)
        """
        if not self.is_trained:
            logger.warning("Model not trained. Attempting to load saved model...")
            if not self.load_model():
                raise ValueError("Model not trained and no saved model found")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Get probability of class 1 (upward movement)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        
        logger.info(f"Generated {len(proba)} predictions")
        
        return proba
    
    def predict_single(self, df: pd.DataFrame) -> float:
        """
        Predict probability for the latest data point.
        
        Args:
            df: DataFrame with features (should have at least one row)
            
        Returns:
            float: Probability of upward movement
        """
        proba = self.predict(df)
        return proba[-1] if len(proba) > 0 else 0.5
    
    def save_model(self, filepath: Optional[str] = None) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save model (default: MODEL_FILE from config)
            
        Returns:
            bool: Success status
        """
        if filepath is None:
            filepath = MODEL_FILE
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: Optional[str] = None) -> bool:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            bool: Success status
        """
        if filepath is None:
            filepath = MODEL_FILE
        
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data.get('is_trained', True)
            
            logger.info(f"Model loaded from {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        if not self.is_trained:
            logger.warning("Model not trained. Cannot compute feature importance.")
            return pd.DataFrame()
        
        try:
            # Get feature names used in training
            X = self.prepare_features(pd.DataFrame(columns=['Close']))
            feature_names = X.columns.tolist()
            
            # Get importance scores
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                logger.warning("Model does not support feature importance")
                return pd.DataFrame()
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            logger.info(f"Top {top_n} features:\n{importance_df.head(top_n)}")
            
            return importance_df.head(top_n)
            
        except Exception as e:
            logger.error(f"Error computing feature importance: {str(e)}")
            return pd.DataFrame()
