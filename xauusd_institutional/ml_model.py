"""
═══════════════════════════════════════════════════════════════════
ML MODEL - ENSEMBLE AI PREDICTION SYSTEM
═══════════════════════════════════════════════════════════════════
Multi-model ensemble with XGBoost, LightGBM, CatBoost, RandomForest,
ExtraTrees, and HistGradientBoosting with probability calibration.
═══════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from config import ML_CONFIG, CV_SPLITS, RETRAIN_DAYS, MODEL_PATH, CACHE_DIR

logger = logging.getLogger(__name__)


class MLModel:
    """Institutional-grade ensemble ML model for price direction prediction."""
    
    FEATURE_COLUMNS = [
        "rsi", "macd_hist", "ema_cross_signal", "adx",
        "bb_pct", "bb_width", "atr_pct", "momentum_10",
        "stoch_k", "stoch_d", "cci", "roc", "williams_r",
        "volume_ratio", "mfi", "bull_score", "bear_score",
        "price_vs_support_pct", "price_vs_resistance_pct",
        "obv_change", "trend_direction", "aroon_oscillator",
        "ichimoku_conversion", "ichimoku_base"
    ]
    
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.MLModel")
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importances_ = {}
        self.model_weights = {
            "xgboost": 0.25, "lightgbm": 0.20, "catboost": 0.20,
            "random_forest": 0.15, "extra_trees": 0.10, "hist_gradient_boost": 0.10,
        }
        
    def create_target(self, df: pd.DataFrame, forward: int = 1) -> pd.Series:
        target = (df["Close"].shift(-forward) > df["Close"]).astype(int)
        return target
    
    def prepare_features(self, df: pd.DataFrame, forward: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        available_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        X = df[available_cols].copy()
        y = self.create_target(df, forward)
        
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        finite_mask = np.isfinite(X).all(axis=1)
        X = X[finite_mask]
        y = y[finite_mask]
        
        return X, y
    
    def train(self, df: pd.DataFrame, forward: int = 1) -> Dict[str, float]:
        self.logger.info("Training ensemble ML models...")
        
        X, y = self.prepare_features(df, forward)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data for training: {len(X)} rows")
        
        self.logger.info(f"Training on {len(X)} samples, {len(X.columns)} features")
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        cv_results = {}
        
        self._initialize_models()
        
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train = X_scaled.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X_scaled.iloc[val_idx]
                y_val = y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                fold_scores.append(acc)
                self.logger.debug(f"  Fold {fold+1}: accuracy={acc:.4f}")
            
            mean_acc = np.mean(fold_scores)
            std_acc = np.std(fold_scores)
            cv_results[model_name] = mean_acc
            
            self.logger.info(f"  {model_name}: CV accuracy = {mean_acc:.4f} ± {std_acc:.4f}")
            model.fit(X_scaled, y)
        
        self._compute_feature_importances(X.columns)
        self.save_model()
        
        self.logger.info("Ensemble training complete")
        return cv_results
    
    def _initialize_models(self):
        try:
            import xgboost as xgb
            self.models["xgboost"] = xgb.XGBClassifier(
                n_estimators=ML_CONFIG["xgboost"]["n_estimators"],
                max_depth=ML_CONFIG["xgboost"]["max_depth"],
                learning_rate=ML_CONFIG["xgboost"]["learning_rate"],
                subsample=ML_CONFIG["xgboost"]["subsample"],
                colsample_bytree=ML_CONFIG["xgboost"]["colsample_bytree"],
                min_child_weight=ML_CONFIG["xgboost"]["min_child_weight"],
                gamma=ML_CONFIG["xgboost"]["gamma"],
                reg_alpha=ML_CONFIG["xgboost"]["reg_alpha"],
                reg_lambda=ML_CONFIG["xgboost"]["reg_lambda"],
                random_state=ML_CONFIG["xgboost"]["random_state"],
                use_label_encoder=False,
                eval_metric="logloss",
            )
        except ImportError:
            self.logger.warning("XGBoost not available, skipping")
        
        try:
            import lightgbm as lgb
            self.models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=ML_CONFIG["lightgbm"]["n_estimators"],
                max_depth=ML_CONFIG["lightgbm"]["max_depth"],
                learning_rate=ML_CONFIG["lightgbm"]["learning_rate"],
                num_leaves=ML_CONFIG["lightgbm"]["num_leaves"],
                min_child_samples=ML_CONFIG["lightgbm"]["min_child_samples"],
                subsample=ML_CONFIG["lightgbm"]["subsample"],
                colsample_bytree=ML_CONFIG["lightgbm"]["colsample_bytree"],
                random_state=ML_CONFIG["lightgbm"]["random_state"],
                verbose=-1,
            )
        except ImportError:
            self.logger.warning("LightGBM not available, skipping")
        
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
        
        self.models["random_forest"] = RandomForestClassifier(
            n_estimators=ML_CONFIG["random_forest"]["n_estimators"],
            max_depth=ML_CONFIG["random_forest"]["max_depth"],
            min_samples_leaf=ML_CONFIG["random_forest"]["min_samples_leaf"],
            random_state=ML_CONFIG["random_forest"]["random_state"],
            n_jobs=-1,
        )
        
        self.models["extra_trees"] = ExtraTreesClassifier(
            n_estimators=ML_CONFIG["extra_trees"]["n_estimators"],
            max_depth=ML_CONFIG["extra_trees"]["max_depth"],
            min_samples_leaf=ML_CONFIG["extra_trees"]["min_samples_leaf"],
            random_state=ML_CONFIG["extra_trees"]["random_state"],
            n_jobs=-1,
        )
        
        self.models["hist_gradient_boost"] = HistGradientBoostingClassifier(
            max_iter=ML_CONFIG["hist_gradient_boost"]["max_iter"],
            max_depth=ML_CONFIG["hist_gradient_boost"]["max_depth"],
            learning_rate=ML_CONFIG["hist_gradient_boost"]["learning_rate"],
            random_state=ML_CONFIG["hist_gradient_boost"]["random_state"],
        )
    
    def _compute_feature_importances(self, feature_names: pd.Index):
        importances = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, "feature_importances_"):
                    importances[name] = model.feature_importances_
            except Exception as e:
                self.logger.debug(f"Could not get importance from {name}: {e}")
        
        if importances:
            weighted_imp = np.zeros(len(feature_names))
            total_weight = 0
            
            for name, imp in importances.items():
                if len(imp) == len(feature_names):
                    weight = self.model_weights.get(name, 0.1)
                    weighted_imp += imp * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_imp /= total_weight
            
            self.feature_importances_ = dict(zip(feature_names, weighted_imp))
    
    def predict(self, df: pd.DataFrame, forward: int = 1) -> np.ndarray:
        X, _ = self.prepare_features(df, forward)
        
        if len(X) == 0:
            raise ValueError("No valid samples for prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        all_probas = []
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_scaled)[:, 1]
                weight = self.model_weights.get(name, 0.1)
                all_probas.append((proba, weight))
            except Exception as e:
                self.logger.warning(f"Prediction failed for {name}: {e}")
        
        if not all_probas:
            raise ValueError("No models produced predictions")
        
        weighted_proba = np.zeros(len(X))
        total_weight = 0
        
        for proba, weight in all_probas:
            weighted_proba += proba * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_proba /= total_weight
        
        return weighted_proba
    
    def predict_with_uncertainty(self, df: pd.DataFrame, forward: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        X, _ = self.prepare_features(df, forward)
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        all_probas = []
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_scaled)[:, 1]
                all_probas.append(proba)
            except:
                continue
        
        if not all_probas:
            raise ValueError("No models produced predictions")
        
        proba_matrix = np.array(all_probas)
        mean_proba = np.mean(proba_matrix, axis=0)
        std_proba = np.std(proba_matrix, axis=0)
        
        return mean_proba, std_proba
    
    def save_model(self):
        save_data = {
            "models": self.models,
            "scaler": self.scaler,
            "feature_importances": self.feature_importances_,
            "model_weights": self.model_weights,
            "timestamp": datetime.now().isoformat(),
        }
        
        joblib.dump(save_data, self.model_path)
        self.logger.info(f"Saved ensemble model to {self.model_path}")
    
    def load_model(self) -> bool:
        if not self.model_path.exists():
            self.logger.info("No saved model found, will train fresh")
            return False
        
        try:
            save_data = joblib.load(self.model_path)
            
            timestamp = datetime.fromisoformat(save_data["timestamp"])
            age_days = (datetime.now() - timestamp).days
            
            if age_days > RETRAIN_DAYS:
                self.logger.info(f"Model is {age_days} days old, retraining recommended")
            
            self.models = save_data["models"]
            self.scaler = save_data["scaler"]
            self.feature_importances_ = save_data["feature_importances"]
            self.model_weights = save_data.get("model_weights", self.model_weights)
            
            self.logger.info(f"Loaded ensemble model from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_feature_importance(self, top_n: int = 15) -> Dict[str, float]:
        if not self.feature_importances_:
            return {}
        
        sorted_imp = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_imp[:top_n])
    
    def print_feature_importance(self, top_n: int = 15):
        importance = self.get_feature_importance(top_n)
        
        if not importance:
            print("No feature importance available")
            return
        
        print("\n" + "="*50)
        print("TOP FEATURE IMPORTANCES")
        print("="*50)
        print(f"{'Feature':<30} {'Importance':>10}")
        print("-"*50)
        
        for feat, imp in importance.items():
            print(f"{feat:<30} {imp:>10.4f}")
        
        print("="*50 + "\n")
