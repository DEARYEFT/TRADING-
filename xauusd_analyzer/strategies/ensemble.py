"""
Ensemble Strategy Module
Advanced ensemble strategies for XAU/USD trading
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WEIGHT_TECHNICAL, WEIGHT_ML, WEIGHT_NEWS,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class EnsembleStrategy:
    """
    Advanced ensemble strategy combining multiple signals with dynamic weighting.
    Supports different ensemble methods: weighted average, voting, stacking.
    """
    
    def __init__(self):
        logger.info("EnsembleStrategy initialized")
        
        self.weight_technical = WEIGHT_TECHNICAL
        self.weight_ml = WEIGHT_ML
        self.weight_news = WEIGHT_NEWS
        
        # Performance tracking for adaptive weighting
        self.signal_history = []
        self.performance_window = 20  # Days to consider for adaptive weights
    
    def weighted_average_ensemble(self, tech_signal: float, ml_signal: float,
                                  news_signal: float) -> float:
        """
        Combine signals using fixed weighted average.
        
        Args:
            tech_signal: Technical analysis score [0, 1]
            ml_signal: ML model probability [0, 1]
            news_signal: News sentiment score [0, 1]
            
        Returns:
            float: Combined signal score
        """
        combined = (
            tech_signal * self.weight_technical +
            ml_signal * self.weight_ml +
            news_signal * self.weight_news
        )
        
        return combined
    
    def adaptive_weighted_ensemble(self, tech_signal: float, ml_signal: float,
                                   news_signal: float, 
                                   recent_performance: Dict[str, float]) -> float:
        """
        Combine signals using adaptive weights based on recent performance.
        
        Args:
            tech_signal: Technical analysis score [0, 1]
            ml_signal: ML model probability [0, 1]
            news_signal: News sentiment score [0, 1]
            recent_performance: Dict with 'technical', 'ml', 'news' accuracy scores
            
        Returns:
            float: Combined signal score with adaptive weights
        """
        # Get recent accuracies (default to equal if not available)
        tech_acc = recent_performance.get('technical', 0.5)
        ml_acc = recent_performance.get('ml', 0.5)
        news_acc = recent_performance.get('news', 0.5)
        
        # Normalize accuracies to sum to 1
        total_acc = tech_acc + ml_acc + news_acc
        if total_acc > 0:
            adj_tech = tech_acc / total_acc
            adj_ml = ml_acc / total_acc
            adj_news = news_acc / total_acc
        else:
            adj_tech = adj_ml = adj_news = 1/3
        
        # Combine with adaptive weights
        combined = (
            tech_signal * adj_tech +
            ml_signal * adj_ml +
            news_signal * adj_news
        )
        
        logger.debug(f"Adaptive weights: tech={adj_tech:.2f}, ml={adj_ml:.2f}, news={adj_news:.2f}")
        
        return combined
    
    def voting_ensemble(self, signals: List[str], 
                       weights: Optional[List[float]] = None) -> str:
        """
        Combine signals using weighted voting.
        
        Args:
            signals: List of individual signals ('BUY', 'SELL', 'HOLD')
            weights: Optional list of weights for each signal
            
        Returns:
            str: Final consensus signal
        """
        if not signals:
            return 'HOLD'
        
        if weights is None:
            weights = [1.0] * len(signals)
        
        # Count weighted votes
        vote_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for signal, weight in zip(signals, weights):
            if signal in vote_scores:
                vote_scores[signal] += weight
        
        # Return signal with highest weighted votes
        final_signal = max(vote_scores, key=vote_scores.get)
        
        logger.debug(f"Voting results: {vote_scores} -> {final_signal}")
        
        return final_signal
    
    def confidence_weighted_ensemble(self, signals: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Combine signals weighted by their individual confidences.
        
        Args:
            signals: List of signal dictionaries with 'action', 'confidence', 'source'
            
        Returns:
            dict: Combined signal with confidence-weighted action
        """
        if not signals:
            return {'action': 'HOLD', 'confidence': 0, 'method': 'confidence_weighted'}
        
        # Map actions to numeric values
        action_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
        reverse_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for signal in signals:
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 50) / 100  # Normalize to [0, 1]
            
            action_value = action_map.get(action, 0)
            weighted_sum += action_value * confidence
            total_weight += confidence
        
        if total_weight > 0:
            avg_score = weighted_sum / total_weight
        else:
            avg_score = 0
        
        # Convert back to action
        if avg_score > 0.3:
            final_action = 'BUY'
        elif avg_score < -0.3:
            final_action = 'SELL'
        else:
            final_action = 'HOLD'
        
        # Calculate combined confidence
        combined_confidence = abs(avg_score) * 100
        
        result = {
            'action': final_action,
            'confidence': round(combined_confidence, 1),
            'score': round(avg_score, 3),
            'method': 'confidence_weighted',
            'individual_signals': signals
        }
        
        logger.info(f"Confidence-weighted ensemble: {final_action} ({combined_confidence:.1f}%)")
        
        return result
    
    def regime_based_ensemble(self, df: pd.DataFrame, tech_signal: float,
                             ml_signal: float, news_signal: float) -> Dict[str, any]:
        """
        Adjust ensemble weights based on market regime.
        
        In trending markets: emphasize technical and ML
        In ranging markets: emphasize mean-reversion signals
        In high volatility: reduce news weight
        
        Args:
            df: DataFrame with indicators
            tech_signal: Technical score
            ml_signal: ML score
            news_signal: News score
            
        Returns:
            dict: Combined signal with regime-adjusted weights
        """
        latest = df.iloc[-1]
        
        # Determine market regime
        adx = latest.get('ADX', 25)
        atr_percent = latest.get('ATR_Percent', 1.0)
        
        # Regime detection
        is_trending = adx >= 25
        is_high_vol = atr_percent >= 1.5
        
        # Adjust weights based on regime
        if is_trending and not is_high_vol:
            # Trending, normal volatility: trust technicals more
            weights = {'technical': 0.50, 'ml': 0.35, 'news': 0.15}
            regime = 'TRENDING'
        elif is_trending and is_high_vol:
            # Trending but volatile: be cautious
            weights = {'technical': 0.35, 'ml': 0.35, 'news': 0.30}
            regime = 'TRENDING_HIGH_VOL'
        elif not is_trending and not is_high_vol:
            # Ranging, low vol: mean reversion likely
            weights = {'technical': 0.30, 'ml': 0.40, 'news': 0.30}
            regime = 'RANGING'
        else:
            # High vol, no trend: uncertain
            weights = {'technical': 0.33, 'ml': 0.34, 'news': 0.33}
            regime = 'UNCERTAIN'
        
        # Calculate weighted signal
        combined = (
            tech_signal * weights['technical'] +
            ml_signal * weights['ml'] +
            news_signal * weights['news']
        )
        
        # Determine action
        if combined >= 0.65:
            action = 'BUY'
        elif combined <= 0.35:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        confidence = abs(combined - 0.5) * 2 * 100
        
        result = {
            'action': action,
            'confidence': round(confidence, 1),
            'regime': regime,
            'weights': weights,
            'combined_score': round(combined, 3)
        }
        
        logger.info(f"Regime-based ensemble [{regime}]: {action} (conf={confidence:.1f}%)")
        
        return result
    
    def record_signal_performance(self, signal: str, actual_return: float,
                                 source: str) -> None:
        """
        Record signal performance for adaptive weighting.
        
        Args:
            signal: Signal that was given ('BUY', 'SELL', 'HOLD')
            actual_return: Actual return that followed
            source: Source of signal ('technical', 'ml', 'news')
        """
        # Determine if signal was correct
        if signal == 'BUY' and actual_return > 0:
            correct = True
        elif signal == 'SELL' and actual_return < 0:
            correct = True
        elif signal == 'HOLD':
            correct = True  # HOLD is always "correct" in a sense
        else:
            correct = False
        
        self.signal_history.append({
            'signal': signal,
            'actual_return': actual_return,
            'source': source,
            'correct': correct,
            'timestamp': pd.Timestamp.now()
        })
        
        # Keep only recent history
        if len(self.signal_history) > self.performance_window * 3:
            self.signal_history = self.signal_history[-self.performance_window * 3:]
    
    def get_recent_accuracy(self, source: str) -> float:
        """
        Get recent accuracy for a signal source.
        
        Args:
            source: Signal source ('technical', 'ml', 'news')
            
        Returns:
            float: Accuracy rate (0-1)
        """
        relevant = [s for s in self.signal_history if s['source'] == source]
        
        if not relevant:
            return 0.5  # Default
        
        correct = sum(1 for s in relevant if s['correct'])
        accuracy = correct / len(relevant)
        
        return accuracy
