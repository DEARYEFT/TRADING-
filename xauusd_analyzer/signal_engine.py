"""
Signal Engine Module
Generates trading signals by combining technical, ML, and news analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WEIGHT_TECHNICAL, WEIGHT_ML, WEIGHT_NEWS,
    BUY_THRESHOLD, SELL_THRESHOLD, CONFIDENCE_THRESHOLD,
    RSI_PERIOD, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Generates trading signals by combining multiple analysis methods:
    - Technical analysis (RSI, MACD, EMA alignment, breakouts)
    - Machine learning predictions
    - News sentiment analysis
    
    Uses weighted ensemble approach for final signal.
    """
    
    def __init__(self):
        logger.info("SignalEngine initialized")
        
        self.weight_technical = WEIGHT_TECHNICAL
        self.weight_ml = WEIGHT_ML
        self.weight_news = WEIGHT_NEWS
        
        self.buy_threshold = BUY_THRESHOLD
        self.sell_threshold = SELL_THRESHOLD
        self.confidence_threshold = CONFIDENCE_THRESHOLD
    
    def compute_technical_signal(self, df: pd.DataFrame) -> float:
        """
        Compute technical analysis signal score.
        
        Args:
            df: DataFrame with all technical indicators
            
        Returns:
            float: Technical signal score normalized to [0, 1]
        """
        logger.debug("Computing technical signal...")
        
        latest = df.iloc[-1]
        
        score = 0.0
        max_score = 10.0  # Maximum possible raw score
        
        # RSI signals (max +2 or -2)
        rsi = latest.get('RSI', 50)
        if not pd.isna(rsi):
            if rsi < 30:
                score += 2  # Oversold - bullish
            elif rsi > 70:
                score -= 2  # Overbought - bearish
        
        # MACD signals (max +1 or -1)
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal:
                score += 1  # Bullish crossover
            elif macd < macd_signal:
                score -= 1  # Bearish crossover
        
        # EMA alignment (max +3 or -3)
        ema_alignment = latest.get('ema_alignment', 'NEUTRAL')
        if ema_alignment == 'BULLISH_STACK':
            score += 3
        elif ema_alignment == 'BEARISH_STACK':
            score -= 3
        
        # Breakout detection (max +2 or -2)
        breakout = latest.get('breakout_detected', 'NONE')
        if breakout == 'BULLISH':
            score += 2
        elif breakout == 'BEARISH':
            score -= 2
        
        # ADX trend confirmation (bonus if strong trend)
        adx = latest.get('ADX', 0)
        if not pd.isna(adx) and adx > 25:
            # Trend is confirmed, amplify existing signal
            if score > 0:
                score *= 1.2
            elif score < 0:
                score *= 1.2
        
        # Momentum confirmation
        momentum = latest.get('Momentum', 0)
        if not pd.isna(momentum):
            if momentum > 0 and score > 0:
                score += 1  # Momentum confirms direction
            elif momentum < 0 and score < 0:
                score -= 1
        
        # Stochastic confirmation
        stoch_k = latest.get('Stoch_K', 50)
        stoch_d = latest.get('Stoch_D', 50)
        if not pd.isna(stoch_k) and not pd.isna(stoch_d):
            if stoch_k < 20 and stoch_k > stoch_d:
                score += 1  # Oversold with upward cross
            elif stoch_k > 80 and stoch_k < stoch_d:
                score -= 1  # Overbought with downward cross
        
        # Normalize to [0, 1] range
        # Raw score ranges approximately from -9 to +9
        normalized_score = (score + max_score) / (2 * max_score)
        normalized_score = max(0, min(1, normalized_score))  # Clamp to [0, 1]
        
        logger.debug(f"Technical signal: raw={score:.2f}, normalized={normalized_score:.3f}")
        
        return normalized_score
    
    def compute_ml_signal(self, ml_proba: float) -> float:
        """
        Convert ML probability to signal score.
        
        Args:
            ml_proba: ML model probability of upward movement (0-1)
            
        Returns:
            float: ML signal score (same as input, already normalized)
        """
        logger.debug(f"ML signal: {ml_proba:.3f}")
        return ml_proba
    
    def compute_news_signal(self, sentiment_score: float) -> float:
        """
        Convert news sentiment score to signal score.
        
        Args:
            sentiment_score: Sentiment score normalized to [0, 1]
            
        Returns:
            float: News signal score
        """
        logger.debug(f"News signal: {sentiment_score:.3f}")
        return sentiment_score
    
    def generate_final_signal(self, df: pd.DataFrame, ml_proba: float,
                             sentiment: Dict[str, any]) -> Dict[str, any]:
        """
        Generate final ensemble signal combining all inputs.
        
        Args:
            df: DataFrame with technical indicators
            ml_proba: ML model probability
            sentiment: News sentiment dictionary
            
        Returns:
            dict: Final signal with action, confidence, and breakdown
        """
        logger.info("Generating final ensemble signal...")
        
        # Compute individual signals
        tech_signal = self.compute_technical_signal(df)
        ml_signal = self.compute_ml_signal(ml_proba)
        news_signal = self.compute_news_signal(sentiment.get('normalized_score', 0.5))
        
        # Weighted ensemble
        final_score = (
            tech_signal * self.weight_technical +
            ml_signal * self.weight_ml +
            news_signal * self.weight_news
        )
        
        # Determine action
        if final_score >= self.buy_threshold:
            action = 'BUY'
        elif final_score <= self.sell_threshold:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Calculate confidence
        # Confidence is distance from neutral (0.5), scaled to percentage
        confidence = abs(final_score - 0.5) * 2 * 100
        
        # Determine signal strength labels
        tech_label = self._get_tech_label(tech_signal)
        ml_label = 'BULLISH' if ml_signal > 0.5 else 'BEARISH'
        news_label = sentiment.get('label', 'NEUTRAL')
        
        # Build result dictionary
        signal = {
            'action': action,
            'confidence': round(confidence, 1),
            'final_score': round(final_score, 3),
            'timestamp': datetime.now(),
            'breakdown': {
                'technical': {
                    'score': round(tech_signal, 3),
                    'weight': self.weight_technical,
                    'label': tech_label
                },
                'ml': {
                    'score': round(ml_signal, 3),
                    'weight': self.weight_ml,
                    'label': ml_label
                },
                'news': {
                    'score': round(news_signal, 3),
                    'weight': self.weight_news,
                    'label': news_label
                }
            },
            'thresholds': {
                'buy': self.buy_threshold,
                'sell': self.sell_threshold,
                'confidence': self.confidence_threshold * 100
            }
        }
        
        logger.info(f"Signal generated: {action} (confidence={confidence:.1f}%)")
        
        return signal
    
    def _get_tech_label(self, score: float) -> str:
        """Get descriptive label for technical score."""
        if score >= 0.7:
            return 'Strong Uptrend'
        elif score >= 0.55:
            return 'Moderate Uptrend'
        elif score >= 0.45:
            return 'Neutral'
        elif score >= 0.3:
            return 'Moderate Downtrend'
        else:
            return 'Strong Downtrend'
    
    def format_signal_output(self, signal: Dict[str, any]) -> None:
        """
        Print formatted signal output to console.
        
        Args:
            signal: Signal dictionary from generate_final_signal()
        """
        action = signal['action']
        confidence = signal['confidence']
        breakdown = signal['breakdown']
        
        # Action emoji
        action_emoji = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡'
        }
        
        print("\n" + "═" * 50)
        print(f"⚡ XAU/USD SIGNAL: {action_emoji.get(action, '❓')} {action}")
        print(f"📊 Confidence: {confidence:.1f}%")
        print("═" * 50)
        
        print("\nBREAKDOWN:")
        
        # ML Model
        ml = breakdown['ml']
        ml_bar = '█' * int(ml['score'] * 10)
        print(f"  🤖 ML Model:     {ml['label']:12} ({ml['score']:.2f}) {ml_bar}")
        
        # Technical
        tech = breakdown['technical']
        tech_bar = '█' * int(tech['score'] * 10)
        print(f"  📈 Technical:   {tech['label']:12} ({tech['score']:.2f}) {tech_bar}")
        
        # News
        news = breakdown['news']
        news_bar = '█' * int(news['score'] * 10)
        print(f"  📰 News:        {news['label']:12} ({news['score']:.2f}) {news_bar}")
        
        print("\n" + "─" * 50)
        
        # Risk level based on confidence
        if confidence >= 80:
            risk_level = "LOW"
            risk_emoji = "🟢"
        elif confidence >= 60:
            risk_level = "MEDIUM"
            risk_emoji = "🟡"
        else:
            risk_level = "HIGH"
            risk_emoji = "🔴"
        
        print(f"⚠️  Risk Level: {risk_emoji} {risk_level}")
        print(f"📅 Timestamp: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 50)
    
    def generate_signals_series(self, df: pd.DataFrame, ml_probas: np.ndarray,
                               sentiment_scores: pd.Series) -> pd.Series:
        """
        Generate signal series for backtesting.
        
        Args:
            df: DataFrame with technical indicators
            ml_probas: Array of ML probabilities
            sentiment_scores: Series of sentiment scores
            
        Returns:
            pd.Series: Signal series (BUY/SELL/HOLD)
        """
        signals = []
        
        for i in range(len(df)):
            # Get row data
            row_df = df.iloc[:i+1]  # Use data up to current point
            
            # Get ML probability
            ml_proba = ml_probas[i] if i < len(ml_probas) else 0.5
            
            # Get sentiment score
            sent_score = sentiment_scores.iloc[i] if i < len(sentiment_scores) else 0.5
            
            # Generate signal
            tech_signal = self.compute_technical_signal(row_df)
            ml_signal = self.compute_ml_signal(ml_proba)
            news_signal = self.compute_news_signal(sent_score)
            
            final_score = (
                tech_signal * self.weight_technical +
                ml_signal * self.weight_ml +
                news_signal * self.weight_news
            )
            
            if final_score >= self.buy_threshold:
                signals.append('BUY')
            elif final_score <= self.sell_threshold:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return pd.Series(signals, index=df.index)
