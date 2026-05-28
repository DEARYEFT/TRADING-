"""
═══════════════════════════════════════════════════════════════════
SIGNAL ENGINE - INSTITUTIONAL SIGNAL GENERATION
═══════════════════════════════════════════════════════════════════
Generates BUY/SELL/HOLD signals using weighted ensemble of
technical, ML, news, structure, and macro factors.
═══════════════════════════════════════════════════════════════════
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from colorama import Fore, Style, init

from config import (
    ENSEMBLE_WEIGHTS, CONFIDENCE_THRESHOLD, MIN_ADX_FOR_TRADE,
    SIGNAL_QUALITY_THRESHOLDS, MAX_ATR_PERCENTILE
)

init(autoreset=True)
logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Institutional signal generation engine combining multiple
    alpha sources with conflict detection and quality scoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SignalEngine")
    
    def generate_final_signal(
        self,
        df: np.ndarray,
        ml_proba: float,
        sentiment: Dict,
        technical_score: Optional[float] = None,
        structure_score: Optional[float] = None,
        macro_score: Optional[float] = None,
        orderflow_score: Optional[float] = None
    ) -> Dict:
        """
        Generate final trading signal from all alpha sources.
        
        Args:
            df: DataFrame with technical indicators
            ml_proba: ML model probability
            sentiment: News sentiment dict
            technical_score: Pre-computed technical score (optional)
            structure_score: Market structure score (optional)
            macro_score: Macro alignment score (optional)
            orderflow_score: Order flow score (optional)
            
        Returns:
            Complete signal dict with direction, confidence, metadata
        """
        self.logger.info("Generating final signal...")
        
        # Get technical score from DataFrame
        if technical_score is None:
            technical_score = float(df["technical_signal_norm"].iloc[-1])
        
        # Default scores if not provided
        if structure_score is None:
            structure_score = self._compute_structure_score(df)
        if macro_score is None:
            macro_score = 0.5  # Neutral default
        if orderflow_score is None:
            orderflow_score = 0.5  # Neutral default
        
        # News score normalized
        news_score = sentiment.get("norm", 0.5)
        
        # Weighted ensemble
        final_score = (
            technical_score * ENSEMBLE_WEIGHTS["technical"] +
            ml_proba * ENSEMBLE_WEIGHTS["ml"] +
            news_score * ENSEMBLE_WEIGHTS["news"] +
            structure_score * ENSEMBLE_WEIGHTS["structure"] +
            macro_score * ENSEMBLE_WEIGHTS["macro"] +
            orderflow_score * ENSEMBLE_WEIGHTS["orderflow"]
        )
        
        # Determine direction
        if final_score > 0.62:
            direction = "BUY"
        elif final_score < 0.38:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        # Confidence calculation
        confidence = min(abs(final_score - 0.5) * 2.2, 1.0)
        
        # Quality grade
        quality = self._assign_quality_grade(confidence)
        
        # Strength label
        if confidence > 0.80:
            strength = "VERY STRONG"
        elif confidence > 0.65:
            strength = "STRONG"
        elif confidence > 0.50:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        # Detect conflicts
        conflicts = self._detect_conflicts(
            technical_score, ml_proba, news_score
        )
        
        # Apply conflict penalty
        if conflicts["has_conflict"]:
            confidence *= (1 - conflicts["conflict_penalty"])
        
        signal = {
            "direction": direction,
            "confidence": confidence,
            "quality": quality,
            "strength": strength,
            "final_score": final_score,
            "breakdown": {
                "technical": technical_score,
                "ml": ml_proba,
                "news": news_score,
                "structure": structure_score,
                "macro": macro_score,
                "orderflow": orderflow_score
            },
            "sentiment_label": sentiment.get("label", "NEUTRAL"),
            "conflicts": conflicts,
            "timestamp": datetime.now(),
            "no_trade_warning": confidence < CONFIDENCE_THRESHOLD
        }
        
        return signal
    
    def _compute_structure_score(self, df) -> float:
        """Compute market structure score from trend and SMC."""
        try:
            score = 0.5
            
            # Trend alignment
            trend_dir = df["trend_direction"].iloc[-1]
            if trend_dir == 1:
                score += 0.2
            elif trend_dir == -1:
                score -= 0.2
            
            # ADX confirmation
            adx = df["adx"].iloc[-1]
            if adx > 25:
                score += 0.15
            elif adx < 20:
                score -= 0.1
            
            # Breakout confirmation
            if df["breakout_up"].iloc[-1]:
                score += 0.15
            elif df["breakout_down"].iloc[-1]:
                score -= 0.15
            
            return np.clip(score, 0, 1)
        except Exception as e:
            self.logger.debug(f"Structure score error: {e}")
            return 0.5
    
    def _detect_conflicts(
        self, 
        tech: float, 
        ml: float, 
        news: float
    ) -> Dict:
        """Detect conflicting signals between alpha sources."""
        tech_vs_ml = abs(tech - ml)
        ml_vs_news = abs(ml - news)
        tech_vs_news = abs(tech - news)
        
        max_disagreement = max(tech_vs_ml, ml_vs_news, tech_vs_news)
        
        has_conflict = max_disagreement > 0.3
        penalty = min(max_disagreement / 3, 0.3) if has_conflict else 0
        
        return {
            "tech_vs_ml": tech_vs_ml,
            "ml_vs_news": ml_vs_news,
            "tech_vs_news": tech_vs_news,
            "max_disagreement": max_disagreement,
            "has_conflict": has_conflict,
            "conflict_penalty": penalty
        }
    
    def _assign_quality_grade(self, confidence: float) -> str:
        """Assign letter grade based on confidence."""
        for grade, threshold in sorted(
            SIGNAL_QUALITY_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if confidence >= threshold:
                return grade
        return "F"
    
    def format_signal_output(self, signal: Dict):
        """Print beautifully formatted signal output."""
        direction = signal["direction"]
        confidence = signal["confidence"]
        strength = signal["strength"]
        quality = signal["quality"]
        
        # Color coding
        if direction == "BUY":
            dir_color = Fore.GREEN
            dir_symbol = "🟢"
        elif direction == "SELL":
            dir_color = Fore.RED
            dir_symbol = "🔴"
        else:
            dir_color = Fore.YELLOW
            dir_symbol = "🟡"
        
        # Confidence bar
        bar_length = int(confidence * 10)
        bar = "█" * bar_length + "░" * (10 - bar_length)
        
        print("\n" + "="*60)
        print(f"{Fore.CYAN}⚡ XAU/USD INSTITUTIONAL TRADING SIGNAL ⚡{Style.RESET_ALL}")
        print("="*60)
        print(f"  {dir_color}{dir_symbol} SIGNAL:{Style.RESET_ALL}     {dir_color}{direction}{Style.RESET_ALL} ({strength})")
        print(f"  📊 Quality:      {quality}")
        print(f"  Confidence:      {bar} {confidence*100:.1f}%")
        print("-"*60)
        print(f"  {Fore.WHITE}BREAKDOWN:{Style.RESET_ALL}")
        print(f"    🤖 ML Model:     {signal['breakdown']['ml']:.2f}  [{'BULLISH' if signal['breakdown']['ml'] > 0.55 else 'BEARISH'}]")
        print(f"    📈 Technical:    {signal['breakdown']['technical']:.2f}  [{'UPTREND' if signal['breakdown']['technical'] > 0.55 else 'DOWNTREND'}]")
        print(f"    🏗️  Structure:    {signal['breakdown']['structure']:.2f}")
        print(f"    📰 News:         {signal['breakdown']['news']:.2f}  [{signal['sentiment_label']}]")
        print(f"    🌍 Macro:        {signal['breakdown']['macro']:.2f}")
        print(f"    📊 Order Flow:   {signal['breakdown']['orderflow']:.2f}")
        print("-"*60)
        
        if signal["conflicts"]["has_conflict"]:
            print(f"  {Fore.YELLOW}⚠️  SIGNAL CONFLICT DETECTED{Style.RESET_ALL}")
            print(f"     Max disagreement: {signal['conflicts']['max_disagreement']:.2f}")
        
        if signal["no_trade_warning"]:
            print(f"  {Fore.RED}⛔ NO TRADE RECOMMENDED - Low confidence{Style.RESET_ALL}")
        
        print(f"  📅 Generated:    {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("="*60 + "\n")
