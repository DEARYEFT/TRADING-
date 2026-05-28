"""
Risk Manager Module
Manages risk assessment, position sizing, and signal filtering
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CONFIDENCE_THRESHOLD, INITIAL_CAPITAL,
    STOP_LOSS_ATR_MULT, TAKE_PROFIT_ATR_MULT, MAX_POSITION_SIZE,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Handles risk management including volatility filtering,
    position sizing using Kelly Criterion, and signal approval.
    """
    
    def __init__(self):
        logger.info("RiskManager initialized")
        
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.initial_capital = INITIAL_CAPITAL
        self.stop_loss_atr_mult = STOP_LOSS_ATR_MULT
        self.take_profit_atr_mult = TAKE_PROFIT_ATR_MULT
        self.max_position_size = MAX_POSITION_SIZE
    
    def volatility_filter(self, atr: float, atr_percentile: float) -> Dict[str, any]:
        """
        Filter signals based on volatility regime.
        
        Args:
            atr: Current ATR value
            atr_percentile: ATR percentile in historical context (0-100)
            
        Returns:
            dict: Volatility assessment
        """
        if pd.isna(atr) or pd.isna(atr_percentile):
            return {
                'risk_level': 'UNKNOWN',
                'approved': True,
                'reason': 'Insufficient data for volatility assessment'
            }
        
        # High volatility: ATR > 95th percentile
        if atr_percentile >= 95:
            return {
                'risk_level': 'HIGH',
                'approved': True,
                'reason': f'Very high volatility (ATR at {atr_percentile:.1f}th percentile). Consider reducing position size.',
                'recommendation': 'REDUCE_SIZE'
            }
        
        # Elevated volatility: ATR > 80th percentile
        elif atr_percentile >= 80:
            return {
                'risk_level': 'ELEVATED',
                'approved': True,
                'reason': f'Elevated volatility (ATR at {atr_percentile:.1f}th percentile)',
                'recommendation': 'NORMAL_SIZE'
            }
        
        # Normal volatility
        else:
            return {
                'risk_level': 'NORMAL',
                'approved': True,
                'reason': f'Normal volatility (ATR at {atr_percentile:.1f}th percentile)',
                'recommendation': 'NORMAL_SIZE'
            }
    
    def sideways_filter(self, adx: float) -> Dict[str, any]:
        """
        Filter signals when market is ranging (no clear trend).
        
        Args:
            adx: ADX value
            
        Returns:
            dict: Trend assessment
        """
        if pd.isna(adx):
            return {
                'is_sideways': False,
                'override_signal': False,
                'reason': 'Insufficient data for trend assessment'
            }
        
        # ADX < 20 indicates weak/no trend (sideways market)
        if adx < 20:
            return {
                'is_sideways': True,
                'override_signal': True,
                'recommended_action': 'HOLD',
                'reason': f'Market is ranging (ADX={adx:.1f} < 20). Trend-following signals may be unreliable.'
            }
        
        # ADX 20-25: Moderate trend
        elif adx < 25:
            return {
                'is_sideways': False,
                'override_signal': False,
                'trend_strength': 'MODERATE',
                'reason': f'Moderate trend strength (ADX={adx:.1f})'
            }
        
        # ADX >= 25: Strong trend
        else:
            return {
                'is_sideways': False,
                'override_signal': False,
                'trend_strength': 'STRONG',
                'reason': f'Strong trend confirmed (ADX={adx:.1f})'
            }
    
    def confidence_filter(self, confidence: float) -> Dict[str, any]:
        """
        Filter signals based on confidence level.
        
        Args:
            confidence: Signal confidence (0-100)
            
        Returns:
            dict: Confidence assessment
        """
        if pd.isna(confidence):
            return {
                'passed': False,
                'reason': 'No confidence value available'
            }
        
        threshold_pct = self.confidence_threshold * 100
        
        if confidence >= threshold_pct:
            return {
                'passed': True,
                'reason': f'Confidence {confidence:.1f}% >= threshold {threshold_pct:.0f}%'
            }
        else:
            return {
                'passed': False,
                'reason': f'Confidence {confidence:.1f}% < threshold {threshold_pct:.0f}%'
            }
    
    def position_size_kelly(self, win_rate: float, avg_win: float, 
                           avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly Formula: f* = (p/a) - (q/b)
        where:
            p = win probability
            q = loss probability (1-p)
            a = average loss ratio
            b = average win ratio
        
        Args:
            win_rate: Win rate (0-1)
            avg_win: Average win amount (positive)
            avg_loss: Average loss amount (positive)
            
        Returns:
            float: Recommended position size as fraction of capital (0 to max_position_size)
        """
        if pd.isna(win_rate) or pd.isna(avg_win) or pd.isna(avg_loss):
            logger.warning("Missing parameters for Kelly calculation")
            return 0.05  # Default 5%
        
        if avg_win <= 0 or avg_loss <= 0:
            logger.warning("Invalid win/loss values for Kelly calculation")
            return 0.05
        
        # Win and loss probabilities
        p = win_rate
        q = 1 - win_rate
        
        # Win/loss ratios relative to a unit bet
        b = avg_win / avg_loss  # Odds received on the wager
        a = 1.0  # Odds against (we lose our stake)
        
        # Kelly formula
        kelly_fraction = (p / a) - (q / b)
        
        # Clamp to [0, max_position_size]
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        # Apply fractional Kelly (half-Kelly for safety)
        kelly_fraction = kelly_fraction * 0.5
        
        logger.debug(f"Kelly calculation: win_rate={win_rate:.2f}, "
                    f"avg_win={avg_win:.2f}, avg_loss={avg_loss:.2f}, "
                    f"kelly={kelly_fraction:.3f}")
        
        return kelly_fraction
    
    def assess_risk(self, signal: Dict[str, any], df: pd.DataFrame, 
                   confidence: float) -> Dict[str, any]:
        """
        Comprehensive risk assessment for a trading signal.
        
        Args:
            signal: Signal dictionary from SignalEngine
            df: DataFrame with OHLCV and indicators
            confidence: Signal confidence percentage
            
        Returns:
            dict: Risk assessment results
        """
        logger.info("Performing risk assessment...")
        
        # Get latest row
        latest = df.iloc[-1] if len(df) > 0 else pd.Series()
        
        # Initialize result
        risk_result = {
            'risk_level': 'MEDIUM',
            'approved': True,
            'position_size': 0.10,  # Default 10%
            'stop_loss': None,
            'take_profit': None,
            'reasons': []
        }
        
        # 1. Check confidence filter
        conf_filter = self.confidence_filter(confidence)
        if not conf_filter['passed']:
            risk_result['approved'] = False
            risk_result['reasons'].append(conf_filter['reason'])
            risk_result['risk_level'] = 'HIGH'
        
        # 2. Check volatility
        atr = latest.get('ATR', np.nan)
        atr_percentile = self._calculate_atr_percentile(df, 'ATR')
        vol_filter = self.volatility_filter(atr, atr_percentile)
        
        risk_result['volatility'] = vol_filter['risk_level']
        risk_result['reasons'].append(vol_filter['reason'])
        
        if vol_filter.get('recommendation') == 'REDUCE_SIZE':
            risk_result['position_size'] = 0.05  # Reduce to 5%
        
        # 3. Check for sideways market
        adx = latest.get('ADX', np.nan)
        sideways_filter = self.sideways_filter(adx)
        
        if sideways_filter.get('override_signal'):
            risk_result['approved'] = False
            risk_result['reasons'].append(sideways_filter['reason'])
            risk_result['overridden_action'] = sideways_filter.get('recommended_action', 'HOLD')
        
        # 4. Calculate stop loss and take profit levels
        current_price = latest.get('Close', np.nan)
        if not pd.isna(current_price) and not pd.isna(atr):
            if signal.get('action') == 'BUY':
                risk_result['stop_loss'] = current_price - (atr * self.stop_loss_atr_mult)
                risk_result['take_profit'] = current_price + (atr * self.take_profit_atr_mult)
            elif signal.get('action') == 'SELL':
                risk_result['stop_loss'] = current_price + (atr * self.stop_loss_atr_mult)
                risk_result['take_profit'] = current_price - (atr * self.take_profit_atr_mult)
        
        # 5. Determine overall risk level
        if not risk_result['approved']:
            risk_result['risk_level'] = 'HIGH'
        elif vol_filter['risk_level'] in ['HIGH', 'ELEVATED']:
            risk_result['risk_level'] = vol_filter['risk_level']
        elif confidence >= 80:
            risk_result['risk_level'] = 'LOW'
        elif confidence >= 60:
            risk_result['risk_level'] = 'MEDIUM'
        else:
            risk_result['risk_level'] = 'HIGH'
        
        # 6. Estimate position size based on backtested metrics
        # (In production, this would use live performance data)
        estimated_win_rate = 0.55  # Placeholder
        estimated_avg_win = atr * self.take_profit_atr_mult if not pd.isna(atr) else 10
        estimated_avg_loss = atr * self.stop_loss_atr_mult if not pd.isna(atr) else 10
        
        kelly_size = self.position_size_kelly(
            estimated_win_rate,
            estimated_avg_win,
            estimated_avg_loss
        )
        
        # Use minimum of Kelly and max position size
        risk_result['position_size'] = min(kelly_size, self.max_position_size)
        
        logger.info(f"Risk assessment complete: {risk_result['risk_level']}, "
                   f"approved={risk_result['approved']}")
        
        return risk_result
    
    def _calculate_atr_percentile(self, df: pd.DataFrame, 
                                  atr_col: str = 'ATR',
                                  window: int = 252) -> float:
        """
        Calculate current ATR percentile over rolling window.
        
        Args:
            df: DataFrame with ATR column
            atr_col: Name of ATR column
            window: Lookback window in days
            
        Returns:
            float: ATR percentile (0-100)
        """
        if atr_col not in df.columns:
            return 50.0
        
        recent_atr = df[atr_col].iloc[-1]
        historical_atr = df[atr_col].iloc[-window:]
        
        if len(historical_atr) == 0 or pd.isna(recent_atr):
            return 50.0
        
        percentile = (historical_atr.dropna() < recent_atr).sum() / len(historical_atr.dropna()) * 100
        
        return percentile
    
    def print_risk_report(self, risk_assessment: Dict[str, any]) -> None:
        """
        Print formatted risk report to console.
        
        Args:
            risk_assessment: Risk assessment dictionary
        """
        print("\n" + "=" * 50)
        print("⚠️  RISK ASSESSMENT")
        print("=" * 50)
        
        risk_emoji = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴',
            'ELEVATED': '🟠',
            'NORMAL': '🟢'
        }
        
        emoji = risk_emoji.get(risk_assessment['risk_level'], '❓')
        print(f"{emoji} Risk Level: {risk_assessment['risk_level']}")
        print(f"{'✅' if risk_assessment['approved'] else '❌'} Approved: {risk_assessment['approved']}")
        print(f"💰 Position Size: {risk_assessment['position_size']*100:.1f}% of capital")
        
        if risk_assessment.get('stop_loss'):
            print(f"🛑 Stop Loss: ${risk_assessment['stop_loss']:.2f}")
        if risk_assessment.get('take_profit'):
            print(f"🎯 Take Profit: ${risk_assessment['take_profit']:.2f}")
        
        if risk_assessment.get('volatility'):
            print(f"📊 Volatility: {risk_assessment['volatility']}")
        
        if risk_assessment.get('reasons'):
            print("\nReasons:")
            for reason in risk_assessment['reasons']:
                print(f"  • {reason}")
        
        print("=" * 50)
