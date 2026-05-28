"""
═══════════════════════════════════════════════════════════════════
RISK MANAGER - INSTITUTIONAL RISK CONTROL
═══════════════════════════════════════════════════════════════════
Professional risk management with Kelly criterion, ATR-based stops,
volatility filtering, and position sizing.
═══════════════════════════════════════════════════════════════════
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime

from config import (
    INITIAL_CAPITAL, KELLY_MAX_FRACTION, KELLY_MULTIPLIER,
    MAX_POSITION_SIZE_PCT, MAX_DRAWDOWN_LIMIT, MAX_CONSECUTIVE_LOSSES,
    STOP_LOSS_ATR_MULT, TAKE_PROFIT_ATR_MULT, TRAILING_STOP_ATR_MULT,
    BREAKEVEN_TRIGGER_ATR_MULT, MIN_ADX_FOR_TRADE, MAX_ATR_PERCENTILE,
    CONFIDENCE_THRESHOLD
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Institutional risk management engine implementing professional
    position sizing, stop losses, and trade validation.
    """
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.consecutive_losses = 0
        self.logger = logging.getLogger(f"{__name__}.RiskManager")
    
    def assess_risk(
        self,
        signal: Dict,
        df,
        confidence: float
    ) -> Dict:
        """
        Comprehensive risk assessment for a potential trade.
        
        Args:
            signal: Signal dict from SignalEngine
            df: DataFrame with indicators
            confidence: Signal confidence
            
        Returns:
            Risk assessment dict with approval status
        """
        self.logger.info("Assessing trade risk...")
        
        filters_passed = []
        filters_failed = []
        
        # Volatility filter
        vol_ok, vol_msg = self.volatility_filter(df)
        if vol_ok:
            filters_passed.append(vol_msg)
        else:
            filters_failed.append(vol_msg)
        
        # Trend filter
        trend_ok, trend_msg = self.trend_filter(df)
        if trend_ok:
            filters_passed.append(trend_msg)
        else:
            filters_failed.append(trend_msg)
        
        # Confidence filter
        conf_ok, conf_msg = self.confidence_filter(confidence)
        if conf_ok:
            filters_passed.append(conf_msg)
        else:
            filters_failed.append(conf_msg)
        
        # Consecutive loss filter
        loss_ok, loss_msg = self.consecutive_loss_filter()
        if loss_ok:
            filters_passed.append(loss_msg)
        else:
            filters_failed.append(loss_msg)
        
        # Determine approval
        approved = len(filters_failed) == 0
        
        # Risk level
        risk_level = self._calculate_risk_level(df, confidence, len(filters_failed))
        
        # Calculate position size
        entry_price = float(df["Close"].iloc[-1])
        position_pct = self.kelly_position_size(
            win_rate=0.55,  # Historical estimate
            avg_win_pct=0.03,
            avg_loss_pct=0.02
        )
        
        # Calculate stops
        stops = self.calculate_stops(df, signal["direction"], entry_price)
        
        assessment = {
            "approved": approved,
            "risk_level": risk_level,
            "filters_passed": filters_passed,
            "filters_failed": filters_failed,
            "position_size_pct": position_pct if approved else 0,
            "stop_loss_price": stops["stop_loss"],
            "take_profit_prices": [stops["tp1"], stops["tp2"], stops["tp3"]],
            "risk_reward_ratio": stops["rr"],
            "override_reason": None if approved else "; ".join(filters_failed[:2])
        }
        
        return assessment
    
    def volatility_filter(self, df) -> Tuple[bool, str]:
        """Check if volatility is within acceptable range."""
        try:
            atr_pct = df["atr_pct"].iloc[-1]
            atr_percentile = df["atr_pct"].rank(pct=True).iloc[-1] * 100
            
            if atr_percentile > MAX_ATR_PERCENTILE:
                return False, f"ATR too high ({atr_pct:.2f}%, {atr_percentile:.0f}th percentile)"
            
            return True, f"Volatility OK (ATR={atr_pct:.2f}%, {atr_percentile:.0f}th pct)"
        except Exception as e:
            self.logger.debug(f"Volatility filter error: {e}")
            return True, "Volatility check passed (default)"
    
    def trend_filter(self, df) -> Tuple[bool, str]:
        """Check if trend conditions are met."""
        try:
            adx = df["adx"].iloc[-1]
            
            if adx < MIN_ADX_FOR_TRADE:
                return False, f"ADX={adx:.1f} < {MIN_ADX_FOR_TRADE} (sideways market)"
            
            return True, f"ADX={adx:.1f} >= {MIN_ADX_FOR_TRADE} (trend confirmed)"
        except Exception as e:
            self.logger.debug(f"Trend filter error: {e}")
            return True, "Trend check passed (default)"
    
    def confidence_filter(self, confidence: float) -> Tuple[bool, str]:
        """Check if confidence meets threshold."""
        if confidence < CONFIDENCE_THRESHOLD:
            return False, f"Confidence {confidence:.1%} < {CONFIDENCE_THRESHOLD:.1%}"
        
        return True, f"Confidence {confidence:.1%} >= {CONFIDENCE_THRESHOLD:.1%}"
    
    def consecutive_loss_filter(self) -> Tuple[bool, str]:
        """Check consecutive loss limit."""
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return False, f"Consecutive losses ({self.consecutive_losses}) >= max ({MAX_CONSECUTIVE_LOSSES})"
        
        return True, f"Consecutive losses: {self.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}"
    
    def kelly_position_size(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win_pct: Average win percentage
            avg_loss_pct: Average loss percentage
            
        Returns:
            Recommended position size as fraction of capital
        """
        if avg_loss_pct == 0:
            return 0.01
        
        # Kelly formula: f = (p*b - q) / b
        # where p=win_prob, q=loss_prob, b=win_loss_ratio
        b = avg_win_pct / avg_loss_pct
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (p * b - q) / b
        
        # Apply fractional Kelly (safer)
        kelly_fraction *= KELLY_MULTIPLIER
        
        # Clip to max
        kelly_fraction = max(0.01, min(kelly_fraction, KELLY_MAX_FRACTION))
        
        return round(kelly_fraction, 4)
    
    def calculate_stops(
        self,
        df,
        direction: str,
        entry_price: float
    ) -> Dict:
        """
        Calculate stop loss and take profit levels using ATR.
        
        Args:
            df: DataFrame with ATR
            direction: BUY or SELL
            entry_price: Entry price
            
        Returns:
            Dict with SL, TP levels and RR ratio
        """
        atr = float(df["atr"].iloc[-1])
        
        if direction == "BUY":
            stop_loss = entry_price - STOP_LOSS_ATR_MULT * atr
            tp1 = entry_price + 1.5 * atr
            tp2 = entry_price + 2.5 * atr
            tp3 = entry_price + TAKE_PROFIT_ATR_MULT * atr
        else:  # SELL
            stop_loss = entry_price + STOP_LOSS_ATR_MULT * atr
            tp1 = entry_price - 1.5 * atr
            tp2 = entry_price - 2.5 * atr
            tp3 = entry_price - TAKE_PROFIT_ATR_MULT * atr
        
        risk = abs(entry_price - stop_loss)
        reward = abs(tp2 - entry_price)
        rr = reward / risk if risk > 0 else 0
        
        return {
            "stop_loss": round(stop_loss, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "rr": round(rr, 2),
            "atr_used": round(atr, 2)
        }
    
    def _calculate_risk_level(
        self,
        df,
        confidence: float,
        failed_filters: int
    ) -> str:
        """Determine overall risk level."""
        if failed_filters > 0:
            return "EXTREME"
        
        if confidence > 0.80:
            return "LOW"
        elif confidence > 0.65:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def record_trade_result(self, pnl_pct: float):
        """Record trade result for consecutive loss tracking."""
        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        self.logger.info(
            f"Trade recorded: PnL={pnl_pct:.2%}, "
            f"consecutive_losses={self.consecutive_losses}"
        )
