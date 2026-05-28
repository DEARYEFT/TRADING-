"""
═══════════════════════════════════════════════════════════════════
BACKTEST ENGINE - WALK-FORWARD INSTITUTIONAL BACKTESTER
═══════════════════════════════════════════════════════════════════
Professional backtesting with realistic execution, commissions,
slippage, and comprehensive performance metrics.
═══════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from colorama import Fore, Style, init

from config import (
    INITIAL_CAPITAL, COMMISSION_PER_LOT, SLIPPAGE_BPS, SPREAD_BPS,
    MAX_HOLD_BARS, STOP_LOSS_ATR_MULT, TAKE_PROFIT_ATR_MULT
)

init(autoreset=True)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Institutional walk-forward backtester with realistic execution
    modeling and comprehensive performance analytics.
    """
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(f"{__name__}.Backtester")
    
    def run(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame = None
    ) -> Dict:
        """
        Run walk-forward backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV and indicators
            signals_df: Optional pre-computed signals
            
        Returns:
            Dict with trades list and equity curve
        """
        self.logger.info("Running backtest...")
        
        capital = self.initial_capital
        trades = []
        equity_curve = [capital]
        position = None
        
        # Iterate through bars (skip first 200 for indicator warmup)
        for i in range(200, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # Generate signal if not provided
            if signals_df is not None:
                signal = signals_df.iloc[i]
            else:
                # Simple signal logic for backtest
                signal = self._generate_simple_signal(row)
            
            # Check exit conditions for open position
            if position is not None:
                exit_result = self._check_exit(
                    position, row, next_row, i
                )
                
                if exit_result["exited"]:
                    # Calculate PnL
                    pnl = exit_result["pnl"]
                    capital += pnl
                    
                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": row.name,
                        "direction": position["direction"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_result["exit_price"],
                        "pnl": pnl,
                        "pnl_pct": pnl / position["notional"] * 100,
                        "duration": i - position["entry_idx"],
                        "exit_reason": exit_result["reason"]
                    })
                    
                    position = None
            
            # Enter new position if none open and signal triggered
            if position is None and signal["action"] != "HOLD":
                position = self._enter_position(
                    direction=signal["action"],
                    entry_price=float(next_row["Open"]),
                    capital=capital,
                    atr=float(row["atr"]),
                    idx=i,
                    date=row.name
                )
            
            equity_curve.append(capital)
        
        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "final_capital": capital,
            "total_bars": len(df) - 200
        }
    
    def _generate_simple_signal(self, row) -> Dict:
        """Generate simple signal based on technical indicators."""
        # Buy signal
        if (row["rsi"] < 35 and 
            row["macd_hist"] > 0 and 
            row["Close"] > row["ema_50"]):
            return {"action": "BUY", "confidence": 0.7}
        
        # Sell signal
        elif (row["rsi"] > 65 and 
              row["macd_hist"] < 0 and 
              row["Close"] < row["ema_50"]):
            return {"action": "SELL", "confidence": 0.7}
        
        return {"action": "HOLD", "confidence": 0}
    
    def _enter_position(
        self,
        direction: str,
        entry_price: float,
        capital: float,
        atr: float,
        idx: int,
        date
    ) -> Dict:
        """Enter a new position with realistic costs."""
        # Position size (10% of capital per trade)
        position_value = capital * 0.10
        
        # Calculate lots (1 lot XAU = 100 oz)
        lots = position_value / (entry_price * 100)
        lots = max(0.01, round(lots, 2))
        
        # Costs
        commission = lots * COMMISSION_PER_LOT
        spread_cost = entry_price * (SPREAD_BPS / 10000) * lots * 100
        slippage_cost = entry_price * (SLIPPAGE_BPS / 10000) * lots * 100
        
        total_costs = commission + spread_cost + slippage_cost
        
        # Adjust entry price for costs
        effective_entry = entry_price + spread_cost / (lots * 100)
        
        # Stop loss and take profit
        if direction == "BUY":
            stop_loss = entry_price - STOP_LOSS_ATR_MULT * atr
            take_profit = entry_price + TAKE_PROFIT_ATR_MULT * atr
        else:
            stop_loss = entry_price + STOP_LOSS_ATR_MULT * atr
            take_profit = entry_price - TAKE_PROFIT_ATR_MULT * atr
        
        return {
            "direction": direction,
            "entry_price": effective_entry,
            "raw_entry": entry_price,
            "lots": lots,
            "notional": lots * entry_price * 100,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_idx": idx,
            "entry_date": date,
            "costs": total_costs,
            "max_hold": idx + MAX_HOLD_BARS
        }
    
    def _check_exit(
        self,
        position: Dict,
        row: pd.Series,
        next_row: pd.Series,
        current_idx: int
    ) -> Dict:
        """Check if position should be exited."""
        high = float(row["High"])
        low = float(row["Low"])
        close = float(next_row["Open"])  # Exit at next open
        
        direction = position["direction"]
        sl = position["stop_loss"]
        tp = position["take_profit"]
        
        exit_price = None
        reason = None
        
        # Check stop loss
        if direction == "BUY":
            if low <= sl:
                exit_price = sl
                reason = "STOP_LOSS"
            elif high >= tp:
                exit_price = tp
                reason = "TAKE_PROFIT"
            elif current_idx >= position["max_hold"]:
                exit_price = close
                reason = "TIME_EXIT"
        else:  # SELL
            if high >= sl:
                exit_price = sl
                reason = "STOP_LOSS"
            elif low <= tp:
                exit_price = tp
                reason = "TAKE_PROFIT"
            elif current_idx >= position["max_hold"]:
                exit_price = close
                reason = "TIME_EXIT"
        
        if exit_price is None:
            return {"exited": False}
        
        # Calculate PnL
        if direction == "BUY":
            pnl = (exit_price - position["raw_entry"]) * position["lots"] * 100
        else:
            pnl = (position["raw_entry"] - exit_price) * position["lots"] * 100
        
        # Subtract costs
        pnl -= position["costs"]
        
        return {
            "exited": True,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason
        }
    
    def calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float]
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return self._empty_metrics()
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit/Loss calculations
        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Returns
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Drawdown
        peak = equity_curve[0]
        max_dd = 0
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe Ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino Ratio
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0 and neg_returns.std() > 0:
            sortino = (returns.mean() / neg_returns.std()) * np.sqrt(252)
        else:
            sortino = 0
        
        # Calmar Ratio
        calmar = total_return / max_dd if max_dd > 0 else 0
        
        # Average trade
        avg_pnl = np.mean([t["pnl"] for t in trades])
        avg_duration = np.mean([t["duration"] for t in trades])
        
        # Largest win/loss
        largest_win = max([t["pnl"] for t in trades])
        largest_loss = min([t["pnl"] for t in trades])
        
        return {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "avg_pnl": avg_pnl,
            "avg_duration": avg_duration,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "final_capital": equity_curve[-1],
            "initial_capital": self.initial_capital
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_return": 0,
            "total_return_pct": 0,
            "max_drawdown": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0,
            "avg_pnl": 0,
            "avg_duration": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "gross_profit": 0,
            "gross_loss": 0,
            "final_capital": self.initial_capital,
            "initial_capital": self.initial_capital
        }
    
    def print_report(self, metrics: Dict):
        """Print formatted backtest report."""
        print("\n" + "="*60)
        print(f"{Fore.CYAN}📊 BACKTEST RESULTS REPORT{Style.RESET_ALL}")
        print("="*60)
        print(f"  Initial Capital:    ${metrics['initial_capital']:,.2f}")
        print(f"  Final Capital:      ${metrics['final_capital']:,.2f}")
        print("-"*60)
        print(f"  Total Trades:       {metrics['total_trades']}")
        
        # Win rate with color
        wr = metrics['win_rate'] * 100
        wr_color = Fore.GREEN if wr > 50 else Fore.RED
        print(f"  Win Rate:           {wr_color}{wr:.1f}%{Style.RESET_ALL}")
        
        # Profit factor
        pf = metrics['profit_factor']
        pf_color = Fore.GREEN if pf > 1.5 else Fore.YELLOW if pf > 1 else Fore.RED
        print(f"  Profit Factor:      {pf_color}{pf:.2f}{Style.RESET_ALL}")
        
        # Drawdown
        dd = metrics['max_drawdown_pct']
        dd_color = Fore.GREEN if dd < 15 else Fore.YELLOW if dd < 25 else Fore.RED
        print(f"  Max Drawdown:       {dd_color}{dd:.1f}%{Style.RESET_ALL}")
        
        # Sharpe
        sr = metrics['sharpe_ratio']
        sr_color = Fore.GREEN if sr > 1 else Fore.YELLOW if sr > 0.5 else Fore.RED
        print(f"  Sharpe Ratio:       {sr_color}{sr:.2f}{Style.RESET_ALL}")
        
        # Sortino
        sor = metrics['sortino_ratio']
        sor_color = Fore.GREEN if sor > 1.5 else Fore.YELLOW if sor > 0.7 else Fore.RED
        print(f"  Sortino Ratio:      {sor_color}{sor:.2f}{Style.RESET_ALL}")
        
        # Calmar
        cal = metrics['calmar_ratio']
        cal_color = Fore.GREEN if cal > 0.5 else Fore.YELLOW if cal > 0.2 else Fore.RED
        print(f"  Calmar Ratio:       {cal_color}{cal:.2f}{Style.RESET_ALL}")
        
        print("-"*60)
        
        # Return
        ret = metrics['total_return_pct']
        ret_color = Fore.GREEN if ret > 0 else Fore.RED
        print(f"  Total Return:       {ret_color}{ret:+.1f}%{Style.RESET_ALL}")
        print(f"  Avg Trade PnL:      ${metrics['avg_pnl']:.2f}")
        print(f"  Avg Duration:       {metrics['avg_duration']:.1f} bars")
        print(f"  Largest Win:        ${metrics['largest_win']:.2f}")
        print(f"  Largest Loss:       ${metrics['largest_loss']:.2f}")
        print("="*60 + "\n")
