"""
Backtest Module
Historical simulation of trading strategy performance
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INITIAL_CAPITAL, STOP_LOSS_ATR_MULT, TAKE_PROFIT_ATR_MULT,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Runs backtests on historical data to evaluate strategy performance.
    Simulates trades with realistic entry/exit conditions.
    """
    
    def __init__(self, initial_capital: float = None):
        logger.info("Backtester initialized")
        
        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.stop_loss_atr_mult = STOP_LOSS_ATR_MULT
        self.take_profit_atr_mult = TAKE_PROFIT_ATR_MULT
        
        # Trade log
        self.trades = []
        self.equity_curve = []
    
    def run(self, df: pd.DataFrame, signals: Optional[pd.Series] = None) -> Dict[str, any]:
        """
        Run backtest simulation on historical data.
        
        Args:
            df: DataFrame with OHLCV and indicators
            signals: Optional pre-computed signals (BUY/SELL/HOLD)
            
        Returns:
            dict: Backtest results and metrics
        """
        logger.info(f"Starting backtest on {len(df)} bars...")
        
        if len(df) < 100:
            logger.error("Insufficient data for backtest (minimum 100 bars)")
            raise ValueError("Insufficient data for backtest")
        
        # Reset state
        self.trades = []
        self.equity_curve = []
        
        capital = self.initial_capital
        position = 0  # Current position size (positive=long, negative=short)
        entry_price = 0
        entry_date = None
        stop_loss = None
        take_profit = None
        
        # Generate signals if not provided
        if signals is None:
            signals = self._generate_signals(df)
        
        # Iterate through data (skip first few rows due to indicator warmup)
        start_idx = max(200, len(df) // 4)  # Start after enough history
        
        for i in range(start_idx, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            current_price = row['Close']
            atr = row.get('ATR', current_price * 0.01)
            signal = signals.iloc[i] if hasattr(signals, 'iloc') else signals.get(i, 'HOLD')
            
            # Update equity curve
            current_equity = capital + (position * (current_price - entry_price) if position != 0 else 0)
            self.equity_curve.append({
                'date': row.name,
                'equity': current_equity,
                'price': current_price,
                'position': position
            })
            
            # Check exit conditions for open positions
            if position > 0:  # Long position
                # Check stop loss
                if current_price <= stop_loss:
                    pnl = (stop_loss - entry_price) * position
                    capital += pnl
                    self._record_trade(entry_date, row.name, entry_price, stop_loss, 
                                      position, pnl, 'STOP_LOSS')
                    position = 0
                    
                # Check take profit
                elif current_price >= take_profit:
                    pnl = (take_profit - entry_price) * position
                    capital += pnl
                    self._record_trade(entry_date, row.name, entry_price, take_profit,
                                      position, pnl, 'TAKE_PROFIT')
                    position = 0
            
            elif position < 0:  # Short position
                # Check stop loss (for short, SL is above entry)
                if current_price >= stop_loss:
                    pnl = (entry_price - stop_loss) * abs(position)
                    capital += pnl
                    self._record_trade(entry_date, row.name, entry_price, stop_loss,
                                      abs(position), pnl, 'STOP_LOSS')
                    position = 0
                    
                # Check take profit
                elif current_price <= take_profit:
                    pnl = (entry_price - take_profit) * abs(position)
                    capital += pnl
                    self._record_trade(entry_date, row.name, entry_price, take_profit,
                                      abs(position), pnl, 'TAKE_PROFIT')
                    position = 0
            
            # Enter new position if none exists
            if position == 0 and signal != 'HOLD':
                # Risk 2% of capital per trade
                risk_amount = capital * 0.02
                position_size = risk_amount / (atr * self.stop_loss_atr_mult)
                
                if signal == 'BUY':
                    position = position_size
                    entry_price = next_row['Open']  # Enter at next bar open
                    entry_date = next_row.name
                    stop_loss = entry_price - (atr * self.stop_loss_atr_mult)
                    take_profit = entry_price + (atr * self.take_profit_atr_mult)
                    
                elif signal == 'SELL':
                    position = -position_size
                    entry_price = next_row['Open']
                    entry_date = next_row.name
                    stop_loss = entry_price + (atr * self.stop_loss_atr_mult)
                    take_profit = entry_price - (atr * self.take_profit_atr_mult)
        
        # Close any remaining position at the end
        if position != 0:
            final_price = df.iloc[-1]['Close']
            final_date = df.index[-1]
            
            if position > 0:
                pnl = (final_price - entry_price) * position
            else:
                pnl = (entry_price - final_price) * abs(position)
            
            capital += pnl
            self._record_trade(entry_date, final_date, entry_price, final_price,
                              abs(position), pnl, 'END_OF_DATA')
        
        # Calculate final equity
        final_equity = capital
        
        # Compute metrics
        results = self.calculate_metrics(df, final_equity)
        
        logger.info(f"Backtest complete. Final equity: ${final_equity:.2f}, "
                   f"Return: {results['total_return']:.2f}%")
        
        return results
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate simple signals based on technical indicators.
        This is a fallback if no signals are provided.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            pd.Series: Signal series (BUY/SELL/HOLD)
        """
        signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            rsi = row.get('RSI', 50)
            macd = row.get('MACD', 0)
            macd_signal = row.get('MACD_Signal', 0)
            
            # Simple RSI + MACD strategy
            if rsi < 30 and macd > macd_signal:
                signals.append('BUY')
            elif rsi > 70 and macd < macd_signal:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return pd.Series(signals, index=df.index)
    
    def _record_trade(self, entry_date, exit_date, entry_price: float,
                     exit_price: float, size: float, pnl: float,
                     exit_reason: str) -> None:
        """
        Record a completed trade.
        
        Args:
            entry_date: Trade entry date
            exit_date: Trade exit date
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            pnl: Profit/Loss
            exit_reason: Reason for exit
        """
        trade = {
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'duration': (exit_date - entry_date).days if hasattr(entry_date, '__sub__') else 1
        }
        
        self.trades.append(trade)
        logger.debug(f"Trade recorded: {exit_reason}, PnL=${pnl:.2f}")
    
    def calculate_metrics(self, df: pd.DataFrame, final_equity: float) -> Dict[str, any]:
        """
        Calculate comprehensive backtest metrics.
        
        Args:
            df: Original DataFrame
            final_equity: Final equity value
            
        Returns:
            dict: Performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'avg_trade_duration': 0
            }
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Gross profit and loss
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total return
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        # Average trade duration
        avg_duration = np.mean([t['duration'] for t in self.trades]) if self.trades else 0
        
        # Maximum drawdown from equity curve
        max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Largest win and loss
        largest_win = max((t['pnl'] for t in self.trades), default=0)
        largest_loss = min((t['pnl'] for t in self.trades), default=0)
        
        # Average win and loss
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_return': round(total_return, 2),
            'final_equity': round(final_equity, 2),
            'avg_trade_duration': round(avg_duration, 1),
            'largest_win': round(largest_win, 2),
            'largest_loss': round(largest_loss, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Returns:
            float: Maximum drawdown percentage
        """
        if not self.equity_curve:
            return 0.0
        
        equities = [e['equity'] for e in self.equity_curve]
        
        peak = equities[0]
        max_dd = 0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Returns:
            float: Sharpe ratio
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate daily returns
        equities = [e['equity'] for e in self.equity_curve]
        returns = pd.Series(equities).pct_change().dropna()
        
        if returns.std() == 0:
            return 0.0
        
        # Annualize (assuming daily data)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        return sharpe
    
    def print_report(self, results: Dict[str, any]) -> None:
        """
        Print formatted backtest report to console.
        
        Args:
            results: Backtest results dictionary
        """
        print("\n" + "=" * 60)
        print("📊 BACKTEST REPORT")
        print("=" * 60)
        
        print(f"\n💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"📈 Final Equity: ${results['final_equity']:,.2f}")
        print(f"📊 Total Return: {results['total_return']:+.2f}%")
        
        print("\n" + "-" * 60)
        print("TRADE STATISTICS")
        print("-" * 60)
        print(f"📝 Total Trades: {results['total_trades']}")
        print(f"✅ Winning: {results['winning_trades']} | ❌ Losing: {results['losing_trades']}")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"📐 Profit Factor: {results['profit_factor']}")
        
        print("\n" + "-" * 60)
        print("RISK METRICS")
        print("-" * 60)
        print(f"📉 Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"📈 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"⏱️  Avg Trade Duration: {results['avg_trade_duration']:.1f} days")
        
        print("\n" + "-" * 60)
        print("PROFIT/LOSS BREAKDOWN")
        print("-" * 60)
        print(f"💵 Gross Profit: ${results['gross_profit']:,.2f}")
        print(f"🔻 Gross Loss: ${results['gross_loss']:,.2f}")
        print(f"🏆 Largest Win: ${results['largest_win']:,.2f}")
        print(f"💸 Largest Loss: ${results['largest_loss']:,.2f}")
        print(f"📊 Avg Win: ${results['avg_win']:,.2f}")
        print(f"📊 Avg Loss: ${results['avg_loss']:,.2f}")
        
        print("\n" + "=" * 60)
        
        # Show last 5 trades
        if self.trades:
            print("\nRECENT TRADES:")
            for trade in self.trades[-5:]:
                direction = "LONG" if trade['pnl'] > 0 else "SHORT" if trade['pnl'] < 0 else "FLAT"
                emoji = "✅" if trade['pnl'] > 0 else "❌" if trade['pnl'] < 0 else "➖"
                print(f"  {emoji} {trade['exit_date']}: {direction} PnL=${trade['pnl']:+.2f} ({trade['exit_reason']})")
        
        print("=" * 60)
