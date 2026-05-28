"""
Visualizer Module
Creates comprehensive charts and visualizations for XAU/USD analysis
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    EXPORTS_DIR, ANALYSIS_PLOT,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# Use dark style for plots
plt.style.use('dark_background')


class Visualizer:
    """
    Creates comprehensive visualizations for XAU/USD analysis including:
    - Price charts with indicators
    - ML confidence plots
    - Technical indicator panels
    - Backtest equity curves
    """
    
    def __init__(self):
        logger.info("Visualizer initialized")
        
        self.exports_dir = EXPORTS_DIR
        self.analysis_plot = ANALYSIS_PLOT
        
        # Color scheme
        self.colors = {
            'bg': '#1a1a2e',
            'grid': '#2d2d44',
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#888888',
            'ema_short': '#00ccff',
            'ema_medium': '#ffaa00',
            'ema_long': '#ff4488',
            'bb_upper': '#666666',
            'bb_lower': '#666666'
        }
    
    def plot_full_analysis(self, df: pd.DataFrame, signal: Dict[str, any],
                          backtest_results: Dict[str, any]) -> str:
        """
        Create comprehensive 2x2 subplot analysis figure.
        
        Args:
            df: DataFrame with OHLCV and indicators
            signal: Current signal dictionary
            backtest_results: Backtest results dictionary
            
        Returns:
            str: Path to saved plot file
        """
        logger.info("Creating full analysis visualization...")
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(self.colors['bg'])
        
        # Subplot 1: Price Chart with EMAs and Bollinger Bands
        ax1 = plt.subplot(2, 2, 1)
        self._plot_price_chart(ax1, df, signal)
        
        # Subplot 2: ML Confidence
        ax2 = plt.subplot(2, 2, 2)
        self._plot_ml_confidence(ax2, df, signal)
        
        # Subplot 3: Technical Indicators (RSI + MACD)
        ax3 = plt.subplot(2, 2, 3)
        self._plot_technical_indicators(ax3, df)
        
        # Subplot 4: Backtest Equity Curve
        ax4 = plt.subplot(2, 2, 4)
        self._plot_equity_curve(ax4, backtest_results)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filepath = self.analysis_plot
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'], 
                   edgecolor='none', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Analysis plot saved to {filepath}")
        
        return filepath
    
    def _plot_price_chart(self, ax: plt.Axes, df: pd.DataFrame, 
                         signal: Dict[str, any]) -> None:
        """Plot price chart with EMAs, Bollinger Bands, and signals."""
        
        # Get recent data (last 200 days for clarity)
        n = min(200, len(df))
        df_recent = df.iloc[-n:].copy()
        
        # Plot candlesticks using bar chart
        x = range(len(df_recent))
        
        # Calculate candle colors
        colors = [self.colors['bullish'] if c >= o else self.colors['bearish'] 
                 for o, c in zip(df_recent['Open'], df_recent['Close'])]
        
        # Plot candles
        ax.bar(x, df_recent['High'] - df_recent['Low'],
              bottom=df_recent['Low'], width=0.8,
              color=colors, alpha=0.7, label='Price')
        
        # Plot EMAs
        ax.plot(x, df_recent['EMA_20'], color=self.colors['ema_short'], 
               linewidth=1.5, label='EMA 20')
        ax.plot(x, df_recent['EMA_50'], color=self.colors['ema_medium'],
               linewidth=1.5, label='EMA 50')
        ax.plot(x, df_recent['EMA_200'], color=self.colors['ema_long'],
               linewidth=1.5, label='EMA 200')
        
        # Plot Bollinger Bands
        ax.fill_between(x, df_recent['BB_Upper'], df_recent['BB_Lower'],
                       color=self.colors['bb_upper'], alpha=0.2,
                       label='Bollinger Bands')
        
        # Add signal markers
        action = signal.get('action', 'HOLD')
        if action == 'BUY':
            ax.scatter(len(df_recent)-1, df_recent['Close'].iloc[-1],
                      marker='^', s=200, color=self.colors['bullish'],
                      zorder=5, label='BUY Signal')
        elif action == 'SELL':
            ax.scatter(len(df_recent)-1, df_recent['Close'].iloc[-1],
                      marker='v', s=200, color=self.colors['bearish'],
                      zorder=5, label='SELL Signal')
        
        # Formatting
        ax.set_title('XAU/USD Price Chart', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Days', fontsize=10)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg'])
        
        # Format x-axis
        tick_locs = range(0, len(df_recent), 20)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"D-{len(df_recent)-i}" for i in tick_locs], rotation=45)
    
    def _plot_ml_confidence(self, ax: plt.Axes, df: pd.DataFrame,
                           signal: Dict[str, any]) -> None:
        """Plot ML confidence over time."""
        
        # For simplicity, show current confidence prominently
        # In production, this would show historical ML predictions
        confidence = signal.get('confidence', 50)
        final_score = signal.get('final_score', 0.5)
        
        # Create a simple gauge-like visualization
        x = np.linspace(0, 1, 100)
        
        # Background zones
        ax.axvspan(0, 0.35, color=self.colors['bearish'], alpha=0.3, label='SELL Zone')
        ax.axvspan(0.35, 0.65, color=self.colors['neutral'], alpha=0.2, label='HOLD Zone')
        ax.axvspan(0.65, 1, color=self.colors['bullish'], alpha=0.3, label='BUY Zone')
        
        # Threshold lines
        ax.axvline(0.35, color=self.colors['bearish'], linestyle='--', linewidth=2)
        ax.axvline(0.65, color=self.colors['bullish'], linestyle='--', linewidth=2)
        
        # Current score line
        ax.axvline(final_score, color='white', linestyle='-', linewidth=3,
                  label=f'Current: {final_score:.2f}')
        
        # Confidence indicator
        y_pos = 0.5
        ax.text(final_score, y_pos, f'{confidence:.1f}%', 
               ha='center', va='center', fontsize=20, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Formatting
        ax.set_title('ML Signal Score', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg'])
        
        # Hide y-axis
        ax.set_yticks([])
    
    def _plot_technical_indicators(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot RSI and MACD indicators."""
        
        # Get recent data
        n = min(200, len(df))
        df_recent = df.iloc[-n:].copy()
        
        # Create twin axes for RSI and MACD
        ax_rsi = ax
        ax_macd = ax.twinx()
        
        x = range(len(df_recent))
        
        # Plot RSI (left axis)
        rsi_color = self.colors['bullish']
        ax_rsi.plot(x, df_recent['RSI'], color=rsi_color, linewidth=1.5, label='RSI')
        
        # RSI zones
        ax_rsi.axhline(70, color=self.colors['bearish'], linestyle='--', alpha=0.5)
        ax_rsi.axhline(30, color=self.colors['bullish'], linestyle='--', alpha=0.5)
        ax_rsi.fill_between(x, 30, 70, color=self.colors['neutral'], alpha=0.1)
        
        ax_rsi.set_ylabel('RSI', fontsize=10, color=rsi_color)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_yticks([30, 50, 70])
        
        # Plot MACD histogram (right axis)
        macd_hist = df_recent['MACD_Hist']
        colors = [self.colors['bullish'] if h > 0 else self.colors['bearish'] 
                 for h in macd_hist]
        ax_macd.bar(x, macd_hist, color=colors, alpha=0.7, width=1, label='MACD Hist')
        ax_macd.axhline(0, color='white', linestyle='-', linewidth=0.5)
        
        macd_color = self.colors['ema_short']
        ax_macd.plot(x, df_recent['MACD'], color=macd_color, linewidth=1, label='MACD')
        
        ax_macd.set_ylabel('MACD', fontsize=10, color=macd_color)
        
        # Combined title
        ax.set_title('Technical Indicators (RSI + MACD)', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Days', fontsize=10)
        
        # Grid and background
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg'])
        
        # Format x-axis
        tick_locs = range(0, len(df_recent), 40)
        ax.set_xticks(tick_locs)
        
        # Combined legend
        lines_rsi, labels_rsi = ax_rsi.get_legend_handles_labels()
        lines_macd, labels_macd = ax_macd.get_legend_handles_labels()
        ax.legend(lines_rsi + lines_macd, labels_rsi + labels_macd,
                 loc='upper left', fontsize=8)
    
    def _plot_equity_curve(self, ax: plt.Axes, 
                          backtest_results: Dict[str, any]) -> None:
        """Plot backtest equity curve with drawdown."""
        
        equity_curve = backtest_results.get('equity_curve', [])
        
        if not equity_curve:
            ax.text(0.5, 0.5, 'No backtest data available',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)
            ax.set_facecolor(self.colors['bg'])
            return
        
        # Extract equity values
        dates = [e['date'] for e in equity_curve]
        equities = [e['equity'] for e in equity_curve]
        
        x = range(len(equities))
        
        # Plot equity curve
        ax.plot(x, equities, color=self.colors['bullish'], 
               linewidth=2, label='Equity')
        
        # Calculate and plot drawdown
        peak = equities[0]
        drawdowns = []
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)
        
        ax.fill_between(x, 0, [-dd for dd in drawdowns],
                       color=self.colors['bearish'], alpha=0.3,
                       label='Drawdown')
        
        # Add metrics annotation
        total_return = backtest_results.get('total_return', 0)
        max_dd = backtest_results.get('max_drawdown', 0)
        sharpe = backtest_results.get('sharpe_ratio', 0)
        win_rate = backtest_results.get('win_rate', 0)
        
        metrics_text = (
            f"Return: {total_return:+.1f}%\n"
            f"Max DD: {max_dd:.1f}%\n"
            f"Sharpe: {sharpe:.2f}\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Formatting
        ax.set_title('Backtest Equity Curve', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Trade Number', fontsize=10)
        ax.set_ylabel('Equity ($)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg'])
    
    def plot_single_indicator(self, df: pd.DataFrame, indicator: str,
                             title: str = None) -> str:
        """
        Plot a single indicator for detailed analysis.
        
        Args:
            df: DataFrame with indicator
            indicator: Column name of indicator
            title: Plot title
            
        Returns:
            str: Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        
        n = min(200, len(df))
        df_recent = df.iloc[-n:].copy()
        
        x = range(len(df_recent))
        
        # Determine color based on indicator type
        if 'RSI' in indicator or 'Stoch' in indicator:
            color = self.colors['ema_short']
        elif 'MACD' in indicator:
            color = self.colors['bullish']
        elif 'ATR' in indicator or 'BB' in indicator:
            color = self.colors['ema_medium']
        else:
            color = 'white'
        
        ax.plot(x, df_recent[indicator], color=color, linewidth=1.5)
        
        ax.set_title(title or indicator, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Days', fontsize=10)
        ax.set_ylabel(indicator, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        filepath = os.path.join(self.exports_dir, f"{indicator}_chart.png")
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'],
                   bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Single indicator plot saved: {filepath}")
        
        return filepath
