"""
═══════════════════════════════════════════════════════════════════
VISUALIZATION - INSTITUTIONAL DASHBOARD GENERATOR
═══════════════════════════════════════════════════════════════════
Professional charting with candlesticks, indicators, signals,
and performance analytics.
═══════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)
logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    Institutional dashboard generator producing professional
    multi-panel charts for XAU/USD analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DashboardGenerator")
        plt.style.use('dark_background')
        
    def generate_dashboard(
        self,
        df: pd.DataFrame,
        signal: Dict,
        backtest_metrics: Optional[Dict] = None,
        save_path: str = None
    ):
        """
        Generate comprehensive institutional dashboard.
        
        Args:
            df: DataFrame with OHLCV and indicators
            signal: Current signal dict
            backtest_metrics: Optional backtest results
            save_path: Path to save figure
        """
        self.logger.info("Generating dashboard...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(5, 2, height_ratios=[3, 1, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Panel 1: Price + Signals
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_panel(ax1, df, signal)
        
        # Panel 2: RSI
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_rsi(ax2, df)
        
        # Panel 3: MACD
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_macd(ax3, df)
        
        # Panel 4: Volume + ATR
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_volume_atr(ax4, df)
        
        # Panel 5: Signal Breakdown
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_signal_breakdown(ax5, signal)
        
        # Panel 6: Equity Curve (if backtest available)
        ax6 = fig.add_subplot(gs[3:, :]) if backtest_metrics else None
        if backtest_metrics and ax6:
            self._plot_equity_curve(ax6, backtest_metrics)
        
        # Add title
        fig.suptitle(
            f"XAU/USD INSTITUTIONAL ANALYSIS DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=14, fontweight='bold', color='cyan'
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            self.logger.info(f"Dashboard saved to {save_path}")
        
        plt.close()
    
    def _plot_price_panel(self, ax, df: pd.DataFrame, signal: Dict):
        """Plot price candles with EMAs and signal markers."""
        # Get last 100 candles
        df_plot = df.tail(100).copy()
        
        # Plot EMAs
        ax.plot(df_plot.index, df_plot['ema_20'], label='EMA 20', color='yellow', linewidth=1.2, alpha=0.8)
        ax.plot(df_plot.index, df_plot['ema_50'], label='EMA 50', color='orange', linewidth=1.2, alpha=0.8)
        ax.plot(df_plot.index, df_plot['ema_200'], label='EMA 200', color='purple', linewidth=1.5, alpha=0.8)
        
        # Plot candlesticks
        colors = ['green' if c >= o else 'red' for c, o in zip(df_plot['Close'], df_plot['Open'])]
        ax.bar(df_plot.index, df_plot['Close'] - df_plot['Open'], 
               bottom=df_plot[['Open', 'Close']].min(axis=1),
               width=0.8, color=colors, alpha=0.7, label='Candles')
        
        # Bollinger Bands
        ax.fill_between(df_plot.index, df_plot['bb_upper'], df_plot['bb_lower'], 
                       alpha=0.15, color='gray', label='BB')
        
        # Signal marker
        direction = signal.get('direction', 'HOLD')
        if direction == 'BUY':
            ax.scatter(df_plot.index[-1], df_plot['Close'].iloc[-1], 
                      color='lime', s=200, marker='^', label='BUY SIGNAL', zorder=10)
        elif direction == 'SELL':
            ax.scatter(df_plot.index[-1], df_plot['Close'].iloc[-1], 
                      color='red', s=200, marker='v', label='SELL SIGNAL', zorder=10)
        
        ax.set_ylabel('Price (USD)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title('XAU/USD Price Action', fontsize=11, fontweight='bold')
    
    def _plot_rsi(self, ax, df: pd.DataFrame):
        """Plot RSI indicator."""
        df_plot = df.tail(100).copy()
        
        ax.plot(df_plot.index, df_plot['rsi'], color='cyan', linewidth=1.5)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Fill zones
        ax.fill_between(df_plot.index, 70, 100, alpha=0.2, color='red')
        ax.fill_between(df_plot.index, 0, 30, alpha=0.2, color='green')
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI', fontsize=9)
        ax.set_title('RSI (14)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_macd(self, ax, df: pd.DataFrame):
        """Plot MACD indicator."""
        df_plot = df.tail(100).copy()
        
        ax.plot(df_plot.index, df_plot['macd_line'], label='MACD', color='cyan', linewidth=1.2)
        ax.plot(df_plot.index, df_plot['macd_signal'], label='Signal', color='orange', linewidth=1.2)
        
        # Histogram
        colors = ['green' if v > 0 else 'red' for v in df_plot['macd_hist']]
        ax.bar(df_plot.index, df_plot['macd_hist'], color=colors, alpha=0.5, width=0.8, label='Histogram')
        
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_ylabel('MACD', fontsize=9)
        ax.set_title('MACD (12,26,9)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_volume_atr(self, ax, df: pd.DataFrame):
        """Plot volume and ATR."""
        df_plot = df.tail(100).copy()
        
        # Volume bars
        colors = ['green' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i] else 'red' 
                 for i in range(len(df_plot))]
        ax.bar(df_plot.index, df_plot['volume_ratio'], color=colors, alpha=0.6, width=0.8, label='Vol Ratio')
        
        ax.axhline(y=1, color='yellow', linestyle='--', alpha=0.7, linewidth=1, label='Avg')
        
        ax.set_ylabel('Volume Ratio', fontsize=9)
        ax.set_title('Relative Volume', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_signal_breakdown(self, ax, signal: Dict):
        """Plot signal breakdown bar chart."""
        breakdown = signal.get('breakdown', {})
        if not breakdown:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        
        categories = list(breakdown.keys())
        values = [breakdown[k] for k in categories]
        colors = ['#2ecc71' if v > 0.55 else '#e74c3c' if v < 0.45 else '#f39c12' for v in values]
        
        y_pos = np.arange(len(categories))
        ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([cat.capitalize() for cat in categories], fontsize=9)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.3)
        ax.axvline(x=0.62, color='green', linestyle=':', alpha=0.5)
        ax.axvline(x=0.38, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Score', fontsize=9)
        ax.set_title('Signal Components', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_equity_curve(self, ax, metrics: Dict):
        """Plot equity curve from backtest."""
        # This would need actual equity curve data
        # For now show summary stats
        ax.axis('off')
        
        textstr = f"""
BACKTEST PERFORMANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Return:     {metrics.get('total_return_pct', 0):+.1f}%
Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.1f}%
Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}
Sortino Ratio:    {metrics.get('sortino_ratio', 0):.2f}
Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%
Profit Factor:    {metrics.get('profit_factor', 0):.2f}
Total Trades:     {metrics.get('total_trades', 0)}
Final Capital:    ${metrics.get('final_capital', 0):,.2f}
        """.strip()
        
        ax.text(0.1, 0.5, textstr, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor='#2a2a3e', edgecolor='cyan', alpha=0.8))
    
    def print_summary(self, signal: Dict, df: pd.DataFrame):
        """Print text-based dashboard summary."""
        print("\n" + "="*70)
        print(f"{Fore.CYAN}📊 XAU/USD INSTITUTIONAL DASHBOARD{Style.RESET_ALL}")
        print("="*70)
        
        # Current price info
        current_price = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        change_color = Fore.GREEN if change >= 0 else Fore.RED
        print(f"\n  💰 Current Price:  {Fore.WHITE}${current_price:.2f}{Style.RESET_ALL}")
        print(f"  📈 Daily Change:   {change_color}{change:+.2f} ({change_pct:+.2f}%){Style.RESET_ALL}")
        
        # Key indicators
        print(f"\n  {Fore.YELLOW}KEY INDICATORS:{Style.RESET_ALL}")
        print(f"    RSI (14):        {df['rsi'].iloc[-1]:.1f}")
        print(f"    ADX (14):        {df['adx'].iloc[-1]:.1f}")
        print(f"    ATR (14):        ${df['atr'].iloc[-1]:.2f}")
        print(f"    MACD Hist:       {df['macd_hist'].iloc[-1]:.4f}")
        print(f"    Bull Score:      {df['bull_score'].iloc[-1]:.0f}/100")
        print(f"    Bear Score:      {df['bear_score'].iloc[-1]:.0f}/100")
        
        # Trend status
        trend_dir = df['trend_direction'].iloc[-1]
        trend_str = "BULLISH" if trend_dir == 1 else "BEARISH" if trend_dir == -1 else "NEUTRAL"
        trend_color = Fore.GREEN if trend_dir == 1 else Fore.RED if trend_dir == -1 else Fore.YELLOW
        print(f"\n  {Fore.YELLOW}TREND STATUS:{Style.RESET_ALL} {trend_color}{trend_str}{Style.RESET_ALL}")
        
        # Volatility regime
        vol_regime = df['volatility_regime'].iloc[-1]
        print(f"  {Fore.YELLOW}VOLATILITY:{Style.RESET_ALL} {vol_regime}")
        
        # Session
        session = df['session'].iloc[-1]
        print(f"  {Fore.YELLOW}SESSION:{Style.RESET_ALL} {session}")
        
        print("\n" + "="*70)
