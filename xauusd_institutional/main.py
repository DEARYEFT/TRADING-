#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
XAU/USD INSTITUTIONAL AI TRADING SYSTEM
═══════════════════════════════════════════════════════════════════
Production-grade institutional analysis platform for gold futures.
Combines technical analysis, ML ensemble, news sentiment, and 
professional risk management.

Usage:
    pip install -r requirements.txt
    python main.py

Author: Institutional Quant Team
Version: 2.0.0
═══════════════════════════════════════════════════════════════════
"""

import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style, init

# Suppress warnings
warnings.filterwarnings('ignore')
init(autoreset=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOGS_DIR, EXPORT_DIR, CACHE_DIR, SYMBOL,
    ENABLE_MACRO_FILTER, ENABLE_SMC_FILTER, ENABLE_NEWS_FILTER
)
from data.data_engine import DataEngine
from features.feature_engine import FeatureEngine
from ml_model import MLModel
from news_engine import NewsEngine
from signal_engine import SignalEngine
from risk_manager import RiskManager
from backtest import Backtester
from visualization import DashboardGenerator


def setup_logging():
    """Configure logging infrastructure."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'system.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def print_banner():
    """Print system banner."""
    print("\n" + "="*70)
    print(f"{Fore.CYAN}╔════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║   XAU/USD INSTITUTIONAL AI TRADING SYSTEM v2.0             ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║   Production-Grade Analysis Platform                       ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print("="*70)
    print(f"  {Fore.WHITE}📅 Timestamp:{Style.RESET_ALL} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  {Fore.WHITE}🎯 Symbol:{Style.RESET_ALL}    {SYMBOL}")
    print(f"  {Fore.WHITE}⚙️  Mode:{Style.RESET_ALL}     ANALYSIS + BACKTEST")
    print("="*70 + "\n")


def main():
    """Main execution pipeline."""
    logger = setup_logging()
    print_banner()
    
    try:
        # ───────────────────────────────────────────────────────────
        # PHASE 1: DATA ACQUISITION
        # ───────────────────────────────────────────────────────────
        print(f"\n{Fore.BLUE}[1/6]{Style.RESET_ALL} Loading market data...")
        
        data_engine = DataEngine()
        timeframes = data_engine.load_multi_timeframe()
        df = timeframes["1D"]  # Use daily timeframe
        
        print(f"      ✓ Loaded {len(df)} candles from {df.index[0].date()} to {df.index[-1].date()}")
        
        # ───────────────────────────────────────────────────────────
        # PHASE 2: FEATURE ENGINEERING
        # ───────────────────────────────────────────────────────────
        print(f"\n{Fore.BLUE}[2/6]{Style.RESET_ALL} Computing technical indicators...")
        
        feature_engine = FeatureEngine()
        df = feature_engine.compute_all(df)
        
        indicator_count = len([c for c in df.columns if c not in ['Open','High','Low','Close','Volume','timeframe','session']])
        print(f"      ✓ Computed {indicator_count} technical indicators")
        
        # ───────────────────────────────────────────────────────────
        # PHASE 3: MACHINE LEARNING
        # ───────────────────────────────────────────────────────────
        print(f"\n{Fore.BLUE}[3/6]{Style.RESET_ALL} Training ML ensemble...")
        
        ml_model = MLModel()
        
        # Try to load existing model
        if not ml_model.load_model():
            cv_results = ml_model.train(df, forward=1)
            
            # Print CV results
            print("      Cross-validation results:")
            for name, acc in sorted(cv_results.items(), key=lambda x: x[1], reverse=True):
                print(f"        • {name}: {acc:.4f}")
        else:
            print("      ✓ Loaded pre-trained model")
        
        # Get current prediction
        latest_proba = ml_model.predict(df.tail(1), forward=1)[0]
        _, uncertainty = ml_model.predict_with_uncertainty(df.tail(1), forward=1)
        
        print(f"      ✓ ML Prediction: {latest_proba:.2%} bullish (±{uncertainty[0]:.2%} uncertainty)")
        
        # ───────────────────────────────────────────────────────────
        # PHASE 4: NEWS SENTIMENT
        # ───────────────────────────────────────────────────────────
        print(f"\n{Fore.BLUE}[4/6]{Style.RESET_ALL} Analyzing news sentiment...")
        
        news_engine = NewsEngine()
        sentiment = news_engine.get_market_sentiment()
        
        print(f"      ✓ Sentiment: {sentiment['label']} (norm={sentiment['norm']:.2f})")
        print(f"      ✓ Articles analyzed: {sentiment['article_count']}")
        
        # ───────────────────────────────────────────────────────────
        # PHASE 5: SIGNAL GENERATION
        # ───────────────────────────────────────────────────────────
        print(f"\n{Fore.BLUE}[5/6]{Style.RESET_ALL} Generating trading signal...")
        
        signal_engine = SignalEngine()
        signal = signal_engine.generate_final_signal(
            df=df,
            ml_proba=float(latest_proba),
            sentiment=sentiment
        )
        
        # Format and display signal
        signal_engine.format_signal_output(signal)
        
        # ───────────────────────────────────────────────────────────
        # PHASE 6: RISK ASSESSMENT
        # ───────────────────────────────────────────────────────────
        print(f"\n{Fore.BLUE}[6/6]{Style.RESET_ALL} Performing risk assessment...")
        
        risk_manager = RiskManager()
        risk_assessment = risk_manager.assess_risk(
            signal=signal,
            df=df,
            confidence=signal['confidence']
        )
        
        if risk_assessment['approved']:
            print(f"      {Fore.GREEN}✓ TRADE APPROVED{Style.RESET_ALL}")
            print(f"      Risk Level: {risk_assessment['risk_level']}")
            print(f"      Position Size: {risk_assessment['position_size_pct']*100:.1f}% of capital")
            print(f"      Stop Loss: ${risk_assessment['stop_loss_price']:.2f}")
            print(f"      Take Profit: ${risk_assessment['take_profit_prices'][1]:.2f}")
            print(f"      Risk/Reward: {risk_assessment['risk_reward_ratio']:.2f}:1")
        else:
            print(f"      {Fore.RED}✗ TRADE REJECTED{Style.RESET_ALL}")
            print(f"      Reason: {risk_assessment['override_reason']}")
        
        # ───────────────────────────────────────────────────────────
        # BACKTEST EXECUTION
        # ───────────────────────────────────────────────────────────
        print(f"\n{Fore.YELLOW}{'─'*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Running historical backtest...{Style.RESET_ALL}\n")
        
        backtester = Backtester()
        backtest_results = backtester.run(df)
        
        metrics = backtester.calculate_metrics(
            backtest_results['trades'],
            backtest_results['equity_curve']
        )
        
        backtester.print_report(metrics)
        
        # ───────────────────────────────────────────────────────────
        # DASHBOARD GENERATION
        # ───────────────────────────────────────────────────────────
        print(f"\n{Fore.YELLOW}Generating dashboard...{Style.RESET_ALL}")
        
        viz = DashboardGenerator()
        viz.print_summary(signal, df)
        
        dashboard_path = EXPORT_DIR / "dashboard.png"
        viz.generate_dashboard(
            df=df,
            signal=signal,
            backtest_metrics=metrics,
            save_path=str(dashboard_path)
        )
        
        print(f"      ✓ Dashboard saved to: {dashboard_path}")
        
        # ───────────────────────────────────────────────────────────
        # FINAL SUMMARY
        # ───────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print(f"{Fore.GREEN}✅ ANALYSIS COMPLETE{Style.RESET_ALL}")
        print("="*70)
        print(f"\n  📊 Signal:        {signal['direction']} ({signal['quality']})")
        print(f"  🎯 Confidence:    {signal['confidence']*100:.1f}%")
        print(f"  💰 Entry Zone:    ${float(df['Close'].iloc[-1]):.2f}")
        print(f"  🛑 Stop Loss:     ${risk_assessment['stop_loss_price']:.2f}")
        print(f"  🎯 Take Profit:   ${risk_assessment['take_profit_prices'][1]:.2f}")
        print(f"  📈 Backtest RR:   {metrics['total_return_pct']:+.1f}%")
        print(f"  📉 Max DD:        {metrics['max_drawdown_pct']:.1f}%")
        print(f"  ⚡ Sharpe:        {metrics['sharpe_ratio']:.2f}")
        print("\n" + "="*70)
        
        if signal['no_trade_warning']:
            print(f"\n{Fore.YELLOW}⚠️  WARNING: Low confidence signal - Consider waiting for better setup{Style.RESET_ALL}\n")
        
        return {
            'signal': signal,
            'risk': risk_assessment,
            'backtest': metrics,
            'ml_prediction': float(latest_proba),
            'sentiment': sentiment
        }
        
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"\n{Fore.RED}❌ FATAL ERROR: {e}{Style.RESET_ALL}")
        print("\nCheck logs/system.log for details.")
        raise


if __name__ == "__main__":
    result = main()
