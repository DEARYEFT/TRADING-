"""
XAU/USD Analyzer - Main Entry Point
Complete trading analysis system with ML, technical indicators, and news sentiment
"""

import argparse
import logging
import sys
import os
import time
import csv
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SYMBOL, PERIOD, INTERVAL, CONFIDENCE_THRESHOLD,
    LOGS_DIR, EXPORTS_DIR, SIGNALS_LOG, SIGNALS_HISTORY,
    LIVE_UPDATE_INTERVAL, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
)
from data.data_engine import DataEngine
from features.feature_engine import FeatureEngine
from models.ml_model import MLModel
from news_engine import NewsEngine
from signal_engine import SignalEngine
from risk_manager import RiskManager
from backtest import Backtester
from visualizer import Visualizer
from strategies.ensemble import EnsembleStrategy

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def print_banner():
    """Print ASCII art banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     ███████╗██╗   ██╗███████╗ ██████╗ ██╗  ██╗            ║
║     ██╔════╝╚██╗ ██╔╝██╔════╝██╔═══██╗██║  ██║            ║
║     █████╗   ╚████╔╝ █████╗  ██║   ██║███████║            ║
║     ██╔══╝    ╚██╔╝  ██╔══╝  ██║   ██║██╔══██║            ║
║     ███████╗   ██║   ██║     ╚██████╔╝██║  ██║            ║
║     ╚══════╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝  ╚═╝            ║
║                                                           ║
║              XAU/USD ANALYZER v1.0                        ║
║     AI-Powered Gold Trading Analysis System               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def export_to_csv(signal: dict, results: dict) -> str:
    """Export signal and results to CSV."""
    filepath = os.path.join(EXPORTS_DIR, f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write signal info
            writer.writerow(['=== CURRENT SIGNAL ==='])
            writer.writerow(['Action', signal.get('action', 'N/A')])
            writer.writerow(['Confidence', f"{signal.get('confidence', 0):.1f}%"])
            writer.writerow(['Final Score', signal.get('final_score', 0)])
            writer.writerow(['Timestamp', signal.get('timestamp', datetime.now())])
            writer.writerow([])
            
            # Write breakdown
            writer.writerow(['=== SIGNAL BREAKDOWN ==='])
            breakdown = signal.get('breakdown', {})
            for component, data in breakdown.items():
                writer.writerow([component.upper(), data.get('label', 'N/A'), 
                               f"{data.get('score', 0):.3f}"])
            writer.writerow([])
            
            # Write backtest results
            writer.writerow(['=== BACKTEST RESULTS ==='])
            writer.writerow(['Total Trades', results.get('total_trades', 0)])
            writer.writerow(['Win Rate', f"{results.get('win_rate', 0):.1f}%"])
            writer.writerow(['Total Return', f"{results.get('total_return', 0):.2f}%"])
            writer.writerow(['Max Drawdown', f"{results.get('max_drawdown', 0):.2f}%"])
            writer.writerow(['Sharpe Ratio', results.get('sharpe_ratio', 0)])
            writer.writerow(['Final Equity', f"${results.get('final_equity', 0):,.2f}"])
        
        logger.info(f"Signal exported to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        return ""


def log_signal(signal: dict) -> None:
    """Log signal to signals.log file."""
    try:
        with open(SIGNALS_LOG, 'a') as f:
            timestamp = datetime.now().strftime(LOG_DATE_FORMAT)
            action = signal.get('action', 'N/A')
            confidence = signal.get('confidence', 0)
            score = signal.get('final_score', 0)
            
            f.write(f"{timestamp} | {action} | Confidence: {confidence:.1f}% | Score: {score:.3f}\n")
        
        logger.debug(f"Signal logged to {SIGNALS_LOG}")
        
    except Exception as e:
        logger.error(f"Error logging signal: {str(e)}")


def append_signals_history(signal: dict) -> None:
    """Append signal to history CSV for live mode."""
    file_exists = os.path.exists(SIGNALS_HISTORY)
    
    try:
        with open(SIGNALS_HISTORY, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['timestamp', 'action', 'confidence', 'final_score',
                               'tech_score', 'ml_score', 'news_score'])
            
            breakdown = signal.get('breakdown', {})
            writer.writerow([
                datetime.now().isoformat(),
                signal.get('action', 'N/A'),
                signal.get('confidence', 0),
                signal.get('final_score', 0),
                breakdown.get('technical', {}).get('score', 0),
                breakdown.get('ml', {}).get('score', 0),
                breakdown.get('news', {}).get('score', 0)
            ])
        
        logger.debug(f"Signal appended to {SIGNALS_HISTORY}")
        
    except Exception as e:
        logger.error(f"Error appending to signals history: {str(e)}")


def run_analysis():
    """Run complete analysis pipeline."""
    logger.info("Starting XAU/USD analysis pipeline...")
    
    # 1. Load and clean data
    print("\n[1/8] Loading market data...")
    data_engine = DataEngine()
    df = data_engine.load_data()
    df = data_engine.clean_data(df)
    print(f"    ✓ Loaded {len(df)} bars from {df.index.min().date()} to {df.index.max().date()}")
    
    # 2. Compute features
    print("\n[2/8] Computing technical indicators...")
    feature_engine = FeatureEngine()
    df = feature_engine.compute_all(df)
    print(f"    ✓ Computed {len(feature_engine.get_feature_names())} features")
    
    # 3. Train/load ML model
    print("\n[3/8] Preparing ML model...")
    ml_model = MLModel()
    
    # Try to load existing model first
    if ml_model.load_model():
        print("    ✓ Loaded existing model from disk")
    else:
        print("    Training new model...")
        metrics = ml_model.train(df)
        print(f"    ✓ Model trained (CV Accuracy: {metrics['accuracy']:.2%})")
    
    # Get ML prediction for latest data
    ml_proba = ml_model.predict_single(df)
    print(f"    ✓ ML Prediction: {ml_proba:.3f} ({'BULLISH' if ml_proba > 0.5 else 'BEARISH'})")
    
    # 4. Fetch and analyze news
    print("\n[4/8] Analyzing news sentiment...")
    news_engine = NewsEngine()
    sentiment = news_engine.get_market_sentiment()
    print(f"    ✓ Sentiment: {sentiment['label']} (score={sentiment['score']:.2f})")
    print(f"    ✓ Articles analyzed: {sentiment['articles_analyzed']}")
    
    # 5. Generate ensemble signal
    print("\n[5/8] Generating ensemble signal...")
    signal_engine = SignalEngine()
    signal = signal_engine.generate_final_signal(df, ml_proba, sentiment)
    signal_engine.format_signal_output(signal)
    
    # 6. Risk assessment
    print("\n[6/8] Performing risk assessment...")
    risk_manager = RiskManager()
    risk = risk_manager.assess_risk(signal, df, signal['confidence'])
    risk_manager.print_risk_report(risk)
    
    # 7. Run backtest
    print("\n[7/8] Running backtest simulation...")
    backtester = Backtester()
    backtest_results = backtester.run(df)
    backtester.print_report(backtest_results)
    
    # 8. Create visualizations
    print("\n[8/8] Creating visualizations...")
    visualizer = Visualizer()
    plot_path = visualizer.plot_full_analysis(df, signal, backtest_results)
    print(f"    ✓ Analysis chart saved to {plot_path}")
    
    # Export results
    export_path = export_to_csv(signal, backtest_results)
    log_signal(signal)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📊 Plot: {plot_path}")
    print(f"📁 Export: {export_path}")
    print(f"📝 Log: {SIGNALS_LOG}")
    
    return signal, backtest_results, risk


def run_live_mode():
    """Run in live mode with continuous updates."""
    print("\n🔴 LIVE MODE ACTIVATED")
    print(f"Update interval: {LIVE_UPDATE_INTERVAL} seconds")
    print("Press Ctrl+C to stop\n")
    
    last_signal = None
    
    try:
        while True:
            print(f"\n{'='*60}")
            print(f"🕐 Update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print('='*60)
            
            try:
                signal, results, risk = run_analysis()
                
                # Check if signal changed
                if last_signal is None or signal['action'] != last_signal.get('action'):
                    print("\n🚨 SIGNAL CHANGE DETECTED!")
                    append_signals_history(signal)
                    last_signal = signal
                
                # Wait for next update
                print(f"\n⏳ Waiting {LIVE_UPDATE_INTERVAL} seconds...")
                time.sleep(LIVE_UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in live update cycle: {str(e)}")
                print(f"\n❌ Error: {str(e)}")
                print(f"Retrying in {LIVE_UPDATE_INTERVAL} seconds...")
                time.sleep(LIVE_UPDATE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Live mode stopped by user")
        print(f"Total signals logged to: {SIGNALS_HISTORY}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='XAU/USD Analyzer - AI-Powered Gold Trading Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py           # Run single analysis
  python main.py --live    # Run in live mode (continuous updates)
        """
    )
    
    parser.add_argument('--live', action='store_true',
                       help='Run in live mode with continuous updates')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    if args.live:
        run_live_mode()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
