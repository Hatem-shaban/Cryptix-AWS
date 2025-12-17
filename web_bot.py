from flask import Flask, render_template_string, jsonify, redirect, send_file
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import config  # Import trading configuration
import os, time, threading, subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import requests  # Added for Coinbase API calls
import pytz
import csv
from pathlib import Path
import io
import zipfile
# from keep_alive import keep_alive  # Disabled to avoid Flask conflicts
import sys
import json
from typing import Optional  # Add this import for Python version compatibility

# Import enhanced trading modules
try:
    from position_manager import get_position_manager
    from signal_filter import get_signal_filter
    from risk_manager import get_risk_manager
    from market_intelligence import get_market_intelligence
    from ml_predictor import EnhancedMLPredictor
    # Use Supabase position tracker (with fallback to file-based)
    from supabase_position_tracker import get_position_tracker
    from smart_profit_taker import get_smart_profit_taker
    
    ENHANCED_MODULES_AVAILABLE = True
    print("✅ Enhanced trading modules loaded successfully")
    print("🧠 ML Intelligence and Market Analytics enabled")
    
    # Initialize enhanced modules
    position_manager = get_position_manager()
    signal_filter = get_signal_filter()
    risk_manager = get_risk_manager()
    market_intelligence = get_market_intelligence(lookback_days=config.ML_LOOKBACK_DAYS)
    ml_predictor = EnhancedMLPredictor()
    
except ImportError as e:
    print(f"⚠️ Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False
    position_manager = None
    signal_filter = None
    risk_manager = None
    market_intelligence = None
    ml_predictor = None

# Import ML Trading Strategy
try:
    from ml_trading_strategy import ml_pure_strategy, get_ml_trading_strategy
    ML_STRATEGY_AVAILABLE = True
    print("🧠 ML Pure Trading Strategy loaded successfully")
    print("🎯 Target: Smart, Profitable, Non-Conservative ML Trading")
except ImportError as e:
    print(f"⚠️ ML Trading Strategy not available: {e}")
    ML_STRATEGY_AVAILABLE = False
    def ml_pure_strategy(*args, **kwargs): return "HOLD", "ML Strategy not available"

# Import smart signal optimizer
try:
    from smart_signal_optimizer import get_signal_optimizer
    SMART_OPTIMIZER_AVAILABLE = True
    print("✅ Smart Signal Optimizer loaded successfully")
except ImportError as e:
    SMART_OPTIMIZER_AVAILABLE = False
    print(f"⚠️ Smart Signal Optimizer not available: {e}")

# Import Telegram notifications
try:
    from telegram_notify import (
        notify_signal, notify_trade, notify_error, notify_bot_status, 
        notify_daily_summary, notify_market_update, process_queued_notifications,
        get_telegram_stats, telegram_notifier
    )
    TELEGRAM_AVAILABLE = True
    print("✅ Telegram notifications module loaded successfully")
except ImportError as e:
    print(f"⚠️ Telegram notifications not available: {e}")
    TELEGRAM_AVAILABLE = False
    # Create dummy functions to prevent errors
    def notify_signal(*args, **kwargs): return False
    def notify_trade(*args, **kwargs): return False
    def notify_error(*args, **kwargs): return False
    def notify_bot_status(*args, **kwargs): return False
    def notify_daily_summary(*args, **kwargs): return False
    def notify_market_update(*args, **kwargs): return False
    def process_queued_notifications(): pass
    def get_telegram_stats(): return {}
    telegram_notifier = None

# Install psutil if not present
try:
    import psutil
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

# Watchdog and auto-restart functionality has been removed

# Load environment variables
load_dotenv()

# Verbosity helper (set VERBOSE_LOGS=1 to enable extra logs)
def _verbose() -> bool:
    try:
        return str(os.getenv("VERBOSE_LOGS", "")).strip().lower() in {"1", "true", "yes", "on", "debug"}
    except Exception:
        return False

def update_bot_status_common(symbol, signal, price, df=None, score=0):
    """Helper to reduce duplicate bot_status update patterns"""
    bot_status.update({
        'current_symbol': symbol,
        'last_signal': signal,
        'last_price': price,
        'last_update': format_cairo_time(),
        'rsi': float(df['rsi'].iloc[-1]) if df is not None and 'rsi' in df.columns else 50,
        'macd': {
            'macd': float(df['macd'].iloc[-1]) if df is not None and 'macd' in df.columns else 0,
            'signal': float(df['macd_signal'].iloc[-1]) if df is not None and 'macd_signal' in df.columns else 0,
            'trend': df['macd_trend'].iloc[-1] if df is not None and 'macd_trend' in df.columns else 'NEUTRAL'
        },
        'opportunity_score': score
    })

def update_monitored_pair(symbol, signal, price, df, score=0):
    """Helper to update monitored pairs tracking"""
    bot_status['monitored_pairs'][symbol] = {
        'last_signal': signal,
        'last_price': price,
        'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50,
        'macd': {
            'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0,
            'signal': float(df['macd_signal'].iloc[-1]) if 'macd_signal' in df.columns else 0,
            'trend': df['macd_trend'].iloc[-1] if 'macd_trend' in df.columns else 'NEUTRAL'
        },
        'last_update': format_cairo_time(),
        'opportunity_score': score
    }

def safe_operation(operation_name, operation_func, default_return=None, log_error=True):
    """Helper to handle exceptions consistently"""
    try:
        return operation_func()
    except Exception as e:
        if log_error:
            error_msg = f"{operation_name} failed: {str(e)}"
            if _verbose():
                print(f"⚠️ {error_msg}")
            log_error_to_csv(error_msg, "OPERATION_ERROR", operation_name, "ERROR")
        return default_return

# Cairo timezone
CAIRO_TZ = pytz.timezone('Africa/Cairo')

def get_cairo_time():
    """Get current time in Cairo, Egypt timezone"""
    return datetime.now(CAIRO_TZ)

def format_cairo_time(dt=None):
    """Format datetime to Cairo timezone string"""
    if dt is None:
        dt = get_cairo_time()
    elif dt.tzinfo is None:
        # If naive datetime, assume it's UTC and convert to Cairo
        dt = pytz.UTC.localize(dt).astimezone(CAIRO_TZ)
    elif dt.tzinfo != CAIRO_TZ:
        # Convert to Cairo timezone
        dt = dt.astimezone(CAIRO_TZ)
    
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')

def get_time_remaining_for_next_signal():
    """Calculate time remaining until next signal in a human-readable format"""
    try:
        if not bot_status.get('next_signal_time') or not bot_status.get('running'):
            return "Not scheduled"
        
        next_signal = bot_status['next_signal_time']
        current_time = get_cairo_time()
        
        # If next_signal is naive datetime, make it timezone-aware
        if next_signal.tzinfo is None:
            next_signal = CAIRO_TZ.localize(next_signal)
        
        time_diff = next_signal - current_time
        
        if time_diff.total_seconds() <= 0:
            return "Signal due now"
        
        # Convert to minutes and seconds
        total_seconds = int(time_diff.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception as e:
        return "Unknown"

# CSV Trade History Logging
def setup_csv_logging():
    """Initialize CSV logging directories and files while preserving existing data"""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Define CSV file paths
    csv_files = {
        'trades': logs_dir / 'trade_history.csv',
        'signals': logs_dir / 'signal_history.csv',
        'errors': logs_dir / 'error_log.csv'
    }
    
    # Define headers for each file type
    trade_headers = [
        'timestamp', 'cairo_time', 'signal', 'symbol', 'quantity', 'price', 
        'value', 'fee', 'status', 'order_id', 'rsi', 'macd_trend', 'sentiment',
        'balance_before', 'balance_after', 'profit_loss'
    ]
    
    signal_headers = [
        'timestamp', 'cairo_time', 'signal', 'symbol', 'price', 'rsi', 'macd', 'macd_trend',
        'sentiment', 'sma5', 'sma20', 'reason'
    ]
    
    error_headers = [
        'timestamp', 'cairo_time', 'error_type', 'error_message', 'function_name',
        'severity', 'bot_status'
    ]
    
    headers_map = {
        'trades': trade_headers,
        'signals': signal_headers,
        'errors': error_headers
    }
    
    # Initialize CSV files while preserving existing data
    for file_type, file_path in csv_files.items():
        if not file_path.exists():
            # Create new file with headers if it doesn't exist
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers_map[file_type])
        else:
            # File exists - verify headers
            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    existing_headers = next(reader, None)
                    
                    # If file is empty or headers don't match, initialize with headers while preserving data
                    if not existing_headers or existing_headers != headers_map[file_type]:
                        # Read existing data
                        f.seek(0)
                        existing_data = list(reader)
                        
                        # Rewrite file with correct headers and preserved data
                        with open(file_path, 'w', newline='', encoding='utf-8') as f_write:
                            writer = csv.writer(f_write)
                            writer.writerow(headers_map[file_type])
                            writer.writerows(existing_data)
            except Exception as e:
                print(f"Error verifying {file_type} log file: {e}")
                # If there's an error, backup the existing file and create a new one
                backup_path = file_path.with_suffix('.csv.bak')
                try:
                    if file_path.exists():
                        file_path.rename(backup_path)
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers_map[file_type])
                except Exception as be:
                    print(f"Error creating backup of {file_type} log file: {be}")
    
    return csv_files

def log_trade_to_csv(trade_info, additional_data=None):
    """Log trade information to CSV file and update position tracking"""
    try:
        csv_files = setup_csv_logging()
        
        # 🎯 Update position tracking
        try:
            position_tracker = get_position_tracker()
            signal = trade_info.get('signal', '').upper()
            symbol = trade_info.get('symbol', '')
            quantity = float(trade_info.get('quantity', 0))
            price = float(trade_info.get('price', 0))
            
            if signal == 'BUY' and symbol and quantity > 0 and price > 0:
                position_tracker.update_position(symbol, quantity, price, 'BUY')
                print(f"📊 Updated position: BUY {quantity:.8f} {symbol} @ ${price:.4f}")
            elif signal in ['SELL', 'SELL_PARTIAL'] and symbol and quantity > 0:
                position_tracker.update_position(symbol, quantity, price, 'SELL')
                print(f"📊 Updated position: SELL {quantity:.8f} {symbol} @ ${price:.4f}")
        except Exception as e:
            print(f"⚠️ Position tracking update failed: {e}")
        
        # Prepare trade data
        trade_data = [
            trade_info.get('timestamp', ''),
            format_cairo_time(),
            trade_info.get('signal', ''),
            trade_info.get('symbol', ''),
            trade_info.get('quantity', 0),
            trade_info.get('price', 0),
            trade_info.get('value', 0),
            trade_info.get('fee', 0),
            trade_info.get('status', ''),
            trade_info.get('order_id', ''),
            additional_data.get('rsi', 0) if additional_data else 0,
            additional_data.get('macd_trend', '') if additional_data else '',
            additional_data.get('sentiment', '') if additional_data else '',
            additional_data.get('balance_before', 0) if additional_data else 0,
            additional_data.get('balance_after', 0) if additional_data else 0,
            additional_data.get('profit_loss', 0) if additional_data else 0
        ]
        
        # Write to CSV with most recent at top
        import tempfile
        temp_file = csv_files['trades'].with_suffix('.tmp')
        
        # Read existing data
        existing_data = []
        if csv_files['trades'].exists():
            with open(csv_files['trades'], 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_data = list(reader)
        
        # Write new data at top
        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if existing_data and existing_data[0]:  # Write header if exists
                writer.writerow(existing_data[0])
                writer.writerow(trade_data)  # New entry at top
                writer.writerows(existing_data[1:])  # Rest of data
            else:
                writer.writerow(trade_data)
        
        # Replace original file
        temp_file.replace(csv_files['trades'])
            
        print(f"Trade logged to CSV: {trade_info.get('signal', 'UNKNOWN')} at {trade_info.get('price', 0)}")
        
    except Exception as e:
        print(f"Error logging trade to CSV: {e}")

# Global signal tracking to prevent duplicates
last_signals = {}
last_signal_time = None  # Track when ANY signal was last generated globally

def log_signal_to_csv(signal, price, indicators, reason=""):
    """Log trading signal to CSV file and Supabase with enhanced duplicate prevention"""
    global last_signals, last_signal_time
    try:
        symbol = indicators.get('symbol', 'UNKNOWN')
        current_time = datetime.now()
        
        print(f"🔍 Attempting to log signal: {signal} for {symbol} at ${price:.4f}")  # Debug

        # Reduce noise: only log HOLD if it's a transition or reason is important
        try:
            last_logged = bot_status.get('last_logged_signal', {}).get(symbol)
            important = signal in ("BUY", "SELL") or (isinstance(reason, str) and ("blocked" in reason.lower() or "error" in reason.lower()))
            if signal == "HOLD" and last_logged == "HOLD" and not important:
                print(f"ℹ️ Skipping HOLD log for {symbol} (no transition)")
                return
            bot_status.setdefault('last_logged_signal', {})[symbol] = signal
        except Exception:
            pass
        
        # OPTIMIZED rate limiting - allow more frequent signals but prevent spam
        if last_signal_time is not None:
            global_time_diff = (current_time - last_signal_time).total_seconds()
            if global_time_diff < 20:  # Reduced from 45 to 20 seconds between ANY signals globally
                print(f"🛑 GLOBAL rate limit: Any signal suppressed (last signal {global_time_diff:.1f}s ago, need 20s gap)")
                return
        
        # Enhanced duplicate prevention - prevent ANY signal for same symbol within 60 seconds
        # This prevents rapid-fire signal generation regardless of signal type
        symbol_key = f"{symbol}"  # Just symbol, not signal type
        
        if symbol_key in last_signals:
            time_diff = (current_time - last_signals[symbol_key]).total_seconds()
            if time_diff < 60:  # Reduced from 90 to 60 seconds cooldown for ANY signal on this symbol
                print(f"⚠️ Symbol rate limit: {signal} for {symbol} suppressed (last signal {time_diff:.1f}s ago, need 60s gap)")
                return
        
        # Also check for the specific signal type (additional safety)
        signal_key = f"{symbol}_{signal}"
        if signal_key in last_signals:
            time_diff = (current_time - last_signals[signal_key]).total_seconds()
            if time_diff < 120:  # Reduced from 180 to 120 seconds cooldown for same signal type
                print(f"⚠️ Signal type rate limit: {signal} for {symbol} suppressed (same signal {time_diff:.1f}s ago)")
                return
        
        # Update all tracking variables
        last_signal_time = current_time
        last_signals[symbol_key] = current_time
        last_signals[signal_key] = current_time
        
        # Log to Supabase first (preferred)
        try:
            position_tracker = get_position_tracker()
            if hasattr(position_tracker, 'log_signal'):
                position_tracker.log_signal(signal, symbol, price, indicators, reason)
                print(f"✅ Signal logged to Supabase: {signal} for {symbol} at ${price:.4f} - {reason}")
        except Exception as e:
            print(f"⚠️ Failed to log signal to Supabase: {e}, falling back to CSV")
        
        # Also log to CSV as backup
        csv_files = setup_csv_logging()
        
        signal_data = [
            datetime.now().isoformat(),
            format_cairo_time(),
            signal,
            symbol,
            price,
            indicators.get('rsi', 0),
            indicators.get('macd', 0),
            indicators.get('macd_trend', ''),
            indicators.get('sentiment', ''),
            indicators.get('sma5', 0),
            indicators.get('sma20', 0),
            reason
        ]
        
        with open(csv_files['signals'], 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(signal_data)
            
        print(f"✅ Signal also logged to CSV backup")
            
    except Exception as e:
        print(f"❌ Error logging signal: {e}")
        # Log the error for debugging
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")

def log_error_to_csv(error_message, error_type="GENERAL", function_name="", severity="ERROR"):
    """Log errors to CSV file and Supabase"""
    try:
        # Log to Supabase first (preferred)
        try:
            position_tracker = get_position_tracker()
            if hasattr(position_tracker, 'log_error'):
                position_tracker.log_error(
                    error_message=error_message,
                    error_type=error_type,
                    function_name=function_name,
                    severity=severity,
                    bot_status=bot_status.get('running', False)
                )
                print(f"✅ Error logged to Supabase: {error_type} - {error_message}")
        except Exception as e:
            print(f"⚠️ Failed to log error to Supabase: {e}, falling back to CSV only")
        
        # Also log to CSV as backup
        csv_files = setup_csv_logging()
        
        error_data = [
            datetime.now().isoformat(),
            format_cairo_time(),
            error_type,
            str(error_message),
            function_name,
            severity,
            bot_status.get('running', False)
        ]
        
        # Write to CSV with most recent at top
        import tempfile
        temp_file = csv_files['errors'].with_suffix('.tmp')
        
        # Read existing data
        existing_data = []
        if csv_files['errors'].exists():
            with open(csv_files['errors'], 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_data = list(reader)
        
        # Write new data at top
        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if existing_data and existing_data[0]:  # Write header if exists
                writer.writerow(existing_data[0])
                writer.writerow(error_data)  # New entry at top
                writer.writerows(existing_data[1:])  # Rest of data
            else:
                writer.writerow(error_data)
        
        # Replace original file
        temp_file.replace(csv_files['errors'])
            
        print(f"✅ Error also logged to CSV backup")
        
        # Send Telegram notification for critical errors
        if TELEGRAM_AVAILABLE and severity in ['ERROR', 'CRITICAL']:
            try:
                notify_error(str(error_message), error_type, function_name, severity)
            except Exception as telegram_error:
                print(f"Telegram error notification failed: {telegram_error}")
            
    except Exception as e:
        print(f"❌ Error logging error: {e}")

def get_supabase_trade_history(days=0):
    """Read and return trade history from Supabase."""
    try:
        position_tracker = get_position_tracker()
        if hasattr(position_tracker, 'get_trade_history'):
            # Use the Supabase tracker to get history
            history = position_tracker.get_trade_history(days=days)
            
            # The data from Supabase is a list of dicts.
            # We need to ensure it has the fields the template expects.
            processed_history = []
            for trade in history:
                # Basic field mapping
                processed_trade = {
                    'timestamp': trade.get('timestamp'),
                    'cairo_time': format_cairo_time(pd.to_datetime(trade.get('timestamp'))),
                    'signal': trade.get('action'),
                    'symbol': trade.get('symbol'),
                    'quantity': float(trade.get('quantity', 0)),
                    'price': float(trade.get('price', 0)),
                    'status': trade.get('status', 'success'), # Assume success if not specified
                    'fee': float(trade.get('fee', 0)),
                    'order_id': trade.get('order_id', '')
                }
                # Calculate 'value'
                processed_trade['value'] = processed_trade['quantity'] * processed_trade['price']
                processed_history.append(processed_trade)
            
            return processed_history
        else:
            # Fallback to CSV if not using Supabase tracker
            return get_csv_trade_history(days=days)
            
    except Exception as e:
        log_error_to_csv(f"Error reading Supabase trade history: {e}", 
                       "SUPABASE_READ_ERROR", "get_supabase_trade_history", "ERROR")
        return []

def get_csv_trade_history(days=30):
    """Read and return trade history from CSV"""
    try:
        csv_files = setup_csv_logging()
        
        if not csv_files['trades'].exists():
            return []
        
        # Read CSV file
        df = pd.read_csv(csv_files['trades'])
        
        # Filter by date if needed
        if days > 0 and not df.empty:
            cutoff_date = datetime.now() - timedelta(days=days)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] >= cutoff_date]
        
        # Sort by timestamp (newest first) and convert to records
        if not df.empty:
            df = df.sort_values('timestamp', ascending=False)
            return df.to_dict('records')
        
        return []
        
    except Exception as e:
        log_error_to_csv(f"Error reading CSV trade history: {e}", 
                       "CSV_READ_ERROR", "get_csv_trade_history", "ERROR")
        return []

def get_supabase_signal_history(limit=100):
    """Read and return signal history from Supabase."""
    try:
        position_tracker = get_position_tracker()
        if hasattr(position_tracker, 'get_signal_history'):
            # Use the Supabase tracker to get signal history
            signals = position_tracker.get_signal_history(limit=limit)
            return signals
        else:
            # Fallback to CSV if not using Supabase tracker
            return get_csv_signal_history(limit=limit)
            
    except Exception as e:
        log_error_to_csv(f"Error reading Supabase signal history: {e}", 
                       "SUPABASE_READ_ERROR", "get_supabase_signal_history", "ERROR")
        return []

def get_csv_signal_history(limit=100):
    """Read and return signal history from CSV"""
    try:
        csv_files = setup_csv_logging()
        
        if not csv_files['signals'].exists():
            return []
        
        # Read CSV file
        df = pd.read_csv(csv_files['signals'])
        
        # Sort by timestamp (newest first) and limit
        if not df.empty:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
            df = df.head(limit)
            return df.to_dict('records')
        
        return []
        
    except Exception as e:
        log_error_to_csv(f"Error reading CSV signal history: {e}", 
                       "CSV_READ_ERROR", "get_csv_signal_history", "ERROR")
        return []

def get_supabase_error_history(limit=50):
    """Read and return error history from Supabase."""
    try:
        position_tracker = get_position_tracker()
        if hasattr(position_tracker, 'get_error_history'):
            # Use the Supabase tracker to get error history
            errors = position_tracker.get_error_history(limit=limit)
            return errors
        else:
            # Fallback to CSV if not using Supabase tracker
            return get_csv_error_history(limit=limit)
            
    except Exception as e:
        print(f"Error reading Supabase error history: {e}")
        return []

def get_csv_error_history(limit=50):
    """Read and return error history from CSV"""
    try:
        csv_files = setup_csv_logging()
        
        if not csv_files['errors'].exists():
            return []
        
        # Read CSV file
        df = pd.read_csv(csv_files['errors'])
        
        # Sort by timestamp (newest first) and limit
        if not df.empty:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
            df = df.head(limit)
            return df.to_dict('records')
        
        return []
        
    except Exception as e:
        print(f"Error reading CSV error history: {e}")
        return []

# Global bot status
bot_status = {
    'running': False,
    'signal_scanning_active': False,  # Track signal scanning status
    'last_signal': 'UNKNOWN',
    'last_scan_time': None,  # Track when last scan occurred
    'current_symbol': 'BTCUSDT',  # Track currently analyzed symbol
    'last_price': 0,
    'last_update': None,
    'api_connected': False,
    'total_trades': 0,
    'errors': [],
    'start_time': get_cairo_time(),
    'consecutive_errors': 0,
    'rsi': 50,
    'macd': {'macd': 0, 'signal': 0, 'trend': 'NEUTRAL'},
    'sentiment': 'neutral',
    'monitored_pairs': {},  # Track all monitored pairs' status
    'trading_strategy': config.DEFAULT_STRATEGY,  # Current trading strategy
    'next_signal_time': None,  # Track when next signal will be generated
    'signal_interval': 300,  # Base signal generation interval in seconds (5 minutes - adaptive)
    'market_regime': 'NORMAL',  # Current market regime (QUIET, NORMAL, VOLATILE, EXTREME)
    'hunting_mode': True,  # Aggressive opportunity hunting mode
    'last_volatility_check': None,  # Track when we last checked volatility
    'adaptive_intervals': {
        'QUIET': 3600,       # 60 minutes during quiet markets (increased from 30min)
        'NORMAL': 1800,      # 30 minutes during normal markets (increased from 5min)
        'VOLATILE': 900,     # 15 minutes during volatile markets (increased from 3min)
        'EXTREME': 600,      # 10 minutes during extreme volatility (increased from 1min)
        'HUNTING': 300       # 5 minutes when hunting opportunities (increased from 30s)
    },
    'trading_summary': {
        'total_revenue': 0.0,
        'successful_trades': 0,
        'failed_trades': 0,
        'total_buy_volume': 0.0,
        'total_sell_volume': 0.0,
        'average_trade_size': 0.0,
        'win_rate': 0.0,
        'trades_history': []  # Last 10 trades for display
    },
    # Caches
    'exchange_info_cache': None,   # {'time': datetime, 'data': {...}}
    'coinbase_cache': {},          # {'BTC-USD': {'time': dt, 'data': {...}}}
    # Logging deduplication
    'last_logged_signal': {}       # per-symbol last logged signal value
}

app = Flask(__name__)

# Initialize CSV logging on startup
setup_csv_logging()

# Initialize API credentials with multiple fallback methods for Render deployment
print("🚀 CRYPTIX Bot Starting...")

api_key = None
api_secret = None
client = None

# Try multiple methods to get environment variables without noisy prints
try:
    # Method 1: os.getenv (standard)
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    # Method 2: Direct os.environ access (backup)
    if not api_key:
        api_key = os.environ.get("API_KEY")
    if not api_secret:
        api_secret = os.environ.get("API_SECRET")
    if _verbose():
        print(f"🔑 Initial credential check:")
        print(f"   API_KEY loaded: {'✓' if api_key else '✗'}")
        print(f"   API_SECRET loaded: {'✓' if api_secret else '✗'}")
        if api_key and api_secret:
            print(f"   API_KEY format: {len(api_key)} chars, preview: {api_key[:8]}...{api_key[-4:]}")
            print("✅ Credentials loaded successfully at startup")
        else:
            print("⚠️  Credentials not found at startup - will retry during initialization")
except Exception as e:
    if _verbose():
        print(f"⚠️  Error loading credentials at startup: {e}")
        print("   Will attempt to load during client initialization")

# Lightweight sentiment analysis function
def get_sentiment_score(text):
    """Enhanced sentiment scoring with crypto-specific keyword weighting"""
    try:
        blob = TextBlob(text)
        base_sentiment = blob.sentiment.polarity
        
        # Crypto-specific keywords for better sentiment analysis
        bullish_keywords = ['moon', 'bullish', 'buy', 'hodl', 'pump', 'rally', 'breakout', 'surge', 'gains', 'profit']
        bearish_keywords = ['dump', 'crash', 'sell', 'bearish', 'drop', 'fall', 'loss', 'decline', 'dip', 'correction']
        
        text_lower = text.lower()
        keyword_boost = 0
        
        # Apply keyword boosting
        for keyword in bullish_keywords:
            if keyword in text_lower:
                keyword_boost += 0.1
                
        for keyword in bearish_keywords:
            if keyword in text_lower:
                keyword_boost -= 0.1
        
        # Combine base sentiment with keyword boost
        enhanced_sentiment = base_sentiment + keyword_boost
        
        # Ensure sentiment stays within bounds [-1, 1]
        return max(-1, min(1, enhanced_sentiment))
    except Exception as e:
        print(f"Sentiment scoring error: {e}")
        return 0

def initialize_client():
    global client, bot_status, api_key, api_secret
    try:
        # EMERGENCY CHECK: Skip API initialization if in emergency mode during ban
        import config
        if hasattr(config, 'EMERGENCY_MODE') and config.EMERGENCY_MODE:
            # Check if ban is still active
            current_time = datetime.now().timestamp() * 1000
            ban_until = 1760962139050  # Updated ban timestamp (THIRD ban)
            
            if current_time < ban_until:
                ban_lift_time = datetime.fromtimestamp(ban_until / 1000)
                minutes_remaining = (ban_until - current_time) / 1000 / 60
                print(f"🚨 EMERGENCY: API ban active until {ban_lift_time.strftime('%H:%M:%S')}")
                print(f"⏰ {minutes_remaining:.1f} minutes remaining - SKIPPING API initialization")
                
                # Set demo mode flags
                bot_status['api_connected'] = False
                bot_status['demo_mode'] = True
                return False
        
        # Skip if already connected and client exists
        if client and bot_status.get('api_connected', False):
            print("✅ API client already connected")
            return True
            
        # Reload environment variables to ensure we have latest values
        load_dotenv()
        
        # Get API credentials with multiple fallback methods for Render
        api_key = (
            os.getenv("API_KEY") or 
            os.environ.get("API_KEY") or 
            os.getenv("BINANCE_API_KEY") or
            os.environ.get("BINANCE_API_KEY") or
            None
        )
        api_secret = (
            os.getenv("API_SECRET") or 
            os.environ.get("API_SECRET") or 
            os.getenv("BINANCE_API_SECRET") or
            os.environ.get("BINANCE_API_SECRET") or
            None
        )
        
        # Detailed logging for debugging (verbose only)
        if _verbose():
            print(f"🔍 Environment check:")
            print(f"   API_KEY found: {'Yes' if api_key else 'No'}")
            print(f"   API_SECRET found: {'Yes' if api_secret else 'No'}")
            if api_key:
                print(f"   API_KEY length: {len(api_key)}")
                print(f"   API_KEY preview: {api_key[:8]}...{api_key[-4:]}")
        
        if not api_key or not api_secret:
            error_msg = f"API credentials missing - API_KEY: {'✓' if api_key else '✗'}, API_SECRET: {'✓' if api_secret else '✗'}"
            print(f"❌ {error_msg}")
            bot_status['errors'].append(error_msg)
            log_error_to_csv(error_msg, "CREDENTIALS_ERROR", "initialize_client", "ERROR")
            return False
        
        # Determine whether to use Binance Testnet (via env or config)
        def _truthy(v):
            return str(v).strip().lower() in {"1", "true", "yes", "on"}
        env_flag = os.getenv("BINANCE_TESTNET") or os.getenv("USE_TESTNET")
        use_testnet = _truthy(env_flag) if env_flag is not None else getattr(config, 'USE_TESTNET', False)

        # Validate credential format (less strict for testnet); allow variation in lengths on LIVE
        if not use_testnet and len(api_key) < 32:
            error_msg = f"Invalid API key format - too short for LIVE (len={len(api_key)})"
            print(f"❌ {error_msg}")
            bot_status['errors'].append(error_msg)
            log_error_to_csv(error_msg, "CREDENTIALS_ERROR", "initialize_client", "ERROR")
            return False
        if not use_testnet and len(api_secret) < 32:
            error_msg = f"Invalid API secret format - too short for LIVE (len={len(api_secret)})"
            print(f"❌ {error_msg}")
            bot_status['errors'].append(error_msg)
            log_error_to_csv(error_msg, "CREDENTIALS_ERROR", "initialize_client", "ERROR")
            return False
        if use_testnet:
            # Basic sanity check only
            if len(api_key) < 24 or len(api_secret) < 24:
                error_msg = f"Testnet credentials look too short (key {len(api_key)}, secret {len(api_secret)})"
                print(f"❌ {error_msg}")
                bot_status['errors'].append(error_msg)
                log_error_to_csv(error_msg, "CREDENTIALS_ERROR", "initialize_client", "ERROR")
                return False

        print(f"🔗 Initializing Binance client for {'TESTNET' if use_testnet else 'LIVE'} trading...")
        client = Client(api_key, api_secret, testnet=use_testnet)
        # Ensure Spot Testnet base URL when requested
        if use_testnet:
            try:
                client.API_URL = 'https://testnet.binance.vision/api'
            except Exception:
                pass
        try:
            base_url = getattr(client, 'API_URL', None) or getattr(client, 'BASE_URL', None)
            if base_url:
                print(f"   Base URL: {base_url}")
        except Exception:
            pass
        
        # Test API connection with minimal call
        if _verbose():
            print("📊 Testing API connection...")
        server_time = client.get_server_time()
        
        # Only get account info if server connection is successful
        account = client.get_account()
        
        if _verbose():
            print("✅ API connection successful!")
            print(f"   Account Type: {account.get('accountType', 'Unknown')}")
            print(f"   Can Trade: {account.get('canTrade', 'Unknown')}")
            perms = account.get('permissions', [])
            try:
                perms_str = ", ".join(perms)
            except Exception:
                perms_str = str(perms)
            print(f"   Permissions: {perms_str}")
        
        bot_status['api_connected'] = True
        bot_status['account_type'] = account.get('accountType', 'Unknown')
        bot_status['can_trade'] = account.get('canTrade', False)
        
        return True
        
    except BinanceAPIException as e:
        error_msg = f"Binance API Error {e.code}: {e.message}"
        print(f"❌ {error_msg}")
        bot_status['errors'].append(error_msg)
        bot_status['api_connected'] = False
        client = None
        
        # Log specific error solutions
        if e.code == -2015:
            solution_msg = "Error -2015: Check API key/secret format, IP restrictions, or regenerate API key"
            print(f"💡 {solution_msg}")
            log_error_to_csv(f"{error_msg} | {solution_msg}", "API_ERROR", "initialize_client", "ERROR")
        else:
            log_error_to_csv(error_msg, "API_ERROR", "initialize_client", "ERROR")
        
        return False
        
    except Exception as e:
        error_msg = f"Unexpected error initializing client: {str(e)}"
        print(f"❌ {error_msg}")
        bot_status['errors'].append(error_msg)
        bot_status['api_connected'] = False
        client = None
        log_error_to_csv(error_msg, "CLIENT_ERROR", "initialize_client", "ERROR")
        return False

# Market data based sentiment analysis is used instead of social sentiment

def fetch_coinbase_data(product: str = "BTC-USD", ttl_seconds: int = 30):
    """Fetch Coinbase public market data with simple TTL cache and backoff.
    Returns dict with order_book, recent_trades, timestamp or None on error.
    """
    try:
        # TTL cache
        cache = bot_status.get('coinbase_cache') or {}
        entry = cache.get(product)
        now = get_cairo_time()
        if entry and (now - entry['time']).total_seconds() < ttl_seconds:
            return entry['data']

        base_url = "https://api.exchange.coinbase.com"
        headers = {
            'User-Agent': 'CRYPTIX-ML/1.0',
            'Accept': 'application/json'
        }

        if _verbose():
            print(f"Fetching Coinbase order book for {product}...")

        # Helper for GET with backoff
        def get_with_backoff(url, max_retries=3):
            delay = 0.35
            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, headers=headers, timeout=5)
                except Exception as req_err:
                    if attempt == max_retries - 1:
                        raise req_err
                    time.sleep(delay)
                    delay = min(delay * 2, 2.0)
                    continue
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 429:
                    retry_after = resp.headers.get('Retry-After')
                    wait_s = float(retry_after) if retry_after else delay
                    log_error_to_csv("Coinbase rate limit exceeded", "API_RATE_LIMIT", "fetch_coinbase_data", "WARNING")
                    time.sleep(wait_s)
                    delay = min(delay * 2, 2.0)
                    continue
                # Other errors: raise after final attempt
                if attempt == max_retries - 1:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(delay)
                delay = min(delay * 2, 2.0)
            return None

        # Requests
        order_book_resp = get_with_backoff(f"{base_url}/products/{product}/book?level=2")
        order_book = order_book_resp.json() if order_book_resp is not None else None
        if not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book:
            log_error_to_csv(f"Invalid Coinbase order book response for {product}", "COINBASE_ERROR", "fetch_coinbase_data", "ERROR")
            return None

        time.sleep(0.2)  # slight pacing between requests
        trades_resp = get_with_backoff(f"{base_url}/products/{product}/trades")
        trades = trades_resp.json() if trades_resp is not None else []

        data = {
            'order_book': order_book,
            'recent_trades': trades,
            'timestamp': datetime.now().timestamp()
        }
        # Save in cache
        cache[product] = {'time': now, 'data': data}
        bot_status['coinbase_cache'] = cache
        return data
    except Exception as e:
        print(f"Coinbase data fetch error: {e}")
        return None

def analyze_market_sentiment():
    """Analyze market sentiment from multiple sources"""
    try:
        # Initialize sentiment components
        order_book_sentiment = 0
        trade_flow_sentiment = 0
        print("\nAnalyzing market sentiment from order book and trade data...")  # Debug log
        
        # 1. Order Book Analysis
        cb_data = fetch_coinbase_data("BTC-USD")
        if cb_data:
            order_book = cb_data['order_book']
            if 'bids' in order_book and 'asks' in order_book:
                # Calculate buy/sell pressure
                bid_volume = sum(float(bid[1]) for bid in order_book['bids'][:10])
                ask_volume = sum(float(ask[1]) for ask in order_book['asks'][:10])
                
                # Normalize order book sentiment
                total_volume = bid_volume + ask_volume
                if total_volume > 0:
                    order_book_sentiment = (bid_volume - ask_volume) / total_volume
        
            # 3. Recent Trade Flow Analysis
            if 'recent_trades' in cb_data:
                recent_trades = cb_data['recent_trades']
                buy_volume = sum(float(trade['size']) for trade in recent_trades if trade['side'] == 'buy')
                sell_volume = sum(float(trade['size']) for trade in recent_trades if trade['side'] == 'sell')
                
                total_trade_volume = buy_volume + sell_volume
                if total_trade_volume > 0:
                    trade_flow_sentiment = (buy_volume - sell_volume) / total_trade_volume
        
        # Market data based sentiment weights
        weights = {
            'order_book': 0.6,  # Order book pressure weight
            'trade_flow': 0.4   # Recent trade flow weight
        }
        
        # Calculate combined sentiment using market data
        combined_sentiment = (
            weights['order_book'] * order_book_sentiment +
            weights['trade_flow'] * trade_flow_sentiment
        )
        
        # Advanced sentiment thresholds with confidence levels
        sentiment_data = {
            'value': combined_sentiment,
            'components': {
                'order_book_sentiment': order_book_sentiment,
                'trade_flow_sentiment': trade_flow_sentiment
            },
            'confidence': min(1.0, abs(combined_sentiment) * 2)  # Confidence score 0-1
        }
        # Determine sentiment with confidence threshold
        if abs(combined_sentiment) < 0.1:
            return "neutral"
        elif combined_sentiment > 0:
            return "bullish" if sentiment_data['confidence'] > 0.5 else "neutral"
        else:
            return "bearish" if sentiment_data['confidence'] > 0.5 else "neutral"
            
    except Exception as e:
        bot_status['errors'].append(f"Market sentiment analysis failed: {e}")
        return "neutral"

def get_exchange_info_cached(ttl_seconds: int = 300):
    """Return Binance exchange_info using a simple TTL cache to reduce API calls."""
    if not client:
        raise RuntimeError("Client not initialized")
    try:
        cache = bot_status.get('exchange_info_cache')
        now = get_cairo_time()
        if cache and (now - cache['time']).total_seconds() < ttl_seconds:
            return cache['data']
        data = client.get_exchange_info()
        bot_status['exchange_info_cache'] = {'time': now, 'data': data}
        return data
    except Exception as e:
        log_error_to_csv(f"exchange_info cache error: {e}", "CACHE_ERROR", "get_exchange_info_cached", "WARNING")
        return client.get_exchange_info()

def calculate_smart_minimum_trade_usdt(symbol="BTCUSDT", current_price=None, available_usdt=None):
    """
    Calculate the smart minimum USDT required for a trade based on:
    1. Binance LOT_SIZE filter (minimum quantity)
    2. Binance MIN_NOTIONAL filter (minimum value)
    3. Current market price
    4. Adaptive safety margin based on available balance
    
    Returns the actual minimum USDT needed to place a trade for this symbol
    """
    try:
        if not client:
            return config.MIN_TRADE_USDT  # Fallback to config default
            
        # Get current price if not provided
        if current_price is None:
            try:
                ticker = client.get_ticker(symbol=symbol)
                current_price = float(ticker['lastPrice'])
            except Exception as e:
                log_error_to_csv(f"Error getting price for {symbol}: {e}", "PRICE_ERROR", "calculate_smart_minimum_trade_usdt", "WARNING")
                return config.MIN_TRADE_USDT
        
        # Get exchange info for symbol filters
        try:
            exchange_info = get_exchange_info_cached()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if not symbol_info:
                print(f"⚠️ Symbol {symbol} not found in exchange info, using config default")
                return config.MIN_TRADE_USDT
                
        except Exception as e:
            log_error_to_csv(f"Error getting exchange info for {symbol}: {e}", "EXCHANGE_INFO_ERROR", "calculate_smart_minimum_trade_usdt", "WARNING")
            return config.MIN_TRADE_USDT
        
        # Extract relevant filters
        min_qty = 0.001  # Default minimum quantity
        min_notional = 10.0  # Default minimum notional value (Binance standard)
        step_size = 0.001  # Default step size
        
        for filter_info in symbol_info['filters']:
            if filter_info['filterType'] == 'LOT_SIZE':
                min_qty = float(filter_info['minQty'])
                step_size = float(filter_info['stepSize'])
            elif filter_info['filterType'] == 'MIN_NOTIONAL':
                min_notional = float(filter_info['minNotional'])
                
        # Calculate minimum USDT needed based on different constraints
        
        # 1. Minimum quantity requirement
        min_usdt_from_qty = min_qty * current_price
        
        # 2. Minimum notional requirement (this is usually the binding constraint)
        min_usdt_from_notional = min_notional
        
        # 3. Take the maximum of both requirements
        calculated_minimum = max(min_usdt_from_qty, min_usdt_from_notional)
        
        # 4. Adaptive safety margin based on available balance and situation
        if available_usdt is not None:
            # If user has exactly or very close to the minimum, use minimal margin
            balance_buffer = available_usdt - calculated_minimum
            
            if balance_buffer <= 0:
                # User doesn't have enough for even the base minimum
                smart_minimum = calculated_minimum
                safety_margin_used = 1.0
            elif balance_buffer < 1.0:
                # User has very close to minimum, use tiny margin (0.5%)
                safety_margin_used = 1.005
                smart_minimum = calculated_minimum * safety_margin_used
            elif balance_buffer < 5.0:
                # User has some buffer, use small margin (1%)
                safety_margin_used = 1.01
                smart_minimum = calculated_minimum * safety_margin_used
            else:
                # User has good buffer, use normal margin (2%)
                safety_margin_used = 1.02
                smart_minimum = calculated_minimum * safety_margin_used
        else:
            # No balance info provided, use conservative margin
            safety_margin_used = 1.02
            smart_minimum = calculated_minimum * safety_margin_used
        
        # 5. Ensure we don't go below the config minimum (fallback protection)
        final_minimum = max(smart_minimum, config.MIN_TRADE_USDT)
        
        # 6. CRITICAL: Ensure final minimum meets actual notional requirements
        # Double-check that our calculation actually meets the exchange's minimum notional
        if final_minimum < min_notional:
            print(f"   ⚠️ Final minimum ${final_minimum:.2f} below exchange notional ${min_notional:.2f}, adjusting")
            final_minimum = min_notional * 1.01  # Add 1% buffer to be safe
        
        print(f"💡 Smart minimum calculation for {symbol}:")
        print(f"   Current price: ${current_price:.4f}")
        print(f"   Min quantity: {min_qty} (worth ${min_usdt_from_qty:.2f})")
        print(f"   Min notional: ${min_notional:.2f}")
        print(f"   Calculated minimum: ${calculated_minimum:.2f}")
        if available_usdt is not None:
            print(f"   Available USDT: ${available_usdt:.2f}")
            print(f"   Balance buffer: ${available_usdt - calculated_minimum:.2f}")
            print(f"   Adaptive margin: {safety_margin_used:.1%}")
        print(f"   Smart minimum: ${smart_minimum:.2f}")
        print(f"   Final minimum: ${final_minimum:.2f}")
        
        return round(final_minimum, 2)
        
    except Exception as e:
        error_msg = f"Error calculating smart minimum for {symbol}: {e}"
        log_error_to_csv(error_msg, "SMART_MINIMUM_ERROR", "calculate_smart_minimum_trade_usdt", "ERROR")
        print(f"❌ {error_msg}")
        return config.MIN_TRADE_USDT  # Safe fallback

def format_quantity_for_binance(quantity: float, step_size: float = None) -> str:
    """
    Format quantity to avoid scientific notation and ensure proper precision for Binance API
    
    Args:
        quantity: The quantity to format
        step_size: Optional step size to determine precision
        
    Returns:
        String representation of quantity without scientific notation
    """
    try:
        # Handle very small numbers that might become scientific notation
        if quantity == 0:
            return "0"
        
        # Determine precision based on step size if provided
        if step_size is not None and step_size > 0:
            # Calculate decimal places from step size
            step_str = f"{step_size:.20f}".rstrip('0').rstrip('.')
            if '.' in step_str:
                precision = len(step_str.split('.')[1])
            else:
                precision = 0
            
            # Format with calculated precision, but limit to reasonable max
            max_precision = min(precision, 8)  # Limit to 8 decimal places max
            formatted = f"{quantity:.{max_precision}f}"
        else:
            # For cases without step size, use appropriate precision based on magnitude
            if quantity >= 1:
                formatted = f"{quantity:.8f}".rstrip('0').rstrip('.')
            elif quantity >= 0.01:
                formatted = f"{quantity:.8f}"
            elif quantity >= 0.0001:
                formatted = f"{quantity:.8f}"
            else:
                # For very small quantities, use fixed notation with enough precision
                formatted = f"{quantity:.8f}"
        
        # Ensure no scientific notation
        if 'e' in formatted.lower():
            # Convert scientific notation to fixed notation
            formatted = f"{float(formatted):.20f}".rstrip('0').rstrip('.')
        
        # Clean up trailing zeros for readability (but keep at least one decimal for very small numbers)
        if '.' in formatted and not step_size:
            formatted = formatted.rstrip('0').rstrip('.')
            
        # Final validation - ensure it's a valid decimal
        try:
            float(formatted)
            return formatted
        except ValueError:
            # Fallback to simple formatting
            return f"{float(quantity):.8f}".rstrip('0').rstrip('.')
            
    except Exception as e:
        print(f"Error formatting quantity {quantity}: {e}")
        # Safe fallback
        return f"{float(quantity):.8f}".rstrip('0').rstrip('.')

def calculate_rsi(prices, period=None):
    """Calculate RSI using proper Wilder's smoothing method"""
    period = period or config.RSI_PERIOD
    try:
        # Handle different input types - ensure we have a numpy array of floats
        if hasattr(prices, 'values'):  # pandas Series
            prices = prices.values
        elif isinstance(prices, list):
            prices = np.array(prices)
        elif isinstance(prices, (int, float)):  # Single value
            return 50  # Can't calculate RSI for single value
        
        # Convert to float and handle any string values
        try:
            prices = np.array([float(p) for p in prices])
        except (ValueError, TypeError) as e:
            log_error_to_csv(f"Price conversion error in RSI: {e}, prices type: {type(prices)}", 
                           "DATA_TYPE_ERROR", "calculate_rsi", "ERROR")
            return 50
        
        if len(prices) < period + 1:
            return 50  # Neutral RSI when insufficient data
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use Wilder's smoothing (similar to EMA) for more accurate RSI
        alpha = 1.0 / period
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Apply Wilder's smoothing to the rest of the data
        for i in range(period, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure RSI is within bounds
        return max(0, min(100, rsi))
    except Exception as e:
        log_error_to_csv(f"RSI calculation error: {e}", "RSI_ERROR", "calculate_rsi", "ERROR")
        return 50

def calculate_sma(df, period=20):
    """Calculate Simple Moving Average efficiently"""
    try:
        if df is None or len(df) < period:
            return pd.Series([])
        
        # Use pandas rolling for efficiency
        return df['close'].rolling(window=period).mean()
    except Exception as e:
        print(f"SMA calculation error: {e}")
        return pd.Series([])

def calculate_macd(prices, fast=None, slow=None, signal=None):
    """Calculate MACD using configuration parameters"""
    fast = fast or config.MACD_FAST
    slow = slow or config.MACD_SLOW
    signal = signal or config.MACD_SIGNAL
    
    try:
        # Handle different input types - ensure we have a numpy array of floats
        if hasattr(prices, 'values'):  # pandas Series
            prices = prices.values
        elif isinstance(prices, list):
            prices = np.array(prices)
        elif isinstance(prices, (int, float)):  # Single value
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
        
        # Convert to float and handle any string values
        try:
            prices = np.array([float(p) for p in prices])
        except (ValueError, TypeError) as e:
            log_error_to_csv(f"Price conversion error in MACD: {e}, prices type: {type(prices)}", 
                           "DATA_TYPE_ERROR", "calculate_macd", "ERROR")
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
        
        if len(prices) < slow:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
        
        # Calculate exponential moving averages for more accurate MACD
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_values = [float(data[0])]  # Start with first value as float
            for price in data[1:]:
                ema_values.append(alpha * float(price) + (1 - alpha) * ema_values[-1])
            return np.array(ema_values)
        
        fast_ema = ema(prices, fast)
        slow_ema = ema(prices, slow)
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema
        
        # Signal line = EMA of MACD line
        signal_line = ema(macd_line, signal)
        
        # Histogram = MACD - Signal
        histogram = macd_line - signal_line
        
        # Current values
        current_macd = float(macd_line[-1])
        current_signal = float(signal_line[-1])
        current_histogram = float(histogram[-1])
        
        # Determine trend based on MACD crossover and histogram
        if current_macd > current_signal and current_histogram > 0:
            trend = "BULLISH"
        elif current_macd < current_signal and current_histogram < 0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        return {
            "macd": round(current_macd, 6),
            "signal": round(current_signal, 6),
            "histogram": round(current_histogram, 6),
            "trend": trend
        }
    except Exception as e:
        log_error_to_csv(f"MACD calculation error: {e}", "MACD_ERROR", "calculate_macd", "ERROR")
        return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}

def fetch_data(symbol="BTCUSDT", interval="1h", limit=100):
    """Fetch historical price data from Binance."""
    try:
        if _verbose():
            print(f"\n=== Fetching data for {symbol} ===")  # Debug log
        if client:
            if _verbose():
                print("Using Binance client...")  # Debug log
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if _verbose():
                print(f"Received {len(klines)} candles from Binance")  # Debug log
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                             'taker_buy_quote_asset_volume', 'ignore'])
            
            # Convert numeric columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        else:
            error_msg = "Trading client not initialized. Cannot fetch market data."
            log_error_to_csv(error_msg, "CLIENT_ERROR", "fetch_data", "ERROR")
            return None
        
        # Calculate technical indicators
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()

        # EMA family (uses config periods)
        try:
            ema_fast = config.EMA_PERIODS.get('fast', 12)
            ema_slow = config.EMA_PERIODS.get('slow', 26)
            ema_mid = config.EMA_PERIODS.get('mid', 50)
            ema_long = config.EMA_PERIODS.get('long', 200)
        except Exception:
            ema_fast, ema_slow, ema_mid, ema_long = 12, 26, 50, 200
        df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=ema_mid, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=ema_long, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
        df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # Calculate RSI with proper error handling
        prices = df['close'].values
        try:
            rsi_value = calculate_rsi(prices)
            if isinstance(rsi_value, (int, float)):
                df['rsi'] = rsi_value  # Single value for entire series
            else:
                df['rsi'] = 50  # Default fallback
        except Exception as rsi_error:
            log_error_to_csv(f"RSI calculation failed for {symbol}: {rsi_error}", 
                           "RSI_ERROR", "fetch_data", "WARNING")
            df['rsi'] = 50
        
        # Calculate MACD with proper error handling
        try:
            macd_data = calculate_macd(prices)
            df['macd'] = macd_data.get('macd', 0)
            df['macd_signal'] = macd_data.get('signal', 0)
            df['macd_histogram'] = macd_data.get('histogram', 0)
            df['macd_trend'] = macd_data.get('trend', 'NEUTRAL')
        except Exception as macd_error:
            log_error_to_csv(f"MACD calculation failed for {symbol}: {macd_error}", 
                           "MACD_ERROR", "fetch_data", "WARNING")
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_histogram'] = 0
            df['macd_trend'] = 'NEUTRAL'
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # True Range helpers for ATR and ADX
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(config.ATR_PERIOD).mean()

        # Stochastic Oscillator %K and %D
        try:
            k_period = config.STOCH.get('k_period', 14)
            d_period = config.STOCH.get('d_period', 3)
        except Exception:
            k_period, d_period = 14, 3
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = np.where(
            (highest_high - lowest_low) > 0,
            (df['close'] - lowest_low) / (highest_high - lowest_low) * 100,
            50
        )
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

        # VWAP (rolling approximation)
        try:
            vwap_window = config.VWAP.get('window', 20)
        except Exception:
            vwap_window = 20
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        pv = typical_price * df['volume']
        df['vwap'] = pv.rolling(window=vwap_window).sum() / df['volume'].rolling(window=vwap_window).sum()

        # ADX
        try:
            adx_period = config.ADX.get('period', 14)
        except Exception:
            adx_period = 14
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr_smooth = tr.rolling(window=adx_period).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=adx_period).sum() / atr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=adx_period).sum() / atr_smooth)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
        df['adx'] = dx.rolling(window=adx_period).mean()
        
    # Volume trend
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'] / df['volume_sma']
        
        return df
        
    except Exception as e:
        error_msg = f"Error fetching data for {symbol}: {e}"
        log_error_to_csv(error_msg, "DATA_FETCH_ERROR", "fetch_data", "ERROR")
        bot_status['errors'].append(error_msg)
        return None

def detect_market_regime():
    """Professional market regime detection for intelligent timing"""
    try:
        print("\n=== Detecting Market Regime ===")
        
        # Get multi-timeframe data for regime analysis
        btc_1h = fetch_data("BTCUSDT", "1h", 48)  # 48 hours
        btc_5m = fetch_data("BTCUSDT", "5m", 288)  # 24 hours in 5-min candles
        
        if btc_1h is None or btc_5m is None or len(btc_1h) < 24 or len(btc_5m) < 144:
            return 'NORMAL'  # Default regime
        
        # Calculate market volatility measures
        hourly_vol = btc_1h['close'].pct_change().rolling(24).std() * np.sqrt(24 * 365)
        five_min_vol = btc_5m['close'].pct_change().rolling(144).std() * np.sqrt(288 * 365)
        
        current_hourly_vol = hourly_vol.iloc[-1] if not pd.isna(hourly_vol.iloc[-1]) else 0.5
        current_5m_vol = five_min_vol.iloc[-1] if not pd.isna(five_min_vol.iloc[-1]) else 0.5
        
        # Volume surge detection
        avg_volume_1h = btc_1h['volume'].rolling(24).mean().iloc[-1]
        current_volume_1h = btc_1h['volume'].iloc[-1]
        volume_surge = current_volume_1h / avg_volume_1h if avg_volume_1h > 0 else 1
        
        # Price movement analysis
        price_change_1h = abs(btc_1h['close'].pct_change().iloc[-1])
        price_change_24h = abs((btc_1h['close'].iloc[-1] - btc_1h['close'].iloc[-24]) / btc_1h['close'].iloc[-24])
        
        # Market regime classification
        if (current_hourly_vol > 1.2 or current_5m_vol > 1.5 or 
            volume_surge > 2.5 or price_change_1h > 0.03):
            regime = 'EXTREME'
        elif (current_hourly_vol > 0.6 or current_5m_vol > 0.9 or 
              volume_surge > 1.5 or price_change_1h > 0.02):
            regime = 'VOLATILE'
        elif (current_hourly_vol < 0.2 and current_5m_vol < 0.3 and 
              volume_surge < 1.1 and price_change_1h < 0.01):
            regime = 'QUIET'
        else:
            regime = 'NORMAL'
        
        # Store regime data for analytics
        bot_status['market_regime'] = regime
        bot_status['volatility_metrics'] = {
            'hourly_vol': current_hourly_vol,
            'five_min_vol': current_5m_vol,
            'volume_surge': volume_surge,
            'price_change_1h': price_change_1h,
            'price_change_24h': price_change_24h
        }
        
        print(f"Market Regime: {regime}")
        print(f"Hourly Volatility: {current_hourly_vol:.3f}")
        print(f"5min Volatility: {current_5m_vol:.3f}")
        print(f"Volume Surge: {volume_surge:.2f}x")
        print(f"1h Price Change: {price_change_1h:.3f}")
        
        return regime
        
    except Exception as e:
        log_error_to_csv(str(e), "REGIME_DETECTION", "detect_market_regime", "ERROR")
        return 'NORMAL'

def detect_breakout_opportunities():
    """Real-time breakout and momentum opportunity detection with rate limiting"""
    try:
        opportunities = []
        major_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]  # Restored original 4 symbols
        
        # Rate limiting between API calls
        breakout_delay = 0.3  # 300ms delay between fetch calls
        
        for symbol in major_pairs:
            try:
                # Rate limiting before API calls
                time.sleep(breakout_delay)
                
                # Get short-term data for breakout detection (reduced limits)
                df_5m = fetch_data(symbol, "5m", 100)  # Reduced from 144 to 100
                
                time.sleep(breakout_delay)  # Rate limit between calls
                
                df_1m = fetch_data(symbol, "1m", 40)   # Reduced from 60 to 40
                
                if df_5m is None or df_1m is None or len(df_5m) < 40 or len(df_1m) < 20:  # Reduced minimums
                    continue
                
                current_price = df_1m['close'].iloc[-1]
                
                # Bollinger Band breakout detection
                bb_upper = df_5m['bb_upper'].iloc[-1]
                bb_lower = df_5m['bb_lower'].iloc[-1]
                bb_middle = df_5m['bb_middle'].iloc[-1]
                
                # Volume spike detection
                avg_volume = df_5m['volume'].rolling(48).mean().iloc[-1]
                current_volume = df_1m['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Momentum detection
                momentum_5m = (current_price - df_5m['close'].iloc[-6]) / df_5m['close'].iloc[-6]  # 30min momentum
                momentum_1m = (current_price - df_1m['close'].iloc[-10]) / df_1m['close'].iloc[-10]  # 10min momentum
                
                # RSI divergence detection
                rsi_current = df_5m['rsi'].iloc[-1]
                rsi_prev = df_5m['rsi'].iloc[-12]  # 1 hour ago
                
                opportunity_score = 0
                signals = []
                
                # Breakout signals
                if current_price > bb_upper and volume_ratio > 2.0:
                    opportunity_score += 30
                    signals.append("BB_BREAKOUT_UP")
                elif current_price < bb_lower and volume_ratio > 2.0:
                    opportunity_score += 30
                    signals.append("BB_BREAKOUT_DOWN")
                
                # Momentum signals
                if momentum_5m > 0.02 and momentum_1m > 0.01:
                    opportunity_score += 25
                    signals.append("STRONG_MOMENTUM_UP")
                elif momentum_5m < -0.02 and momentum_1m < -0.01:
                    opportunity_score += 25
                    signals.append("STRONG_MOMENTUM_DOWN")
                
                # Volume surge
                if volume_ratio > 3.0:
                    opportunity_score += 20
                    signals.append("VOLUME_SURGE")
                
                # RSI extremes with volume
                if rsi_current < 25 and volume_ratio > 1.5:
                    opportunity_score += 15
                    signals.append("RSI_OVERSOLD_VOLUME")
                elif rsi_current > 75 and volume_ratio > 1.5:
                    opportunity_score += 15
                    signals.append("RSI_OVERBOUGHT_VOLUME")
                
                if opportunity_score >= 40:  # High opportunity threshold
                    opportunities.append({
                        'symbol': symbol,
                        'score': opportunity_score,
                        'signals': signals,
                        'price': current_price,
                        'volume_ratio': volume_ratio,
                        'momentum_5m': momentum_5m,
                        'momentum_1m': momentum_1m,
                        'rsi': rsi_current,
                        'bb_position': 'ABOVE' if current_price > bb_upper else 'BELOW' if current_price < bb_lower else 'INSIDE'
                    })
                    
            except Exception as e:
                log_error_to_csv(str(e), "BREAKOUT_DETECTION", f"detect_breakout_opportunities_{symbol}", "WARNING")
                continue
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        if opportunities:
            print(f"\n=== BREAKOUT OPPORTUNITIES DETECTED ===")
            for opp in opportunities[:3]:  # Top 3
                print(f"{opp['symbol']}: Score {opp['score']}, Signals: {', '.join(opp['signals'])}")
        
        return opportunities
        
    except Exception as e:
        log_error_to_csv(str(e), "BREAKOUT_DETECTION", "detect_breakout_opportunities", "ERROR")
        return []

def calculate_smart_interval():
    """Calculate intelligent scanning interval based on market conditions"""
    try:
        # Get current market regime
        current_regime = bot_status.get('market_regime', 'NORMAL')
        base_intervals = bot_status.get('adaptive_intervals', {
            'QUIET': 1800, 'NORMAL': 900, 'VOLATILE': 300, 'EXTREME': 60, 'HUNTING': 30
        })
        
        # Check for hunting mode triggers
        hunting_triggers = 0
        
        # Time-based factors (market opening/closing times)
        current_hour = get_cairo_time().hour
        
        # US market hours (convert to Cairo time: UTC+2)
        us_market_hours = list(range(16, 24)) + list(range(0, 1))  # 2:30 PM - 11 PM Cairo time
        asian_market_hours = list(range(2, 10))  # 2 AM - 10 AM Cairo time
        
        if current_hour in us_market_hours:
            hunting_triggers += 1  # US market active
        if current_hour in asian_market_hours:
            hunting_triggers += 1  # Asian market active
            
        # Check for high volatility events
        volatility_metrics = bot_status.get('volatility_metrics', {})
        if (volatility_metrics.get('volume_surge', 1) > 2.5 or 
            volatility_metrics.get('price_change_1h', 0) > 0.03):
            hunting_triggers += 2
            
        # Check for recent profitable trades (momentum)
        recent_trades = bot_status.get('trading_summary', {}).get('trades_history', [])
        if len(recent_trades) >= 2:
            recent_profitable = sum(1 for trade in recent_trades[-2:] if trade.get('profit_loss', 0) > 0)
            if recent_profitable >= 2:
                hunting_triggers += 1  # Hot streak
                
        # Determine final interval
        if hunting_triggers >= 2 or current_regime in ['EXTREME', 'VOLATILE']:  # Lowered from 3 to 2 triggers
            bot_status['hunting_mode'] = True
            interval = base_intervals.get('HUNTING', 30)
            mode = 'HUNTING'
        else:
            bot_status['hunting_mode'] = False
            interval = base_intervals.get(current_regime, 900)
            mode = current_regime
            
        # Log interval decision
        print(f"\n=== Smart Interval Calculation ===")
        print(f"Market Regime: {current_regime}")
        print(f"Hunting Triggers: {hunting_triggers}")
        print(f"Selected Mode: {mode}")
        print(f"Interval: {interval} seconds ({interval/60:.1f} minutes)")
        
        return interval, mode
        
    except Exception as e:
        log_error_to_csv(str(e), "SMART_INTERVAL", "calculate_smart_interval", "ERROR")
        return 900, 'NORMAL'  # Default fallback

def should_scan_now():
    """Intelligent decision on whether to scan now based on market conditions"""
    try:
        current_time = get_cairo_time()
        
        # Always scan if no previous scan time
        if not bot_status.get('next_signal_time'):
            return True, "Initial scan"
            
        # Check if scheduled time has passed
        if current_time >= bot_status['next_signal_time']:
            return True, "Scheduled scan time reached"
            
        # Override scheduling for extreme conditions
        last_regime_check = bot_status.get('last_volatility_check')
        if (not last_regime_check or 
            (current_time - last_regime_check).total_seconds() > 300):  # Check regime every 5 minutes
            
            regime = detect_market_regime()
            bot_status['last_volatility_check'] = current_time
            
            if regime in ['EXTREME', 'VOLATILE']:
                return True, f"Market regime override: {regime}"
                
        # Check for breakout opportunities in extreme volatility
        if bot_status.get('market_regime') == 'EXTREME':
            opportunities = detect_breakout_opportunities()
            if opportunities:
                return True, f"Breakout opportunity detected: {opportunities[0]['symbol']}"
                
        return False, "Waiting for next scheduled scan"
        
    except Exception as e:
        log_error_to_csv(str(e), "SCAN_DECISION", "should_scan_now", "ERROR")
        return True, "Error in scan decision - defaulting to scan"

# Removed duplicate scan_trading_pairs definition (using the later optimized version)

def analyze_trading_pairs():
    """Analyze all available trading pairs and find the best opportunities"""
    pairs_analysis = []
    default_result = {"symbol": "BTCUSDT", "signal": "HOLD", "score": 0}
    
    try:
        if not client:
            return default_result
        
        try:
            exchange_info = get_exchange_info_cached()
        except Exception as e:
            log_error_to_csv(str(e), "PAIR_ANALYSIS", "analyze_trading_pairs", "ERROR")
            return default_result
        
        # Get all USDT pairs with good volume
        for symbol_info in exchange_info['symbols']:
            # Skip non-USDT or non-trading pairs
            if not (symbol_info['quoteAsset'] == 'USDT' and symbol_info['status'] == 'TRADING'):
                continue
            
            symbol = symbol_info['symbol']
            
            # Get 24hr stats
            try:
                # Get basic market stats
                ticker = client.get_ticker(symbol=symbol)
                volume_usdt = float(ticker['quoteVolume'])
                trades_24h = int(ticker['count'])
                
                # Filter out low volume/activity pairs
                if volume_usdt < 1000000 or trades_24h < 1000:  # Minimum $1M volume and 1000 trades
                    continue
                    
            except Exception as e:
                log_error_to_csv(str(e), "PAIR_ANALYSIS", f"analyze_trading_pairs_{symbol}_stats", "WARNING")
                continue

            try:
                # Get detailed market data
                df = fetch_data(symbol=symbol)
                if df is None or df.empty:
                    continue
                
                # Calculate metrics
                volatility = df['close'].pct_change().std() * np.sqrt(252)
                rsi = calculate_rsi(df['close'].values)
                macd_data = calculate_macd(df['close'].values)
                
                # Get sentiment for major coins
                sentiment = 'neutral'
                if symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
                    sentiment = analyze_market_sentiment()
                
                # Calculate trend metrics
                trend_strength = 0
                trend_score = 0
                if 'sma5' in df.columns and 'sma20' in df.columns:
                    trend_strength = abs(df['sma5'].iloc[-1] - df['sma20'].iloc[-1]) / df['sma20'].iloc[-1]
                    trend_score = 1 if df['sma5'].iloc[-1] > df['sma20'].iloc[-1] else -1
                
                momentum = df['close'].pct_change(5).iloc[-1]
                volume_trend = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                
                # Composite score calculation
                price_potential = 0
                if rsi < 30:  # Oversold
                    price_potential = 1
                elif rsi > 70:  # Overbought
                    price_potential = -1
                    
                momentum_score = momentum * 100  # Convert to percentage
                
                # Calculate final opportunity score
                base_score = (
                    price_potential * 0.3 +  # RSI weight
                    trend_score * 0.3 +      # Trend weight
                    momentum_score * 0.2 +    # Momentum weight
                    (volume_trend - 1) * 0.2  # Volume trend weight
                )
                
                # Apply volatility adjustment if configured
                if config.ADAPTIVE_STRATEGY['volatility_adjustment']:
                    score = base_score * (1 - (volatility/0.4))  # Use default volatility max
                else:
                    score = base_score
                
                # Add sentiment boost for major coins
                if sentiment == 'bullish':
                    score *= 1.2
                elif sentiment == 'bearish':
                    score *= 0.8
                
                # Generate signal based on composite analysis
                signal = "HOLD"
                if score > 0.5:  # Strong bullish signal
                    signal = "BUY"
                elif score < -0.5:  # Strong bearish signal
                    signal = "SELL"
                
                # Store analysis results
                pairs_analysis.append({
                    "symbol": symbol,
                    "signal": signal,
                    "score": score,
                    "volume_usdt": volume_usdt,
                    "volatility": volatility,
                    "rsi": rsi,
                    "trend_strength": trend_strength,
                    "volume_trend": volume_trend,
                    "sentiment": sentiment
                })
            
            except Exception as e:
                log_error_to_csv(str(e), "PAIR_ANALYSIS", f"analyze_trading_pairs_{symbol}_analysis", "WARNING")
                continue
        
        # Sort by absolute score (highest opportunity regardless of buy/sell)
        if pairs_analysis:
            pairs_analysis.sort(key=lambda x: abs(x['score']), reverse=True)
            return pairs_analysis[0]
        
        return {"symbol": "BTCUSDT", "signal": "HOLD", "score": 0}
            
    except Exception as e:
        log_error_to_csv(str(e), "PAIR_ANALYSIS", "analyze_trading_pairs", "ERROR")
        return {"symbol": "BTCUSDT", "signal": "HOLD", "score": 0}

def adaptive_strategy(df, symbol, indicators):
    """
    Smart strategy that adapts based on market conditions using configuration parameters
    - Uses volatility and trend strength
    - Adjusts thresholds dynamically based on config
    - Considers market regime with configurable settings
    """
    if df is None or len(df) < 30:
        return "HOLD", "Insufficient data"
        
    # Extract indicators
    rsi = indicators['rsi']
    macd_trend = indicators['macd_trend']
    sentiment = indicators['sentiment']
    volatility = indicators['volatility']
    current_price = indicators['current_price']
    sma5 = indicators['sma5']
    sma20 = indicators['sma20']
    ema50 = indicators.get('ema50')
    ema200 = indicators.get('ema200')
    stoch_k = indicators.get('stoch_k')
    vwap = indicators.get('vwap')
    adx = indicators.get('adx')
    
    # Get adaptive strategy settings
    adaptive_config = config.ADAPTIVE_STRATEGY
    
    # Calculate market regime using default thresholds
    is_high_volatility = volatility > 0.4  # Default volatility max
    trend_strength = abs((sma5 - sma20) / sma20)
    is_strong_trend = trend_strength > 0.02  # Default trend strength
    
    # Adjust thresholds based on market conditions and current market regime
    regime = bot_status.get('market_regime', 'NORMAL')
    if is_high_volatility or regime in ['VOLATILE', 'EXTREME']:
        rsi_buy = 35  # More conservative in high volatility
        rsi_sell = 65
        dynamic_threshold = max(25, adaptive_config.get('score_threshold', 30))
    elif regime == 'QUIET':
        rsi_buy = 45  # Harder to trigger in quiet markets
        rsi_sell = 55
        dynamic_threshold = min(35, adaptive_config.get('score_threshold', 30))
    else:
        rsi_buy = 40  # Default
        rsi_sell = 60
        dynamic_threshold = adaptive_config.get('score_threshold', 30)
        
    # Score-based system (0-100) with weights and per-component breakdown
    weights = config.ADAPTIVE_STRATEGY.get('weights', {
        'rsi': 0.2, 'macd': 0.2, 'ema_trend': 0.15, 'stoch': 0.15, 'adx': 0.15, 'vwap': 0.15
    })
    components = {
        'rsi': 0.0,
        'macd': 0.0,
        'ema_trend': 0.0,
        'stoch': 0.0,
        'adx': 0.0,
        'vwap': 0.0
    }

    # RSI (scaled by distance from thresholds) - Enhanced for buy low/sell high
    rsi_weight = weights.get('rsi', 0.2)
    if rsi < rsi_buy:
        # Stronger signal for deeper oversold conditions (buy lower)
        rsi_norm = (rsi_buy - rsi) / max(1.0, rsi_buy)
        # Boost score for very oversold conditions
        oversold_boost = 1.5 if rsi < 25 else (1.3 if rsi < 30 else 1.0)
        components['rsi'] = 100 * rsi_weight * rsi_norm * oversold_boost
    elif rsi > rsi_sell:
        # Stronger signal for deeper overbought conditions (sell higher)
        rsi_norm = (rsi - rsi_sell) / max(1.0, (100.0 - rsi_sell))
        # Boost score for very overbought conditions
        overbought_boost = 1.5 if rsi > 75 else (1.3 if rsi > 70 else 1.0)
        components['rsi'] = -100 * rsi_weight * rsi_norm * overbought_boost
    # else stays 0 near neutral band

    # MACD (fixed contribution based on trend)
    macd_w = weights.get('macd', 0.2)
    if macd_trend == 'BULLISH':
        components['macd'] = 100 * macd_w * 0.6
    elif macd_trend == 'BEARISH':
        components['macd'] = -100 * macd_w * 0.6

    # EMA trend (prefer EMA50/200 alignment; fallback to SMA cross)
    ema_w = weights.get('ema_trend', 0.15)
    if ema50 is not None and ema200 is not None:
        if current_price > ema50 > ema200:
            components['ema_trend'] = 100 * ema_w * 0.6
        elif current_price < ema50 < ema200:
            components['ema_trend'] = -100 * ema_w * 0.6
    else:
        if sma5 > sma20:
            components['ema_trend'] = 100 * ema_w * 0.3
        else:
            components['ema_trend'] = -100 * ema_w * 0.3

    # Stochastic (scaled by distance from overbought/oversold)
    stoch_w = weights.get('stoch', 0.15)
    if stoch_k is not None:
        st_oversold = float(config.STOCH.get('oversold', 20))
        st_overbought = float(config.STOCH.get('overbought', 80))
        if stoch_k < st_oversold:
            st_norm = (st_oversold - stoch_k) / max(1.0, st_oversold)
            components['stoch'] = 100 * stoch_w * st_norm
        elif stoch_k > st_overbought:
            st_norm = (stoch_k - st_overbought) / max(1.0, (100.0 - st_overbought))
            components['stoch'] = -100 * stoch_w * st_norm

    # ADX (symmetric: reward strong trend, small penalty for weak trend)
    adx_w = weights.get('adx', 0.15)
    if adx is not None:
        adx_min = float(config.ADAPTIVE_STRATEGY.get('adx_min', 20))
        if adx >= adx_min:
            components['adx'] = 100 * adx_w * 0.5
        else:
            deficit_ratio = max(0.0, (adx_min - adx) / max(1.0, adx_min))
            components['adx'] = -100 * adx_w * 0.2 * deficit_ratio

    # VWAP relation - Enhanced for buy low/sell high optimization
    vwap_w = weights.get('vwap', 0.15)
    if vwap is not None:
        vwap_distance = (current_price - vwap) / vwap
        if current_price >= vwap:
            # Above VWAP - good for selling, moderate for buying
            if abs(vwap_distance) > 0.02:  # Significantly above VWAP
                components['vwap'] = 100 * vwap_w * 0.6  # Strong sell signal
            else:
                components['vwap'] = 100 * vwap_w * 0.2  # Weak buy signal
        else:
            # Below VWAP - good for buying, bad for selling
            if abs(vwap_distance) > 0.02:  # Significantly below VWAP
                components['vwap'] = 100 * vwap_w * 0.6  # Strong buy signal (price is low)
            else:
                components['vwap'] = 100 * vwap_w * 0.2  # Weak buy signal

    # Price Range Position Analysis (NEW) - Ensure we buy low and sell high
    try:
        # Calculate recent price range (24h equivalent)
        recent_high = df['high'].tail(48).max() if 'high' in df.columns else current_price * 1.02
        recent_low = df['low'].tail(48).min() if 'low' in df.columns else current_price * 0.98
        price_range = recent_high - recent_low
        
        if price_range > 0:
            # Calculate where current price sits in the range (0 = bottom, 1 = top)
            range_position = (current_price - recent_low) / price_range
            
            # Reward buying near the bottom of the range
            if range_position < 0.3:  # In bottom 30% of range
                components['price_position'] = 100 * 0.1 * (0.3 - range_position) / 0.3  # Max +10 points
            elif range_position > 0.7:  # In top 30% of range
                components['price_position'] = -100 * 0.1 * (range_position - 0.7) / 0.3  # Max -10 points
            else:
                components['price_position'] = 0  # Neutral zone
        else:
            components['price_position'] = 0
    except Exception:
        components['price_position'] = 0

    # Volume Confirmation Analysis (NEW) - Validate signals with volume
    try:
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(20)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volume surge confirmation for signals
            if volume_ratio > 2.0:  # Strong volume surge
                # Determine if it's buying or selling volume based on price action
                price_change = (current_price - df['close'].tail(5).iloc[0]) / df['close'].tail(5).iloc[0]
                
                if price_change > 0.01:  # Price rising with volume - good for sells, bad for buys
                    components['volume_confirmation'] = -5  # Slight negative for buying
                elif price_change < -0.01:  # Price falling with volume - good for buys, bad for sells
                    components['volume_confirmation'] = 5  # Slight positive for buying
                else:
                    components['volume_confirmation'] = 0
            elif volume_ratio < 0.5:  # Low volume - reduces signal strength
                components['volume_confirmation'] = -3
            else:
                components['volume_confirmation'] = 0
        else:
            components['volume_confirmation'] = 0
    except Exception:
        components['volume_confirmation'] = 0

    # Sum base score and apply regime-based scaling to keep breakdown consistent
    score = sum(components.values())
    if is_high_volatility:
        components = {k: v * 0.8 for k, v in components.items()}
    if is_strong_trend:
        components = {k: v * 1.2 for k, v in components.items()}
    score = sum(components.values())

    # Use dynamic score threshold for decisions with concise breakdown
    score_threshold = dynamic_threshold
    breakdown = (
        f"RSI {components['rsi']:+.1f}, "
        f"MACD {components['macd']:+.1f}, "
        f"EMA {components['ema_trend']:+.1f}, "
        f"Stoch {components['stoch']:+.1f}, "
        f"ADX {components['adx']:+.1f}, "
        f"VWAP {components['vwap']:+.1f}"
    )

    if score >= score_threshold:
        return "BUY", f"Adaptive buy signal (Score: {score:.0f}/{score_threshold}; {breakdown})"
    elif score <= -score_threshold:
        return "SELL", f"Adaptive sell signal (Score: {score:.0f}/{score_threshold}; {breakdown})"
    
    return "HOLD", f"Neutral (Score: {score:.0f}/±{score_threshold}; {breakdown})"

def get_account_balances_summary():
    """Get a summary of all non-zero account balances"""
    try:
        if not client:
            return {"error": "Client not initialized"}
        
        account_info = client.get_account()
        balances = {}
        total_usdt_value = 0
        
        print("\n💰 Account Balance Summary:")
        print("=" * 40)
        
        for balance in account_info['balances']:
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            if total_balance > 0:
                asset = balance['asset']
                balances[asset] = {
                    'free': free_balance,
                    'locked': locked_balance,
                    'total': total_balance
                }
                
                # Try to get USDT value for major coins
                usdt_value = 0
                if asset == 'USDT':
                    usdt_value = total_balance
                elif asset in ['BTC', 'ETH', 'BNB']:
                    try:
                        ticker = client.get_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['lastPrice'])
                        usdt_value = total_balance * price
                    except:
                        usdt_value = 0  # Skip if price fetch fails
                
                total_usdt_value += usdt_value
                
                print(f"{asset:>8}: {free_balance:>12.8f} free, {locked_balance:>12.8f} locked "
                      f"(~${usdt_value:>8.2f})")
        
        print("=" * 40)
        print(f"{'TOTAL':>8}: ~${total_usdt_value:>8.2f} USDT value")
        print()
        
        return {
            'balances': balances,
            'total_usdt_value': total_usdt_value,
            'timestamp': format_cairo_time()
        }
        
    except Exception as e:
        error_msg = f"Error getting balance summary: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "BALANCE_SUMMARY_ERROR", "get_account_balances_summary", "ERROR")
        return {"error": error_msg}

def calculate_dynamic_trade_value(confidence_score, available_usdt=None):
    """Calculate the trade value that would be used by DYNAMIC_SIZING, adaptive to available balance"""
    if hasattr(config, 'DYNAMIC_SIZING') and config.DYNAMIC_SIZING.get('base_amount'):
        base_amount = config.DYNAMIC_SIZING.get('base_amount', 50)
        confidence_multiplier = config.DYNAMIC_SIZING.get('confidence_multiplier', 3.0)
        max_amount = config.DYNAMIC_SIZING.get('max_amount', 200)
        adaptive_to_balance = config.DYNAMIC_SIZING.get('adaptive_to_balance', False)
        min_balance_ratio = config.DYNAMIC_SIZING.get('min_balance_ratio', 0.8)
        fallback_percentage = config.DYNAMIC_SIZING.get('fallback_percentage', 60)
        
        # Calculate ideal dynamic position value
        ideal_value = base_amount * (1 + (confidence_score * confidence_multiplier))
        ideal_value = min(max_amount, ideal_value)
        
        # If adaptive sizing is enabled and we have balance info
        if adaptive_to_balance and available_usdt is not None:
            max_affordable = available_usdt * min_balance_ratio
            
            if ideal_value > max_affordable:
                # If we can't afford the ideal amount, scale down intelligently
                if max_affordable >= base_amount:
                    # Use what we can afford, but at least base amount
                    adaptive_value = max_affordable
                    print(f"   🔄 DYNAMIC_SIZING: Scaling down from ${ideal_value:.2f} to ${adaptive_value:.2f} (available: ${available_usdt:.2f})")
                    return adaptive_value
                else:
                    # Use fallback percentage of available balance
                    fallback_value = available_usdt * (fallback_percentage / 100)
                    print(f"   🔄 DYNAMIC_SIZING: Using fallback {fallback_percentage}% of balance: ${fallback_value:.2f} (ideal: ${ideal_value:.2f})")
                    return fallback_value
        
        return ideal_value
    
    return 50  # Default base amount if DYNAMIC_SIZING not configured

def check_usdt_balance(symbol="BTCUSDT", confidence_score=None):
    """Check if we have sufficient USDT balance to place a BUY order, considering DYNAMIC_SIZING"""
    try:
        # Cache to reduce repeated API calls within a short window
        if 'usdt_balance_cache' not in bot_status:
            bot_status['usdt_balance_cache'] = {}
        cache_key = f"{symbol}_USDT_CHECK_{confidence_score or 'min'}"
        cache_entry = bot_status['usdt_balance_cache'].get(cache_key)
        now = get_cairo_time()
        if cache_entry and (now - cache_entry['time']).total_seconds() < 180:  # 3 minute cache
            return cache_entry['has'], cache_entry['amount'], cache_entry['msg']
            
        if not client:
            return False, 0, "Client not initialized"
            
        # Get account balances
        account_info = client.get_account()
        
        # Find USDT balance
        usdt_balance = None
        for balance in account_info['balances']:
            if balance['asset'] == 'USDT':
                usdt_balance = float(balance['free'])
                break
                
        if usdt_balance is None:
            result = (False, 0, "USDT balance not found in account")
        else:
            # Calculate the required balance considering DYNAMIC_SIZING
            if confidence_score is not None:
                # Pass available balance to DYNAMIC_SIZING for adaptive calculation
                dynamic_trade_value = calculate_dynamic_trade_value(confidence_score, available_usdt=usdt_balance)
                min_notional = calculate_smart_minimum_trade_usdt(symbol, available_usdt=usdt_balance)
                
                # The required amount is the adaptive dynamic value, but must meet minimum notional
                min_required = max(min_notional, dynamic_trade_value)
                required_type = f"DYNAMIC_SIZING (${dynamic_trade_value:.2f})"
            else:
                # Fallback to minimum check only
                min_required = calculate_smart_minimum_trade_usdt(symbol, available_usdt=usdt_balance)
                required_type = "minimum trade"
            has_sufficient = usdt_balance >= min_required
            
            result = (
                (True, usdt_balance, f"Sufficient USDT: {usdt_balance:.2f} (min: {min_required:.2f})")
                if has_sufficient
                else (False, usdt_balance, f"Insufficient USDT: {usdt_balance:.2f} < {min_required:.2f}")
            )
            
        # Save to cache
        bot_status['usdt_balance_cache'][cache_key] = {
            'has': result[0], 'amount': result[1], 'msg': result[2], 'time': now
        }
        return result
        
    except Exception as e:
        error_msg = f"Error checking USDT balance: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "USDT_BALANCE_CHECK_ERROR", "check_usdt_balance", "ERROR")
        return False, 0, error_msg

def check_coin_balance(symbol):
    """Check if we have sufficient balance to place a SELL order for the given symbol"""
    try:
        # Cache to reduce repeated API calls within a short window
        if 'balance_cache' not in bot_status:
            bot_status['balance_cache'] = {}
        cache_entry = bot_status['balance_cache'].get(symbol)
        now = get_cairo_time()
        if cache_entry and (now - cache_entry['time']).total_seconds() < 300:
            return cache_entry['has'], cache_entry['amount'], cache_entry['msg']
        if not client:
            print(f"⚠️ Client not initialized - cannot check balance for {symbol}")
            return False, 0, "Client not initialized"
        
        # Extract base asset from symbol (e.g., "BTC" from "BTCUSDT")
        if symbol.endswith('USDT'):
            base_asset = symbol[:-4]  # Remove "USDT"
        elif symbol.endswith('BUSD'):
            base_asset = symbol[:-4]  # Remove "BUSD" 
        else:
            # For other quote currencies, try to find the quote asset
            try:
                exchange_info = get_exchange_info_cached()
                symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                if symbol_info:
                    base_asset = symbol_info['baseAsset']
                else:
                    print(f"⚠️ Cannot determine base asset for {symbol}")
                    return False, 0, "Unknown symbol format"
            except Exception as e:
                print(f"⚠️ Error getting symbol info for {symbol}: {e}")
                return False, 0, f"Symbol info error: {e}"
        
        print(f"🔍 Checking {base_asset} balance for potential sell order...")
        
        # Get account balances
        account_info = client.get_account()
        asset_balance = 0
        
        for balance in account_info['balances']:
            if balance['asset'] == base_asset:
                asset_balance = float(balance['free'])
                break
        
        print(f"💰 Available {base_asset} balance: {asset_balance}")
        
        # Get minimum quantity requirements
        min_sellable_qty = 0.001  # Default minimum
        try:
            exchange_info = get_exchange_info_cached()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_size_filter:
                    min_sellable_qty = float(lot_size_filter['minQty'])
        except Exception as e:
            print(f"⚠️ Warning: Could not get minimum quantity for {symbol}: {e}")
        
        # Check if we have enough balance to place a meaningful sell order
        has_sufficient_balance = asset_balance >= min_sellable_qty
        
        print(f"📊 Balance check result:")
        print(f"   Available: {asset_balance} {base_asset}")
        print(f"   Minimum required: {min_sellable_qty} {base_asset}")
        print(f"   Can sell: {'✅ Yes' if has_sufficient_balance else '❌ No'}")
        
        result = (
            (True, asset_balance, f"Sufficient balance: {asset_balance} {base_asset}")
            if has_sufficient_balance
            else (False, asset_balance, f"Insufficient balance: {asset_balance} < {min_sellable_qty} {base_asset}")
        )
        # Save to cache
        bot_status['balance_cache'][symbol] = {
            'has': result[0], 'amount': result[1], 'msg': result[2], 'time': now
        }
        return result
            
    except Exception as e:
        error_msg = f"Error checking balance for {symbol}: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "BALANCE_CHECK_ERROR", "check_coin_balance", "ERROR")
        return False, 0, error_msg

def get_account_balance_for_ui():
    """Get account balance specifically for UI display - smart caching"""
    try:
        # Check cache first (5-minute cache for UI)
        cache_key = 'ui_balance_cache'
        cache_entry = bot_status.get(cache_key)
        now = get_cairo_time()
        if cache_entry and (now - cache_entry['time']).total_seconds() < 300:  # 5-minute cache
            return cache_entry['data']
        
        if not client:
            return {"usdt_balance": 0.0, "total_value": 0.0, "error": "Client not initialized"}
        
        # Get account info
        account_info = client.get_account()
        usdt_balance = 0.0
        total_value = 0.0
        
        # Find USDT balance and calculate total portfolio value
        for balance in account_info['balances']:
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            if total_balance > 0:
                asset = balance['asset']
                
                if asset == 'USDT':
                    usdt_balance = total_balance
                    total_value += total_balance
                elif asset in ['BTC', 'ETH', 'BNB', 'ADA', 'LINK', 'AVAX', 'XRP']:
                    # Get price for major coins to calculate value
                    try:
                        ticker = client.get_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['lastPrice'])
                        asset_value = total_balance * price
                        total_value += asset_value
                    except Exception:
                        pass  # Skip if price fetch fails
        
        result = {
            "usdt_balance": usdt_balance,
            "total_value": total_value,
            "timestamp": format_cairo_time(),
            "error": None
        }
        
        # Cache the result
        bot_status[cache_key] = {'data': result, 'time': now}
        
        return result
        
    except Exception as e:
        error_msg = f"Error getting account balance for UI: {e}"
        log_error_to_csv(error_msg, "UI_BALANCE_ERROR", "get_account_balance_for_ui", "WARNING")
        return {"usdt_balance": 0.0, "total_value": 0.0, "error": error_msg}

def signal_generator(df, symbol="BTCUSDT"):
    print("\n=== Generating Trading Signal ===")
    print(f"Analyzing market sentiment from order book and trade data...")
    
    # Display available modules status
    if SMART_OPTIMIZER_AVAILABLE:
        print("🧠 Smart Signal Optimizer: ✅ Available")
    else:
        print("🧠 Smart Signal Optimizer: ❌ Not Available")
    
    if ENHANCED_MODULES_AVAILABLE:
        print("🎯 Enhanced ML Modules: ✅ Available")
    else:
        print("🎯 Enhanced ML Modules: ❌ Not Available")
        
    if df is None or len(df) < 30:
        print(f"Insufficient data for {symbol}")  # Debug log
        signal = "HOLD"
        update_bot_status_common(symbol, signal, 0)
        log_signal_to_csv(signal, 0, {"symbol": symbol}, "Insufficient data")
        return signal
    
    # Enhanced risk management checks
    daily_pnl = bot_status['trading_summary'].get('total_revenue', 0)
    consecutive_losses = bot_status.get('consecutive_losses', 0)
    
    # Stop trading if daily loss limit exceeded
    if daily_pnl < -config.MAX_DAILY_LOSS:
        log_signal_to_csv("HOLD", 0, {"symbol": symbol}, f"Daily loss limit exceeded: ${daily_pnl}")
        return "HOLD"
    
    # Reduce activity after consecutive losses
    if consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
        # Still compute a signal but enforce HOLD at the end; don't spam with repeated logs
        risk_locked = True
    else:
        risk_locked = False
    
    sentiment = analyze_market_sentiment()
    
    # Get the latest technical indicators with error handling
    try:
        # Handle RSI - could be a single value or Series
        if 'rsi' in df.columns:
            if hasattr(df['rsi'], 'iloc'):
                rsi = float(df['rsi'].iloc[-1]) if not pd.isna(df['rsi'].iloc[-1]) else 50
            else:
                rsi = float(df['rsi']) if not pd.isna(df['rsi']) else 50
        else:
            rsi = 50
            
        # Handle MACD data
        if 'macd' in df.columns:
            if hasattr(df['macd'], 'iloc'):
                macd = float(df['macd'].iloc[-1]) if not pd.isna(df['macd'].iloc[-1]) else 0
            else:
                macd = float(df['macd']) if not pd.isna(df['macd']) else 0
        else:
            macd = 0
            
        # Handle MACD trend
        if 'macd_trend' in df.columns:
            if hasattr(df['macd_trend'], 'iloc'):
                macd_trend = df['macd_trend'].iloc[-1] if not pd.isna(df['macd_trend'].iloc[-1]) else 'NEUTRAL'
            else:
                macd_trend = df['macd_trend'] if not pd.isna(df['macd_trend']) else 'NEUTRAL'
        else:
            macd_trend = 'NEUTRAL'
            
        # Handle SMAs
        if 'sma5' in df.columns and hasattr(df['sma5'], 'iloc'):
            sma5 = float(df['sma5'].iloc[-1]) if not pd.isna(df['sma5'].iloc[-1]) else 0
        else:
            sma5 = 0
            
        if 'sma20' in df.columns and hasattr(df['sma20'], 'iloc'):
            sma20 = float(df['sma20'].iloc[-1]) if not pd.isna(df['sma20'].iloc[-1]) else 0
        else:
            sma20 = 0
            
        # Handle current price
        if hasattr(df['close'], 'iloc'):
            current_price = float(df['close'].iloc[-1])
        else:
            current_price = float(df['close'])
            
        # Handle volatility
        if 'volatility' in df.columns and hasattr(df['volatility'], 'iloc'):
            volatility = float(df['volatility'].iloc[-1]) if not pd.isna(df['volatility'].iloc[-1]) else 0.5
        else:
            # Calculate basic volatility as fallback
            if hasattr(df['close'], 'pct_change'):
                volatility = float(df['close'].pct_change().std() * np.sqrt(252))
            else:
                volatility = 0.5

        # New indicators
        ema50 = float(df['ema50'].iloc[-1]) if 'ema50' in df.columns and not pd.isna(df['ema50'].iloc[-1]) else None
        ema200 = float(df['ema200'].iloc[-1]) if 'ema200' in df.columns and not pd.isna(df['ema200'].iloc[-1]) else None
        stoch_k = float(df['stoch_k'].iloc[-1]) if 'stoch_k' in df.columns and not pd.isna(df['stoch_k'].iloc[-1]) else None
        stoch_d = float(df['stoch_d'].iloc[-1]) if 'stoch_d' in df.columns and not pd.isna(df['stoch_d'].iloc[-1]) else None
        vwap = float(df['vwap'].iloc[-1]) if 'vwap' in df.columns and not pd.isna(df['vwap'].iloc[-1]) else None
        adx = float(df['adx'].iloc[-1]) if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]) else None
                
    except Exception as e:
        log_error_to_csv(f"Error extracting indicators: {str(e)}", "INDICATOR_ERROR", "signal_generator", "ERROR")
        return "HOLD"
    
    # Handle NaN values
    if pd.isna(rsi) or pd.isna(macd) or pd.isna(sma5) or pd.isna(sma20):
        log_signal_to_csv("HOLD", current_price, {'symbol': symbol, 'rsi': rsi, 'macd': macd, 'sentiment': sentiment}, "NaN values detected")
        return "HOLD"
        
    # Prepare indicators dictionary for strategies
    indicators = {
        'symbol': symbol,  # Add symbol to indicators for proper logging
        'rsi': rsi,
        'macd': macd,
        'macd_trend': macd_trend,
        'sentiment': sentiment,
        'sma5': sma5,
        'sma20': sma20,
        'current_price': current_price,
        'volatility': volatility,
        'ema50': ema50,
        'ema200': ema200,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'vwap': vwap,
        'adx': adx
    }
    
    # Use selected strategy with enhanced error handling
    try:
        strategy = bot_status.get('trading_strategy', config.DEFAULT_STRATEGY)
        print(f"Using strategy: {strategy}")  # Debug log
        
        if strategy == 'ML_PURE':
            if ML_STRATEGY_AVAILABLE:
                print("🧠 Using ML Pure Strategy - AI-Driven Trading")
                signal, reason = ml_pure_strategy(df, symbol, indicators)
                
                # Early exit for very low probability signals to avoid unnecessary computation
                if hasattr(signal, '__dict__') or isinstance(reason, str):
                    # Extract probability from reason if available
                    import re
                    prob_match = re.search(r'success probability.*?(\d+\.?\d*)', reason.lower())
                    if prob_match:
                        success_prob = float(prob_match.group(1))
                        if success_prob < 0.05:  # Less than 5% success probability
                            print(f"🚫 Early exit: Extremely low success probability ({success_prob:.2f}) detected")
                            print(f"   Skipping complex ML processing to save resources")
                            signal = "HOLD"
                            reason = f"Early exit due to extremely low success probability ({success_prob:.2f})"
                            # Skip all further ML processing
                            return signal, reason, indicators
            else:
                print("⚠️ ML Pure Strategy not available, falling back to ADAPTIVE")
                signal, reason = adaptive_strategy(df, symbol, indicators)
        elif strategy == 'ADAPTIVE':
            signal, reason = adaptive_strategy(df, symbol, indicators)
        else:
            print(f"Unknown strategy {strategy}, defaulting to ADAPTIVE")  # Debug log
            signal, reason = adaptive_strategy(df, symbol, indicators)  # Default to adaptive
        
        # Strategy-Specific Signal Processing
        if ENHANCED_MODULES_AVAILABLE and signal != "HOLD":
            if strategy == 'ML_PURE':
                print(f"\n🧠 Applying ML_PURE optimized signal processing...")
                
                # ML_PURE uses its own internal validation - apply minimal external filtering
                # Only check for critical risk factors, not conservative quality filters
                try:
                    # Prepare minimal market data for critical risk assessment only
                    market_data = {
                        'volume_24h': getattr(df, 'volume', pd.Series([1000000])).iloc[-1] if hasattr(df, 'volume') else 1000000,
                        'price_change_24h_pct': indicators.get('price_change_24h', 0),
                        'volume_ratio': indicators.get('volume_ratio', 1.0),
                        'spread_pct': 0.05,
                        'volume_consistency': 0.5
                    }
                    
                    # Use much more lenient filtering for ML_PURE (confidence threshold 0.35 vs 0.65)
                    signal_filter = get_signal_filter()
                    filtered_result = signal_filter.filter_and_validate_signal(
                        symbol, signal, indicators, market_data, confidence_threshold=0.35
                    )
                    
                    print(f"   ML Raw Signal: {signal}")
                    print(f"   ML Validated Signal: {filtered_result['signal']}")
                    print(f"   ML Confidence: {filtered_result['confidence']:.2f}")
                    print(f"   Risk Assessment: {filtered_result['filters_passed']}/7 checks passed")
                    
                    # For ML_PURE, only override if critical risk factors are detected
                    if filtered_result['filters_passed'] >= 3:  # Much more lenient (3/7 vs 5/7)
                        # Keep ML signal - just log the validation
                        print(f"   ✅ ML_PURE signal validated: {signal}")
                        reason = f"ML {reason} - Risk Assessment: {filtered_result['filters_passed']}/7"
                    else:
                        # Only override for serious risk concerns
                        signal = filtered_result['signal']
                        reason = f"ML Risk Override: {filtered_result['reason']}"
                        print(f"   ⚠️ ML_PURE signal overridden due to high risk")
                    
                except Exception as filter_error:
                    print(f"   ⚠️ ML signal validation error: {filter_error}")
                    # Continue with ML signal if validation fails
                    
            else:
                # Standard Enhanced Signal Filtering for non-ML strategies
                print(f"\n🔍 Applying enhanced signal filtering...")
                
                # Prepare market data for filtering
                market_data = {
                    'volume_24h': getattr(df, 'volume', pd.Series([1000000])).iloc[-1] if hasattr(df, 'volume') else 1000000,
                    'price_change_24h_pct': indicators.get('price_change_24h', 0),
                    'volume_ratio': indicators.get('volume_ratio', 1.0),
                    'spread_pct': 0.05,  # Default spread
                    'volume_consistency': 0.5  # Default consistency
                }
                
                try:
                    signal_filter = get_signal_filter()
                    filtered_result = signal_filter.filter_and_validate_signal(
                        symbol, signal, indicators, market_data, confidence_threshold=0.65
                    )
                    
                    print(f"   Raw Signal: {signal}")
                    print(f"   Filtered Signal: {filtered_result['signal']}")
                    print(f"   Confidence: {filtered_result['confidence']:.2f}")
                    print(f"   Quality Score: {filtered_result['quality_score']:.2f}")
                    print(f"   Filters Passed: {filtered_result['filters_passed']}/7")
                    print(f"   Reason: {filtered_result['reason']}")
                    
                    # Update signal and reason with filtered results
                    signal = filtered_result['signal']
                    reason = f"Enhanced Filter: {filtered_result['reason']}"
                    
                except Exception as filter_error:
                    print(f"   ⚠️ Signal filtering error: {filter_error}")
                    # Continue with original signal if filtering fails
            
            # Store filtering metrics in bot status (for both strategies)
            if 'filtered_result' in locals():
                bot_status['last_signal_quality'] = {
                    'confidence': filtered_result['confidence'],
                    'quality_score': filtered_result['quality_score'],
                    'filters_passed': filtered_result['filters_passed']
                }
        
        # Strategy-Specific Smart Signal Optimization
        if SMART_OPTIMIZER_AVAILABLE:
            if strategy == 'ML_PURE':
                print(f"\n🧠 Applying ML_PURE smart optimization...")
                
                try:
                    # Get current balances for optimization
                    current_balance = {}
                    if client:
                        account_info = client.get_account()
                        for balance in account_info['balances']:
                            if float(balance['free']) > 0:
                                current_balance[balance['asset']] = float(balance['free'])
                    
                    signal_optimizer = get_signal_optimizer()
                    optimization_result = signal_optimizer.optimize_entry_exit(
                        symbol, signal, df, indicators, current_balance
                    )
                    
                    print(f"   📊 ML Original Signal: {signal}")
                    print(f"   🎯 ML Optimized Signal: {optimization_result['optimized_signal']}")
                    print(f"   📈 ML Optimization Confidence: {optimization_result['confidence']:.2f}")
                    print(f"   💡 ML Optimization Reason: {optimization_result['reason']}")
                    
                    if 'factors' in optimization_result:
                        factors = optimization_result['factors']
                        print(f"   💰 Profitability Score: {factors.get('profitability_score', 0):.2f}")
                        print(f"   ⏰ Timing Score: {factors.get('timing_score', 0):.2f}")
                        print(f"   ⚖️ Risk/Reward Ratio: {factors.get('risk_reward_ratio', 0):.2f}")
                        print(f"   🚀 Momentum Score: {factors.get('momentum_score', 0):.2f}")
                    
                    # For ML_PURE, be more conservative about overriding ML decisions
                    # Only override if optimizer has very high confidence (0.75+) and suggests something different
                    if optimization_result['optimized_signal'] != signal and optimization_result['confidence'] > 0.75:
                        original_signal = signal
                        signal = optimization_result['optimized_signal']
                        reason = f"ML Optimizer Override: {optimization_result['reason']} (ML Original: {original_signal})"
                        print(f"   🔄 ML signal changed from {original_signal} to {signal} by high-confidence optimizer")
                    elif optimization_result['optimized_signal'] == signal:
                        print(f"   ✅ ML Optimizer confirms {signal} signal")
                        reason = f"ML Confirmed: {reason}"
                    else:
                        print(f"   ℹ️ ML Optimizer suggests {optimization_result['optimized_signal']} but respecting ML decision ({optimization_result['confidence']:.2f} < 0.75)")
                    
                except Exception as optimizer_error:
                    print(f"   ⚠️ ML optimization error: {optimizer_error}")
                    # Continue with ML signal if optimization fails
                    
            else:
                # Standard Smart Signal Optimization for non-ML strategies
                print(f"\n🧠 Applying smart profitability optimization...")
                
                try:
                    # Get current balances for optimization
                    current_balance = {}
                    if client:
                        account_info = client.get_account()
                        for balance in account_info['balances']:
                            if float(balance['free']) > 0:
                                current_balance[balance['asset']] = float(balance['free'])
                    
                    signal_optimizer = get_signal_optimizer()
                    optimization_result = signal_optimizer.optimize_entry_exit(
                        symbol, signal, df, indicators, current_balance
                    )
                    
                    print(f"   📊 Original Signal: {signal}")
                    print(f"   🎯 Optimized Signal: {optimization_result['optimized_signal']}")
                    print(f"   📈 Optimization Confidence: {optimization_result['confidence']:.2f}")
                    print(f"   💡 Optimization Reason: {optimization_result['reason']}")
                    
                    if 'factors' in optimization_result:
                        factors = optimization_result['factors']
                        print(f"   💰 Profitability Score: {factors.get('profitability_score', 0):.2f}")
                        print(f"   ⏰ Timing Score: {factors.get('timing_score', 0):.2f}")
                        print(f"   ⚖️ Risk/Reward Ratio: {factors.get('risk_reward_ratio', 0):.2f}")
                        print(f"   🚀 Momentum Score: {factors.get('momentum_score', 0):.2f}")
                    
                    if optimization_result['should_wait']:
                        print(f"   📊 Optimization suggests waiting: {optimization_result.get('timing_details', {}).get('wait_reason', 'Better timing expected')}")
                    
                    # Update signal with optimized result if it's different and confidence is good
                    if optimization_result['optimized_signal'] != signal and optimization_result['confidence'] > 0.6:
                        original_signal = signal
                        signal = optimization_result['optimized_signal']
                        reason = f"Smart Optimizer: {optimization_result['reason']} (Original: {original_signal})"
                        print(f"   🔄 Signal changed from {original_signal} to {signal} by Smart Optimizer")
                    elif optimization_result['optimized_signal'] == signal:
                        print(f"   ✅ Smart Optimizer confirms {signal} signal")
                    else:
                        print(f"   ⚠️ Smart Optimizer suggests {optimization_result['optimized_signal']} but confidence too low ({optimization_result['confidence']:.2f})")
                    
                except Exception as optimizer_error:
                    print(f"   ⚠️ Signal optimization error: {optimizer_error}")
                    # Continue with filtered signal if optimization fails
            
            # Store optimization metrics in bot status (for both strategies)
            if 'optimization_result' in locals():
                bot_status['last_optimization'] = {
                    'confidence': optimization_result['confidence'],
                    'factors': optimization_result.get('factors', {}),
                    'should_wait': optimization_result['should_wait']
                }
        
        # Smart Balance Validation (NEW) - Ensure we can actually execute the signal
        if signal == "SELL":
            print(f"\n💰 Validating balance for SELL signal...")
            
            try:
                has_balance, available_balance, balance_msg = check_coin_balance(symbol)
                
                if not has_balance:
                    print(f"   ❌ Cannot execute SELL - {balance_msg}")
                    signal = "HOLD"
                    reason = f"SELL blocked: {balance_msg}"
                    
                    # Log this for analysis
                    log_signal_to_csv(signal, current_price, indicators, 
                                    f"Strategy wanted SELL but {balance_msg}")
                else:
                    print(f"   ✅ Balance validated - {balance_msg}")
                    
                    # Additional profitability check for selling
                    # Only sell if we're likely to be in profit
                    try:
                        base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol.split('USDT')[0]
                        
                        # Check if we're in the upper part of recent range
                        if 'high' in df.columns and 'low' in df.columns:
                            recent_high = df['high'].tail(48).max()
                            recent_low = df['low'].tail(48).min()
                            price_range = recent_high - recent_low
                            
                            if price_range > 0:
                                range_position = (current_price - recent_low) / price_range
                                
                                # Only sell if we're in upper part of range (selling high)
                                if range_position < 0.4:  # In bottom 40% of range
                                    print(f"   📊 Price in lower range ({range_position:.1%}) - holding for better price")
                                    signal = "HOLD"
                                    reason = f"Smart Hold: Price too low in range ({range_position:.1%}) - waiting to sell higher"
                                else:
                                    print(f"   📊 Price in upper range ({range_position:.1%}) - good time to sell")
                    except Exception as range_error:
                        print(f"   ⚠️ Range analysis error: {range_error}")
                        
            except Exception as balance_error:
                print(f"   ⚠️ Balance validation error: {balance_error}")
        
        elif signal == "BUY":
            print(f"\n💰 Validating conditions for BUY signal...")
            
            # First, check if we have sufficient USDT balance considering DYNAMIC_SIZING
            try:
                # Get confidence score for DYNAMIC_SIZING calculation - prefer ML optimization confidence
                confidence_score = 0.5  # Default
                if 'last_optimization' in bot_status and bot_status['last_optimization'].get('confidence'):
                    confidence_score = bot_status['last_optimization']['confidence']
                elif 'last_signal_quality' in bot_status and bot_status['last_signal_quality'].get('confidence'):
                    confidence_score = bot_status['last_signal_quality']['confidence']
                
                # Check balance with DYNAMIC_SIZING considerations
                has_usdt, usdt_amount, usdt_msg = check_usdt_balance(symbol, confidence_score)
                
                if not has_usdt:
                    print(f"   ❌ Cannot execute BUY - {usdt_msg}")
                    signal = "HOLD"
                    reason = f"BUY blocked: {usdt_msg}"
                    
                    # Log this for analysis
                    log_signal_to_csv(signal, current_price, indicators, 
                                    f"Strategy wanted BUY but {usdt_msg}")
                else:
                    print(f"   ✅ USDT balance validated - {usdt_msg}")
                    
                    # Additional range position check for buying
                    try:
                        # Check if we're buying low (in lower part of recent range)
                        if 'high' in df.columns and 'low' in df.columns:
                            recent_high = df['high'].tail(48).max()
                            recent_low = df['low'].tail(48).min()
                            price_range = recent_high - recent_low
                            
                            if price_range > 0:
                                range_position = (current_price - recent_low) / price_range
                                
                                # Prefer buying in lower part of range (buying low)
                                if range_position > 0.6:  # In top 40% of range
                                    print(f"   📊 Price in upper range ({range_position:.1%}) - may wait for better entry")
                                    # Don't block but reduce confidence
                                    if 'Smart Optimizer:' in reason:
                                        reason += f" (Price high in range: {range_position:.1%})"
                                    else:
                                        reason += f" - Price high in range: {range_position:.1%}"
                                else:
                                    print(f"   📊 Price in lower range ({range_position:.1%}) - good entry point")
                                    if 'Smart Optimizer:' in reason:
                                        reason += f" (Good entry: {range_position:.1%} in range)"
                    except Exception as buy_analysis_error:
                        print(f"   ⚠️ Buy analysis error: {buy_analysis_error}")
                        
            except Exception as usdt_balance_error:
                print(f"   ⚠️ USDT balance validation error: {usdt_balance_error}")
                # If balance check fails, be conservative and hold
                signal = "HOLD"
                reason = f"BUY blocked: Balance check failed - {usdt_balance_error}"
        
        # Strategy-Specific ML Intelligence Analysis
        if ENHANCED_MODULES_AVAILABLE and signal != "HOLD" and config.ML_ENABLED:
            if strategy == 'ML_PURE':
                print(f"\n🧠 Applying ML_PURE Intelligence Validation...")
                
                try:
                    # For ML_PURE, use intelligence as validation rather than override
                    market_intel = market_intelligence.get_market_intelligence_summary(df, {
                        'action': signal,
                        'symbol': symbol,
                        'rsi': rsi,
                        'macd': macd,
                        'macd_trend': macd_trend,
                        'volume_ratio': indicators.get('volume_ratio', 1.0),
                        'current_price': current_price,
                        'volatility': volatility
                    })
                    
                    print(f"   📊 Market Regime: {market_intel['market_regime']['primary_regime']} "
                          f"(Confidence: {market_intel['market_regime']['confidence']:.2f})")
                    print(f"   🔍 Pattern Confidence: {market_intel['pattern_analysis']['pattern_confidence']:.2f}")
                    print(f"   📈 Signal Success Probability: {market_intel['signal_probability']:.2f}")
                    print(f"   🎯 Intelligence Score: {market_intel['intelligence_score']:.2f}")
                    
                    # Update adaptive thresholds based on market regime
                    if config.ADAPTIVE_THRESHOLDS_ENABLED:
                        adaptive_thresholds = market_intel['adaptive_thresholds']
                        print(f"   ⚙️ Adaptive RSI Thresholds: {adaptive_thresholds['rsi_oversold']:.1f}/{adaptive_thresholds['rsi_overbought']:.1f}")
                        print(f"   ⚙️ Adaptive MACD Threshold: {adaptive_thresholds['macd_threshold']:.4f}")
                    
                    # ML Pattern Recognition & Market Regime Prediction
                    if config.PATTERN_RECOGNITION_ENABLED:
                        signal_success_probability = ml_predictor.predict_signal_success(
                            {'action': signal, 'rsi': rsi, 'macd': macd}, indicators
                        )
                        print(f"   🤖 ML Signal Success Prediction: {signal_success_probability:.2f}")
                    
                    # Market Regime Prediction
                    if config.REGIME_DETECTION_ENABLED:
                        regime_prediction = ml_predictor.predict_market_regime(df)
                        print(f"   🌍 ML Regime Prediction: {regime_prediction['regime']} "
                              f"(Confidence: {regime_prediction['confidence']:.2f})")
                    
                    intelligence_score = market_intel['intelligence_score']
                    signal_probability = market_intel['signal_probability']
                    
                    # For ML_PURE: Only override for extreme risk situations (much more lenient)
                    if signal_probability < 0.05:  # Only for extremely low probability (0.05 vs 0.25)
                        print(f"   🚨 EXTREME RISK: Very low success probability ({signal_probability:.2f}) - ML Override for safety")
                        original_signal = signal
                        signal = "HOLD"
                        reason = f"ML Safety Override: Extreme risk detected ({signal_probability:.2f}) - Original: {original_signal}"
                        print(f"   🔄 ML_PURE signal overridden due to extreme risk")
                    elif signal_probability < 0.15:  # Warning zone but don't override
                        print(f"   ⚠️ Low success probability ({signal_probability:.2f}) - Proceeding with ML signal but flagged")
                        indicators['ml_low_probability'] = True
                        reason = f"ML {reason} - Low prob warning: {signal_probability:.2f}"
                    else:
                        print(f"   ✅ ML Intelligence validation passed - Signal maintained")
                        if signal_probability > 0.6:
                            reason = f"ML {reason} - High success prob: {signal_probability:.2f}"
                    
                except Exception as intel_error:
                    print(f"   ⚠️ ML Intelligence validation error: {intel_error}")
                    # Continue with ML signal if intelligence fails
                    
            else:
                # Standard ML Intelligence Analysis for non-ML strategies
                print(f"\n🧠 Applying ML Intelligence Analysis...")
                
                try:
                    # Comprehensive Market Intelligence Analysis
                    market_intel = market_intelligence.get_market_intelligence_summary(df, {
                        'action': signal,
                        'symbol': symbol,
                        'rsi': rsi,
                        'macd': macd,
                        'macd_trend': macd_trend,
                        'volume_ratio': indicators.get('volume_ratio', 1.0),
                        'current_price': current_price,
                        'volatility': volatility
                    })
                    
                    print(f"   📊 Market Regime: {market_intel['market_regime']['primary_regime']} "
                          f"(Confidence: {market_intel['market_regime']['confidence']:.2f})")
                    print(f"   🔍 Pattern Confidence: {market_intel['pattern_analysis']['pattern_confidence']:.2f}")
                    print(f"   📈 Signal Success Probability: {market_intel['signal_probability']:.2f}")
                    print(f"   🎯 Intelligence Score: {market_intel['intelligence_score']:.2f}")
                    
                    # Update adaptive thresholds based on market regime
                    if config.ADAPTIVE_THRESHOLDS_ENABLED:
                        adaptive_thresholds = market_intel['adaptive_thresholds']
                        print(f"   ⚙️ Adaptive RSI Thresholds: {adaptive_thresholds['rsi_oversold']:.1f}/{adaptive_thresholds['rsi_overbought']:.1f}")
                        print(f"   ⚙️ Adaptive MACD Threshold: {adaptive_thresholds['macd_threshold']:.4f}")
                    
                    # ML Pattern Recognition & Market Regime Prediction
                    if config.PATTERN_RECOGNITION_ENABLED:
                        signal_success_probability = ml_predictor.predict_signal_success(
                            {'action': signal, 'rsi': rsi, 'macd': macd}, indicators
                        )
                        print(f"   🤖 ML Signal Success Prediction: {signal_success_probability:.2f}")
                    
                    # Market Regime Prediction
                    if config.REGIME_DETECTION_ENABLED:
                        regime_prediction = ml_predictor.predict_market_regime(df)
                        print(f"   🌍 ML Regime Prediction: {regime_prediction['regime']} "
                              f"(Confidence: {regime_prediction['confidence']:.2f})")
                    
                    intelligence_score = market_intel['intelligence_score']
                    signal_probability = market_intel['signal_probability']
                    
                    # Apply ML-based signal validation (standard thresholds for non-ML strategies)
                    if intelligence_score < config.INTELLIGENCE_CONFIDENCE_THRESHOLD:
                        if signal != "HOLD":
                            print(f"   ⚠️ Low intelligence confidence ({intelligence_score:.2f} < {config.INTELLIGENCE_CONFIDENCE_THRESHOLD}) - Reducing position conviction")
                            # Don't change signal to HOLD, but flag for reduced position sizing
                            indicators['ml_confidence_low'] = True
                    
                    # Apply signal probability threshold (standard conservative thresholds)
                    if signal_probability < 0.4:  # Low success probability
                        print(f"   ⚠️ Low signal success probability ({signal_probability:.2f}) - Consider HOLD")
                        if signal_probability < 0.25:  # Very low probability
                            original_signal = signal
                            signal = "HOLD"
                            reason = f"ML Intelligence: Very low success probability ({signal_probability:.2f}) - Original: {original_signal}"
                            print(f"   🔄 Signal changed from {original_signal} to HOLD due to low ML probability")
                            
                except Exception as intel_error:
                    print(f"   ⚠️ ML Intelligence analysis error: {intel_error}")
                    # Continue with signal if intelligence fails
            
            # Store ML intelligence in bot status (for both strategies)
            if 'market_intel' in locals():
                bot_status['last_ml_intelligence'] = {
                    'market_regime': market_intel['market_regime']['primary_regime'],
                    'regime_confidence': market_intel['market_regime']['confidence'],
                    'pattern_confidence': market_intel['pattern_analysis']['pattern_confidence'],
                    'signal_probability': market_intel['signal_probability'],
                    'intelligence_score': market_intel['intelligence_score'],
                    'adaptive_thresholds': market_intel['adaptive_thresholds'],
                    'market_stress': market_intel['market_stress']['stress_level']
                }
                
                # Update reason with ML insights (for both strategies)
                if signal != "HOLD":
                    ml_insights = []
                    if market_intel['market_regime']['confidence'] > 0.8:
                        ml_insights.append(f"Regime: {market_intel['market_regime']['primary_regime']}")
                    if market_intel['signal_probability'] > 0.7:
                        ml_insights.append(f"High success prob: {market_intel['signal_probability']:.2f}")
                    if market_intel['intelligence_score'] > 0.8:
                        ml_insights.append(f"High intelligence: {market_intel['intelligence_score']:.2f}")
                    
                    if ml_insights:
                        reason += f" | ML: {', '.join(ml_insights)}"
                
                print(f"   ✅ ML Intelligence analysis completed")
        
        # Update bot status with latest signal and timestamp
        current_time = format_cairo_time()
        update_bot_status_common(symbol, signal, current_price, df)
        bot_status.update({'last_strategy': strategy})
            
        print(f"Strategy {strategy} generated final signal: {signal} - {reason}")  # Debug log
        
        # Final Profitability Summary (NEW)
        if signal != "HOLD":
            print(f"\n📊 PROFITABILITY ANALYSIS SUMMARY:")
            print(f"   Signal: {signal} for {symbol}")
            print(f"   Current Price: ${current_price:.4f}")
            
            # Show optimization metrics if available
            if 'last_optimization' in bot_status:
                opt_data = bot_status['last_optimization']
                print(f"   Optimization Confidence: {opt_data.get('confidence', 0):.2f}")
                if 'factors' in opt_data:
                    factors = opt_data['factors']
                    print(f"   - Profitability Score: {factors.get('profitability_score', 0):.2f}/1.0")
                    print(f"   - Timing Score: {factors.get('timing_score', 0):.2f}/1.0")
                    print(f"   - Risk/Reward Ratio: {factors.get('risk_reward_ratio', 0):.2f}")
            
            # Show signal quality if available
            if 'last_signal_quality' in bot_status:
                quality_data = bot_status['last_signal_quality']
                print(f"   Signal Quality: {quality_data.get('quality_score', 0):.2f}/1.0")
                print(f"   Filters Passed: {quality_data.get('filters_passed', 0)}/7")
            
            # Show price position analysis
            try:
                if 'high' in df.columns and 'low' in df.columns:
                    recent_high = df['high'].tail(48).max()
                    recent_low = df['low'].tail(48).min()
                    price_range = recent_high - recent_low
                    
                    if price_range > 0:
                        range_position = (current_price - recent_low) / price_range
                        print(f"   Price Position: {range_position:.1%} in 48h range (${recent_low:.4f} - ${recent_high:.4f})")
                        
                        if signal == "BUY" and range_position < 0.4:
                            print(f"   ✅ GOOD BUY: Buying in lower part of range")
                        elif signal == "BUY" and range_position > 0.6:
                            print(f"   ⚠️ HIGH BUY: Buying in upper part of range")
                        elif signal == "SELL" and range_position > 0.6:
                            print(f"   ✅ GOOD SELL: Selling in upper part of range")
                        elif signal == "SELL" and range_position < 0.4:
                            print(f"   ⚠️ LOW SELL: Selling in lower part of range")
            except Exception as summary_error:
                print(f"   ⚠️ Price analysis error: {summary_error}")
            
            print(f"   Final Reason: {reason}")
            print(f"=" * 60)
    
    except Exception as e:
        error_msg = f"Error in strategy execution: {str(e)}"
        print(f"Error in strategy execution: {str(e)}")  # Debug log
        log_error_to_csv(error_msg, "STRATEGY_ERROR", "signal_generator", "ERROR")
        signal, reason = "HOLD", f"Strategy error: {str(e)}"
    
    # ===== PROFIT VALIDATION FOR SELL SIGNALS BEFORE LOGGING =====
    # Validate SELL signals for profitability before logging them
    if signal == "SELL":
        try:
            # Use the global Supabase-aware position tracker imported at module level
            position_tracker = get_position_tracker()
            
            minimum_profit_pct = config.REBALANCING.get('minimum_profit_pct', 2.0)
            should_sell, profit_reason = position_tracker.should_allow_partial_sell(
                symbol=symbol,
                current_price=current_price,
                minimum_profit_pct=minimum_profit_pct
            )
            
            if not should_sell:
                # Convert unprofitable SELL to HOLD
                print(f"🚫 SELL signal converted to HOLD: {profit_reason}")
                signal = "HOLD"
                reason = f"UNPROFITABLE_SELL_BLOCKED - {profit_reason}"
        except Exception as validation_error:
            # Safety-first: Convert to HOLD on validation error
            print(f"⚠️ Sell validation error, converting to HOLD: {validation_error}")
            signal = "HOLD"
            reason = f"PROFIT_VALIDATION_ERROR - {validation_error}"
    # ===== END PROFIT VALIDATION =====
    
    # Block BUY signals except for BTC and ETH
    try:
        allowed_buy_assets = {"BTC", "ETH"}
        base_asset = symbol.replace("USDT", "") if isinstance(symbol, str) and symbol.endswith("USDT") else (symbol.split("/")[0] if isinstance(symbol, str) and "/" in symbol else symbol)
        if signal == "BUY" and base_asset not in allowed_buy_assets:
            print(f"🚫 BUY blocked for {symbol}: only BTC and ETH allowed")
            reason = f"BUY_BLOCKED - Allowed assets: BTC, ETH"
            log_error_to_csv(reason, "BUY_BLOCKED", "signal_generator", "WARNING")
            signal = "HOLD"
    except Exception as e:
        log_error_to_csv(f"Error in buy-block check: {e}", "INTERNAL_ERROR", "signal_generator", "ERROR")

    # Final signal logging and notifications
    log_signal_to_csv(signal, current_price, indicators, f"Strategy {strategy} - {reason}")
    
    # Send Telegram notification for trading signals (configurable)
    if TELEGRAM_AVAILABLE and signal in ["BUY", "SELL"] and getattr(config, 'TELEGRAM_SEND_SIGNALS', False):
        try:
            notify_signal(signal, symbol, current_price, indicators, reason)
        except Exception as telegram_error:
            print(f"Telegram signal notification failed: {telegram_error}")
    
    return signal

def update_trade_tracking(trade_result, profit_loss=0):
    """Track consecutive wins/losses for smart risk management"""
    try:
        if trade_result == 'success':
            # Only update streaks on realized PnL events
            # Treat profit_loss == 0 (e.g., BUY entries or unknown PnL) as neutral: no change
            if profit_loss is None:
                return
            if profit_loss > 0:
                bot_status['consecutive_losses'] = 0  # Reset on profitable close
                bot_status['consecutive_wins'] = bot_status.get('consecutive_wins', 0) + 1
            elif profit_loss < 0:
                bot_status['consecutive_losses'] = bot_status.get('consecutive_losses', 0) + 1
                bot_status['consecutive_wins'] = 0
            else:
                # Flat/unknown PnL: don't modify counters
                bot_status['consecutive_wins'] = bot_status.get('consecutive_wins', 0)
        else:
            bot_status['consecutive_losses'] = bot_status.get('consecutive_losses', 0) + 1
            bot_status['consecutive_wins'] = 0
            
        # Log if consecutive losses are getting high
        if bot_status['consecutive_losses'] >= 3:
            log_error_to_csv(
                f"Consecutive losses: {bot_status['consecutive_losses']}", 
                "RISK_WARNING", 
                "update_trade_tracking", 
                "WARNING"
            )
    except Exception as e:
        log_error_to_csv(str(e), "TRACKING_ERROR", "update_trade_tracking", "ERROR")

def apply_dynamic_sizing_and_minimum_check(position_result, signal, symbol, current_price, usdt_balance, confidence_score, available_coin_balance=None):
    """Apply DYNAMIC_SIZING configuration and intelligent minimum notional checking"""
    
    # Apply DYNAMIC_SIZING configuration override
    if hasattr(config, 'DYNAMIC_SIZING') and config.DYNAMIC_SIZING.get('base_amount'):
        print(f"\n💰 Applying DYNAMIC_SIZING configuration...")
        
        # For SELL orders, use available coin balance; for BUY orders, use USDT balance
        if signal == "SELL" and available_coin_balance is not None:
            # For SELL orders, calculate dynamic value WITHOUT adaptive balance constraints
            # because we're selling existing holdings, not limited by USDT balance
            base_amount = config.DYNAMIC_SIZING.get('base_amount', 50)
            confidence_multiplier = config.DYNAMIC_SIZING.get('confidence_multiplier', 3.0)
            max_amount = config.DYNAMIC_SIZING.get('max_amount', 200)
            
            # Calculate ideal dynamic position value (no adaptive constraints for SELL)
            dynamic_value = base_amount * (1 + (confidence_score * confidence_multiplier))
            dynamic_value = min(max_amount, dynamic_value)
            
            # Calculate desired quantity based on dynamic value
            desired_quantity = dynamic_value / current_price
            
            # Cap at available coin balance (safety check only)
            dynamic_quantity = min(desired_quantity, available_coin_balance)
            
            print(f"   📊 Dynamic sizing calculation (SELL):")
            print(f"   Confidence score: {confidence_score:.3f}")
            print(f"   Calculated value: ${dynamic_value:.2f} (no USDT balance constraints)")
            print(f"   Desired quantity: {desired_quantity:.8f}")
            print(f"   Available balance: {available_coin_balance:.8f}")
            print(f"   Dynamic quantity (capped): {dynamic_quantity:.8f} (${dynamic_quantity * current_price:.2f})")
            print(f"   Original quantity: {position_result['quantity']:.8f}")
        else:
            # BUY order - use USDT balance with adaptive sizing
            dynamic_value = calculate_dynamic_trade_value(confidence_score, available_usdt=usdt_balance)
            dynamic_quantity = dynamic_value / current_price
            
            print(f"   📊 Dynamic sizing calculation (BUY):")
            print(f"   Confidence score: {confidence_score:.3f}")
            print(f"   Calculated value: ${dynamic_value:.2f} (adaptive to ${usdt_balance:.2f} balance)")
            print(f"   Dynamic quantity: {dynamic_quantity:.8f}")
            print(f"   Original quantity: {position_result['quantity']:.8f}")
        
        # Use the larger of the two calculations
        if dynamic_quantity > position_result['quantity']:
            print(f"   ✅ Using DYNAMIC_SIZING (larger): ${dynamic_quantity * current_price:.2f}")
            position_result['quantity'] = dynamic_quantity
            position_result['risk_amount'] = dynamic_quantity * current_price
            position_result['risk_percentage'] = (dynamic_quantity * current_price / usdt_balance) * 100
            position_result['sizing_method'] = 'DYNAMIC_SIZING'
        else:
            print(f"   📊 Keeping advanced sizing (larger): ${position_result['risk_amount']:.2f}")
            position_result['sizing_method'] = 'ADVANCED_POSITION_MANAGER'
    
    # Simple minimum notional checking - respect only exchange requirements
    qty = position_result['quantity']
    smart_min_trade_value = calculate_smart_minimum_trade_usdt(symbol, current_price, 
                                                             available_usdt=usdt_balance if signal == "BUY" else None)
    
    exchange_min_qty = smart_min_trade_value / current_price
    current_trade_value = qty * current_price
    
    print(f"\n🔍 Exchange Minimum Check:")
    print(f"   Current trade value: ${current_trade_value:.2f}")
    print(f"   Exchange minimum: ${smart_min_trade_value:.2f}")
    
    if qty < exchange_min_qty:
        print(f"   📈 Adjusting to meet exchange minimum: ${smart_min_trade_value:.2f}")
        qty = exchange_min_qty
        position_result['quantity'] = qty
        position_result['risk_amount'] = qty * current_price
        position_result['risk_percentage'] = (qty * current_price / usdt_balance) * 100
    else:
        print(f"   ✅ Position already meets exchange requirements")
    
    return position_result, qty


def execute_trade(signal, symbol="BTCUSDT", qty=None):
    print("\n=== Trade Execution Debug Log ===")
    print(f"Attempting trade: {signal} for {symbol}")
    print(f"Initial quantity: {qty}")
    
    if signal == "HOLD":
        print("Signal is HOLD - no action needed")
        return f"Signal: {signal} - No action taken"
        
    # Get symbol info for precision and filters
    symbol_info = None
    try:
        if client:
            print("Getting exchange info from Binance API...")
            exchange_info = get_exchange_info_cached()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                print(f"Symbol info found for {symbol}:")
                print(f"Base Asset: {symbol_info['baseAsset']}")
                print(f"Quote Asset: {symbol_info['quoteAsset']}")
                print(f"Minimum Lot Size: {next((f['minQty'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), 'unknown')}")
                
                # Get current ticker info
                ticker = client.get_ticker(symbol=symbol)
                print(f"Current {symbol} price: ${float(ticker['lastPrice']):.2f}")
                print(f"24h Volume: {float(ticker['volume']):.2f} {symbol_info['baseAsset']}")
                print(f"24h Price Change: {float(ticker['priceChangePercent']):.2f}%")
            else:
                print(f"Warning: No symbol info found for {symbol}")
        else:
            print("Warning: Client not initialized - running in demo mode")
    except Exception as e:
        log_error_to_csv(str(e), "SYMBOL_INFO_ERROR", "execute_trade", "ERROR")
        print(f"Error getting symbol info: {e}")
        return f"Failed to get symbol info: {e}"

    # Safety: Block BUY execution except for BTC and ETH
    try:
        if signal == "BUY":
            base_asset = None
            if symbol_info and 'baseAsset' in symbol_info:
                base_asset = symbol_info.get('baseAsset')
            else:
                base_asset = symbol.replace('USDT', '') if isinstance(symbol, str) and symbol.endswith('USDT') else (symbol.split('/')[0] if isinstance(symbol, str) and '/' in symbol else symbol)

            if base_asset not in {"BTC", "ETH"}:
                msg = f"BUY blocked by policy for {symbol} (only BTC/ETH allowed)"
                print(f"🚫 {msg}")
                log_error_to_csv(msg, "BUY_BLOCKED", "execute_trade", "WARNING")
                return f"Blocked: {msg}"
    except Exception as e:
        log_error_to_csv(f"Error in buy-block execution check: {e}", "INTERNAL_ERROR", "execute_trade", "ERROR")
    
    # Create trade info structure early to avoid scope issues
    trade_info = {
        'timestamp': format_cairo_time(),
        'signal': signal,
        'symbol': symbol,
        'quantity': 0.001,  # Default minimum, will be updated
        'status': 'initializing',
        'price': 0,
        'value': 0,
        'fee': 0
    }

    # Calculate position size based on available balance and risk management
    try:
        if client:
            print("\n=== Enhanced Position Sizing & Risk Management ===")
            balance = client.get_account()
            
            # More robust balance extraction
            usdt_balance = 0
            btc_balance = 0
            for b in balance['balances']:
                if b['asset'] == 'USDT':
                    usdt_balance = float(b['free'])
                elif b['asset'] == 'BTC':
                    btc_balance = float(b['free'])
            
            print(f"Available USDT balance: {usdt_balance}")
            print(f"Available BTC balance: {btc_balance}")
            
            # Get current market price
            print("\n=== Price Check ===")
            ticker = client.get_ticker(symbol=symbol)
            current_price = float(ticker['lastPrice'])
            print(f"Current {symbol} price: {current_price}")
            print(f"24h price change: {ticker['priceChangePercent']}%")
            
            # Enhanced Position Sizing (if modules available)
            if ENHANCED_MODULES_AVAILABLE:
                try:
                    print("\n🧠 Calculating optimal position size...")
                    
                    # Get current positions for risk assessment
                    current_positions = {}  # This should be populated with actual positions
                    
                    # Prepare market conditions
                    market_conditions = {
                        'regime': bot_status.get('market_regime', 'NORMAL'),
                        'volatility': bot_status.get('volatility_metrics', {}).get('hourly_vol', 0.5),
                        'volume_surge': bot_status.get('volatility_metrics', {}).get('volume_surge', 1.0),
                        'volume_24h': float(ticker['volume']) * current_price,
                        'price_change_24h_pct': abs(float(ticker['priceChangePercent'])),
                        'spread_pct': 0.05  # Default spread
                    }
                    
                    # Get volatility from indicators if available
                    volatility = 0.5
                    if 'volatility' in locals():
                        volatility = locals()['volatility']
                    
                    # Get confidence score from ML optimization (preferred) or signal quality (fallback)
                    confidence_score = 0.5  # Default
                    if 'last_optimization' in bot_status and bot_status['last_optimization'].get('confidence'):
                        confidence_score = bot_status['last_optimization']['confidence']
                        print(f"   🎯 Using ML optimization confidence: {confidence_score:.3f}")
                    elif 'last_signal_quality' in bot_status and bot_status['last_signal_quality'].get('confidence'):
                        confidence_score = bot_status['last_signal_quality']['confidence']
                        print(f"   📊 Using signal quality confidence: {confidence_score:.3f}")
                    else:
                        print(f"   ⚠️ Using default confidence: {confidence_score:.3f}")
                    
                    # For SELL orders, get available coin balance
                    available_coin_balance = None
                    if signal == "SELL":
                        base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol.split('USDT')[0]
                        for b in balance['balances']:
                            if b['asset'] == base_asset:
                                available_coin_balance = float(b['free'])
                                break
                        print(f"   Available {base_asset} balance for SELL: {available_coin_balance}")
                    
                    # Calculate optimal position size
                    position_manager = get_position_manager()
                    position_result = position_manager.calculate_optimal_position_size(
                        symbol=symbol,
                        signal=signal,
                        current_price=current_price,
                        account_balance=usdt_balance,
                        volatility=volatility,
                        confidence_score=confidence_score,
                        market_regime=market_conditions['regime']
                    )
                    
                    # Apply DYNAMIC_SIZING and enhanced minimum checking
                    position_result, qty = apply_dynamic_sizing_and_minimum_check(
                        position_result, signal, symbol, current_price, usdt_balance, confidence_score, available_coin_balance
                    )
                    
                    # Comprehensive Risk Assessment
                    print("\n🛡️ Conducting comprehensive risk assessment...")
                    risk_manager = get_risk_manager()
                    
                    proposed_trade = {
                        'symbol': symbol,
                        'signal': signal,
                        'quantity': position_result['quantity'],
                        'price': current_price
                    }
                    
                    risk_assessment = risk_manager.comprehensive_risk_check(
                        account_balance=usdt_balance,
                        current_positions=current_positions,
                        proposed_trade=proposed_trade,
                        market_conditions=market_conditions
                    )
                    
                    print(f"   Risk Assessment: {'✅ APPROVED' if risk_assessment['approved'] else '❌ BLOCKED'}")
                    print(f"   Risk Score: {risk_assessment['risk_score']:.2f}")
                    print(f"   Warnings: {len(risk_assessment['warnings'])}")
                    print(f"   Blocks: {len(risk_assessment['blocks'])}")
                    
                    # Check if trade is approved
                    if not risk_assessment['approved']:
                        print(f"🚫 Trade blocked by risk management:")
                        for block in risk_assessment['blocks']:
                            print(f"   - {block}")
                        return f"Trade blocked: {'; '.join(risk_assessment['blocks'])}"
                    
                    # Apply risk adjustments
                    if 'position_size_multiplier' in risk_assessment.get('adjustments', {}):
                        multiplier = risk_assessment['adjustments']['position_size_multiplier']
                        position_result['quantity'] *= multiplier
                        print(f"   📉 Position size adjusted by {multiplier:.0%} due to risk factors")
                    
                    # Apply DYNAMIC_SIZING and enhanced minimum checking (AFTER risk adjustments)
                    position_result, qty = apply_dynamic_sizing_and_minimum_check(
                        position_result, signal, symbol, current_price, usdt_balance, confidence_score, available_coin_balance
                    )
                    
                    print(f"\n📊 Enhanced Position Sizing Results:")
                    print(f"   Optimal Quantity: {qty:.8f}")
                    print(f"   Risk Amount: ${position_result['risk_amount']:.2f}")
                    print(f"   Risk Percentage: {position_result['risk_percentage']:.2f}%")
                    print(f"   Sizing Method: {position_result.get('sizing_method', 'STANDARD')}")
                    print(f"   Factors Applied:")
                    for factor, value in position_result.get('factors', {}).items():
                        print(f"     - {factor}: {value:.3f}")
                        
                    # Store enhanced metrics in trade info
                    trade_info['enhanced_metrics'] = {
                        'position_sizing': position_result,
                        'risk_assessment': risk_assessment
                    }
                    
                except Exception as enhanced_error:
                    print(f"   ⚠️ Enhanced positioning error: {enhanced_error}")
                    # Fall back to basic calculation
                    risk_amount = usdt_balance * (config.RISK_PERCENTAGE / 100)
                    qty = risk_amount / current_price
                    print(f"   Falling back to basic position sizing: {qty:.8f}")
            else:
                # Original basic calculation with minimum notional enforcement
                risk_amount = usdt_balance * (config.RISK_PERCENTAGE / 100)
                print(f"Risk amount ({config.RISK_PERCENTAGE}% of balance): {risk_amount} USDT")
                
                # Calculate quantity based on risk amount and current price
                raw_qty = risk_amount / current_price
                print(f"Raw quantity (before adjustments): {raw_qty}")
                
                # CRITICAL FIX: Ensure basic calculation also meets minimum notional
                smart_min_trade_value = calculate_smart_minimum_trade_usdt(symbol, current_price, 
                                                                         available_usdt=usdt_balance if signal == "BUY" else None)
                min_qty_for_smart_minimum = smart_min_trade_value / current_price
                
                if raw_qty < min_qty_for_smart_minimum:
                    print(f"   💡 Basic position size below minimum notional requirement:")
                    print(f"   Raw quantity: {raw_qty:.8f} (${raw_qty * current_price:.2f})")
                    print(f"   Minimum required: {min_qty_for_smart_minimum:.8f} (${smart_min_trade_value:.2f})")
                    
                    if signal == "SELL":
                        # Check if we have enough balance for minimum notional
                        has_balance, available_balance_check, _ = check_coin_balance(symbol)
                        if has_balance and available_balance_check >= min_qty_for_smart_minimum:
                            print(f"   ✅ Adjusting to minimum notional quantity")
                            qty = min_qty_for_smart_minimum
                        else:
                            print(f"   ❌ Insufficient balance for minimum notional - skipping trade")
                            trade_info['status'] = 'insufficient_balance_for_notional'
                            return f"Insufficient balance for minimum order: need ${smart_min_trade_value:.2f}"
                    else:  # BUY order
                        if usdt_balance >= smart_min_trade_value:
                            print(f"   ✅ Adjusting to minimum notional quantity")
                            qty = min_qty_for_smart_minimum
                        else:
                            print(f"   ❌ Insufficient USDT for minimum notional - skipping trade")
                            trade_info['status'] = 'insufficient_usdt_for_notional'
                            return f"Insufficient USDT for minimum order: need ${smart_min_trade_value:.2f}"
                else:
                    qty = raw_qty
                    print(f"   ✅ Basic position size meets minimum notional requirements")
            
            if symbol_info:
                print("\n=== Enhanced Order Validation ===")
                
                # For SELL orders, get available balance first
                available_balance = None
                if signal == "SELL":
                    has_balance, available_balance, balance_msg = check_coin_balance(symbol)
                    if not has_balance:
                        print(f"❌ Insufficient balance for SELL: {balance_msg}")
                        trade_info['status'] = 'insufficient_funds'
                        trade_info['error'] = balance_msg
                        return f"Insufficient balance: {balance_msg}"
                    print(f"💰 Available balance for SELL: {available_balance}")
                
                # Use comprehensive order validation
                try:
                    from order_validator import OrderValidator, log_validation_result
                    validator = OrderValidator(client)
                    
                    validation_result = validator.validate_order(symbol, qty, signal, current_price, available_balance)
                    
                    # Add debugging info
                    print(f"🔍 Validation Debug:")
                    print(f"   Input quantity: {qty:.8f}")
                    print(f"   Current price: ${current_price:.2f}")
                    print(f"   Available balance: {available_balance}")
                    print(f"   Min notional required: ${validation_result.get('min_notional_required', 0):.2f}")
                    print(f"   Validation result: {validation_result['is_valid']}")
                    print(f"   Adjusted quantity: {validation_result['adjusted_quantity']:.8f}")
                    print(f"   Errors: {validation_result.get('errors', [])}")
                    print(f"   Warnings: {validation_result.get('warnings', [])}")
                    
                    if validation_result['is_valid']:
                        qty = validation_result['adjusted_quantity']
                        print(f"✅ Order validation passed")
                        print(f"Final validated quantity: {qty:.8f}")
                        print(f"Estimated trade value: ${qty * current_price:.2f} USDT")
                        
                        if validation_result['warnings']:
                            for warning in validation_result['warnings']:
                                print(f"⚠️ {warning}")
                                
                        # Critical: Double-check notional compliance after all adjustments
                        final_notional = qty * current_price
                        min_notional_required = validation_result.get('min_notional_required', 0)
                        print(f"🔍 Final notional check: ${final_notional:.2f} vs required ${min_notional_required:.2f}")
                        if min_notional_required > 0 and final_notional < min_notional_required:
                            print(f"❌ CRITICAL: Final quantity still fails notional check!")
                            print(f"   Final notional: ${final_notional:.2f} < Required: ${min_notional_required:.2f}")
                            # Force recalculation with minimum valid quantity
                            min_valid_qty = validator.calculate_minimum_valid_quantity(symbol, current_price)
                            print(f"   Calculated minimum valid quantity: {min_valid_qty:.8f}")
                            if signal == "SELL" and available_balance and min_valid_qty > available_balance:
                                print(f"❌ Cannot meet minimum notional: need {min_valid_qty:.8f}, have {available_balance:.8f}")
                                trade_info['status'] = 'insufficient_balance'
                                trade_info['error'] = f"Balance too small for minimum notional: ${available_balance * current_price:.2f} < ${min_notional_required:.2f}"
                                return f"Balance too small for minimum notional: ${available_balance * current_price:.2f} < ${min_notional_required:.2f}"
                            else:
                                print(f"🔧 FORCED adjustment to minimum valid quantity: {min_valid_qty:.8f}")
                                qty = min_valid_qty
                                trade_info['forced_adjustment'] = True
                    else:
                        print(f"❌ Order validation failed:")
                        for error in validation_result['errors']:
                            print(f"   - {error}")
                        
                        # Log validation errors
                        log_validation_result(validation_result, symbol, "execute_trade")
                        
                        # Implement fallback position sizing instead of failing completely
                        print(f"🔄 Attempting fallback position sizing...")
                        
                        try:
                            # Try to calculate minimum valid quantity for this symbol
                            min_valid_qty = validator.calculate_minimum_valid_quantity(symbol, current_price)
                            
                            # Check if we can afford the minimum quantity
                            min_order_value = min_valid_qty * current_price
                            
                            if signal == "BUY" and usdt_balance >= min_order_value:
                                print(f"💡 Using minimum valid quantity fallback: {min_valid_qty:.8f}")
                                print(f"   Order value: ${min_order_value:.2f}")
                                qty = min_valid_qty
                                trade_info['fallback_sizing'] = True
                            elif signal == "SELL" and available_balance and available_balance >= min_valid_qty:
                                # For SELL, use minimum of available balance or min valid quantity
                                fallback_qty = min(available_balance, min_valid_qty)
                                fallback_value = fallback_qty * current_price
                                
                                # Check if fallback meets minimum notional
                                min_notional = validation_result.get('min_notional_required', 10.0)
                                if fallback_value >= min_notional:
                                    print(f"💡 Using fallback quantity for SELL: {fallback_qty:.8f}")
                                    print(f"   Order value: ${fallback_value:.2f}")
                                    qty = fallback_qty
                                    trade_info['fallback_sizing'] = True
                                else:
                                    print(f"❌ Even fallback sizing insufficient: ${fallback_value:.2f} < ${min_notional:.2f}")
                                    trade_info['status'] = 'validation_failed'
                                    trade_info['error'] = f"Balance too small for minimum requirements: ${fallback_value:.2f} < ${min_notional:.2f}"
                                    return f"Balance too small for minimum requirements: ${fallback_value:.2f} < ${min_notional:.2f}"
                            else:
                                # Can't afford even minimum quantity
                                trade_info['status'] = 'validation_failed' 
                                trade_info['error'] = f"Insufficient balance for minimum quantity: ${min_order_value:.2f}"
                                return f"Insufficient balance for minimum quantity: ${min_order_value:.2f}"
                                
                        except Exception as fallback_error:
                            print(f"❌ Fallback position sizing failed: {fallback_error}")
                            trade_info['status'] = 'validation_failed'
                            trade_info['error'] = '; '.join(validation_result['errors'])
                            return f"Order validation failed: {'; '.join(validation_result['errors'])}"
                        
                except ImportError:
                    # Fallback to original validation if order_validator is not available
                    print("⚠️ Enhanced validation not available, using basic validation")
                    
                    # For SELL orders, apply balance limit first
                    if signal == "SELL" and available_balance is not None:
                        if qty > available_balance:
                            print(f"⚠️ Adjusting quantity from {qty} to {available_balance} (available balance)")
                            qty = available_balance
                    
                    # Get symbol filters for comprehensive basic validation
                    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ['MIN_NOTIONAL', 'NOTIONAL']), None)
                    
                    min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
                    print(f"Minimum allowed quantity: {min_qty}")
                    
                    # Check LOT_SIZE requirement
                    if qty < min_qty:
                        print(f"❌ Quantity {qty:.8f} below minimum {min_qty:.8f}")
                        trade_info['status'] = 'insufficient_quantity'
                        trade_info['error'] = f"Quantity {qty:.8f} below minimum {min_qty:.8f}"
                        return f"Quantity too small: {qty:.8f} < {min_qty:.8f}"
                    
                    # Check NOTIONAL requirement (critical missing validation)
                    if min_notional_filter:
                        min_notional = float(min_notional_filter['minNotional'])
                        order_value = qty * current_price
                        print(f"Order value: ${order_value:.4f}, Min notional: ${min_notional:.2f}")
                        
                        if order_value < min_notional:
                            # Calculate minimum quantity needed for notional
                            required_qty = min_notional / current_price
                            
                            # Check if we can afford the minimum notional
                            if signal == "BUY" and min_notional > usdt_balance:
                                print(f"❌ Insufficient USDT for minimum notional: ${min_notional:.2f} > ${usdt_balance:.2f}")
                                trade_info['status'] = 'insufficient_funds'
                                trade_info['error'] = f"Insufficient USDT for minimum notional: ${min_notional:.2f}"
                                return f"Insufficient USDT for minimum notional: ${min_notional:.2f}"
                            elif signal == "SELL" and available_balance is not None:
                                max_possible_value = available_balance * current_price
                                if max_possible_value < min_notional:
                                    print(f"❌ Maximum possible order value ${max_possible_value:.2f} below minimum ${min_notional:.2f}")
                                    trade_info['status'] = 'insufficient_quantity'
                                    trade_info['error'] = f"Balance too small for minimum notional"
                                    return f"Balance too small for minimum notional: ${max_possible_value:.2f} < ${min_notional:.2f}"
                            
                            # Adjust quantity to meet minimum notional
                            qty = required_qty
                            print(f"🔧 Adjusted quantity to meet minimum notional: {qty:.8f}")
                            print(f"   New order value: ${qty * current_price:.2f}")
                    
                    # Apply LOT_SIZE rounding if needed
                    if lot_size_filter:
                        step_size = float(lot_size_filter['stepSize'])
                        # Round down to nearest step size
                        qty = (qty // step_size) * step_size
                        # Ensure we still meet minimum after rounding
                        if qty < min_qty:
                            qty = min_qty
                        print(f"Quantity after LOT_SIZE rounding: {qty:.8f}")
                    
                    # Use smart minimum trade value calculation with adaptive margin for BUY orders
                    if signal == "BUY":
                        smart_min_trade_value = calculate_smart_minimum_trade_usdt(symbol, current_price, available_usdt=usdt_balance)
                        if qty * current_price < smart_min_trade_value:
                            adj_qty = smart_min_trade_value / current_price
                            print(f"🔧 Smart minimum adjustment: ${smart_min_trade_value:.2f} requires {adj_qty:.8f}")
                            qty = adj_qty
                    
                    print(f"Final basic validation quantity: {qty:.8f}")
                    print(f"Final estimated trade value: ${qty * current_price:.2f}")
                    
                    # Round to correct precision
                    step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 0.001
                    precision = len(str(step_size).split('.')[-1])
                    qty = round(qty - (qty % float(step_size)), precision)
                    print(f"Final quantity after rounding (step size {step_size}): {qty}")
                    print(f"Estimated trade value: {qty * current_price} USDT")
    except Exception as e:
        log_error_to_csv(str(e), "POSITION_SIZE_ERROR", "execute_trade", "ERROR")
        qty = 0.001  # Fallback to minimum quantity
    
    # Update trade info with final quantity
    trade_info.update({
        'quantity': qty,
        'status': 'ready_for_execution'
    })
    
    if client is None:
        error_msg = "Trading client not initialized. Cannot execute trade."
        log_error_to_csv(error_msg, "CLIENT_ERROR", "execute_trade", "ERROR")
        return error_msg
    
    try:
        print("\n=== Final Pre-Execution Validation ===")
        
        # Final validation checkpoint to catch any remaining issues
        try:
            from order_validator import validate_order_before_execution
            
            is_valid, adjusted_qty, validation_error = validate_order_before_execution(
                client, symbol, qty, signal
            )
            
            if not is_valid:
                print(f"❌ Final validation failed: {validation_error}")
                trade_info['status'] = 'final_validation_failed'
                trade_info['error'] = validation_error
                log_error_to_csv(f"Final validation failed: {validation_error}", "FINAL_VALIDATION_ERROR", "execute_trade", "ERROR")
                return f"Final validation failed: {validation_error}"
            
            if adjusted_qty != qty:
                print(f"🔧 Final quantity adjustment: {qty:.8f} -> {adjusted_qty:.8f}")
                qty = adjusted_qty
                trade_info['quantity'] = qty
                
            print(f"✅ Final validation passed - Quantity: {qty:.8f}, Value: ${qty * current_price:.2f}")
            
        except ImportError:
            print("⚠️ Final validation module not available, proceeding with current validation")
        
        print("\n=== Trade Execution ===")
        if signal == "BUY":
            print("Processing BUY order...")
            
            # Extract base asset from symbol (e.g., "BTC" from "BTCUSDT")  
            base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol.split(symbol_info['quoteAsset'])[0]
            
            account_info = client.get_account()

            # Debug: Print all balances to see what we're getting
            print("=== Account Balances Debug ===")
            for balance in account_info['balances']:
                if float(balance['free']) > 0 or balance['asset'] == 'USDT':
                    print(f"{balance['asset']}: free={balance['free']}, locked={balance['locked']}")

            # More robust USDT balance extraction
            usdt_balance = None
            for balance in account_info['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = balance
                    break

            if usdt_balance is None:
                print("❌ USDT balance not found in account")
                trade_info['status'] = 'no_usdt_balance'
                bot_status['trading_summary']['failed_trades'] += 1
                log_error_to_csv("USDT balance not found in account", "BALANCE_ERROR", "execute_trade", "ERROR")
                return "USDT balance not found"

            usdt = float(usdt_balance['free'])
            
            # Calculate smart minimum required for this symbol with adaptive margin
            smart_minimum = calculate_smart_minimum_trade_usdt(symbol, available_usdt=usdt)
            
            print(f"USDT available for buy: {usdt}")
            print(f"Smart minimum required: {smart_minimum} USDT")
            print(f"Risk amount would be: {usdt * (config.RISK_PERCENTAGE / 100):.2f} USDT")

            if usdt < smart_minimum:
                print(f"❌ Insufficient USDT balance (minimum {smart_minimum} USDT required)")
                trade_info['status'] = 'insufficient_funds'
                bot_status['trading_summary']['failed_trades'] += 1
                log_error_to_csv(f"Insufficient USDT for buy: {usdt} < {smart_minimum}", "BALANCE_ERROR", "execute_trade", "WARNING")
                return f"Insufficient USDT: {usdt:.2f} < {smart_minimum:.2f}"

            # Get step size for proper quantity formatting
            step_size = None
            if symbol_info:
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_size_filter:
                    step_size = float(lot_size_filter['stepSize'])
            
            # Format quantity properly to avoid scientific notation
            formatted_qty = format_quantity_for_binance(qty, step_size)
            print(f"🚀 Placing market buy order: {formatted_qty} {base_asset} (formatted from {qty})")
            
            order = client.order_market_buy(symbol=symbol, quantity=formatted_qty)
            trade_info['price'] = float(order['fills'][0]['price']) if order['fills'] else 0
            trade_info['value'] = float(order['cummulativeQuoteQty'])
            trade_info['fee'] = sum([float(fill['commission']) for fill in order['fills']])
            trade_info['status'] = 'success'

            # Update trading summary
            bot_status['trading_summary']['total_buy_volume'] += trade_info['value']
            bot_status['trading_summary']['successful_trades'] += 1

        elif signal == "SELL":
            print("Processing SELL order...")

            # Extract base asset from symbol (e.g., "BTC" from "BTCUSDT")
            base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol.split(symbol_info['quoteAsset'])[0]

            # ===== CRITICAL: Check if sell would result in loss using Smart Position Tracker =====
            print("\n=== Profit Validation Check ===")
            try:
                # Use the same position tracker already imported at module level
                position_tracker = get_position_tracker()
                
                # Get current market price for profit calculation
                ticker = client.get_ticker(symbol=symbol)
                current_price = float(ticker['lastPrice'])
                
                # Check if we should allow this partial sell based on profitability
                minimum_profit_pct = config.REBALANCING.get('minimum_profit_pct', 2.0)
                should_sell, reason = position_tracker.should_allow_partial_sell(
                    symbol=symbol,
                    current_price=current_price,
                    minimum_profit_pct=minimum_profit_pct
                )
                
                if not should_sell:
                    print(f"🚫 SELL ORDER BLOCKED: {reason}")
                    trade_info['status'] = 'blocked_unprofitable'
                    trade_info['error'] = reason
                    bot_status['trading_summary']['failed_trades'] += 1
                    
                    # Signal already logged as HOLD in signal_generator
                    return f"❌ Sell order blocked: {reason}"
                
                # If we reach here, the sell is profitable
                print(f"✅ Profit check passed: {reason}")
                
            except Exception as profit_check_error:
                print(f"🚫 CRITICAL: Profit validation error - blocking sell for safety")
                print(f"   Error: {profit_check_error}")
                
                # SAFETY-FIRST: Block the sell on ANY error to prevent potential losses
                trade_info['status'] = 'profit_validation_error'
                trade_info['error'] = str(profit_check_error)
                bot_status['trading_summary']['failed_trades'] += 1
                
                # Signal already logged as HOLD in signal_generator
                return f"❌ Sell blocked - validation error: {profit_check_error}"
            # ===== END PROFIT VALIDATION CHECK =====

            print(f"✅ Proceeding with validated SELL order...")
            print(f"Validated quantity: {qty} {base_asset}")

            # Final quantity check (should be handled by validation, but extra safety)
            if qty <= 0:
                print("❌ Quantity is zero or negative after validation")
                trade_info['status'] = 'insufficient_quantity'
                bot_status['trading_summary']['failed_trades'] += 1
                return "Cannot place SELL order: quantity too small"

            # Get step size for proper quantity formatting
            step_size = None
            if symbol_info:
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_size_filter:
                    step_size = float(lot_size_filter['stepSize'])
            
            # Format quantity properly to avoid scientific notation
            formatted_qty = format_quantity_for_binance(qty, step_size)
            print(f"🚀 Placing market sell order: {formatted_qty} {base_asset} (formatted from {qty})")
            
            order = client.order_market_sell(symbol=symbol, quantity=formatted_qty)
            trade_info['price'] = float(order['fills'][0]['price']) if order['fills'] else 0
            trade_info['value'] = float(order['cummulativeQuoteQty'])
            trade_info['fee'] = sum([float(fill['commission']) for fill in order['fills']])
            trade_info['status'] = 'success'

            print(f"✅ SELL order executed successfully!")
            print(f"   Order ID: {order.get('orderId', 'N/A')}")
            print(f"   Price: ${trade_info['price']:.4f}")
            print(f"   Value: ${trade_info['value']:.2f}")
            print(f"   Fee: ${trade_info['fee']:.4f}")

            # Update trading summary
            bot_status['trading_summary']['total_sell_volume'] += trade_info['value']
            bot_status['trading_summary']['successful_trades'] += 1

            # Calculate revenue (sell value minus rough cost basis)
            revenue = 0
            try:
                if bot_status['trading_summary']['total_buy_volume'] > 0 and qty and qty > 0:
                    # Rough average buy price estimate
                    avg_buy_price = bot_status['trading_summary']['total_buy_volume'] / max(1, (bot_status['trading_summary']['successful_trades'] // 2 or 1))
                    revenue = trade_info['value'] - (qty * avg_buy_price)
            except Exception:
                revenue = 0
            bot_status['trading_summary']['total_revenue'] += revenue
            # Track daily_loss (treat negative revenue as loss)
            bot_status['daily_loss'] = bot_status.get('daily_loss', 0.0) + (-revenue if revenue < 0 else 0.0)

        # Update trade history (keep last 10 trades)
        bot_status['trading_summary']['trades_history'].insert(0, trade_info)
        if len(bot_status['trading_summary']['trades_history']) > 10:
            bot_status['trading_summary']['trades_history'].pop()

        # Log real trade to CSV
        try:
            balance_before = balance_after = 0
            if client:
                account = client.get_account()
                # Calculate TOTAL portfolio value (USDT + all crypto positions)
                total_portfolio_value = 0
                usdt_balance = 0
                
                for balance in account['balances']:
                    asset = balance['asset']
                    free_qty = float(balance['free'])
                    locked_qty = float(balance['locked'])
                    total_qty = free_qty + locked_qty
                    
                    if total_qty > 0:
                        if asset == 'USDT':
                            usdt_balance = total_qty
                            total_portfolio_value += total_qty
                        else:
                            # Get current price for this asset
                            try:
                                ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
                                asset_price = float(ticker['price'])
                                asset_value = total_qty * asset_price
                                total_portfolio_value += asset_value
                                print(f"📊 Portfolio: {total_qty:.8f} {asset} @ ${asset_price:.4f} = ${asset_value:.2f}")
                            except:
                                # If price fetch fails, skip this asset
                                pass
                
                balance_after = total_portfolio_value
                print(f"💰 Total Portfolio Value: ${total_portfolio_value:.2f} (USDT: ${usdt_balance:.2f})")

            # Calculate real profit/loss using portfolio tracker
            try:
                from portfolio_tracker import get_portfolio_tracker
                portfolio_tracker = get_portfolio_tracker()
                
                # Set starting balance if not set (user said they started with $1000)
                if portfolio_tracker.data['starting_balance'] is None:
                    portfolio_tracker.set_starting_balance(1000.0)
                
                # Calculate real P&L based on total portfolio change
                real_pnl = portfolio_tracker.update_balance(balance_after)
                
                print(f"📊 Real P&L: ${real_pnl:.2f} (Portfolio: ${balance_after:.2f} - Starting: $1000.00)")
                
            except Exception as e:
                print(f"⚠️ Portfolio tracking failed: {e}")
                real_pnl = 0

            additional_data = {
                'rsi': bot_status.get('rsi', 50),
                'macd_trend': bot_status.get('macd', {}).get('trend', 'NEUTRAL'),
                'sentiment': bot_status.get('sentiment', 'neutral'),
                'balance_before': balance_before,
                'balance_after': balance_after,
                'profit_loss': real_pnl,  # Use real P&L instead of individual trade revenue
                'order_id': order.get('orderId', '') if 'order' in locals() else ''
            }
            trade_info['order_id'] = additional_data['order_id']
            trade_info['profit_loss'] = additional_data['profit_loss']
            log_trade_to_csv(trade_info, additional_data)

            # Send Telegram notification for successful trades
            if TELEGRAM_AVAILABLE:
                try:
                    notify_trade(trade_info, is_executed=True)
                except Exception as telegram_error:
                    print(f"Telegram trade notification failed: {telegram_error}")

        except Exception as csv_error:
            log_error_to_csv(f"CSV logging error: {csv_error}", "CSV_ERROR", "execute_trade", "WARNING")

        # Update statistics
        total_trades = bot_status['trading_summary']['successful_trades'] + bot_status['trading_summary']['failed_trades']
        bot_status['total_trades'] = total_trades

        if total_trades > 0:
            bot_status['trading_summary']['win_rate'] = (bot_status['trading_summary']['successful_trades'] / total_trades) * 100
            bot_status['trading_summary']['average_trade_size'] = (
                bot_status['trading_summary']['total_buy_volume'] + bot_status['trading_summary']['total_sell_volume']
            ) / total_trades if total_trades > 0 else 0

        # Update smart trade tracking (only pass PnL for realized SELLs)
        realized_pnl = None
        if signal == "SELL":
            realized_pnl = revenue
        update_trade_tracking('success', realized_pnl)

        return f"{signal} order executed: {order['orderId']} at ${trade_info['price']:.2f}"

    except BinanceAPIException as e:
        trade_info['status'] = 'api_error'
        bot_status['trading_summary']['failed_trades'] += 1
        bot_status['trading_summary']['trades_history'].insert(0, trade_info)
        bot_status['errors'].append(str(e))

        # Parse specific API errors for better handling
        error_code = getattr(e, 'code', 0)
        error_msg = str(e)
        
        # Note: Insufficient balance (-2010) is handled upstream in signal generation
        # so if we get here, it's likely a different issue
        
        if error_code == -1013:
            # Enhanced handling for filter failures
            trade_info['status'] = 'filter_failure'
            trade_info['error'] = f"Order filter failure: {error_msg}"
            
            # Log detailed filter failure information
            filter_type = "UNKNOWN"
            if "NOTIONAL" in error_msg:
                filter_type = "NOTIONAL"
            elif "LOT_SIZE" in error_msg:
                filter_type = "LOT_SIZE"
            elif "PRICE_FILTER" in error_msg:
                filter_type = "PRICE_FILTER"
            elif "MIN_NOTIONAL" in error_msg:
                filter_type = "MIN_NOTIONAL"
            
            detailed_msg = f"Filter failure ({filter_type}): {error_msg}"
            print(f"❌ FILTER FAILURE: {detailed_msg}")
            print(f"   Symbol: {symbol}, Quantity: {qty}, Signal: {signal}")
            
            # Try to get symbol info for debugging
            try:
                if client:
                    from order_validator import OrderValidator
                    validator = OrderValidator(client)
                    current_price = client.get_ticker(symbol=symbol)['lastPrice']
                    min_valid_qty = validator.calculate_minimum_valid_quantity(symbol, float(current_price))
                    print(f"   Current price: ${current_price}")
                    print(f"   Minimum valid quantity: {min_valid_qty:.8f}")
                    print(f"   Attempted quantity: {qty:.8f}")
                    print(f"   Order value: ${qty * float(current_price):.2f}")
            except Exception as debug_error:
                print(f"   Debug info failed: {debug_error}")
            
            log_error_to_csv(f"API Filter Error (Code: {error_code}, Type: {filter_type}): {error_msg}", 
                           "FILTER_ERROR", "execute_trade", "ERROR")
        else:
            trade_info['error'] = error_msg
            log_error_to_csv(f"API Error (Code: {error_code}): {error_msg}", "API_ERROR", "execute_trade", "ERROR")

        # Update smart trade tracking for failed trades
        update_trade_tracking('failed', -1)

        # Log failed trade to CSV
        additional_data = {
            'rsi': bot_status.get('rsi', 50),
            'macd_trend': bot_status.get('macd', {}).get('trend', 'NEUTRAL'),
            'sentiment': bot_status.get('sentiment', 'neutral'),
            'balance_before': 0,
            'balance_after': 0,
            'profit_loss': 0,
            'error_code': error_code,
            'error_type': trade_info.get('status', 'api_error')
        }
        log_trade_to_csv(trade_info, additional_data)

        # Send Telegram notification for failed trades
        if TELEGRAM_AVAILABLE:
            try:
                notify_trade(trade_info, is_executed=False)
            except Exception as telegram_error:
                print(f"Telegram failed trade notification failed: {telegram_error}")

        return f"Order failed: {str(e)}"

def scan_trading_pairs(base_assets=None, quote_asset="USDT", min_volume_usdt=1000000):
    """Smart multi-coin scanner for best trading opportunities with rate limiting"""
    opportunities = []
    
    # Default assets if none provided
    if base_assets is None:
        base_assets = ["BTC", "ETH", "BNB", "XRP", "ADA", "AVAX", "LINK"]  # Reduced symbols
    
    # Add rate limiting to prevent API ban
    scan_delay = 0.5  # 500ms delay between API calls
    
    for base in base_assets:
        try:
            symbol = f"{base}{quote_asset}"
            
            # Rate limiting - wait before each API call
            time.sleep(scan_delay)
            
            # Get 24h ticker statistics
            ticker = client.get_ticker(symbol=symbol)
            volume_usdt = float(ticker['quoteVolume'])
            price_change_pct = float(ticker['priceChangePercent'])
            
            # Skip if volume too low
            if volume_usdt < min_volume_usdt:
                continue
            
            # Rate limiting before data fetch
            time.sleep(scan_delay)
            
            # Fetch market data with smaller limit to reduce API weight
            df = fetch_data(symbol=symbol, limit=30)  # Reduced from 50 to 30
            if df is None or len(df) < 15:  # Reduced minimum from 20 to 15
                continue
            
            # Calculate technical indicators with proper error handling
            current_price = float(df['close'].iloc[-1])
            
            # Get RSI - it should already be calculated in fetch_data
            if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
                current_rsi = float(df['rsi'].iloc[-1])
            else:
                # Fallback calculation
                prices = df['close'].values
                current_rsi = calculate_rsi(prices, period=14)
            
            # Get MACD trend - it should already be calculated in fetch_data  
            if 'macd_trend' in df.columns and not pd.isna(df['macd_trend'].iloc[-1]):
                macd_trend = df['macd_trend'].iloc[-1]
            else:
                # Fallback calculation
                prices = df['close'].values
                macd_result = calculate_macd(prices)
                macd_trend = macd_result.get('trend', 'NEUTRAL')
            
            # Get SMA values with error handling
            try:
                sma_fast = calculate_sma(df, period=10)
                sma_slow = calculate_sma(df, period=20)
                
                if len(sma_fast) == 0 or len(sma_slow) == 0:
                    continue  # Skip if we can't calculate SMAs
                    
                sma_fast_value = float(sma_fast.iloc[-1])
                sma_slow_value = float(sma_slow.iloc[-1])
            except Exception as sma_error:
                log_error_to_csv(f"SMA calculation error for {symbol}: {sma_error}", 
                               "SMA_ERROR", "scan_trading_pairs", "WARNING")
                continue
            
            # Score the opportunity (0-100)
            opportunity_score = 0
            signals = []
            
            # Check if we have balance for this coin (for potential sell signals)
            has_balance, available_balance, balance_msg = check_coin_balance(symbol)
            can_sell = has_balance and available_balance > 0
            
            # RSI scoring with balance-aware adjustments
            if current_rsi < 30:  # Oversold - good for buying
                opportunity_score += 30
                signals.append("RSI_OVERSOLD")
            elif current_rsi > 70:  # Overbought - good for selling if we have balance
                if can_sell:
                    opportunity_score += 25  # Higher score if we can actually sell
                    signals.append("RSI_OVERBOUGHT_SELLABLE")
                else:
                    opportunity_score += 5  # Lower score if we can't sell
                    signals.append("RSI_OVERBOUGHT_NO_BALANCE")
            elif 45 <= current_rsi <= 55:  # Neutral zone
                opportunity_score += 10
                signals.append("RSI_NEUTRAL")
            
            # MACD scoring with balance awareness
            if macd_trend == "BULLISH":
                opportunity_score += 20
                signals.append("MACD_BULLISH")
            elif macd_trend == "BEARISH":
                if can_sell:
                    opportunity_score += 15  # Bearish trend good for selling if we have balance
                    signals.append("MACD_BEARISH_SELLABLE")
                else:
                    signals.append("MACD_BEARISH_NO_BALANCE")
            
            # Price momentum scoring
            if abs(price_change_pct) > 5:  # High volatility
                opportunity_score += 15
                signals.append("HIGH_VOLATILITY")
            
            # Volume scoring
            if volume_usdt > min_volume_usdt * 5:  # Very high volume
                opportunity_score += 15
                signals.append("HIGH_VOLUME")
            
            # SMA trend scoring with balance considerations
            if current_price > sma_fast_value > sma_slow_value:
                opportunity_score += 10
                signals.append("UPTREND")
            elif current_price < sma_fast_value < sma_slow_value:
                if can_sell:
                    opportunity_score += 15  # Downtrend good for selling if we have balance
                    signals.append("DOWNTREND_SELLABLE")
                else:
                    opportunity_score += 5  # Lower score if we can't sell
                    signals.append("DOWNTREND_NO_BALANCE")
            
            # Add balance information to the opportunity
            balance_info = {
                'has_balance': can_sell,
                'available_balance': available_balance if has_balance else 0,
                'balance_msg': balance_msg
            }
            
            opportunities.append({
                'symbol': symbol,
                'score': opportunity_score,
                'price': current_price,
                'volume_usdt': volume_usdt,
                'price_change_pct': price_change_pct,
                'rsi': current_rsi,
                'macd_trend': macd_trend,
                'signals': signals,
                'balance_info': balance_info,  # Add balance information
                'data': df  # Include data for immediate analysis if selected
            })
            
        except Exception as e:
            log_error_to_csv(f"Error scanning {base}{quote_asset}: {e}", 
                           "SCAN_ERROR", "scan_trading_pairs", "WARNING")
            continue
    
    # Sort by opportunity score (highest first)
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    # Log top opportunities with balance information
    if opportunities:
        print(f"\n=== Top Trading Opportunities ===")
        for i, opp in enumerate(opportunities[:5]):  # Show top 5
            balance_status = "✅" if opp['balance_info']['has_balance'] else "❌"
            balance_amount = f"{opp['balance_info']['available_balance']:.4f}" if opp['balance_info']['has_balance'] else "0"
            
            print(f"{i+1}. {opp['symbol']}: Score {opp['score']}, RSI {opp['rsi']:.1f}, "
                  f"Change {opp['price_change_pct']:.2f}%, Balance: {balance_status}({balance_amount}), "
                  f"Signals: {', '.join(opp['signals'])}")
    
    return opportunities

def trading_loop():
    """Professional AI Trading Wolf - Intelligent Timing and Opportunity Hunting"""
    bot_status['running'] = True
    bot_status['signal_scanning_active'] = True  # Activate signal scanning
    consecutive_errors = 0
    max_consecutive_errors = 5
    error_sleep_time = 60  # Start with 1 minute on errors
    
    print("\n🐺 === AI TRADING WOLF ACTIVATED ===")
    print("🎯 Professional timing system engaged")
    print("📊 Market regime detection online")
    print("⚡ Breakout opportunity scanning active")
    print("📡 Signal scanning activated")
    print("\n🛡️ === OPTIMIZED RATE LIMITING ACTIVE ===")
    print("⏱️ Global signal cooldown: 20 seconds between ANY signals")
    print("🔒 Symbol signal cooldown: 60 seconds per symbol")
    print("🚫 Signal type cooldown: 120 seconds for same signal type")
    print("📊 Scan cycle limit: 1 signal per scanning cycle")
    print("🕒 BTC fallback cooldown: 60 seconds")
    print("=" * 50)
    
    # Initialize trading summary if not exists
    if 'trading_summary' not in bot_status:
        bot_status['trading_summary'] = {
            'successful_trades': 0,
            'failed_trades': 0,
            'total_trades': 0,
            'total_buy_volume': 0.0,
            'total_sell_volume': 0.0,
            'total_revenue': 0.0,
            'win_rate': 0.0,
            'average_trade_size': 0.0,
            'trades_history': []
        }
    
    # Ensure API client is initialized (should already be done at startup)
    if not bot_status.get('api_connected', False):
        print("⚠️ API client not connected at trading loop start - attempting reconnection...")
        initialize_client()
        if not bot_status.get('api_connected', False):
            log_error_to_csv("API client not initialized before trading loop start", "CLIENT_ERROR", "trading_loop", "ERROR")
            time.sleep(10)  # Wait longer before giving up
            return  # Exit trading loop if can't connect

    # Initialize multi-coin tracking and regime detection
    bot_status['monitored_pairs'] = {}
    bot_status['market_regime'] = 'NORMAL'
    bot_status['hunting_mode'] = False
    bot_status['last_daily_summary'] = None  # Track when we last sent daily summary
    
    # Initial market regime detection and IMMEDIATE first scan
    initial_regime = detect_market_regime()
    # Mark last regime check now to avoid immediate duplicate checks
    bot_status['last_volatility_check'] = get_cairo_time()
    initial_interval, initial_mode = calculate_smart_interval()
    
    print(f"🎯 Initial scan mode: {initial_mode} ({initial_interval}s)")
    print(f"🚀 Performing immediate startup scan...")
    
    # Perform immediate first scan
    try:
        print(f"\n🐺 === WOLF SCANNING ACTIVATED (STARTUP) ===")
        print(f"🕒 Time: {format_cairo_time()}")
        print(f"🎯 Scan Reason: STARTUP_SCAN")
        print(f"📊 Market Regime: {bot_status.get('market_regime', 'NORMAL')}")
        
        # Scan all trading pairs immediately (restored to original scan)
        scan_results = scan_trading_pairs()  # Uses default 10 symbols
        bot_status['last_scan_time'] = get_cairo_time()  # Record scan time
        print(f"✅ Startup scan completed - found {len(scan_results) if scan_results else 0} opportunities")
        
        # Generate signals for top opportunities at startup
        if scan_results:
            print("\n🎯 Analyzing top opportunities from startup scan...")
            max_startup_targets = 3  # Analyze top 3 opportunities
            signals_generated = 0
            best_trade_executed = False
            
            for i, opportunity in enumerate(scan_results[:max_startup_targets]):
                current_symbol = opportunity['symbol']
                current_score = opportunity.get('score', 0)
                
                print(f"\n📊 Analyzing startup target {i+1}: {current_symbol} (Score: {current_score:.1f})")
                
                # Get fresh data for signal generation
                df = fetch_data(symbol=current_symbol, interval="5m", limit=100)
                if df is not None:
                    signal = signal_generator(df, current_symbol)
                    current_price = float(df['close'].iloc[-1])
                    
                    print(f"💡 Generated signal: {signal} @ ${current_price:.4f}")
                    
                    # Log signal to CSV with startup context
                    indicators = {
                        'symbol': current_symbol,
                        'rsi': float(df['rsi'].iloc[-1]),
                        'macd': float(df['macd'].iloc[-1]),
                        'macd_trend': df['macd_trend'].iloc[-1],
                        'opportunity_score': current_score
                    }
                    log_signal_to_csv(signal, current_price, indicators, "STARTUP_SCAN")
                    
                    # Update pair tracking
                    if current_symbol not in bot_status['monitored_pairs']:
                        update_monitored_pair(current_symbol, signal, current_price, df, current_score)
                    
                    # Update main status with best target
                    if i == 0:
                        update_bot_status_common(current_symbol, signal, current_price, df, current_score)
                    
                    signals_generated += 1
                    
                    # ===== EXECUTE TRADE FOR BEST OPPORTUNITY =====
                    # Execute trade only for the best opportunity (first one) with actionable signal
                    if i == 0 and not best_trade_executed and signal in ['BUY', 'SELL']:
                        print(f"\n🎯 STARTUP TRADE: Executing best opportunity from startup scan")
                        print(f"   Symbol: {current_symbol}")
                        print(f"   Signal: {signal}")
                        print(f"   Score: {current_score:.1f}")
                        print(f"   Price: ${current_price:.4f}")
                        
                        try:
                            # Execute the trade
                            trade_result = execute_trade(signal, current_symbol)
                            
                            if trade_result and "executed" in trade_result.lower():
                                print(f"✅ STARTUP TRADE EXECUTED: {trade_result}")
                                best_trade_executed = True
                                
                                # Send Telegram notification for startup trade
                                if TELEGRAM_AVAILABLE:
                                    try:
                                        notify_trade({
                                            'signal': signal,
                                            'symbol': current_symbol,
                                            'price': current_price,
                                            'timestamp': format_cairo_time(),
                                            'status': 'success',
                                            'context': 'STARTUP_SCAN'
                                        }, is_executed=True)
                                    except Exception as telegram_error:
                                        print(f"⚠️ Telegram notification failed: {telegram_error}")
                            else:
                                print(f"⚠️ STARTUP TRADE RESULT: {trade_result}")
                                
                        except Exception as trade_error:
                            print(f"❌ STARTUP TRADE FAILED: {trade_error}")
                            log_error_to_csv(
                                f"Startup trade failed for {current_symbol}: {trade_error}",
                                "STARTUP_TRADE_ERROR",
                                "start_trading_bot",
                                "ERROR"
                            )
                    # ===== END STARTUP TRADE EXECUTION =====
            
            print(f"\n✨ Startup signal generation complete - Analyzed {signals_generated} opportunities")
            if best_trade_executed:
                print(f"🎯 Best opportunity trade executed successfully")
        
    except Exception as e:
        print(f"⚠️ Startup scan failed: {e}")
    
    # Set next scan time after immediate scan
    bot_status['next_signal_time'] = get_cairo_time() + timedelta(seconds=initial_interval)
    bot_status['signal_interval'] = initial_interval
    print(f"📅 Next scan: {format_cairo_time(bot_status['next_signal_time'])}")
    
    last_major_scan = get_cairo_time()
    quick_scan_count = 0
    
    while bot_status['running']:
        try:
            current_time = get_cairo_time()

            # Safety: decay consecutive losses after cooldown period (e.g., 2 hours without trades)
            try:
                last_trade_time = None
                if bot_status.get('trading_summary', {}).get('trades_history'):
                    last_trade = bot_status['trading_summary']['trades_history'][0]
                    last_trade_time = last_trade.get('timestamp')
                if last_trade_time:
                    # Parse time if string
                    if isinstance(last_trade_time, str):
                        try:
                            last_trade_dt = datetime.fromisoformat(last_trade_time.replace('Z', '+00:00'))
                        except Exception:
                            last_trade_dt = current_time
                    else:
                        last_trade_dt = last_trade_time
                    if (current_time - last_trade_dt).total_seconds() > 2 * 3600:
                        # Gradually reduce penalties
                        bot_status['consecutive_losses'] = max(0, bot_status.get('consecutive_losses', 0) - 1)
                # Hard cap to avoid indefinite lockout
                bot_status['consecutive_losses'] = min(bot_status.get('consecutive_losses', 0), config.MAX_CONSECUTIVE_LOSSES)
            except Exception:
                pass
            
            # Health check - only reinitialize if connection is actually lost
            if not bot_status['api_connected']:
                print("🔄 API connection lost - attempting to reconnect...")
                initialize_client()
                if not bot_status['api_connected']:
                    print("❌ Failed to reconnect to API - retrying in next cycle")
                    time.sleep(30)  # Wait before retrying
                    continue
            
            # Intelligent scan decision
            should_scan, scan_reason = should_scan_now()
            
            if not should_scan:
                # Sleep in short bursts to allow for interruptions
                time.sleep(min(30, bot_status.get('signal_interval', 300) // 10))
                continue
                
            print(f"\n🐺 === WOLF SCANNING ACTIVATED ===")
            print(f"🕒 Time: {format_cairo_time()}")
            print(f"🎯 Scan Reason: {scan_reason}")
            print(f"📊 Market Regime: {bot_status.get('market_regime', 'NORMAL')}")
            print(f"⚡ Hunting Mode: {'ON' if bot_status.get('hunting_mode') else 'OFF'}")
            
            # Update market regime every major scan
            if (current_time - last_major_scan).total_seconds() > 1800:  # Every 30 minutes
                detect_market_regime()
                last_major_scan = current_time
                quick_scan_count = 0
                
            # Quick breakout scan if in hunting mode
            breakout_opportunities = []
            if bot_status.get('hunting_mode') or bot_status.get('market_regime') in ['VOLATILE', 'EXTREME']:
                breakout_opportunities = detect_breakout_opportunities()
                quick_scan_count += 1
                
                if breakout_opportunities:
                    print(f"🚀 BREAKOUT OPPORTUNITIES DETECTED:")
                    for opp in breakout_opportunities[:2]:
                        print(f"   💎 {opp['symbol']}: Score {opp['score']}, Signals: {', '.join(opp['signals'])}")
            
            # Full market scan (intelligent frequency)
            should_full_scan = (
                not breakout_opportunities or  # No breakouts found
                quick_scan_count >= 5 or      # Max quick scans reached
                (current_time - last_major_scan).total_seconds() > 3600  # Force every hour
            )
            
            if should_full_scan:
                print("🔍 Performing FULL MARKET SCAN")
                opportunities = scan_trading_pairs(
                    base_assets=["BTC", "ETH", "BNB", "XRP", "ADA", "AVAX", "LINK"],
                    quote_asset="USDT",
                    min_volume_usdt=500000  # Lower threshold for more opportunities
                )
                bot_status['last_scan_time'] = get_cairo_time()  # Record full scan time
                quick_scan_count = 0
            else:
                print("⚡ Using BREAKOUT SCAN results")
                opportunities = breakout_opportunities
                bot_status['last_scan_time'] = get_cairo_time()  # Record breakout scan time
            
            # Process opportunities
            if not opportunities:
                print("😴 No significant opportunities found - Wolf resting")
                
                # Fallback to default pair (only if not already processed)
                current_symbol = "BTCUSDT"
                
                # Check if BTCUSDT was already processed in recent scan (within last 60 seconds)
                last_btc_scan = bot_status.get('last_btc_scan_time')
                current_time = get_cairo_time()
                
                if (last_btc_scan is None or 
                    (current_time - last_btc_scan).total_seconds() > 60):
                    
                    df = fetch_data(symbol=current_symbol, interval="5m", limit=100)
                    if df is not None:
                        signal = signal_generator(df, current_symbol)
                        current_price = float(df['close'].iloc[-1])
                        
                        update_bot_status_common(current_symbol, signal, current_price, df)
                        bot_status.update({'last_btc_scan_time': current_time})
                        print(f"📊 Default analysis: {signal} for {current_symbol}")
                else:
                    print(f"⚠️ Skipping default {current_symbol} scan - analyzed recently")
            else:
                print(f"🎯 Found {len(opportunities)} hunting targets")
                
                # Process top opportunities with intelligent prioritization
                max_targets = 1  # Limit to 1 target per cycle to prevent signal flooding
                processed_symbols = set()  # Track processed symbols to avoid duplicates
                signals_generated_this_cycle = 0  # Track signals in this cycle
                
                for i, opportunity in enumerate(opportunities[:max_targets]):
                    current_symbol = opportunity['symbol']
                    current_score = opportunity.get('score', 0)
                    
                    # Skip if we've already processed this symbol in this cycle
                    if current_symbol in processed_symbols:
                        print(f"⚠️ Skipping {current_symbol} - already processed in this cycle")
                        continue
                    processed_symbols.add(current_symbol)
                    
                    # Limit signals per cycle
                    if signals_generated_this_cycle >= 1:
                        print(f"🛑 Signal limit reached for this cycle - skipping remaining opportunities")
                        break
                    
                    print(f"\n🎯 === TARGET {i+1}: {current_symbol} ===")
                    print(f"💪 Score: {current_score:.1f}")
                    
                    # Get fresh data for analysis
                    interval = "1m" if bot_status.get('hunting_mode') else "5m"
                    df = fetch_data(symbol=current_symbol, interval=interval, limit=100)
                    
                    if df is None:
                        continue
                        
                    # Enhanced signal generation with market regime consideration
                    signal = signal_generator(df, current_symbol)
                    signals_generated_this_cycle += 1  # Track signals generated in this cycle
                    current_price = float(df['close'].iloc[-1])
                    
                    print(f"🚦 Signal: {signal} (#{signals_generated_this_cycle} this cycle)")
                    print(f"💰 Price: ${current_price:.4f}")
                    
                    if 'rsi' in opportunity:
                        print(f"📈 RSI: {opportunity['rsi']:.1f}")
                    if 'signals' in opportunity:
                        print(f"⚡ Triggers: {', '.join(opportunity['signals'])}")
                    
                    # Update pair tracking
                    if current_symbol not in bot_status['monitored_pairs']:
                        bot_status['monitored_pairs'][current_symbol] = {
                            'last_signal': 'HOLD',
                            'last_price': 0,
                            'rsi': 50,
                            'macd': {'macd': 0, 'signal': 0, 'trend': 'NEUTRAL'},
                            'sentiment': 'neutral',
                            'total_trades': 0,
                            'successful_trades': 0,
                            'last_trade_time': None
                        }
                    
                    update_monitored_pair(current_symbol, signal, current_price, df, current_score)
                    
                    # Update main status with best target
                    if i == 0:
                        update_bot_status_common(current_symbol, signal, current_price, df, current_score)
                    
                    # Execute trade with enhanced conditions
                    if signal in ["BUY", "SELL"]:
                        # Initialize risk tracking if not present
                        if 'consecutive_losses' not in bot_status:
                            bot_status['consecutive_losses'] = 0
                        if 'daily_loss' not in bot_status:
                            bot_status['daily_loss'] = 0.0
                        
                        # Risk management checks with debug logging
                        consecutive_losses = bot_status.get('consecutive_losses', 0)
                        daily_loss = bot_status.get('daily_loss', 0.0)
                        
                        print(f"🔍 Risk Management Check:")
                        print(f"   Consecutive losses: {consecutive_losses}/{config.MAX_CONSECUTIVE_LOSSES}")
                        print(f"   Daily loss: ${daily_loss:.2f}/${config.MAX_DAILY_LOSS}")
                        print(f"   API Connected: {bot_status.get('api_connected', False)}")
                        print(f"   Can Trade (Account): {bot_status.get('can_trade', False)}")
                        
                        can_trade = (
                            consecutive_losses < config.MAX_CONSECUTIVE_LOSSES and
                            daily_loss < config.MAX_DAILY_LOSS and
                            bot_status.get('api_connected', False) and
                            bot_status.get('can_trade', False)
                        )
                        
                        # Additional hunting mode conditions
                        if bot_status.get('hunting_mode'):
                            can_trade = can_trade and current_score >= 50  # Higher threshold in hunting mode
                            print(f"   Hunting mode score: {current_score}/50")
                        
                        if can_trade:
                            print(f"🚀 EXECUTING {signal} for {current_symbol}")
                            result = execute_trade(signal, current_symbol)
                            print(f"📊 Result: {result}")
                            
                            # Ensure monitored_pairs structure exists and is properly initialized
                            if current_symbol not in bot_status['monitored_pairs']:
                                bot_status['monitored_pairs'][current_symbol] = {
                                    'last_signal': 'HOLD',
                                    'last_price': 0,
                                    'rsi': 50,
                                    'macd': {'macd': 0, 'signal': 0, 'trend': 'NEUTRAL'},
                                    'sentiment': 'neutral',
                                    'total_trades': 0,
                                    'successful_trades': 0,
                                    'last_trade_time': None
                                }
                            
                            # Ensure all required keys exist (defensive programming)
                            required_keys = {
                                'total_trades': 0,
                                'successful_trades': 0,
                                'last_trade_time': None
                            }
                            
                            for key, default_value in required_keys.items():
                                if key not in bot_status['monitored_pairs'][current_symbol]:
                                    bot_status['monitored_pairs'][current_symbol][key] = default_value
                            
                            # Update tracking safely
                            try:
                                bot_status['monitored_pairs'][current_symbol]['total_trades'] += 1
                                if "executed" in str(result).lower():
                                    bot_status['monitored_pairs'][current_symbol]['successful_trades'] += 1
                                    bot_status['monitored_pairs'][current_symbol]['last_trade_time'] = get_cairo_time()
                            except (KeyError, TypeError) as ke:
                                print(f"⚠️ Monitored pairs tracking error: {ke}")
                                print(f"   Current symbol: {current_symbol}")
                                print(f"   Available keys: {list(bot_status['monitored_pairs'].get(current_symbol, {}).keys())}")
                                
                                # Force complete re-initialization
                                bot_status['monitored_pairs'][current_symbol] = {
                                    'last_signal': signal,
                                    'last_price': current_price,
                                    'rsi': 50,
                                    'macd': {'macd': 0, 'signal': 0, 'trend': 'NEUTRAL'},
                                    'sentiment': 'neutral',
                                    'total_trades': 1,
                                    'successful_trades': 1 if "executed" in str(result).lower() else 0,
                                    'last_trade_time': get_cairo_time() if "executed" in str(result).lower() else None
                                }
                            
                            # In hunting mode, only take the best trade
                            if bot_status.get('hunting_mode'):
                                break
                        else:
                            print(f"🛑 Trade blocked by risk management")
                            print(f"   Consecutive losses: {consecutive_losses}/{config.MAX_CONSECUTIVE_LOSSES}")
                            print(f"   Daily loss: ${daily_loss:.2f}/${config.MAX_DAILY_LOSS}")
                            print(f"   API Connected: {bot_status.get('api_connected', False)}")
                            print(f"   Account Can Trade: {bot_status.get('can_trade', False)}")
                            if bot_status.get('hunting_mode'):
                                print(f"   Hunting mode score: {current_score}/50")
            
            consecutive_errors = 0  # Reset error counter on successful cycle
            
            # Calculate next scan time with intelligent timing
            next_interval, next_mode = calculate_smart_interval()
            bot_status['next_signal_time'] = get_cairo_time() + timedelta(seconds=next_interval)
            bot_status['signal_interval'] = next_interval
            
            print(f"\n🎯 Next scan: {next_mode} mode in {next_interval}s ({next_interval/60:.1f}min)")
            print(f"📅 Expected at: {format_cairo_time(bot_status['next_signal_time'])}")
            
            # Send periodic Telegram market updates (every hour)
            if TELEGRAM_AVAILABLE and (current_time - last_major_scan).total_seconds() > 3600:
                try:
                    volatility_metrics = bot_status.get('volatility_metrics', {})
                    next_scan_str = format_cairo_time(bot_status['next_signal_time'])
                    notify_market_update(
                        bot_status.get('market_regime', 'NORMAL'),
                        bot_status.get('hunting_mode', False),
                        next_scan_str,
                        volatility_metrics
                    )
                except Exception as telegram_error:
                    print(f"Telegram market update failed: {telegram_error}")
            
            # Send daily summary each day at 08:00 Cairo time (once per day)
            if config.TELEGRAM.get('notifications', {}).get('daily_summary', True):
                try:
                    last_summary_date = bot_status.get('last_daily_summary')
                    current_date = current_time.strftime('%Y-%m-%d')
                    # Trigger within the first 15 minutes after 08:00 to allow for scheduling jitter
                    if (current_time.hour == 8 and current_time.minute < 15 and
                        (last_summary_date != current_date)):
                        # Send Telegram daily summary if Telegram is available
                        if TELEGRAM_AVAILABLE:
                            notify_daily_summary(bot_status.get('trading_summary', {}))
                        bot_status['last_daily_summary'] = current_date
                        print(f"📊 Daily summary processed for {current_date} (08:00 Cairo)")
                except Exception as telegram_error:
                    print(f"Daily summary notification failed: {telegram_error}")
            
            # Process any queued Telegram messages
            if TELEGRAM_AVAILABLE:
                try:
                    process_queued_notifications()
                except Exception as telegram_error:
                    print(f"Telegram queue processing failed: {telegram_error}")
            
            # === AUTOMATIC POSITION REBALANCING ===
            try:
                # Check if it's time for automatic rebalancing
                if config.REBALANCING.get('enabled', True):
                    rebalance_interval = config.REBALANCING.get('check_interval', 3600)  # Default 1 hour
                    last_rebalance = bot_status.get('last_rebalance_check', 0)
                    
                    # Convert to timestamp if it's a datetime object
                    if isinstance(last_rebalance, datetime):
                        last_rebalance = last_rebalance.timestamp()
                    
                    current_timestamp = current_time.timestamp()
                    
                    if current_timestamp - last_rebalance >= rebalance_interval:
                        print(f"\n🔄 Automatic rebalancing check (every {rebalance_interval//60} minutes)")
                        
                        # Execute rebalancing
                        rebalance_results = execute_position_rebalancing()
                        
                        # Update last rebalance time
                        bot_status['last_rebalance_check'] = current_timestamp
                        bot_status['last_rebalance_results'] = rebalance_results
                        
                        # Log summary
                        actions_count = len(rebalance_results.get('partial_sells', [])) + len(rebalance_results.get('dust_liquidations', []))
                        if actions_count > 0:
                            print(f"✅ Rebalancing completed: {actions_count} actions, ${rebalance_results.get('total_freed_usdt', 0):.2f} USDT freed")
                        else:
                            print("ℹ️ No rebalancing actions needed")
                            
            except Exception as rebalance_error:
                print(f"⚠️ Automatic rebalancing error: {rebalance_error}")
                log_error_to_csv(str(rebalance_error), "AUTO_REBALANCE_ERROR", "trading_loop", "WARNING")
            
            # Smart sleep with early wake capabilities
            sleep_chunks = max(1, next_interval // 30)  # Wake up periodically
            chunk_size = next_interval / sleep_chunks
            
            for _ in range(int(sleep_chunks)):
                if not bot_status['running']:
                    break
                time.sleep(chunk_size)
        
        except KeyboardInterrupt:
            print("\n🛑 === KEYBOARD INTERRUPT ===")
            bot_status['running'] = False
            break
            
        except Exception as e:
            consecutive_errors += 1
            error_msg = f"Trading wolf error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
            print(f"⚠️ {error_msg}")
            
            # Log error to CSV
            log_error_to_csv(str(e), "TRADING_LOOP_ERROR", "trading_loop", "ERROR")
            
            # Update bot status
            bot_status['errors'].append(error_msg)
            bot_status['last_error'] = error_msg
            bot_status['last_update'] = format_cairo_time()
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"💀 Maximum errors reached ({max_consecutive_errors}). Wolf hibernating.")
                bot_status['running'] = False
                bot_status['status'] = 'stopped_due_to_errors'
                break
            
            # Smart error recovery with exponential backoff
            sleep_time = min(error_sleep_time * (2 ** (consecutive_errors - 1)), 300)  # Max 5 minutes
            print(f"😴 Wolf resting for {sleep_time} seconds before retry...")
            time.sleep(sleep_time)
    
    print("\n🐺 === AI TRADING WOLF DEACTIVATED ===")
    bot_status['running'] = False
    bot_status['status'] = 'stopped'

def smart_portfolio_manager():
    """Advanced portfolio management with dynamic risk allocation"""
    try:
        if not client:
            return {"error": "API not connected"}
        
        account = client.get_account()
        balances = {b['asset']: float(b['free']) for b in account['balances'] if float(b['free']) > 0}
        
        # Calculate total portfolio value in USDT
        total_usdt_value = balances.get('USDT', 0)
        for asset, amount in balances.items():
            if asset != 'USDT' and amount > 0:
                try:
                    ticker = client.get_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['price'])
                    total_usdt_value += amount * price
                except:
                    continue
        
        # Smart position sizing based on portfolio value and risk
        max_position_size = total_usdt_value * (config.RISK_PERCENTAGE / 100)
        
        # Adjust for volatility and consecutive losses
        volatility_adjustment = 1.0
        loss_adjustment = 1.0
        
        consecutive_losses = bot_status.get('consecutive_losses', 0)
        if consecutive_losses > 0:
            loss_adjustment = max(0.1, 1.0 - (consecutive_losses * 0.2))  # Reduce size by 20% per loss
        
        adjusted_position_size = max_position_size * volatility_adjustment * loss_adjustment
        
        portfolio_info = {
            'total_value_usdt': total_usdt_value,
            'max_position_size': max_position_size,
            'adjusted_position_size': adjusted_position_size,
            'risk_percentage': config.RISK_PERCENTAGE,
            'consecutive_losses': consecutive_losses,
            'loss_adjustment': loss_adjustment,
            'balances': balances,
            'portfolio_allocation': {}
        }
        
        # Calculate portfolio allocation percentages
        for asset, amount in balances.items():
            if asset == 'USDT':
                portfolio_info['portfolio_allocation'][asset] = (amount / total_usdt_value) * 100
            else:
                try:
                    ticker = client.get_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['price'])
                    asset_value = amount * price
                    portfolio_info['portfolio_allocation'][asset] = (asset_value / total_usdt_value) * 100
                except:
                    portfolio_info['portfolio_allocation'][asset] = 0
        
        return portfolio_info
        
    except Exception as e:
        return {"error": f"Portfolio management error: {e}"}

# Flask Routes and Dashboard Functions
def stop_trading_bot():
    """Stop the trading bot"""
    bot_status['running'] = False
    bot_status['signal_scanning_active'] = False  # Deactivate signal scanning
    bot_status['next_signal_time'] = None  # Clear next signal time when stopped
    
    # Send Telegram notification for bot stop
    if TELEGRAM_AVAILABLE:
        try:
            notify_bot_status("STOPPED", "Manually stopped by user")
        except Exception as telegram_error:
            print(f"Telegram bot stop notification failed: {telegram_error}")

def start_trading_bot():
    """Start the trading bot in a separate thread"""
    try:
        if bot_status.get('running', False):
            print("⚠️ Trading bot is already running")
            return
            
        # Check if there's already a trading thread running
        import threading
        for thread in threading.enumerate():
            if thread.name == 'trading_loop_thread' and thread.is_alive():
                print("⚠️ Trading thread already exists and is running")
                bot_status['running'] = True  # Ensure status is consistent
                return
            
        # Only initialize client if not already connected
        if not bot_status.get('api_connected', False):
            print("🔧 Initializing API client...")
            if not initialize_client():
                print("❌ Failed to initialize API client; bot not started")
                log_error_to_csv("Failed to initialize API client on start", "CLIENT_ERROR", "start_trading_bot", "ERROR")
                return
        
        # Start trading loop in background thread with a unique name
        trading_thread = threading.Thread(target=trading_loop, daemon=True, name='trading_loop_thread')
        trading_thread.start()
        bot_status['running'] = True
        bot_status['status'] = 'running'
        print("✅ Trading bot started successfully")
        
        # Send Telegram notification for bot start
        if TELEGRAM_AVAILABLE:
            try:
                current_strategy = bot_status.get('trading_strategy', 'ADAPTIVE')
                notify_bot_status("STARTED", f"Strategy: {current_strategy}")
            except Exception as telegram_error:
                print(f"Telegram bot start notification failed: {telegram_error}")
                
    except Exception as e:
        print(f"❌ Failed to start trading bot: {e}")
        log_error_to_csv(str(e), "START_ERROR", "start_trading_bot", "ERROR")

@app.route('/download_logs')
def download_logs():
    """Create a zip file containing all CSV log files and send it to the user"""
    try:
        # Ensure CSV files are set up
        csv_files = setup_csv_logging()
        
        # Create an in-memory zip file
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Get logs directory
            logs_dir = Path('logs')
            if not logs_dir.exists():
                return jsonify({'error': 'No log files found'}), 404
            
            files_added = 0
            
            # First, add the main CSV files (trade_history, signal_history, error_log)
            main_csv_files = ['trade_history.csv', 'signal_history.csv', 'error_log.csv']
            for csv_filename in main_csv_files:
                csv_file = logs_dir / csv_filename
                if csv_file.exists() and csv_file.stat().st_size > 0:
                    zf.write(csv_file, csv_filename)
                    files_added += 1
                    print(f"Added {csv_filename} to zip")
                else:
                    # Create empty file with headers if it doesn't exist
                    print(f"Creating empty {csv_filename} with headers")
                    zf.writestr(csv_filename, "")
            
            # Then add any other CSV files in the logs directory
            for csv_file in logs_dir.glob('*.csv'):
                if csv_file.name not in main_csv_files and csv_file.exists():
                    # Add file to zip with relative path
                    zf.write(csv_file, csv_file.name)
                    files_added += 1
                    print(f"Added additional CSV: {csv_file.name}")
            
            # Add JSON files (like positions.json)
            for json_file in logs_dir.glob('*.json'):
                if json_file.exists():
                    zf.write(json_file, json_file.name)
                    files_added += 1
                    print(f"Added JSON file: {json_file.name}")
            
            # Add log files (.log extension)
            for log_file in logs_dir.glob('*.log'):
                if log_file.exists():
                    zf.write(log_file, log_file.name)
                    files_added += 1
                    print(f"Added log file: {log_file.name}")
            
            print(f"Total files added to zip: {files_added}")
        
        # Prepare the zip file for sending
        memory_file.seek(0)
        
        if files_added == 0:
            return jsonify({'error': 'No log files available for download'}), 404
            
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'cryptix_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
    except Exception as e:
        print(f"Error creating log zip file: {e}")
        return jsonify({'error': f'Failed to create zip file: {str(e)}'}), 500

@app.route('/')
def home():
    # Get current strategy for display
    current_strategy = bot_status.get('trading_strategy', config.DEFAULT_STRATEGY)
    strategy_descriptions = {
        'ADAPTIVE': '🧠 Smart strategy that adapts to market conditions',
        'ML_PURE': '🤖 Pure ML Strategy'
    }
    strategy_desc = strategy_descriptions.get(current_strategy, '')
    
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐺 CRYPTIX AI Trading Wolf</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 15px;
            color: #333;
        }
        
        .container {
            max-width: 420px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
            overflow: hidden;
            backdrop-filter: blur(20px);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 20px;
            text-align: center;
            position: relative;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .header-content {
            position: relative;
            z-index: 1;
        }
        
        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .subtitle {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 500;
        }
        
        .main-content {
            padding: 25px 20px;
        }
        
        /* Status Cards */
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 25px;
        }
        
        .status-card {
            padding: 16px;
            border-radius: 16px;
            text-align: center;
            font-weight: 600;
            font-size: 0.85rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }
        
        .status-card:hover {
            transform: translateY(-2px);
        }
        
        .status-running {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-stopped {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status-connected {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        .status-disconnected {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .status-label {
            display: block;
            font-size: 0.75rem;
            opacity: 0.8;
            margin-bottom: 4px;
        }
        
        .status-value {
            font-size: 0.9rem;
            font-weight: 700;
        }
        
        /* Wolf Intelligence Section */
        .wolf-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 25px;
            border: 1px solid #dee2e6;
        }
        
        .wolf-title {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 700;
            color: #495057;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .wolf-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .wolf-card {
            padding: 12px;
            border-radius: 12px;
            text-align: center;
            font-size: 0.75rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .wolf-card .label {
            opacity: 0.8;
            margin-bottom: 4px;
            font-weight: 500;
        }
        
        .wolf-card .value {
            font-weight: 700;
            font-size: 0.85rem;
        }
        
        /* Trading Info Section */
        .trading-section {
            background: white;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
        }
        
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #495057;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #f8f9fa;
        }
        
        .info-item:last-child {
            border-bottom: none;
        }
        
        .info-label {
            font-weight: 600;
            color: #6c757d;
            font-size: 0.85rem;
        }
        
        .info-value {
            font-weight: 700;
            color: #495057;
            font-size: 0.9rem;
        }
        
        .signal-buy { color: #28a745; }
        .signal-sell { color: #dc3545; }
        .signal-hold { color: #6c757d; }
        
        .countdown-timer {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 6px 12px;
            border-radius: 12px;
            font-family: 'SF Mono', Monaco, monospace;
            font-weight: 700;
            font-size: 0.85rem;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
        
        /* Strategy Section */
        .strategy-section {
            background: white;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
        }
        
        .strategy-desc {
            text-align: center;
            color: #6c757d;
            font-size: 0.85rem;
            margin-bottom: 20px;
            line-height: 1.5;
        }
        
        .strategy-info {
            display: block;
        }
        
        .strategy-selection {
            display: grid;
            gap: 8px;
            margin-top: 8px;
        }
        
        .strategy-btn {
            padding: 14px;
            border-radius: 14px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85rem;
            text-align: center;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .strategy-btn.active {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            transform: scale(1.02);
        }
        
        .strategy-btn:not(.active) {
            background: #f8f9fa;
            color: #6c757d;
            border: 1px solid #dee2e6;
        }
        
        .strategy-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Controls */
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 14px;
            border-radius: 14px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .btn-start {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
        }
        
        .btn-stop {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
            color: #212529;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 0.75rem;
            color: #6c757d;
            border-top: 1px solid #f8f9fa;
        }
        
        .footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        /* Mobile Optimizations */
        @media (max-width: 480px) {
            body { padding: 10px; }
            .container { max-width: 100%; }
            .header { padding: 20px 15px; }
            .main-content { padding: 20px 15px; }
            .header h1 { font-size: 1.6rem; }
            .status-grid { 
                grid-template-columns: 1fr 1fr;
                gap: 8px; 
            }
            .wolf-grid { gap: 8px; }
            .controls { grid-template-columns: 1fr; }
        }
        
        /* Animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .countdown-timer {
            animation: pulse 2s infinite;
        }
        
        /* Responsive adjustments */
        @media (min-width: 481px) and (max-width: 768px) {
            .container { max-width: 480px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>
                    🐺 CRYPTIX<br>Trading Wolf
                </h1>
                <div class="subtitle">Professional Trading Intelligence</div>
            </div>
        </div>
        
        <div class="main-content">
            <!-- Status Cards -->
            <div class="status-grid">
                <div class="status-card {{ 'status-running' if status.running else 'status-stopped' }}">
                    <div class="status-label">Bot Status</div>
                    <div class="status-value">{{ 'Running' if status.running else 'Stopped' }}</div>
                </div>
                <div class="status-card {{ 'status-connected' if status.api_connected else 'status-disconnected' }}">
                    <div class="status-label">API Status</div>
                    <div class="status-value">{{ 'Connected' if status.api_connected else 'Disconnected' }}</div>
                </div>
            </div>
            
            <!-- AI Wolf Intelligence -->
            <div class="wolf-section">
                <div class="wolf-title">
                    🧠 AI Wolf Intelligence
                </div>
                <div class="wolf-grid">
                    <div class="wolf-card" style="background: {{ '#d4edda' if status.get('market_regime') == 'EXTREME' else '#d1ecf1' if status.get('market_regime') == 'VOLATILE' else '#fff3cd' if status.get('market_regime') == 'QUIET' else '#e9ecef' }}; 
                                               color: {{ '#155724' if status.get('market_regime') == 'EXTREME' else '#0c5460' if status.get('market_regime') == 'VOLATILE' else '#856404' if status.get('market_regime') == 'QUIET' else '#495057' }};">
                        <div class="label">Market Regime</div>
                        <div class="value">{{ status.get('market_regime', 'NORMAL') }}</div>
                    </div>
                    <div class="wolf-card" style="background: {{ '#f8d7da' if status.get('hunting_mode') else '#e9ecef' }}; 
                                               color: {{ '#721c24' if status.get('hunting_mode') else '#495057' }};">
                        <div class="label">Wolf Mode</div>
                        <div class="value">{{ 'HUNTING 🎯' if status.get('hunting_mode') else 'PASSIVE' }}</div>
                    </div>
                    <div class="wolf-card" style="background: #e9ecef; color: #495057;">
                        <div class="label">Scan Interval</div>
                        <div class="value">{{ (status.get('signal_interval', 900) // 60) }}min</div>
                    </div>
                    <div class="wolf-card" style="background: #e9ecef; color: #495057;">
                        <div class="label">Next Scan</div>
                        <div class="value countdown-timer">{{ time_remaining }}</div>
                    </div>
                </div>
            </div>
            
            <!-- Account Balance -->
            <div class="trading-section">
                <div class="section-title">� Account Balance</div>
                {% if account_balance.error %}
                <div class="info-item">
                    <span class="info-label">Status</span>
                    <span class="info-value" style="color: #dc3545;">{{ account_balance.error }}</span>
                </div>
                {% else %}
                <div class="info-item">
                    <span class="info-label">USDT Balance</span>
                    <span class="info-value" style="color: #28a745; font-weight: bold;">${{ "{:,.2f}".format(account_balance.usdt_balance) }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Total Portfolio</span>
                    <span class="info-value" style="color: #17a2b8; font-weight: bold;">${{ "{:,.2f}".format(account_balance.total_value) }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Last Updated</span>
                    <span class="info-value">{{ account_balance.timestamp.split()[1][:8] if account_balance.timestamp else 'Never' }}</span>
                </div>
                {% endif %}
            </div>
            
            <!-- Strategy Section -->
            <div class="strategy-section">
                <div class="section-title">🎯 Trading Strategy</div>
                <div class="strategy-desc">{{ strategy_desc }}</div>
                <div class="strategy-info">
                    <div class="strategy-selection">
                        {% if status.trading_strategy == 'ADAPTIVE' %}
                        <div class="strategy-btn active">
                            🧠 <span>Adaptive - Smart & Dynamic</span>
                        </div>
                        <a href="/strategy/ML_PURE" class="strategy-btn">
                            🤖 <span>ML Pure - AI Driven</span>
                        </a>
                        {% elif status.trading_strategy == 'ML_PURE' %}
                        <a href="/strategy/ADAPTIVE" class="strategy-btn">
                            🧠 <span>Adaptive - Smart & Dynamic</span>
                        </a>
                        <div class="strategy-btn active">
                            🤖 <span>ML Pure - AI Driven</span>
                        </div>
                        {% else %}
                        <div class="strategy-btn active">
                            🧠 <span>Adaptive - Smart & Dynamic</span>
                        </div>
                        <a href="/strategy/ML_PURE" class="strategy-btn">
                            🤖 <span>ML Pure - AI Driven</span>
                        </a>
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <a href="/start" class="btn btn-start">🚀 Start Bot</a>
                <a href="/stop" class="btn btn-stop">🛑 Stop Bot</a>
            </div>
            
            <div style="margin-bottom: 15px;">
                <a href="/logs" class="btn btn-secondary" style="width: 100%; display: block;">📋 View Logs</a>
            </div>
        </div>
        
        <div class="footer">
            <div style="margin-bottom: 10px;">
                <strong>Cairo Time: {{ current_time }}</strong>
            </div>
            Auto-refresh every 30s • <a href="javascript:location.reload()">Manual Refresh</a>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
        
        // Add touch feedback for mobile
        document.querySelectorAll('.btn, .strategy-btn').forEach(button => {
            button.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.98)';
            });
            button.addEventListener('touchend', function() {
                this.style.transform = '';
            });
        });
    </script>
</body>
</html>
    """, 
    status=bot_status, 
    current_time=format_cairo_time(), 
    time_remaining=get_time_remaining_for_next_signal(), 
    strategy_desc=strategy_desc,
    account_balance=get_account_balance_for_ui())


@app.route('/start')
def start():
    """Manual start route"""
    if not bot_status.get('running', False):
        try:
            start_trading_bot()
            return redirect('/')
        except Exception as e:
            bot_status['errors'].append(f"Failed to start bot: {str(e)}")
            return redirect('/')
    else:
        print("⚠️ Bot is already running")
        return redirect('/')

@app.route('/stop')
def stop():
    """Manual stop route"""
    try:
        stop_trading_bot()  # Call the proper stop function
        print("Bot manually stopped via web interface")
        return redirect('/')
    except Exception as e:
        print(f"Error stopping bot: {e}")
        return f"Error stopping bot: {e}"

@app.route('/force_scan')
def force_scan():
    """Force an immediate signal scan"""
    try:
        if not bot_status.get('running', False):
            return jsonify({'error': 'Bot is not running'}), 400
            
        if not bot_status.get('api_connected', False):
            return jsonify({'error': 'API not connected'}), 400
        
        print("🚀 Manual scan triggered from web interface")
        
        # Force immediate scan by setting next_signal_time to now
        bot_status['next_signal_time'] = get_cairo_time()
        
        return jsonify({
            'success': True, 
            'message': 'Scan triggered successfully',
            'next_scan_time': format_cairo_time(bot_status['next_signal_time'])
        })
        
    except Exception as e:
        print(f"Error triggering manual scan: {e}")
        return jsonify({'error': f'Failed to trigger scan: {str(e)}'}), 500

@app.route('/strategy/<name>')
def set_strategy(name):
    """Switch trading strategy"""
    try:
        if name.upper() in ['ADAPTIVE', 'ML_PURE']:
            previous_strategy = bot_status.get('trading_strategy', config.DEFAULT_STRATEGY)
            new_strategy = name.upper()
            
            # Validate ML strategy availability
            if new_strategy == 'ML_PURE' and not ML_STRATEGY_AVAILABLE:
                log_error_to_csv(
                    f"ML_PURE strategy not available - ML modules missing",
                    "STRATEGY_ERROR",
                    "set_strategy",
                    "ERROR"
                )
                return "ML Strategy not available - missing ML modules", 400
            
            # Update bot status
            bot_status['trading_strategy'] = new_strategy
            
            # Log the strategy change
            log_error_to_csv(
                f"Strategy confirmed as {new_strategy}",
                "STRATEGY_CONFIRM",
                "set_strategy",
                "INFO"
            )
            
            # Print debug info
            print(f"Strategy confirmed: {new_strategy}")
            print(f"Current bot status: {bot_status}")
            
            return redirect('/')
        else:
            log_error_to_csv(
                f"Invalid strategy name: {name}. Supported strategies: ADAPTIVE, ML_PURE",
                "STRATEGY_ERROR",
                "set_strategy",
                "ERROR"
            )
            return "Invalid strategy name", 400
    except Exception as e:
        error_msg = f"Error changing strategy: {str(e)}"
        log_error_to_csv(error_msg, "STRATEGY_ERROR", "set_strategy", "ERROR")
        print(error_msg)
        return error_msg, 500

@app.route('/api/status')
def api_status():
    """JSON API endpoint for bot status"""
    return jsonify(bot_status)

@app.route('/api/balances')
def api_balances():
    """JSON API endpoint for account balances"""
    try:
        balance_summary = get_account_balances_summary()
        return jsonify(balance_summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs')
def view_logs():
    """View CSV logs interface"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📋 CRYPTIX Logs</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 15px;
            color: #333;
        }
        
        .container {
            max-width: 420px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
            overflow: hidden;
            backdrop-filter: blur(20px);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 20px;
            text-align: center;
            position: relative;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .header-content {
            position: relative;
            z-index: 1;
        }
        
        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .subtitle {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 500;
        }
        
        .main-content {
            padding: 25px 20px;
        }
        
        .back-link {
            display: inline-block;
            margin-bottom: 25px;
            padding: 12px 20px;
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            text-decoration: none;
            border-radius: 14px;
            font-size: 0.9rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
            box-sizing: border-box;
        }
        
        .back-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(40, 167, 69, 0.4);
        }
        
        /* Log Files Section */
        .log-section {
            background: white;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
        }
        
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #495057;
            margin-bottom: 15px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .log-links {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }
        
        .log-links a {
            padding: 16px;
            border-radius: 14px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85rem;
            text-align: center;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .log-links a:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .log-links a.download {
            background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
            color: #212529;
        }
        
        .log-links a.download:hover {
            box-shadow: 0 4px 12px rgba(255, 193, 7, 0.3);
        }
        
        /* Stats Section */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #f8f9fa;
        }
        
        .stat-item:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            font-weight: 600;
            color: #6c757d;
            font-size: 0.85rem;
        }
        
        .stat-value {
            font-weight: 700;
            color: #495057;
            font-size: 0.9rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 0.75rem;
            color: #6c757d;
            border-top: 1px solid #f8f9fa;
        }
        
        .footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        /* Mobile Optimizations */
        @media (max-width: 480px) {
            body { padding: 10px; }
            .container { max-width: 100%; }
            .header { padding: 20px 15px; }
            .main-content { padding: 20px 15px; }
            .header h1 { font-size: 1.6rem; }
        }
        
        /* Touch feedback */
        .log-links a {
            -webkit-tap-highlight-color: transparent;
        }
        
        /* Responsive adjustments */
        @media (min-width: 481px) and (max-width: 768px) {
            .container { max-width: 480px; }
            .log-links { grid-template-columns: 1fr 1fr; }
            .log-links a.download { grid-column: 1 / -1; }
        }
        
        @media (min-width: 769px) {
            .container { max-width: 600px; }
            .log-links { grid-template-columns: 1fr 1fr; }
            .log-links a.download { grid-column: 1 / -1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>
                    📋 CRYPTIX<br>Trading Logs
                </h1>
                <div class="subtitle">Activity & Performance Monitoring</div>
            </div>
        </div>
        
        <div class="main-content">
            <a href="/" class="back-link">← Back to Dashboard</a>
            
            <!-- Log Files Section -->
            <div class="log-section">
                <div class="section-title">
                    📊 Available Log Files
                </div>
                <div class="log-links">
                    <a href="/logs/trades">
                        📊 <span>Trade History</span>
                    </a>
                    <a href="/logs/signals">
                        📈 <span>Signal History</span>
                    </a>
                    <a href="/logs/errors">
                        ❌ <span>Error Log</span>
                    </a>
                    <a href="/download_logs" class="download">
                        💾 <span>Download All CSV Files</span>
                    </a>
                </div>
            </div>
            
            <!-- Quick Stats Section -->
            <div class="log-section">
                <div class="section-title">
                    📈 Quick Statistics
                </div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-label">Data Storage</span>
                        <span class="stat-value">Supabase Cloud</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Backup Location</span>
                        <span class="stat-value">/logs/</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Last Updated</span>
                        <span class="stat-value">{{ current_time }}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div style="margin-bottom: 10px;">
                <strong>Cairo Time: {{ current_time }}</strong>
            </div>
            <a href="javascript:location.reload()">Refresh Data</a>
        </div>
    </div>
    
    <script>
        // Add touch feedback for mobile
        document.querySelectorAll('.log-links a, .back-link').forEach(button => {
            button.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.98)';
            });
            button.addEventListener('touchend', function() {
                this.style.transform = '';
            });
        });
    </script>
</body>
</html>
    """, current_time=format_cairo_time())

@app.route('/logs/trades')
def view_trade_logs():
    """View trade history from Supabase or CSV"""
    trades = get_supabase_trade_history(days=0)  # Use Supabase by default, no limit
    
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Trade History</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
            background: #f5f5f5;
            padding: 10px;
            line-height: 1.6;
        }
        
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 15px; 
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header {
            margin-bottom: 20px;
        }
        
        .back-link { 
            display: inline-block; 
            margin-bottom: 15px; 
            padding: 10px 20px; 
            background: #28a745; 
            color: white; 
            text-decoration: none; 
            border-radius: 5px;
            font-size: 0.9rem;
            transition: background 0.3s;
        }
        
        .back-link:hover {
            background: #218838;
        }
        
        h1 {
            font-size: 1.5rem;
            margin: 10px 0;
            color: #333;
        }
        
        .trade-count {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 15px;
        }
        
        /* Desktop Table View */
        .table-wrapper {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            display: none;
        }
        
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 15px; 
            font-size: 0.85rem;
            min-width: 800px;
        }
        
        th, td { 
            padding: 12px; 
            border: 1px solid #ddd; 
            text-align: left;
            white-space: nowrap;
        }
        
        th { 
            background: #f8f9fa; 
            font-weight: 600; 
            position: sticky; 
            top: 0;
            z-index: 1;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }
        
        tr:nth-child(even) { 
            background: #f9f9f9; 
        }
        
        /* Mobile Card View */
        .cards-container {
            display: grid;
            gap: 12px;
        }
        
        .trade-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .trade-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .trade-symbol {
            font-size: 1.1rem;
            font-weight: bold;
            color: #333;
        }
        
        .trade-time {
            font-size: 0.75rem;
            color: #666;
        }
        
        .trade-details {
            display: grid;
            gap: 8px;
        }
        
        .trade-detail-row {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
        }
        
        .detail-label {
            font-size: 0.85rem;
            color: #666;
            font-weight: 500;
        }
        
        .detail-value {
            font-size: 0.9rem;
            color: #333;
            font-weight: 600;
            text-align: right;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-success { 
            background: #d4edda; 
            color: #155724;
        }
        
        .status-simulated { 
            background: #d1ecf1; 
            color: #0c5460;
        }
        
        .status-error { 
            background: #f8d7da; 
            color: #721c24;
        }
        
        /* Signal badges */
        .signal-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: bold;
        }
        
        .signal-buy { 
            background: #28a745; 
            color: white;
        }
        
        .signal-sell { 
            background: #dc3545; 
            color: white;
        }
        
        .signal-hold { 
            background: #ffc107; 
            color: #333;
        }
        
        .no-trades {
            text-align: center;
            padding: 40px 20px;
            color: #666;
            font-size: 1rem;
        }
        
        /* Tablet and Desktop */
        @media (min-width: 769px) {
            body {
                padding: 20px;
            }
            
            .container {
                padding: 25px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .back-link {
                padding: 12px 24px;
                font-size: 1rem;
            }
            
            .table-wrapper {
                display: block;
            }
            
            .cards-container {
                display: none;
            }
        }
        
        /* Mobile optimizations */
        @media (max-width: 768px) {
            body {
                padding: 8px;
            }
            
            .container {
                padding: 12px;
                border-radius: 8px;
            }
            
            .back-link {
                width: 100%;
                text-align: center;
                padding: 12px;
                font-size: 0.95rem;
            }
            
            h1 {
                font-size: 1.3rem;
                margin: 12px 0;
            }
            
            .table-wrapper {
                display: none;
            }
            
            .cards-container {
                display: grid;
            }
        }
        
        /* Very small phones */
        @media (max-width: 360px) {
            .trade-symbol {
                font-size: 1rem;
            }
            
            .detail-label,
            .detail-value {
                font-size: 0.8rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="/logs" class="back-link">← Back to Logs</a>
            <h1>📊 Trade History</h1>
            {% if trades %}
            <div class="trade-count">Showing {{ trades|length }} trades (newest first)</div>
            {% endif %}
        </div>
        
        {% if trades %}
        
        <!-- Desktop Table View -->
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>Time (Cairo)</th>
                        <th>Signal</th>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Value</th>
                        <th>Fee</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trades %}
                    <tr class="status-{{ trade.status.lower() if trade.status else 'unknown' }}">
                        <td>{{ trade.cairo_time }}</td>
                        <td class="signal-{{ trade.signal.lower() }}">{{ trade.signal }}</td>
                        <td>{{ trade.symbol }}</td>
                        <td>{{ "%.6f"|format(trade.quantity) }}</td>
                        <td>${{ "%.2f"|format(trade.price) }}</td>
                        <td>${{ "%.2f"|format(trade.value) }}</td>
                        <td>${{ "%.4f"|format(trade.fee) }}</td>
                        <td>{{ trade.status }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <!-- Mobile Card View -->
        <div class="cards-container">
            {% for trade in trades %}
            <div class="trade-card">
                <div class="trade-card-header">
                    <div>
                        <div class="trade-symbol">{{ trade.symbol }}</div>
                        <div class="trade-time">{{ trade.cairo_time }}</div>
                    </div>
                    <span class="signal-badge signal-{{ trade.signal.lower() }}">{{ trade.signal }}</span>
                </div>
                <div class="trade-details">
                    <div class="trade-detail-row">
                        <span class="detail-label">Quantity</span>
                        <span class="detail-value">{{ "%.6f"|format(trade.quantity) }}</span>
                    </div>
                    <div class="trade-detail-row">
                        <span class="detail-label">Price</span>
                        <span class="detail-value">${{ "%.2f"|format(trade.price) }}</span>
                    </div>
                    <div class="trade-detail-row">
                        <span class="detail-label">Total Value</span>
                        <span class="detail-value">${{ "%.2f"|format(trade.value) }}</span>
                    </div>
                    <div class="trade-detail-row">
                        <span class="detail-label">Fee</span>
                        <span class="detail-value">${{ "%.4f"|format(trade.fee) }}</span>
                    </div>
                    <div class="trade-detail-row">
                        <span class="detail-label">Status</span>
                        <span class="detail-value">
                            <span class="status-badge status-{{ trade.status.lower() if trade.status else 'unknown' }}">
                                {{ trade.status }}
                            </span>
                        </span>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% else %}
        <div class="no-trades">
            <p>📭 No trades found.</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
    """, trades=trades)

@app.route('/logs/signals')
def view_signal_logs():
    """View signal history from Supabase"""
    try:
        signals = get_supabase_signal_history(limit=100)
        
        return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal History</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.8rem; }
        th, td { padding: 6px 8px; border: 1px solid #ddd; text-align: left; }
        th { background: #f8f9fa; font-weight: bold; position: sticky; top: 0; }
        tr:nth-child(even) { background: #f9f9f9; }
        .back-link { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; }
        .signal-buy { color: #28a745; font-weight: bold; }
        .signal-sell { color: #dc3545; font-weight: bold; }
        .signal-hold { color: #ffc107; font-weight: bold; }
        .sentiment-bullish { color: #28a745; }
        .sentiment-bearish { color: #dc3545; }
        .sentiment-neutral { color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/logs" class="back-link">← Back to Logs</a>
        <h1>📈 Signal History (Latest 100 - Newest First)</h1>
        
        {% if signals %}
        <table>
            <thead>
                <tr>
                    <th>Time (Cairo)</th>
                    <th>Signal</th>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>RSI</th>
                    <th>MACD</th>
                    <th>MACD Trend</th>
                    <th>Sentiment</th>
                    <th>SMA5</th>
                    <th>SMA20</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {% for signal in signals %}
                <tr>
                    <td>{{ signal.cairo_time }}</td>
                    <td class="signal-{{ signal.signal.lower() }}">{{ signal.signal }}</td>
                    <td>{{ signal.symbol }}</td>
                    <td>${{ "%.2f"|format(signal.price) }}</td>
                    <td>{{ "%.1f"|format(signal.rsi) }}</td>
                    <td>{{ "%.6f"|format(signal.macd) }}</td>
                    <td>{{ signal.macd_trend }}</td>
                    <td class="sentiment-{{ signal.sentiment.lower() if signal.sentiment else 'neutral' }}">{{ signal.sentiment }}</td>
                    <td>${{ "%.2f"|format(signal.sma5) }}</td>
                    <td>${{ "%.2f"|format(signal.sma20) }}</td>
                    <td style="font-size: 0.7rem;">{{ signal.reason[:100] if signal.reason else '' }}...</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No signals found.</p>
        {% endif %}
    </div>
</body>
</html>
        """, signals=signals)
        
    except Exception as e:
        return f"Error loading signal logs: {e}"

@app.route('/logs/errors')
def view_error_logs():
    """View error log from Supabase"""
    try:
        errors = get_supabase_error_history(limit=50)
        
        return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error Log</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.8rem; }
        th, td { padding: 6px 8px; border: 1px solid #ddd; text-align: left; }
        th { background: #f8f9fa; font-weight: bold; position: sticky; top: 0; }
        tr:nth-child(even) { background: #f9f9f9; }
        .back-link { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; }
        .error { background: #f8d7da; }
        .warning { background: #fff3cd; }
        .critical { background: #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/logs" class="back-link">← Back to Logs</a>
        <h1>❌ Error Log (Last 50 - Newest First)</h1>
        
        {% if errors %}
        <table>
            <thead>
                <tr>
                    <th>Time (Cairo)</th>
                    <th>Severity</th>
                    <th>Error Type</th>
                    <th>Function</th>
                    <th>Error Message</th>
                    <th>Bot Status</th>
                </tr>
            </thead>
            <tbody>
                {% for error in errors %}
                <tr class="{{ error.severity.lower() if error.severity else 'error' }}">
                    <td>{{ error.cairo_time }}</td>
                    <td>{{ error.severity }}</td>
                    <td>{{ error.error_type }}</td>
                    <td>{{ error.function_name }}</td>
                    <td style="max-width: 300px; word-wrap: break-word;">{{ error.error_message }}</td>
                    <td>{{ error.bot_status }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No errors found.</p>
        {% endif %}
    </div>
</body>
</html>
        """, errors=errors)
        
    except Exception as e:
        return f"Error loading error logs: {e}"

@app.route('/ping')
def ping():
    """Simple ping endpoint for uptime monitoring"""
    return {"status": "alive", "timestamp": format_cairo_time()}, 200

@app.route('/health')
def health():
    """Comprehensive health check with system and bot status"""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': format_cairo_time(),
            'bot_running': bot_status.get('running', False),
            'api_connected': bot_status.get('api_connected', False),
            'last_update': bot_status.get('last_update', 'Never'),
            'error_count': len(bot_status.get('errors', [])),
            'consecutive_errors': bot_status.get('consecutive_errors', 0),
            'uptime_seconds': (get_cairo_time() - bot_status.get('start_time', get_cairo_time())).total_seconds(),
            'account_type': bot_status.get('account_type', 'Unknown'),
            'can_trade': bot_status.get('can_trade', False)
        }
        
        # Environment variable check (for debugging Render deployment)
        env_check = {
            'api_key_present': bool(os.getenv("API_KEY")),
            'api_secret_present': bool(os.getenv("API_SECRET")),
            'api_key_length': len(os.getenv("API_KEY", "")),
            'api_secret_length': len(os.getenv("API_SECRET", "")),
            'environment': os.getenv("RENDER", "local"),  # Render sets this automatically
        }
        
        if env_check['api_key_present']:
            api_key_val = os.getenv("API_KEY", "")
            env_check['api_key_preview'] = f"{api_key_val[:8]}...{api_key_val[-4:]}" if len(api_key_val) >= 12 else "invalid"
        
        health_data['environment'] = env_check
        
        # Try to get memory info if psutil is available
        try:
            import psutil  # Optional dependency for system metrics
            process = psutil.Process()
            health_data['memory_usage_mb'] = round(process.memory_info().rss / 1024 / 1024, 2)
            health_data['cpu_percent'] = round(process.cpu_percent(), 2)
        except ImportError:
            health_data['memory_usage_mb'] = 'unknown'
            health_data['cpu_percent'] = 'unknown'
        
        # Determine overall health status
        if not bot_status.get('api_connected', False):
            health_data['status'] = 'degraded'
        elif bot_status.get('consecutive_errors', 0) >= 3:
            health_data['status'] = 'warning'
        elif not bot_status.get('running', False):
            health_data['status'] = 'stopped'
            
        return jsonify(health_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': format_cairo_time()
        }), 500

# ==================== POSITION REBALANCING FUNCTIONS ====================

def get_position_rsi(symbol, period=14):
    """Get current RSI for a specific trading pair"""
    try:
        if not client:
            return None
            
        # Get klines data for RSI calculation
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=period + 10)
        if not klines or len(klines) < period:
            print(f"⚠️ Insufficient data for RSI calculation for {symbol}")
            return None
            
        # Extract closing prices
        prices = [float(kline[4]) for kline in klines]
        rsi = calculate_rsi(prices, period)
        
        print(f"📊 {symbol} RSI({period}): {rsi:.1f}")
        return rsi
        
    except Exception as e:
        print(f"❌ Error calculating RSI for {symbol}: {e}")
        return None

def detect_dust_positions(min_usdt_value=None):
    """Detect small positions that should be liquidated to USDT using smart minimum calculation"""
    try:
        if not client:
            return []
            
        account_info = client.get_account()
        dust_positions = []
        
        # Import OrderValidator for smart minimum calculation
        try:
            from order_validator import OrderValidator
            validator = OrderValidator(client)
            use_smart_detection = True
            print(f"\n🔍 Scanning for dust positions using smart minimum notional requirements...")
        except ImportError:
            use_smart_detection = False
            fallback_min = min_usdt_value or 5.0
            print(f"\n🔍 Scanning for dust positions (< ${fallback_min:.2f}) - OrderValidator not available...")
        
        for balance in account_info['balances']:
            asset = balance['asset']
            free_balance = float(balance['free'])
            total_balance = free_balance + float(balance['locked'])
            
            # Skip USDT and positions with zero balance
            if asset == 'USDT' or total_balance <= 0:
                continue
                
            # Only check free balance for liquidation (can't sell locked funds)
            if free_balance <= 0:
                continue
                
            # Try to get USDT value and calculate minimum requirements
            usdt_value = 0
            is_dust = False
            
            try:
                if asset in ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'AVAX', 'LINK', 'UNI', 'ATOM', 'XLM', 'VET', 'FIL', 'TRX', 'ETC', 'LTC', 'BCH', 'EOS', 'XTZ', 'NEO', 'DASH', 'ZEC', 'QTUM', 'ONT', 'BAT', 'ZIL', 'RVN', 'DOGE', 'SHIB']:
                    symbol = f"{asset}USDT"
                    ticker = client.get_ticker(symbol=symbol)
                    price = float(ticker['lastPrice'])
                    usdt_value = free_balance * price
                    
                    # Skip if position has zero value
                    if usdt_value <= 0:
                        continue
                    
                    # Determine if this is dust using smart detection or fallback
                    if use_smart_detection:
                        try:
                            # Calculate the minimum sellable quantity for this symbol
                            min_sellable_qty = validator.calculate_minimum_valid_quantity(symbol, price)
                            min_sellable_value = min_sellable_qty * price
                            
                            # Position is dust if current value is below minimum sellable value
                            is_dust = usdt_value < min_sellable_value
                            
                            if is_dust:
                                print(f"   💨 {asset}: {free_balance:.8f} (~${usdt_value:.2f}) < MIN ${min_sellable_value:.2f}")
                            elif usdt_value < 20.0:  # Log assets close to dust threshold for visibility
                                print(f"   ✅ {asset}: {free_balance:.8f} (~${usdt_value:.2f}) ≥ MIN ${min_sellable_value:.2f}")
                                
                        except Exception as smart_error:
                            # Fallback to hardcoded minimum if smart detection fails
                            fallback_min = min_usdt_value or 5.0
                            is_dust = usdt_value < fallback_min
                            print(f"   ⚠️ {asset}: Smart detection failed ({smart_error}), using fallback ${fallback_min:.2f}")
                            if is_dust:
                                print(f"   💨 {asset}: {free_balance:.8f} (~${usdt_value:.2f}) < FALLBACK ${fallback_min:.2f}")
                    else:
                        # Use fallback threshold
                        fallback_min = min_usdt_value or 5.0
                        is_dust = usdt_value < fallback_min
                        if is_dust:
                            print(f"   � {asset}: {free_balance:.8f} (~${usdt_value:.2f}) < ${fallback_min:.2f}")
                    
                    if is_dust:
                        # Special handling for BNB - it's the target for dust conversion, so skip manual liquidation
                        if asset == 'BNB':
                            print(f"   💰 {asset}: BNB dust (skip manual liquidation - used for dust conversion)")
                            continue
                            
                        dust_positions.append({
                            'asset': asset,
                            'symbol': symbol,
                            'quantity': free_balance,
                            'total_quantity': total_balance,
                            'usdt_value': usdt_value,
                            'price': price
                        })
                        
            except Exception as e:
                print(f"   ⚠️ Could not check {asset}: {e}")
                continue
                
        return dust_positions
        
    except Exception as e:
        print(f"❌ Error detecting dust positions: {e}")
        return []

def sell_partial_position(symbol, percentage=50.0, reason="Partial profit taking", send_notification=True):
    """Sell a percentage of an existing position"""
    try:
        if not client:
            return {"success": False, "error": "Client not initialized"}
            
        base_asset = symbol.replace('USDT', '')
        
        # Get current balance
        account_info = client.get_account()
        current_balance = 0
        for balance in account_info['balances']:
            if balance['asset'] == base_asset:
                current_balance = float(balance['free'])
                break
                
        if current_balance <= 0:
            return {"success": False, "error": f"No {base_asset} balance to sell"}
            
        # ===== CRITICAL: Profit Validation Check - Prevent Selling at Loss =====
        print("\n=== Profit Validation Check (Partial Sell) ===")
        try:
            # Use position tracker to check profitability
            position_tracker = get_position_tracker()
            
            # Get current market price for profit calculation
            ticker = client.get_ticker(symbol=symbol)
            current_price = float(ticker['lastPrice'])
            
            # Check if we should allow this partial sell based on profitability
            minimum_profit_pct = config.REBALANCING.get('minimum_profit_pct', 2.0)
            should_sell, profit_reason = position_tracker.should_allow_partial_sell(
                symbol=symbol,
                current_price=current_price,
                minimum_profit_pct=minimum_profit_pct
            )
            
            if not should_sell:
                print(f"🚫 PARTIAL SELL BLOCKED: {profit_reason}")
                
                # Log blocked partial sell as HOLD signal (not error) with reason
                try:
                    blocked_indicators = {
                        'symbol': symbol,
                        'rsi': bot_status.get('rsi', 0),
                        'macd': bot_status.get('macd', {}).get('macd', 0),
                        'macd_trend': bot_status.get('macd', {}).get('trend', 'NEUTRAL'),
                        'sentiment': bot_status.get('sentiment', 'neutral')
                    }
                    log_signal_to_csv(
                        "HOLD", 
                        current_price, 
                        blocked_indicators, 
                        f"UNPROFITABLE_PARTIAL_SELL_BLOCKED - {profit_reason}"
                    )
                except Exception as log_error:
                    print(f"⚠️ Failed to log blocked partial sell signal: {log_error}")
                
                return {
                    "success": False,
                    "error": f"Sell blocked - unprofitable: {profit_reason}",
                    "blocked_reason": profit_reason
                }
            
            # If we reach here, the sell is profitable
            print(f"✅ Profit check passed: {profit_reason}")
            
        except Exception as profit_check_error:
            print(f"⚠️ Warning: Could not perform profit validation: {profit_check_error}")
            
            # SAFETY-FIRST: Block the sell on ANY error
            # Log as HOLD signal with validation error reason
            try:
                error_indicators = {
                    'symbol': symbol,
                    'rsi': bot_status.get('rsi', 0),
                    'macd': bot_status.get('macd', {}).get('macd', 0),
                    'macd_trend': bot_status.get('macd', {}).get('trend', 'NEUTRAL'),
                    'sentiment': bot_status.get('sentiment', 'neutral')
                }
                log_signal_to_csv(
                    "HOLD", 
                    0, 
                    error_indicators, 
                    f"PROFIT_VALIDATION_ERROR - Partial sell blocked: {profit_check_error}"
                )
            except Exception as log_error:
                print(f"⚠️ Failed to log validation error signal: {log_error}")
            
            return {
                "success": False,
                "error": f"Sell blocked - validation error: {profit_check_error}",
                "validation_error": str(profit_check_error)
            }
        # ===== END PROFIT VALIDATION CHECK =====
            
        # Calculate sell quantity
        sell_quantity = current_balance * (percentage / 100.0)
        
        # Get symbol info for precision
        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info:
            return {"success": False, "error": f"Could not get symbol info for {symbol}"}
            
        # Find lot size filter
        lot_size_filter = None
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                lot_size_filter = f
                break
                
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            sell_quantity = round(sell_quantity / step_size) * step_size
            
            # Format quantity to correct precision to avoid "too much precision" error
            # Calculate precision from step_size
            step_size_str = f"{step_size:f}".rstrip('0').rstrip('.')
            if '.' in step_size_str:
                precision = len(step_size_str.split('.')[1])
            else:
                precision = 0
            
            # Round to proper precision and remove unnecessary decimal places
            sell_quantity = round(sell_quantity, precision)
            
            # If precision is 0, convert to int to avoid decimal point
            if precision == 0:
                sell_quantity = int(sell_quantity)
            
        # Check minimum quantity
        min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
        if sell_quantity < min_qty:
            return {"success": False, "error": f"Quantity {sell_quantity} below minimum {min_qty}"}
            
        # CRITICAL: Check minimum notional requirement  
        try:
            current_price = float(client.get_ticker(symbol=symbol)['lastPrice'])
            order_value = sell_quantity * current_price
            
            # Get minimum notional from symbol filters
            min_notional = 10.0  # Default Binance minimum
            for f in symbol_info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(f['minNotional'])
                    break
            
            print(f"   Order value check: ${order_value:.2f} vs minimum ${min_notional:.2f}")
            
            if order_value < min_notional:
                print(f"   ❌ Order value ${order_value:.2f} below minimum notional ${min_notional:.2f}")
                # Try to increase quantity to meet minimum notional
                required_qty = min_notional / current_price
                
                if required_qty <= current_balance:
                    print(f"   🔧 Adjusting quantity to meet minimum notional: {required_qty:.8f}")
                    sell_quantity = required_qty
                    
                    # Re-adjust for step size if needed
                    if lot_size_filter:
                        step_size = float(lot_size_filter['stepSize'])
                        sell_quantity = round(sell_quantity / step_size) * step_size
                        sell_quantity = round(sell_quantity, precision)
                        
                        # Final check after step size adjustment
                        final_value = sell_quantity * current_price
                        if final_value < min_notional:
                            return {"success": False, "error": f"Cannot meet minimum notional: max possible ${current_balance * current_price:.2f} < required ${min_notional:.2f}"}
                        
                    print(f"   ✅ Adjusted quantity: {sell_quantity:.8f} (${sell_quantity * current_price:.2f})")
                else:
                    max_possible_value = current_balance * current_price
                    return {"success": False, "error": f"Insufficient balance for minimum notional: have ${max_possible_value:.2f}, need ${min_notional:.2f}"}
            else:
                print(f"   ✅ Order value meets minimum notional requirement")
                
        except Exception as e:
            print(f"   ⚠️ Could not verify notional requirement: {e}")
            # Continue with original logic if price check fails
            
        print(f"\n🎯 Partial Sell Order - {reason}")
        print(f"   Symbol: {symbol}")
        print(f"   Current Balance: {current_balance:.8f} {base_asset}")
        print(f"   Sell Percentage: {percentage}%")
        print(f"   Sell Quantity: {sell_quantity} {base_asset}")  # Remove .8f formatting to show actual precision
        
        # Format quantity properly to avoid scientific notation
        formatted_qty = format_quantity_for_binance(sell_quantity, step_size)
        print(f"   Formatted quantity for API: {formatted_qty}")
        
        # Execute sell order
        order = client.order_market_sell(symbol=symbol, quantity=formatted_qty)
        
        # Calculate order details
        avg_price = float(order['fills'][0]['price']) if order['fills'] else 0
        total_value = float(order['cummulativeQuoteQty'])
        total_fee = sum([float(fill['commission']) for fill in order['fills']])
        
        result = {
            "success": True,
            "order_id": order.get('orderId'),
            "symbol": symbol,
            "quantity": sell_quantity,
            "price": avg_price,
            "value": total_value,
            "fee": total_fee,
            "percentage": percentage,
            "reason": reason,
            "timestamp": format_cairo_time()
        }
        
        print(f"✅ Partial sell executed successfully!")
        print(f"   Order ID: {result['order_id']}")
        print(f"   Price: ${avg_price:.4f}")
        print(f"   Value: ${total_value:.2f}")
        print(f"   Fee: ${total_fee:.4f}")
        
        # Log the trade
        log_trade_to_csv(
            {
                'signal': 'SELL_PARTIAL',
                'symbol': symbol,
                'quantity': sell_quantity,
                'price': avg_price,
                'value': total_value,
                'fee': total_fee,
                'status': 'SUCCESS',
                'order_id': result['order_id'],
                'timestamp': format_cairo_time()
            },
            {
                'percentage': percentage,
                'reason': reason,
                'order_type': 'partial_sell'
            }
        )
        
        # Send Telegram notification if available and requested
        if TELEGRAM_AVAILABLE and send_notification:
            try:
                # Reuse the same trade_info structure for notification
                notification_trade_info = {
                    'signal': 'SELL_PARTIAL',
                    'symbol': symbol,
                    'quantity': sell_quantity,
                    'price': avg_price,
                    'value': total_value,
                    'fee': total_fee,
                    'status': 'SUCCESS',
                    'order_id': result['order_id'],
                    'timestamp': format_cairo_time()
                }
                notify_trade(notification_trade_info, is_executed=True)
            except Exception as telegram_error:
                print(f"Telegram partial sell notification failed: {telegram_error}")
            
        return result
        
    except Exception as e:
        error_msg = f"Error in partial sell for {symbol}: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "PARTIAL_SELL_ERROR", "sell_partial_position", "ERROR")
        return {"success": False, "error": str(e)}

def liquidate_dust_position(dust_position):
    """Smart dust liquidation: tries USDT first, falls back to BNB if notional requirements fail"""
    try:
        if not client:
            return {"success": False, "error": "Client not initialized"}
            
        asset = dust_position['asset']
        original_symbol = dust_position['symbol']  # e.g., ADAUSDT
        quantity = dust_position['quantity']
        
        print(f"\n💨 Liquidating dust position: {asset}")
        print(f"   Quantity: {quantity:.8f} {asset}")
        print(f"   Est. Value: ${dust_position['usdt_value']:.2f}")
        
        # Smart liquidation: Try USDT first, then BNB if it fails
        symbols_to_try = []
        
        # Primary: try USDT pair
        if original_symbol.endswith('USDT'):
            symbols_to_try.append((original_symbol, 'USDT'))
        
        # Fallback: try BNB pair (often has lower minimums)
        bnb_symbol = f"{asset}BNB"
        try:
            bnb_info = client.get_symbol_info(bnb_symbol)
            if bnb_info and bnb_info.get('status') == 'TRADING':
                symbols_to_try.append((bnb_symbol, 'BNB'))
        except:
            pass  # BNB pair not available
        
        if not symbols_to_try:
            return {"success": False, "error": f"No trading pairs available for {asset}"}
        
        last_error = None
        
        # Try each symbol until one works
        for symbol, quote_asset in symbols_to_try:
            try:
                print(f"🔄 Trying {symbol}...")
                
                # Initialize variables for proper scope
                lot_size_filter = None
                validated_quantity = quantity
                
                # Get symbol info for filters (needed regardless of validation method)
                symbol_info = client.get_symbol_info(symbol)
                if not symbol_info:
                    last_error = f"Could not get symbol info for {symbol}"
                    continue
                    
                # Find lot size filter (needed for quantity formatting)
                for f in symbol_info['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        lot_size_filter = f
                        break
                
                # Enhanced order validation for dust liquidation
                try:
                    from order_validator import OrderValidator, log_validation_result
                    validator = OrderValidator(client)
                    
                    # Get fresh price for accurate validation
                    try:
                        ticker = client.get_ticker(symbol=symbol)
                        current_price = float(ticker['lastPrice'])
                    except:
                        # Skip this symbol if we can't get price
                        last_error = f"Could not get price for {symbol}"
                        continue
                    
                    # Pass available balance (which is the dust quantity) to validator
                    validation_result = validator.validate_order(symbol, quantity, 'SELL', current_price, quantity)
                    
                    if validation_result['is_valid']:
                        validated_quantity = validation_result['adjusted_quantity']
                        print(f"✅ {symbol} validation passed")
                        print(f"Final validated quantity: {validated_quantity:.8f}")
                        
                        if validation_result['warnings']:
                            for warning in validation_result['warnings']:
                                print(f"⚠️ {warning}")
                    else:
                        print(f"❌ {symbol} validation failed:")
                        for error in validation_result['errors']:
                            print(f"   - {error}")
                        
                        # Check if this is a notional issue that might work with different pair
                        error_str = '; '.join(validation_result['errors'])
                        if 'notional' in error_str.lower() or 'below minimum' in error_str.lower():
                            last_error = f"{symbol}: {error_str}"
                            continue  # Try next symbol
                        else:
                            # Different kind of error, return immediately
                            log_validation_result(validation_result, symbol, "liquidate_dust_position")
                            return {"success": False, "error": f"Validation failed: {error_str}"}
                            
                except ImportError:
                    print("⚠️ Enhanced validation not available, using basic validation")
                    
                    # Find notional filter (lot_size_filter already found above)
                    min_notional_filter = None
                    for f in symbol_info['filters']:
                        if f['filterType'] in ['MIN_NOTIONAL', 'NOTIONAL']:
                            min_notional_filter = f
                            break
                            
                    if lot_size_filter:
                        step_size = float(lot_size_filter['stepSize'])
                        validated_quantity = round(quantity / step_size) * step_size
                    else:
                        validated_quantity = quantity
                        
                    # Check minimum quantity
                    min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
                    if validated_quantity < min_qty:
                        last_error = f"{symbol}: Quantity {validated_quantity:.8f} below minimum {min_qty:.8f}"
                        continue
                    
                    # Check minimum notional value with fresh price
                    if min_notional_filter:
                        try:
                            ticker = client.get_ticker(symbol=symbol)
                            current_price = float(ticker['lastPrice'])
                        except:
                            last_error = f"Could not get price for {symbol}"
                            continue
                            
                        notional_value = validated_quantity * current_price
                        min_notional = float(min_notional_filter['minNotional'])
                        if notional_value < min_notional:
                            last_error = f"{symbol}: Notional ${notional_value:.4f} below minimum ${min_notional:.2f}"
                            continue
                
                # If we get here, validation passed - execute the trade
                print(f"🎯 Executing {symbol} market sell...")
                
                # Format quantity properly
                step_size = float(lot_size_filter['stepSize']) if lot_size_filter else None
                formatted_qty = format_quantity_for_binance(validated_quantity, step_size)
                print(f"   Formatted quantity: {formatted_qty} (from {validated_quantity})")
                
                order = client.order_market_sell(symbol=symbol, quantity=formatted_qty)
                
                # Calculate order details
                avg_price = float(order['fills'][0]['price']) if order['fills'] else 0
                total_value = float(order['cummulativeQuoteQty'])
                total_fee = sum([float(fill['commission']) for fill in order['fills']])
                
                # Initialize BNB conversion result (used in both paths)
                bnb_to_usdt_result = None
                
                # Handle different quote currencies
                if quote_asset == 'BNB':
                    print(f"✅ Dust liquidation successful to BNB!")
                    print(f"   Received: {total_value:.8f} BNB")
                    print(f"   Fee: {total_fee:.8f} {asset}")
                    
                    # Optional: Convert BNB to USDT for consistency
                    try:
                        if total_value > 0.001:  # Only if we got decent BNB amount
                            print(f"🔄 Converting {total_value:.8f} BNB to USDT...")
                            bnb_usdt_ticker = client.get_ticker(symbol='BNBUSDT')
                            bnb_price = float(bnb_usdt_ticker['lastPrice'])
                            
                            # Check if BNB amount meets BNBUSDT minimum requirements
                            bnb_usdt_info = client.get_symbol_info('BNBUSDT')
                            can_convert = True
                            
                            for f in bnb_usdt_info['filters']:
                                if f['filterType'] in ['MIN_NOTIONAL', 'NOTIONAL']:
                                    min_notional = float(f['minNotional'])
                                    bnb_notional = total_value * bnb_price
                                    if bnb_notional < min_notional:
                                        print(f"💰 Keeping {total_value:.8f} BNB (${bnb_notional:.2f} < ${min_notional:.2f} min)")
                                        can_convert = False
                                        break
                            
                            if can_convert:
                                # Format BNB quantity properly
                                formatted_bnb_qty = format_quantity_for_binance(total_value)
                                bnb_order = client.order_market_sell(symbol='BNBUSDT', quantity=formatted_bnb_qty)
                                bnb_usdt_value = float(bnb_order['cummulativeQuoteQty'])
                                bnb_usdt_fee = sum([float(fill['commission']) for fill in bnb_order['fills']])
                                print(f"✅ BNB→USDT conversion successful!")
                                print(f"   Final USDT: ${bnb_usdt_value:.2f}")
                                
                                bnb_to_usdt_result = {
                                    "bnb_quantity": total_value,
                                    "usdt_received": bnb_usdt_value,
                                    "conversion_fee": bnb_usdt_fee
                                }
                    except Exception as e:
                        print(f"⚠️ BNB→USDT conversion failed: {e}")
                        print(f"💰 Keeping {total_value:.8f} BNB")
                else:
                    print(f"✅ Dust liquidation successful!")
                    print(f"   Received: ${total_value:.2f} USDT")
                    print(f"   Fee: ${total_fee:.4f}")

                result = {
                    "success": True,
                    "order_id": order.get('orderId'),
                    "asset": asset,
                    "symbol": symbol,
                    "quantity": validated_quantity,
                    "price": avg_price,
                    "value": total_value,
                    "fee": total_fee,
                    "quote_asset": quote_asset,
                    "timestamp": format_cairo_time()
                }
                
                # Add BNB conversion result if applicable
                if bnb_to_usdt_result:
                    result["bnb_conversion"] = bnb_to_usdt_result
                
                # Log the trade
                log_trade_to_csv({
                    'signal': 'LIQUIDATE_DUST',
                    'symbol': symbol,
                    'quantity': validated_quantity,
                    'price': avg_price,
                    'value': total_value,
                    'fee': total_fee,
                    'status': 'SUCCESS',
                    'timestamp': format_cairo_time()
                }, {
                    'dust_value': dust_position['usdt_value'],
                    'order_type': 'smart_dust_liquidation',
                    'quote_asset': quote_asset,
                    'bnb_conversion': bnb_to_usdt_result is not None
                })
                
                return result
                
            except Exception as e:
                last_error = f"{symbol} execution failed: {e}"
                print(f"❌ {last_error}")
                continue
        
        # If we get here, all symbols failed
        final_error = f"All liquidation attempts failed. Last error: {last_error}"
        print(f"❌ {final_error}")
        
        # Check if all failures were due to notional requirements - use dust conversion instead
        if last_error and ('notional' in last_error.lower() or 'below minimum' in last_error.lower()):
            print(f"💨 Dust position too small for manual trading - attempting dust conversion...")
            
            # Try Binance's dust conversion API as fallback
            try:
                # Check if asset is eligible for dust conversion
                dust_result = client.transfer_dust(asset=asset)
                if dust_result and dust_result.get('transferResult'):
                    bnb_received = float(dust_result.get('totalTransfered', 0))
                    print(f"✅ Dust conversion successful!")
                    print(f"   {asset} → {bnb_received:.8f} BNB via dust conversion")
                    
                    # Log the dust conversion
                    log_trade_to_csv({
                        'signal': 'DUST_CONVERSION',
                        'symbol': f'{asset}_TO_BNB',
                        'quantity': quantity,
                        'price': 0,
                        'value': bnb_received,
                        'fee': 0,
                        'status': 'SUCCESS',
                        'timestamp': format_cairo_time()
                    }, {
                        'dust_value': dust_position['usdt_value'],
                        'order_type': 'binance_dust_conversion',
                        'method': 'transfer_dust_api'
                    })
                    
                    return {
                        "success": True,
                        "method": "dust_conversion",
                        "asset": asset,
                        "quantity": quantity,
                        "bnb_received": bnb_received,
                        "timestamp": format_cairo_time()
                    }
                else:
                    print(f"❌ Dust conversion also failed for {asset}")
                    return {"success": False, "error": f"Dust too small for both trading and conversion: {asset}", "skip_conversion": True}
            except Exception as dust_error:
                if "Insufficient balance" in str(dust_error) or "does not meet the minimum threshold" in str(dust_error):
                    print(f"💨 {asset} dust below conversion threshold")
                    return {"success": False, "error": f"Dust too small: {dust_error}", "skip_conversion": True}
                else:
                    print(f"❌ Dust conversion failed: {dust_error}")
                    return {"success": False, "error": f"All methods failed. Trading: {final_error}, Conversion: {dust_error}"}
        
        return {"success": False, "error": final_error}
        
    except Exception as e:
        error_msg = f"Error liquidating dust {dust_position['asset']}: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "DUST_LIQUIDATION_ERROR", "liquidate_dust_position", "ERROR")
        return {"success": False, "error": str(e)}

def convert_dust_to_bnb():
    """Convert small balances to BNB using Binance's dust conversion feature"""
    try:
        if not client:
            return {"success": False, "error": "Client not initialized"}
        
        print("\n🔄 Converting dust balances to BNB...")
        
        # Get dust balances that can be converted
        account_info = client.get_account()
        convertible_assets = []
        
        for balance in account_info['balances']:
            asset = balance['asset']
            free_balance = float(balance['free'])
            
            # Skip BNB, USDT and zero balances
            if asset in ['BNB', 'USDT'] or free_balance <= 0:
                continue
                
            # Check if asset has very small balance (likely dust)
            if asset in ['BTC', 'ETH', 'XRP', 'ADA'] and free_balance > 0:
                try:
                    ticker = client.get_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['lastPrice'])
                    usdt_value = free_balance * price
                    
                    # If value is very small (under $1), consider for dust conversion
                    if 0 < usdt_value < 1.0:
                        convertible_assets.append(asset)
                        print(f"   💨 {asset}: {free_balance:.8f} (~${usdt_value:.4f})")
                except:
                    continue
        
        if not convertible_assets:
            print("ℹ️ No dust balances found for conversion")
            return {"success": True, "converted_assets": [], "message": "No dust to convert"}
        
        # Execute dust conversion (one asset at a time)
        try:
            print(f"   Attempting to convert: {', '.join(convertible_assets)}")
            converted_assets = []
            total_bnb_received = 0
            
            for asset in convertible_assets:
                try:
                    result = client.transfer_dust(asset=asset)
                    if result and result.get('transferResult'):
                        asset_bnb = float(result.get('totalTransfered', 0))
                        total_bnb_received += asset_bnb
                        converted_assets.append(asset)
                        print(f"   ✅ {asset} → {asset_bnb:.8f} BNB")
                except Exception as e:
                    print(f"   ❌ {asset} conversion failed: {e}")
                    continue
            
            if converted_assets:
                print(f"✅ Dust conversion completed!")
                print(f"   Converted assets: {', '.join(converted_assets)}")
                print(f"   Total BNB received: {total_bnb_received:.8f}")
                
                # Log the conversion
                log_trade_to_csv({
                    'signal': 'DUST_CONVERSION',
                    'symbol': 'DUST_TO_BNB',
                    'quantity': len(converted_assets),
                    'price': 0,
                    'value': total_bnb_received,
                    'fee': 0,
                    'status': 'SUCCESS',
                    'timestamp': format_cairo_time()
                }, {
                    'converted_assets': converted_assets,
                    'total_bnb_received': total_bnb_received
                })
                
                return {
                    "success": True,
                    "converted_assets": converted_assets,
                    "total_bnb_received": total_bnb_received
                }
            else:
                print("ℹ️ No assets could be converted to BNB")
                return {"success": False, "error": "No assets eligible for dust conversion"}
                
        except Exception as e:
            if "Insufficient balance" in str(e) or "does not meet the minimum threshold" in str(e):
                print(f"ℹ️ Dust conversion not available: {e}")
                return {"success": False, "error": f"Dust too small for conversion: {e}"}
            else:
                raise e
                
    except Exception as e:
        error_msg = f"Error converting dust to BNB: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "DUST_CONVERSION_ERROR", "convert_dust_to_bnb", "ERROR")
        return {"success": False, "error": str(e)}

def execute_position_rebalancing():
    """Execute comprehensive position rebalancing based on market conditions"""
    try:
        print("\n" + "="*60)
        print("🔄 POSITION REBALANCING ANALYSIS")
        print("="*60)
        
        rebalancing_results = {
            "timestamp": format_cairo_time(),
            "partial_sells": [],
            "dust_liquidations": [],
            "errors": [],
            "total_freed_usdt": 0
        }
        
        # 1. Check ALL assets with balances for overbought conditions
        if not client:
            rebalancing_results["errors"].append("Client not initialized")
            return rebalancing_results
            
        # Use existing balance summary function to get all assets with balances
        balance_summary = get_account_balances_summary()
        if "error" in balance_summary:
            rebalancing_results["errors"].append(f"Balance summary error: {balance_summary['error']}")
            return rebalancing_results
            
        # Filter to monitored assets with meaningful balances
        monitored_assets = config.REBALANCING.get('assets_to_monitor', [])
        assets_with_balance = []
        
        for asset, balance_info in balance_summary['balances'].items():
            if asset in monitored_assets and balance_info['free'] > 0:
                # Get current price for value check
                try:
                    if asset != 'USDT':  # Skip USDT itself
                        ticker = client.get_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['lastPrice'])
                        usdt_value = balance_info['total'] * price
                        
                        # Only include assets with meaningful value (>$1)
                        if usdt_value >= 1.0:
                            assets_with_balance.append({
                                'asset': asset,
                                'balance': balance_info['free'],
                                'symbol': f"{asset}USDT",
                                'usdt_value': usdt_value
                            })
                except Exception as e:
                    print(f"⚠️ Could not get price for {asset}: {e}")
                    continue
        
        print(f"\n📊 Found {len(assets_with_balance)} assets with balances to check:")
        for asset_info in assets_with_balance:
            print(f"   • {asset_info['asset']}: {asset_info['balance']:.8f} (~${asset_info['usdt_value']:.2f})")
            
        # Check each asset for overbought conditions
        for asset_info in assets_with_balance:
            asset = asset_info['asset']
            symbol = asset_info['symbol']
            balance = asset_info['balance']
            
            print(f"\n🔍 Checking {asset} for overbought conditions...")
            
            # Get RSI for this asset
            asset_rsi = get_position_rsi(symbol, 14)
            
            if asset_rsi is None:
                print(f"⚠️ Could not get RSI for {asset}")
                continue
                
            # Check against overbought threshold
            rsi_threshold = config.REBALANCING.get('rsi_overbought_threshold', 70)
            
            # Check for asset-specific thresholds
            asset_conditions = config.REBALANCING.get('partial_sell_conditions', {})
            if asset in asset_conditions:
                rsi_threshold = asset_conditions[asset].get('rsi_threshold', rsi_threshold)
                sell_percentage = asset_conditions[asset].get('sell_percentage', 40.0)
            else:
                sell_percentage = config.REBALANCING.get('partial_sell_percentage', 40.0)
                
            if asset_rsi >= rsi_threshold:
                print(f"⚠️ {asset} RSI: {asset_rsi:.1f} ≥ {rsi_threshold} - OVERBOUGHT DETECTED!")
                
                # 🎯 ENHANCED PROFITABILITY CHECK
                try:
                    position_tracker = get_position_tracker()
                    smart_profit_taker = get_smart_profit_taker()
                    current_price = float(client.get_ticker(symbol=symbol)['lastPrice'])
                    
                    # Get asset-specific minimum profit percentage
                    asset_conditions = config.REBALANCING.get('partial_sell_conditions', {})
                    if asset in asset_conditions:
                        minimum_profit_pct = asset_conditions[asset].get('minimum_profit_pct', 2.0)
                    else:
                        minimum_profit_pct = config.REBALANCING.get('minimum_profit_pct', 2.0)
                    
                    # Basic profitability check
                    should_sell, reason = position_tracker.should_allow_partial_sell(
                        symbol, current_price, minimum_profit_pct
                    )
                    
                    if not should_sell:
                        print(f"❌ {asset} sell BLOCKED: {reason}")
                        print(f"   RSI ({asset_rsi:.1f}) suggests sell, but profit check prevents loss")
                        continue
                    
                    # Get position info for smart profit analysis
                    position_info = position_tracker.get_position_info(symbol)
                    if position_info:
                        # Enhanced profit-taking analysis
                        should_take_profits, profit_analysis = smart_profit_taker.should_take_profits(
                            symbol, current_price, position_info['avg_buy_price'],
                            minimum_profit_pct, rsi=asset_rsi
                        )
                        
                        if should_take_profits:
                            # Use smart profit taker's recommended percentage
                            recommended_pct = profit_analysis['recommendation']['sell_percentage']
                            if recommended_pct > 0:
                                sell_percentage = min(sell_percentage, recommended_pct)
                                print(f"🎯 {asset} SMART PROFIT TAKING:")
                                print(f"   📊 Analysis: {profit_analysis['reason']}")
                                print(f"   💰 Profit: {profit_analysis['profit_pct']:.2f}% (${profit_analysis['profit_usd_per_unit']:.4f}/unit)")
                                print(f"   📈 Action: {profit_analysis['decision']} ({sell_percentage}%)")
                                print(f"   🎚️ Confidence: {profit_analysis['recommendation']['confidence']:.1%}")
                            else:
                                print(f"⏳ {asset} smart analysis suggests HOLD despite RSI")
                                continue
                        else:
                            print(f"⏳ {asset} smart profit analysis suggests HOLD: {profit_analysis['reason']}")
                            continue
                    else:
                        print(f"✅ {asset} basic profit check passed: {reason}")
                    
                except Exception as e:
                    print(f"⚠️ Error in smart profit analysis for {asset}: {e}")
                    print(f"   🛡️ BLOCKING sell due to analysis failure (safety first)")
                    continue
                
                # Check if we should preserve core holdings
                preserve_holdings = config.REBALANCING.get('preserve_core_holdings', {})
                min_preserve = preserve_holdings.get(asset, 0)
                
                if balance <= min_preserve:
                    print(f"🛡️ {asset} balance ({balance:.8f}) at or below preservation limit ({min_preserve})")
                    continue
                    
                # Calculate sellable amount (preserve minimum if specified)
                sellable_balance = balance - min_preserve
                if sellable_balance <= 0:
                    print(f"ℹ️ No sellable {asset} after preserving minimum holdings")
                    continue
                    
                # Adjust sell percentage based on preservation requirements
                max_sell_qty = sellable_balance * (sell_percentage / 100.0)
                
                print(f"💡 Selling {sell_percentage}% of available {asset}")
                print(f"   Available: {sellable_balance:.8f} {asset}")
                print(f"   Will sell: {max_sell_qty:.8f} {asset}")
                print(f"   Will preserve: {min_preserve:.8f} {asset}")
                
                # Execute partial sell with notification disabled (summary will be sent instead)
                result = sell_partial_position(
                    symbol,
                    (max_sell_qty / balance) * 100,  # Convert back to percentage of total balance
                    f"RSI {asset_rsi:.1f} overbought signal (threshold: {rsi_threshold})",
                    send_notification=False  # Disable individual notification - summary will be sent
                )
                
                if result["success"]:
                    rebalancing_results["partial_sells"].append(result)
                    rebalancing_results["total_freed_usdt"] += result["value"]
                    print(f"✅ {asset} partial sell completed: ${result['value']:.2f}")
                else:
                    rebalancing_results["errors"].append(f"{asset} partial sell failed: {result['error']}")
                    
            else:
                print(f"✅ {asset} RSI: {asset_rsi:.1f} < {rsi_threshold} - Within normal range")
            
        # 2. Detect and liquidate dust positions using smart minimum calculation
        dust_positions = detect_dust_positions()  # No hardcoded minimum - uses smart MIN_NOTIONAL calculation
        
        if dust_positions:
            print(f"\n💨 Found {len(dust_positions)} dust positions to liquidate")
            
            for dust_pos in dust_positions:
                result = liquidate_dust_position(dust_pos)
                
                if result["success"]:
                    rebalancing_results["dust_liquidations"].append(result)
                    
                    # Handle different result structures (trading vs dust conversion)
                    if result.get("method") == "dust_conversion":
                        # Dust conversion result - estimate USD value from BNB
                        bnb_value = result.get("bnb_received", 0)
                        try:
                            bnb_ticker = client.get_ticker(symbol='BNBUSDT')
                            bnb_price = float(bnb_ticker['lastPrice'])
                            estimated_usd = bnb_value * bnb_price
                            rebalancing_results["total_freed_usdt"] += estimated_usd
                            print(f"✅ Dust converted {dust_pos['asset']}: {bnb_value:.8f} BNB (~${estimated_usd:.2f})")
                        except:
                            print(f"✅ Dust converted {dust_pos['asset']}: {bnb_value:.8f} BNB")
                    else:
                        # Regular trading result
                        value = result.get("value", 0)
                        rebalancing_results["total_freed_usdt"] += value
                        print(f"✅ Liquidated {dust_pos['asset']}: ${value:.2f}")
                else:
                    # Check if this is a "too small" error that should be skipped
                    if result.get("skip_conversion", False):
                        print(f"💨 Skipping {dust_pos['asset']}: Too small for liquidation (will try dust conversion)")
                    else:
                        rebalancing_results["errors"].append(f"Failed to liquidate {dust_pos['asset']}: {result['error']}")
                        print(f"❌ Failed to liquidate {dust_pos['asset']}: {result['error']}")
            
            # Try dust conversion for any remaining small balances (excluding BNB itself)
            print(f"\n🔄 Attempting dust conversion for remaining small balances...")
            dust_conversion_result = convert_dust_to_bnb()
            
            if dust_conversion_result["success"] and dust_conversion_result.get("converted_assets"):
                rebalancing_results["dust_conversions"] = dust_conversion_result["converted_assets"]
                print(f"✅ Converted {len(dust_conversion_result['converted_assets'])} assets to BNB")
            elif not dust_conversion_result["success"] and "too small" in dust_conversion_result.get("error", "").lower():
                print(f"💨 Dust conversion skipped: {dust_conversion_result['error']}")
            elif not dust_conversion_result["success"] and "no dust to convert" in dust_conversion_result.get("message", "").lower():
                print(f"ℹ️ No eligible assets found for dust conversion (BNB is excluded as target currency)")
            else:
                print(f"⚠️ Dust conversion failed: {dust_conversion_result.get('error', 'Unknown error')}")
            
        else:
            print("ℹ️ No dust positions found (BNB dust is excluded - it's the target currency for dust conversion)")
        
        # Initialize dust_conversions if not set
        if 'dust_conversions' not in rebalancing_results:
            rebalancing_results['dust_conversions'] = []
            
        # 4. Summary
        print("\n" + "="*60)
        print("📊 REBALANCING SUMMARY")
        print("="*60)
        print(f"Partial Sells: {len(rebalancing_results['partial_sells'])}")
        print(f"Dust Liquidations: {len(rebalancing_results['dust_liquidations'])}")
        print(f"Dust Conversions: {len(rebalancing_results['dust_conversions'])}")
        print(f"Total USDT Freed: ${rebalancing_results['total_freed_usdt']:.2f}")
        print(f"Errors: {len(rebalancing_results['errors'])}")
        
        if rebalancing_results['errors']:
            print("\n⚠️ Errors encountered:")
            for error in rebalancing_results['errors']:
                print(f"❌ {error}")
        
        if rebalancing_results['dust_conversions']:
            print(f"\n🔄 Dust converted to BNB: {', '.join(rebalancing_results['dust_conversions'])}")
                
        # Send Telegram summary if available
        if TELEGRAM_AVAILABLE and (rebalancing_results['partial_sells'] or rebalancing_results['dust_liquidations'] or rebalancing_results['dust_conversions']):
            try:
                # Prepare proper notification with required parameters
                summary_msg = f"🔄 Position Rebalancing Complete\n"
                summary_msg += f"💰 Total USDT Freed: ${rebalancing_results['total_freed_usdt']:.2f}\n"
                summary_msg += f"📈 Partial Sells: {len(rebalancing_results['partial_sells'])}\n"
                summary_msg += f"💨 Dust Liquidated: {len(rebalancing_results['dust_liquidations'])}\n"
                summary_msg += f"🔄 Dust Converted: {len(rebalancing_results['dust_conversions'])}"
                
                # Use telegram notifier directly for rebalancing notifications
                from telegram_notify import telegram_notifier
                
                # Send rebalancing notification using the proper send_message method
                success = telegram_notifier.send_message(summary_msg)
                if success:
                    print("📱 Rebalancing summary sent via Telegram")
                else:
                    print("⚠️ Telegram notification was queued or failed (check connection)")
                
            except Exception as telegram_error:
                print(f"⚠️ Telegram notification failed: {telegram_error}")
                log_error_to_csv(f"Telegram rebalancing notification failed: {telegram_error}", 
                               "TELEGRAM_ERROR", "execute_position_rebalancing", "WARNING")
            
        return rebalancing_results
        
    except Exception as e:
        error_msg = f"Error in position rebalancing: {e}"
        print(f"❌ {error_msg}")
        log_error_to_csv(error_msg, "REBALANCING_ERROR", "execute_position_rebalancing", "ERROR")
        return {
            "timestamp": format_cairo_time(),
            "partial_sells": [],
            "dust_liquidations": [],
            "errors": [error_msg],
            "total_freed_usdt": 0
        }

# Add Flask route for manual rebalancing trigger
@app.route('/api/rebalance', methods=['POST'])
def api_rebalance():
    """API endpoint to trigger position rebalancing"""
    try:
        if not client:
            return jsonify({"error": "Client not initialized"}), 500
            
        print("\n🔄 Manual rebalancing triggered via API")
        results = execute_position_rebalancing()
        
        return jsonify({
            "success": True,
            "message": "Rebalancing completed",
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": format_cairo_time()
        }), 500

# ML Training Scheduler API Routes
@app.route('/api/ml-training/status', methods=['GET'])
def api_ml_training_status():
    """Get ML training status from incremental learning system"""
    try:
        from incremental_learning import IncrementalMLTrainer
        trainer = IncrementalMLTrainer()
        
        # Get training report for all models
        status = {
            'scheduler_running': True,  # Always running with incremental
            'training_enabled': True,   # Always enabled with incremental
            'training_in_progress': False
        }
        
        # Add model-specific stats
        stats = trainer.get_cumulative_stats()
        for model_name in ['trend', 'signal', 'regime']:
            model_stats = stats.get(model_name, {})
            status[f'{model_name}_model'] = {
                'samples_trained': model_stats.get('total_samples_seen', 0),
                'training_sessions': model_stats.get('training_sessions', 0),
                'last_trained': model_stats.get('last_trained'),
            }
            
            # Add accuracy from latest training if available
            curve = trainer.get_learning_curve(model_name)
            if curve:
                status[f'{model_name}_model']['current_accuracy'] = curve[-1].get('accuracy')
        
        return jsonify({
            "success": True,
            "status": status,
            "timestamp": format_cairo_time()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": format_cairo_time()
        }), 500

@app.route('/api/ml-training/force', methods=['POST'])
def api_force_ml_training():
    """Force immediate incremental ML training execution"""
    try:
        # Import and run enhanced ML training
        from enhanced_ml_training import EnhancedMLTrainer
        trainer = EnhancedMLTrainer(use_incremental=True)
        
        # Execute training in incremental mode 
        results = trainer.train_all_models(incremental=True)
        
        return jsonify({
            "success": True,
            "message": "Incremental ML training executed",
            "results": results,
            "timestamp": format_cairo_time()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": format_cairo_time()
        }), 500

if __name__ == '__main__':
    # EMERGENCY CHECK: Prevent startup if API ban is active
    try:
        from datetime import datetime
        import config
        
        # Check if we're in emergency mode due to API ban
        if hasattr(config, 'EMERGENCY_MODE') and config.EMERGENCY_MODE:
            print("\nEMERGENCY MODE DETECTED - Ultra-conservative rate limits active")
            print("=" * 60)
            
            # Check if ban has been lifted
            current_time = datetime.now().timestamp() * 1000
            ban_until = 1760962139050  # Updated ban timestamp (THIRD ban)
            
            if current_time < ban_until:
                ban_lift_time = datetime.fromtimestamp(ban_until / 1000)
                minutes_remaining = (ban_until - current_time) / 1000 / 60
                print(f"API BAN STILL ACTIVE")
                print(f"Ban will lift at: {ban_lift_time.strftime('%H:%M:%S')}")
                print(f"Time remaining: {minutes_remaining:.1f} minutes")
                print("\nUse post_ban_launcher.py to start safely after ban lifts")
                print("   Or wait and run this script again after the ban time")
                exit(1)
            else:
                print("API ban has been lifted - proceeding with emergency rate limits")
                print("Starting with ultra-conservative settings (30 calls/min)")
    except Exception as e:
        print(f"Warning: Could not check emergency status: {e}")
    
    print("\n🚀 Starting CRYPTIX AI Trading Bot...")
    print("=" * 50)
    
    # Initialize bot systems
    try:
        # Initialize CSV logging system
        print("📊 Initializing CSV logging system...")
        try:
            csv_files = setup_csv_logging()
            print(f"✅ CSV files initialized: {', '.join(csv_files.keys())}")
        except Exception as e:
            print(f"⚠️ CSV logging initialization warning: {e}")
        
        # Initialize API client once at startup
        if not bot_status.get('api_connected', False):
            print("🔧 Initializing API client...")
            if not initialize_client():
                # Check if this is due to emergency mode
                try:
                    import config
                    if hasattr(config, 'EMERGENCY_MODE') and config.EMERGENCY_MODE:
                        print("⚠️ Trading client initialization failed - running in demo mode")
                        print("🛡️ Emergency mode active - API calls disabled during ban period")
                        bot_status['demo_mode'] = True
                        bot_status['api_connected'] = False
                        # Continue startup in demo mode
                    else:
                        print("❌ Failed to initialize API client at startup")
                        raise RuntimeError("API client init failed")
                except Exception as e:
                    print("❌ Failed to initialize API client at startup")
                    raise RuntimeError("API client init failed")

        # Verify incremental ML system is available
        try:
            from incremental_learning import IncrementalMLTrainer
            print("🤖 Verifying incremental learning system...")
            trainer = IncrementalMLTrainer()
            stats = trainer.get_cumulative_stats()
            # Print summary of current state
            for model_name in ['trend', 'signal', 'regime']:
                if model_name in stats:
                    print(f"✅ {model_name.upper()}: {stats[model_name]['total_samples_seen']:,} samples trained")
            print("🎯 Incremental learning system ready")
        except Exception as e:
            print(f"⚠️ Failed to initialize incremental learning: {e}")
            # Don't fail the entire app if ML system isn't available

        # Configure Flask for production by default; override with FLASK_DEBUG=1
        flask_host = os.getenv('FLASK_HOST', '0.0.0.0')
        flask_port = int(os.getenv('PORT', 5000))  # Use PORT env var for cloud platforms, default to 5000
        flask_debug = str(os.getenv('FLASK_DEBUG', '0')).strip().lower() in ['1', 'true', 'yes', 'on']
        print(f"🌐 Starting Flask server on {flask_host}:{flask_port} (debug={'ON' if flask_debug else 'OFF'})")
        app.run(host=flask_host, port=flask_port, debug=flask_debug)
    except Exception as e:
        print(f"Failed to start application: {e}")
        log_error_to_csv(str(e), "STARTUP_ERROR", "main", "CRITICAL")
        
