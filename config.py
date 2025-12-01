# Trading Bot Configuration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Trading Symbols Configuration
SYMBOLS = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT').split(',')

# Binance API Configuration (loaded from environment variables)
API_KEY = os.getenv('API_KEY', '')
API_SECRET = os.getenv('API_SECRET', '')

# Also support old naming convention
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', API_KEY)
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', API_SECRET)

# Environment Configuration
USE_TESTNET = os.getenv('USE_TESTNET', 'false').lower() in ('true', '1', 'yes')

# Risk Management Settings (can be overridden by environment variables)
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '2.0'))  # Percentage of total balance to risk per trade
MIN_TRADE_USDT = float(os.getenv('MIN_TRADE_USDT', '5.0'))  # Base minimum trade size in USDT (fallback/safety minimum)
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '15.0'))  # Maximum drawdown percentage allowed

# Trading Strategy Parameters
RSI_PERIOD = 14  # Standard RSI period
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_THRESHOLD = 1000000  # Minimum 24h volume in USDT

# Time Filters
AVOID_TRADING_HOURS = [0, 1, 2, 3]  # Hours to avoid trading (UTC)

# Position Sizing
DEFAULT_QUANTITY = 0.001  # Default fallback quantity
POSITION_SIZE_ADJUSTMENTS = {
    'volatility_factor': 1.0,  # Reduce position size in high volatility
    'trend_factor': 1.2,  # Increase position size in strong trends
}

# Enhanced Position Sizing Configuration
MAX_POSITION_PCT = float(os.getenv('MAX_POSITION_PCT', '10.0'))  # Maximum single position size
ESTIMATED_WIN_RATE = float(os.getenv('ESTIMATED_WIN_RATE', '0.55'))  # For Kelly Criterion
ESTIMATED_AVG_WIN = float(os.getenv('ESTIMATED_AVG_WIN', '1.5'))  # Average win percentage
ESTIMATED_AVG_LOSS = float(os.getenv('ESTIMATED_AVG_LOSS', '1.0'))  # Average loss percentage
MAX_PORTFOLIO_HEAT = int(os.getenv('MAX_PORTFOLIO_HEAT', '5'))  # Max concurrent positions

# Aggressive Position Sizing for Larger Trades
POSITION_SIZING = {
    'default_risk_pct': 5.0,      # 5% risk = ~$45 trades (increased from 1.15%)
    'max_risk_pct': 10.0,         # 10% max = ~$90 trades (increased from 2.0%)
    'adaptive_sizing': True,      # Enable dynamic sizing
    'confidence_based': True      # Scale with signal confidence
}

# Dynamic Sizing Based on Signal Confidence
DYNAMIC_SIZING = {
    'base_amount': 50,           # Start with $50 instead of $10
    'confidence_multiplier': 3.0, # Up to 3x for high confidence
    'max_amount': 200,           # Cap at $200 per trade
    'adaptive_to_balance': True, # Scale down based on available balance
    'min_balance_ratio': 0.8,   # Use up to 80% of available balance if needed
    'fallback_percentage': 60,  # If can't afford base_amount, use 60% of balance
    # No min_amount - respect exchange minimums only
}

# Enhanced Signal Filtering Configuration
SIGNAL_NOISE_THRESHOLD = float(os.getenv('SIGNAL_NOISE_THRESHOLD', '0.7'))  # Signal quality threshold
MIN_VOLUME_RATIO = float(os.getenv('MIN_VOLUME_RATIO', '1.2'))  # Minimum volume vs average
MAX_PRICE_CHANGE_FILTER = float(os.getenv('MAX_PRICE_CHANGE_FILTER', '15.0'))  # Max 24h price change
MIN_VOLUME_FILTER = float(os.getenv('MIN_VOLUME_FILTER', '1000000'))  # Minimum 24h volume

# Enhanced Risk Management Configuration
BASE_STOP_LOSS_PCT = float(os.getenv('BASE_STOP_LOSS_PCT', '3.0'))  # Base stop loss percentage
ATR_STOP_MULTIPLIER = float(os.getenv('ATR_STOP_MULTIPLIER', '2.5'))  # ATR multiplier for stops
EMERGENCY_STOP_CONDITIONS = {
    'market_crash_threshold': 20.0,  # Market drop % to trigger emergency stop
    'flash_crash_threshold': 10.0,   # 1-hour drop % to trigger emergency stop
    'extreme_volatility_threshold': 2.0,  # Volatility threshold for emergency stop
    'max_api_errors_per_hour': 50    # API error threshold
}

# Trading Pairs
DEFAULT_PAIR = "BTCUSDT"
MONITORED_BASE_ASSETS = ["BTC", "ETH", "BNB", "XRP", "SOL", "DOT", "ADA"]
QUOTE_ASSET = "USDT"

# Technical Analysis
PERIOD_FAST = 5   # Fast moving average period
PERIOD_SLOW = 20  # Slow moving average period
ATR_PERIOD = 14   # Average True Range period

# Advanced Indicator Settings (new)
EMA_PERIODS = {
    'fast': 12,
    'slow': 26,
    'mid': 50,
    'long': 200
}

STOCH = {
    'k_period': 14,
    'd_period': 3,
    'overbought': 80,
    'oversold': 20
}

VWAP = {
    # Rolling window (in candles) for VWAP approximation; session VWAP not available in this context
    'window': 20
}

ADX = {
    'period': 14,
    'min_trend': 20
}

# Statistical Parameters
ZSCORE_THRESHOLD = 2.0  # Z-score threshold for statistical signals
VAR_CONFIDENCE = 0.95   # Value at Risk confidence level

ADAPTIVE_STRATEGY = {
    'score_threshold': 30,
    'volatility_adjustment': True,
    'trend_following': True,
    # Weights for composite scoring (sum ~ 1.0)
    'weights': {
        'rsi': 0.2,
        'macd': 0.2,
        'ema_trend': 0.15,
        'stoch': 0.15,
        'adx': 0.15,
        'vwap': 0.15
    },
    'adx_min': 20
}

# Performance Tracking
MAX_TRADES_HISTORY = 100  # Number of recent trades to keep in memory
PERFORMANCE_METRICS = {
    'win_rate_min': 50.0,     # Minimum win rate percentage
    'profit_factor_min': 1.5,  # Minimum profit factor
    'max_consecutive_losses': 3 # Maximum consecutive losing trades
}

# Daily Risk Limits (can be overridden by environment variables)
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '50.0'))  # Maximum daily loss in USD
MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '5'))  # Stop trading after this many losses
MAX_PORTFOLIO_EXPOSURE = float(os.getenv('MAX_PORTFOLIO_EXPOSURE', '80.0'))  # Maximum percentage of portfolio at risk

# Intelligent Timing System (AI TRADING WOLF)
TIMING_SYSTEM = {
    'base_interval': 300,           # Base scanning interval (5 minutes)
    'regime_check_interval': 300,   # Market regime check frequency (5 minutes)
    'breakout_scan_threshold': 40,  # Minimum score for breakout opportunities
    'hunting_mode_triggers': 3,     # Number of triggers needed for hunting mode
    'max_quick_scans': 5,          # Maximum quick scans before full scan
    'volatility_thresholds': {
        'extreme': 1.2,            # Lowered threshold for extreme regime
        'volatile': 0.6,           # Lowered threshold for volatile regime
        'quiet': 0.2               # Lowered threshold for quiet regime
    },
    'volume_surge_thresholds': {
        'extreme': 3.0,            # Volume surge multiplier for extreme conditions
        'volatile': 2.0,           # Volume surge multiplier for volatile conditions
        'significant': 1.5         # Volume surge multiplier for significant activity
    }
}

# Market Hours for Enhanced Timing
MARKET_HOURS = {
    'us_market': list(range(16, 24)) + list(range(0, 1)),  # 2:30 PM - 11 PM Cairo time
    'asian_market': list(range(2, 10)),                     # 2 AM - 10 AM Cairo time
    'european_market': list(range(10, 18)),                 # 10 AM - 6 PM Cairo time
    'high_activity_hours': list(range(14, 23)),             # Peak trading hours
}

# Auto Trading Settings
AUTO_TRADING = True  # Enable automatic trade execution

# API Rate Limiting (EMERGENCY - Ultra Conservative)
EMERGENCY_MODE = True  # Flag for emergency recovery mode
API_RATE_LIMITS = {
    'calls_per_minute': 0,     # ZERO API calls during ban
    'calls_per_second': 0,     # ZERO API calls during ban  
    'weight_per_minute': 0     # ZERO weight during ban
}
DELAY_BETWEEN_CALLS = 10.0  # Emergency delay between API calls

# Simple WebSocket price fallback (emergency only)
def get_websocket_price(symbol):
    """Simple WebSocket price getter for emergency use (avoid REST API calls)"""
    try:
        import websocket
        import json
        import threading
        
        price_cache = {}
        
        def on_message(ws, message):
            data = json.loads(message)
            if 's' in data and 'c' in data:  # symbol and close price
                price_cache[data['s']] = float(data['c'])
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        # Quick price fetch for emergency use
        stream = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
        ws = websocket.WebSocketApp(stream, on_message=on_message, on_error=on_error)
        
        # Run for a few seconds to get price
        def run_ws():
            ws.run_forever()
        
        thread = threading.Thread(target=run_ws, daemon=True)
        thread.start()
        thread.join(timeout=5)  # Wait max 5 seconds
        
        return price_cache.get(symbol.upper())
        
    except Exception as e:
        print(f"WebSocket fallback failed: {e}")
        return None

# Position Rebalancing Configuration
REBALANCING = {
    'enabled': True,                    # Enable automatic rebalancing
    'check_interval': 3600,             # Check every hour (seconds)
    'rsi_overbought_threshold': 70,     # RSI threshold for partial sells
    'rsi_oversold_threshold': 30,       # RSI threshold for additional buys
    'partial_sell_percentage': 40.0,    # Default percentage for partial sells
    'minimum_profit_pct': 2.0,          # 🎯 Minimum profit percentage to allow sells
    'profit_taking_levels': {           # Smart profit taking at different levels
        'conservative': 2.0,            # Conservative: 2% profit
        'moderate': 5.0,                # Moderate: 5% profit  
        'aggressive': 10.0,             # Aggressive: 10% profit
    },
    'dust_threshold_usdt': 5.0,         # Minimum position value (below = dust)
    'max_dust_liquidations_per_run': 5, # Limit dust liquidations per run
    'assets_to_monitor': [              # Assets to monitor for rebalancing
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOT', 'AVAX', 'MATIC', 'LINK'
    ],
    'partial_sell_conditions': {
        'SOL': {'rsi_threshold': 70, 'sell_percentage': 40, 'minimum_profit_pct': 2.0},
        'BTC': {'rsi_threshold': 75, 'sell_percentage': 30, 'minimum_profit_pct': 1.5},
        'ETH': {'rsi_threshold': 72, 'sell_percentage': 35, 'minimum_profit_pct': 2.0},
        'BNB': {'rsi_threshold': 70, 'sell_percentage': 40, 'minimum_profit_pct': 2.5},
        'AVAX': {'rsi_threshold': 70, 'sell_percentage': 40, 'minimum_profit_pct': 3.0},
        'LINK': {'rsi_threshold': 70, 'sell_percentage': 40, 'minimum_profit_pct': 2.5},
    },
    'preserve_core_holdings': {         # Keep minimum holdings
        'BTC': 0.001,                   # Always keep at least 0.001 BTC
        'ETH': 0.01,                    # Always keep at least 0.01 ETH
        'BNB': 0.1,                     # Always keep at least 0.1 BNB
    }
}

# Binance environment configuration is now loaded from .env file

# Telegram Notification Settings
TELEGRAM = {
    'enabled': True,  # Enable/disable Telegram notifications
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),  # Load from environment variables
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),      # Load from environment variables
    'notifications': {
        'signals': True,           # Send trading signal notifications
        'trades': True,            # Send trade execution notifications
        'errors': True,            # Send error notifications
        'daily_summary': True,     # Send daily performance summary
        'bot_status': True         # Send bot start/stop notifications
    },
    'message_format': {
        'include_emoji': True,     # Include emojis in messages
        'include_price': True,     # Include current price in messages
        'include_indicators': True, # Include technical indicators
        'include_profit_loss': True # Include P&L information
    },
    'rate_limiting': {
        'max_messages_per_minute': 20,  # Rate limit to avoid spam
        'batch_notifications': True      # Batch similar notifications
    }
}

# Simple toggle for sending signal notifications (BUY/SELL); trades remain enabled
TELEGRAM_SEND_SIGNALS = False

# Machine Learning Integration Configuration
ML_ENABLED = os.getenv('ML_ENABLED', 'true').lower() == 'true'
ML_PREDICTION_ENABLED = os.getenv('ML_PREDICTION_ENABLED', 'true').lower() == 'true'
ML_MODEL_RETRAIN_DAYS = int(os.getenv('ML_MODEL_RETRAIN_DAYS', '7'))
ML_LOOKBACK_DAYS = int(os.getenv('ML_LOOKBACK_DAYS', '30'))
ML_MIN_TRAINING_SAMPLES = int(os.getenv('ML_MIN_TRAINING_SAMPLES', '100'))

# Pattern Recognition Configuration
PATTERN_RECOGNITION_ENABLED = os.getenv('PATTERN_RECOGNITION_ENABLED', 'true').lower() == 'true'
PATTERN_LOOKBACK_PERIODS = int(os.getenv('PATTERN_LOOKBACK_PERIODS', '50'))
PATTERN_SIMILARITY_THRESHOLD = float(os.getenv('PATTERN_SIMILARITY_THRESHOLD', '0.7'))
PATTERN_CONFIDENCE_THRESHOLD = float(os.getenv('PATTERN_CONFIDENCE_THRESHOLD', '0.6'))

# Market Regime Detection Configuration
REGIME_DETECTION_ENABLED = os.getenv('REGIME_DETECTION_ENABLED', 'true').lower() == 'true'
REGIME_VOLATILITY_LOOKBACK = int(os.getenv('REGIME_VOLATILITY_LOOKBACK', '20'))
REGIME_TREND_LOOKBACK = int(os.getenv('REGIME_TREND_LOOKBACK', '50'))
REGIME_CONFIDENCE_THRESHOLD = float(os.getenv('REGIME_CONFIDENCE_THRESHOLD', '0.7'))

# Adaptive Thresholds Configuration
ADAPTIVE_THRESHOLDS_ENABLED = os.getenv('ADAPTIVE_THRESHOLDS_ENABLED', 'true').lower() == 'true'
BASE_RSI_OVERSOLD = float(os.getenv('BASE_RSI_OVERSOLD', '25'))
BASE_RSI_OVERBOUGHT = float(os.getenv('BASE_RSI_OVERBOUGHT', '70'))
BASE_MACD_THRESHOLD = float(os.getenv('BASE_MACD_THRESHOLD', '0.001'))
ADAPTIVE_ADJUSTMENT_FACTOR = float(os.getenv('ADAPTIVE_ADJUSTMENT_FACTOR', '0.2'))

# ML Model Performance Thresholds
ML_MIN_ACCURACY = float(os.getenv('ML_MIN_ACCURACY', '0.6'))
ML_REGIME_MIN_ACCURACY = float(os.getenv('ML_REGIME_MIN_ACCURACY', '0.55'))
ML_PATTERN_MIN_ACCURACY = float(os.getenv('ML_PATTERN_MIN_ACCURACY', '0.65'))

# Market Intelligence Scoring
INTELLIGENCE_CONFIDENCE_THRESHOLD = float(os.getenv('INTELLIGENCE_CONFIDENCE_THRESHOLD', '0.7'))
PATTERN_WEIGHT = float(os.getenv('PATTERN_WEIGHT', '0.3'))
REGIME_WEIGHT = float(os.getenv('REGIME_WEIGHT', '0.4'))
SIGNAL_WEIGHT = float(os.getenv('SIGNAL_WEIGHT', '0.3'))

# ML Training Schedule
ML_TRAINING_START_HOUR = int(os.getenv('ML_TRAINING_START_HOUR', '2'))  # 2 AM
ML_TRAINING_ENABLED = os.getenv('ML_TRAINING_ENABLED', 'true').lower() == 'true'
AUTO_MODEL_UPDATE = os.getenv('AUTO_MODEL_UPDATE', 'true').lower() == 'true'

# Advanced ML Features
ML_ENSEMBLE_ENABLED = os.getenv('ML_ENSEMBLE_ENABLED', 'true').lower() == 'true'
ML_FEATURE_SELECTION = os.getenv('ML_feature_SELECTION', 'true').lower() == 'true'
ML_HYPERPARAMETER_TUNING = os.getenv('ML_HYPERPARAMETER_TUNING', 'false').lower() == 'true'
ML_CROSS_VALIDATION_FOLDS = int(os.getenv('ML_CROSS_VALIDATION_FOLDS', '5'))

# Market Intelligence Configuration
MARKET_INTELLIGENCE = {
    'enabled': True,
    'analysis_interval': 300,  # 5 minutes
    'pattern_memory_days': 30,
    'regime_history_periods': 100,
    'adaptive_learning': True,
    'confidence_weighting': True,
    'ensemble_predictions': True
}

print(f"ML Features: Enabled={ML_ENABLED}, Pattern Recognition={PATTERN_RECOGNITION_ENABLED}")
print(f"Regime Detection: {REGIME_DETECTION_ENABLED}, Adaptive Thresholds: {ADAPTIVE_THRESHOLDS_ENABLED}")
print(f"Market Intelligence: {MARKET_INTELLIGENCE['enabled']}")

# ML Pure Trading Strategy Configuration
ML_STRATEGY = {
    'enabled': ML_ENABLED,
    'target_daily_return': 10.0,  # Target 10%+ daily returns
    'min_confidence_threshold': 0.65,
    'high_confidence_threshold': 0.8,
    'max_position_multiplier': 2.0,
    'buy_low_threshold': 0.3,  # Buy when price in bottom 30% of range
    'sell_high_threshold': 0.7,  # Sell when price in top 30% of range
    'regime_adjustments': {
        'BULLISH_EXTREME': {'position_multiplier': 1.5, 'confidence_threshold': 0.7},
        'BULLISH': {'position_multiplier': 1.2, 'confidence_threshold': 0.65},
        'NEUTRAL': {'position_multiplier': 1.0, 'confidence_threshold': 0.65},
        'BEARISH': {'position_multiplier': 0.8, 'confidence_threshold': 0.7},
        'BEARISH_EXTREME': {'position_multiplier': 0.5, 'confidence_threshold': 0.8}
    },
    'profitability_validation': True,
    'smart_timing': True,
    'adaptive_position_sizing': True
}

# Trading Strategy Selection
DEFAULT_STRATEGY = os.getenv('DEFAULT_STRATEGY', 'ML_PURE')  # Can be ADAPTIVE or ML_PURE
