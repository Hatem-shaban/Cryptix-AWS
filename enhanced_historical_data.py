"""
Enhanced Historical Data Fetcher for ML Training
This module fetches comprehensive historical data from Binance for training ML models
with extensive technical indicators and market regime features.
"""

import ccxt
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import time
import os
import logging
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedHistoricalDataFetcher:
    """Enhanced historical data fetcher with ML-focused features"""
    
    def __init__(self, symbols: List[str] = None, timeframes: List[str] = None):
        """
        Initialize the enhanced data fetcher
        
        Args:
            symbols: List of trading symbols (default: major crypto pairs)
            timeframes: List of timeframes to fetch (default: multiple timeframes)
        """
        self.symbols = symbols or [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT",
            "MATIC/USDT", "DOT/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT",
            "ATOM/USDT", "LTC/USDT", "BCH/USDT", "FIL/USDT", "TRX/USDT"
        ]
        
        self.timeframes = timeframes or ['1h', '4h', '1d']
        self.exchange = ccxt.binance({
            'rateLimit': 1200,  # Be respectful to API limits
            'enableRateLimit': True,
        })
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
    
    def fetch_incremental_data(self, symbol: str, timeframe: str, 
                              existing_df: pd.DataFrame = None, 
                              max_days_gap: int = 7) -> pd.DataFrame:
        """
        Fetch only incremental data since last update
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe 
            existing_df: Existing DataFrame to append to
            max_days_gap: Maximum days to fetch incrementally (fallback to full if more)
            
        Returns:
            Updated DataFrame with new data
        """
        try:
            if existing_df is None or existing_df.empty:
                # No existing data, fetch limited recent data
                return self.fetch_historical_ohlcv(symbol, timeframe, days_back=90)
            
            # Get the latest timestamp from existing data
            last_timestamp = pd.to_datetime(existing_df['timestamp']).max()
            days_since_last = (datetime.now() - last_timestamp).days
            
            if days_since_last <= max_days_gap:
                # Fetch only incremental data
                logger.info(f"Fetching {days_since_last} days of incremental data for {symbol}")
                new_data = self.fetch_historical_ohlcv(symbol, timeframe, days_back=days_since_last + 1)
                
                # Merge and deduplicate
                if not new_data.empty:
                    combined = pd.concat([existing_df, new_data], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
                    combined = combined.sort_values('timestamp').reset_index(drop=True)
                    return combined
                else:
                    return existing_df
            else:
                # Gap too large, fetch fresh data
                logger.info(f"Gap of {days_since_last} days too large, fetching fresh 90-day data")
                return self.fetch_historical_ohlcv(symbol, timeframe, days_back=90)
                
        except Exception as e:
            logger.error(f"Error in incremental fetch: {e}")
            return existing_df if existing_df is not None else pd.DataFrame()

    def fetch_historical_ohlcv(self, symbol: str, timeframe: str, 
                             days_back: int = 365) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a specific symbol and timeframe
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            days_back: Number of days to fetch (default: 1 year)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate since timestamp
            since = self.exchange.milliseconds() - (days_back * 24 * 60 * 60 * 1000)
            all_ohlcv = []
            
            logger.info(f"Fetching {symbol} {timeframe} data for {days_back} days...")
            
            while since < self.exchange.milliseconds():
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, timeframe, since=since, limit=1000
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    
                    # Rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                    # Break if we got less than requested (end of data)
                    if len(ohlcv) < 1000:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error fetching batch for {symbol}: {e}")
                    time.sleep(2)
                    continue
            
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
                
                logger.info(f"✅ {symbol} {timeframe}: {len(df)} records fetched")
                return df
            else:
                logger.warning(f"No data fetched for {symbol} {timeframe}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def calculate_ml_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for ML training
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if len(df) < 200:  # Need enough data for indicators
            logger.warning("Insufficient data for indicator calculation")
            return df
        
        try:
            # Basic price features
            df['price'] = df['close']
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price ratios
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            
            # Volume features
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
            
            # Trend indicators
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            # MACD family
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            df['macd_trend'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            df['macd_crossover'] = (df['macd_trend'] != df['macd_trend'].shift(1)).astype(int)
            
            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
                df[f'price_above_sma_{period}'] = (df['close'] > df[f'sma_{period}']).astype(int)
                df[f'price_above_ema_{period}'] = (df['close'] > df[f'ema_{period}']).astype(int)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
            
            # Volatility indicators
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['volatility'] = df['returns'].rolling(20).std()
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            
            # Momentum indicators
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            df['momentum'] = talib.MOM(df['close'], timeperiod=10)
            
            # Trend strength indicators
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'])
            df['di_diff'] = df['plus_di'] - df['minus_di']
            
            # VWAP and volume indicators
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
            
            # Pattern recognition features
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            
            # Market regime features
            df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)
            df['market_regime'] = np.where(
                (df['sma_20'] > df['sma_50']) & (df['sma_50'] > df['sma_100']), 'uptrend',
                np.where(
                    (df['sma_20'] < df['sma_50']) & (df['sma_50'] < df['sma_100']), 'downtrend',
                    'sideways'
                )
            )
            
            # Support/Resistance levels
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            
            # Fill NaN values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
            
            # Create target variables for ML training
            df['future_return_1h'] = df['close'].shift(-1) / df['close'] - 1
            df['future_return_4h'] = df['close'].shift(-4) / df['close'] - 1
            df['future_return_24h'] = df['close'].shift(-24) / df['close'] - 1
            
            # Signal targets
            df['signal_success'] = np.where(df['future_return_4h'] > 0.01, 1, 0)  # 1% threshold
            df['strong_signal'] = np.where(df['future_return_4h'] > 0.02, 1, 0)   # 2% threshold
            
            logger.info(f"✅ Calculated {len(df.columns)} features including ML targets")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def fetch_comprehensive_data(self, days_back: int = 90, use_incremental: bool = True) -> pd.DataFrame:
        """
        Fetch comprehensive historical data for all symbols and timeframes
        with smart incremental loading
        
        Args:
            days_back: Number of days to fetch (reduced default)
            use_incremental: Use incremental loading when possible
            days_back: Number of days to fetch
            
        Returns:
            Combined DataFrame with all data and indicators
        """
        combined_data = []
        total_combinations = len(self.symbols) * len(self.timeframes)
        current_combination = 0
        
        logger.info(f"Starting comprehensive data fetch for {total_combinations} combinations...")
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                current_combination += 1
                logger.info(f"Processing {symbol} {timeframe} ({current_combination}/{total_combinations})")
                
                try:
                    # Fetch OHLCV data
                    df = self.fetch_historical_ohlcv(symbol, timeframe, days_back)
                    
                    if df.empty:
                        continue
                    
                    # Calculate indicators
                    df = self.calculate_ml_indicators(df)
                    
                    # Add metadata
                    df['symbol'] = symbol.split('/')[0]
                    df['base_symbol'] = symbol.split('/')[0]
                    df['quote_symbol'] = symbol.split('/')[1]
                    df['timeframe'] = timeframe
                    
                    combined_data.append(df)
                    
                    logger.info(f"✅ {symbol} {timeframe}: {len(df)} records processed")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} {timeframe}: {e}")
                    continue
        
        if combined_data:
            # Combine all data
            final_df = pd.concat(combined_data, ignore_index=True)
            
            # Remove rows with target NaN (last few rows of each dataset)
            # Only drop if these columns exist (they won't exist for small datasets < 200 rows)
            if 'future_return_1h' in final_df.columns and 'signal_success' in final_df.columns:
                final_df = final_df.dropna(subset=['future_return_1h', 'signal_success'])
            
            logger.info(f"🎯 Combined dataset ready: {len(final_df)} total records")
            logger.info(f"📊 Features: {len(final_df.columns)} columns")
            logger.info(f"🔄 Symbols: {final_df['symbol'].nunique()} unique symbols")
            logger.info(f"⏰ Timeframes: {final_df['timeframe'].nunique()} unique timeframes")
            
            return final_df
        else:
            logger.error("No data was successfully fetched!")
            return pd.DataFrame()
    
    def save_training_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the comprehensive training data to CSV
        
        Args:
            df: DataFrame to save
            filename: Optional filename (default: auto-generated)
            
        Returns:
            Saved filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/ml_training_data_{timestamp}.csv"
        
        try:
            df.to_csv(filename, index=False)
            logger.info(f"💾 Training data saved to {filename}")
            logger.info(f"📈 Dataset size: {len(df)} rows x {len(df.columns)} columns")
            
            # Save summary statistics
            summary_file = filename.replace('.csv', '_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("ML Training Data Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Records: {len(df):,}\n")
                f.write(f"Total Features: {len(df.columns)}\n")
                f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
                f.write(f"Symbols: {', '.join(sorted(df['symbol'].unique()))}\n")
                f.write(f"Timeframes: {', '.join(sorted(df['timeframe'].unique()))}\n\n")
                
                f.write("Signal Distribution:\n")
                f.write(f"Signal Success Rate: {df['signal_success'].mean():.2%}\n")
                f.write(f"Strong Signal Rate: {df['strong_signal'].mean():.2%}\n\n")
                
                f.write("Market Regime Distribution:\n")
                regime_counts = df['market_regime'].value_counts()
                for regime, count in regime_counts.items():
                    f.write(f"{regime}: {count:,} ({count/len(df):.1%})\n")
            
            logger.info(f"📋 Summary saved to {summary_file}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            return ""

def main():
    """Main execution function"""
    logger.info("🚀 Starting Enhanced Historical Data Collection for ML Training")
    
    # Initialize fetcher
    fetcher = EnhancedHistoricalDataFetcher()
    
    # Fetch comprehensive data (1 year of history)
    logger.info("📥 Fetching comprehensive historical data...")
    df = fetcher.fetch_comprehensive_data(days_back=365)
    
    if not df.empty:
        # Save training data
        filename = fetcher.save_training_data(df)
        
        if filename:
            logger.info("\n🎉 Data collection completed successfully!")
            logger.info(f"📁 Training data saved to: {filename}")
            logger.info(f"🤖 Ready for ML model training with {len(df):,} samples")
            
            # Also save as the standard training file for backward compatibility
            standard_file = "logs/trade_history_combined.csv"
            df.to_csv(standard_file, index=False)
            logger.info(f"📁 Also saved as: {standard_file}")
            
        else:
            logger.error("❌ Failed to save training data")
    else:
        logger.error("❌ No data was collected")

if __name__ == "__main__":
    main()
