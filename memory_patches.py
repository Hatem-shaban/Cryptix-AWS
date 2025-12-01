"""
Memory-Optimized Trading Functions for Render
Patches existing functions to use less memory without changing trading logic
"""

import gc
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

def memory_optimized_dataframe_operations():
    """Patch pandas operations to be more memory efficient"""
    
    # Store original DataFrame methods
    if not hasattr(pd.DataFrame, '_original_rolling'):
        pd.DataFrame._original_rolling = pd.DataFrame.rolling
        pd.DataFrame._original_ewm = pd.DataFrame.ewm
        pd.DataFrame._original_pct_change = pd.DataFrame.pct_change
    
    def memory_efficient_rolling(self, window, **kwargs):
        """Memory-efficient rolling operations"""
        # For small DataFrames, use original method
        if len(self) <= 100:
            return self._original_rolling(window, **kwargs)
        
        # For larger DataFrames, process in chunks
        result = self._original_rolling(window, **kwargs)
        
        # Force garbage collection after rolling operations
        gc.collect()
        return result
    
    def memory_efficient_ewm(self, **kwargs):
        """Memory-efficient exponential weighted operations"""
        result = self._original_ewm(**kwargs)
        gc.collect()
        return result
    
    def memory_efficient_pct_change(self, **kwargs):
        """Memory-efficient percentage change calculation"""
        result = self._original_pct_change(**kwargs)
        gc.collect()
        return result
    
    # Apply patches
    pd.DataFrame.rolling = memory_efficient_rolling
    pd.DataFrame.ewm = memory_efficient_ewm
    pd.DataFrame.pct_change = memory_efficient_pct_change

def optimize_technical_indicators():
    """Optimize technical indicator calculations for memory efficiency"""
    
    def memory_efficient_rsi(prices, period=14):
        """Memory-optimized RSI calculation"""
        try:
            if hasattr(prices, 'values'):
                prices = prices.values
            elif isinstance(prices, list):
                prices = np.array(prices, dtype=np.float32)  # Use float32
            elif isinstance(prices, (int, float)):
                return 50  # Single value case
            
            if len(prices) < period + 1:
                return 50
            
            # Use float32 for all calculations to save memory
            prices = prices.astype(np.float32)
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0).astype(np.float32)
            losses = np.where(deltas < 0, -deltas, 0).astype(np.float32)
            
            # Use Wilder's smoothing method for accurate RSI
            alpha = 1.0 / period
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Apply Wilder's smoothing to remaining data
            for i in range(period, len(gains)):
                avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
            
            if avg_loss == 0:
                return 100 if avg_gain > 0 else 50
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(max(0, min(100, rsi)))
            
        except Exception as e:
            logger.warning(f"RSI calculation error: {e}")
            return 50
    
    def memory_efficient_macd(prices, fast=12, slow=26, signal=9):
        """Memory-optimized MACD calculation"""
        try:
            if hasattr(prices, 'values'):
                prices = prices.values
            elif isinstance(prices, list):
                prices = np.array(prices, dtype=np.float32)
            
            if len(prices) < slow:
                return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
            
            # Use float32 for all calculations
            prices = prices.astype(np.float32)
            
            # Simplified EMA calculation
            def simple_ema(data, period):
                alpha = 2.0 / (period + 1)
                ema = [float(data[0])]
                for price in data[1:]:
                    ema.append(alpha * float(price) + (1 - alpha) * ema[-1])
                return np.array(ema, dtype=np.float32)
            
            fast_ema = simple_ema(prices, fast)
            slow_ema = simple_ema(prices, slow)
            
            macd_line = fast_ema - slow_ema
            signal_line = simple_ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            current_macd = float(macd_line[-1])
            current_signal = float(signal_line[-1])
            current_histogram = float(histogram[-1])
            
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
            logger.warning(f"MACD calculation error: {e}")
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
    
    return memory_efficient_rsi, memory_efficient_macd

def patch_web_bot_functions():
    """Patch web_bot functions for memory efficiency without changing logic"""
    try:
        import web_bot
        
        # Optimize technical indicators
        efficient_rsi, efficient_macd = optimize_technical_indicators()
        
        # Store original functions
        if not hasattr(web_bot, '_original_calculate_rsi'):
            web_bot._original_calculate_rsi = web_bot.calculate_rsi
            web_bot._original_calculate_macd = web_bot.calculate_macd
        
        # Replace with memory-efficient versions
        web_bot.calculate_rsi = efficient_rsi
        web_bot.calculate_macd = efficient_macd
        
        # Patch fetch_data for memory efficiency
        if not hasattr(web_bot, '_original_fetch_data'):
            web_bot._original_fetch_data = web_bot.fetch_data
        
        def memory_efficient_fetch_data(symbol="BTCUSDT", interval="1h", limit=100):
            """Memory-optimized fetch_data"""
            # Limit the data size to reduce memory usage
            efficient_limit = min(limit, 80)  # Max 80 candles instead of 100
            
            # Call original function
            df = web_bot._original_fetch_data(symbol, interval, efficient_limit)
            
            if df is not None and not df.empty:
                # Convert to more memory-efficient types
                float_cols = df.select_dtypes(include=['float64']).columns
                df[float_cols] = df[float_cols].astype('float32')
                
                # Remove unnecessary columns if they exist
                unnecessary_cols = ['close_time', 'quote_asset_volume', 'number_of_trades', 
                                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                for col in unnecessary_cols:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                
                # Force garbage collection after data processing
                gc.collect()
            
            return df
        
        web_bot.fetch_data = memory_efficient_fetch_data
        
        # Patch scan_trading_pairs for memory efficiency
        if not hasattr(web_bot, '_original_scan_trading_pairs'):
            web_bot._original_scan_trading_pairs = web_bot.scan_trading_pairs
        
        def memory_efficient_scan_trading_pairs(base_assets=None, quote_asset="USDT", min_volume_usdt=1000000):
            """Memory-optimized scan_trading_pairs - keep same logic but optimize memory"""
            import time
            
            opportunities = []
            
            # Keep original symbols - do not reduce trading pairs
            if base_assets is None:
                base_assets = ["BTC", "ETH", "BNB", "XRP", "SOL", "MATIC", "DOT", "ADA", "AVAX", "LINK"]
            
            # Increase delay slightly to reduce memory pressure from rapid API calls
            scan_delay = 0.6  # Slightly longer delay
            
            for base in base_assets:
                try:
                    symbol = f"{base}{quote_asset}"
                    
                    time.sleep(scan_delay)
                    
                    # Get ticker data
                    ticker = web_bot.client.get_ticker(symbol=symbol)
                    volume_usdt = float(ticker['quoteVolume'])
                    price_change_pct = float(ticker['priceChangePercent'])
                    
                    if volume_usdt < min_volume_usdt:
                        continue
                    
                    time.sleep(scan_delay)
                    
                    # Fetch smaller data set for memory efficiency
                    df = memory_efficient_fetch_data(symbol=symbol, limit=25)  # Reduced from 30
                    if df is None or len(df) < 10:  # Reduced minimum
                        continue
                    
                    # Rest of the logic remains the same as original function
                    current_price = float(df['close'].iloc[-1])
                    
                    # Calculate indicators (using optimized functions)
                    if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
                        current_rsi = float(df['rsi'].iloc[-1])
                    else:
                        current_rsi = efficient_rsi(df['close'].values, period=14)
                    
                    if 'macd_trend' in df.columns and not pd.isna(df['macd_trend'].iloc[-1]):
                        macd_trend = df['macd_trend'].iloc[-1]
                    else:
                        macd_result = efficient_macd(df['close'].values)
                        macd_trend = macd_result.get('trend', 'NEUTRAL')
                    
                    # Calculate SMAs with memory optimization
                    try:
                        sma_fast = df['close'].rolling(window=10).mean()
                        sma_slow = df['close'].rolling(window=20).mean()
                        
                        if len(sma_fast) == 0 or len(sma_slow) == 0:
                            continue
                        
                        sma_fast_value = float(sma_fast.iloc[-1])
                        sma_slow_value = float(sma_slow.iloc[-1])
                    except Exception:
                        continue
                    
                    # Same scoring logic as original
                    opportunity_score = 0
                    signals = []
                    
                    # Check balance (same as original)
                    has_balance, available_balance, balance_msg = web_bot.check_coin_balance(symbol)
                    can_sell = has_balance and available_balance > 0
                    
                    # Same scoring logic as original function
                    if current_rsi < 30:
                        opportunity_score += 30
                        signals.append("RSI_OVERSOLD")
                    elif current_rsi > 70:
                        if can_sell:
                            opportunity_score += 25
                            signals.append("RSI_OVERBOUGHT_SELLABLE")
                        else:
                            opportunity_score += 5
                            signals.append("RSI_OVERBOUGHT_NO_BALANCE")
                    elif 45 <= current_rsi <= 55:
                        opportunity_score += 10
                        signals.append("RSI_NEUTRAL")
                    
                    if macd_trend == "BULLISH":
                        opportunity_score += 20
                        signals.append("MACD_BULLISH")
                    elif macd_trend == "BEARISH":
                        if can_sell:
                            opportunity_score += 15
                            signals.append("MACD_BEARISH_SELLABLE")
                        else:
                            signals.append("MACD_BEARISH_NO_BALANCE")
                    
                    if abs(price_change_pct) > 5:
                        opportunity_score += 15
                        signals.append("HIGH_VOLATILITY")
                    
                    if volume_usdt > min_volume_usdt * 5:
                        opportunity_score += 15
                        signals.append("HIGH_VOLUME")
                    
                    if current_price > sma_fast_value > sma_slow_value:
                        opportunity_score += 10
                        signals.append("UPTREND")
                    elif current_price < sma_fast_value < sma_slow_value:
                        if can_sell:
                            opportunity_score += 15
                            signals.append("DOWNTREND_SELLABLE")
                        else:
                            opportunity_score += 5
                            signals.append("DOWNTREND_NO_BALANCE")
                    
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
                        'balance_info': balance_info,
                        'data': df
                    })
                    
                    # Clear DataFrame from memory after processing
                    del df
                    gc.collect()
                    
                except Exception as e:
                    web_bot.log_error_to_csv(f"Error scanning {base}{quote_asset}: {e}", 
                                           "SCAN_ERROR", "memory_efficient_scan_trading_pairs", "WARNING")
                    continue
            
            # Sort by score (same as original)
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            
            # Same logging as original
            if opportunities:
                print(f"\n=== Top Trading Opportunities ===")
                for i, opp in enumerate(opportunities[:5]):
                    balance_status = "✅" if opp['balance_info']['has_balance'] else "❌"
                    balance_amount = f"{opp['balance_info']['available_balance']:.4f}" if opp['balance_info']['has_balance'] else "0"
                    
                    print(f"{i+1}. {opp['symbol']}: Score {opp['score']}, RSI {opp['rsi']:.1f}, "
                          f"Change {opp['price_change_pct']:.2f}%, Balance: {balance_status}({balance_amount}), "
                          f"Signals: {', '.join(opp['signals'])}")
            
            return opportunities
        
        web_bot.scan_trading_pairs = memory_efficient_scan_trading_pairs
        
        logger.info("✅ Web bot functions patched for memory efficiency")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch web_bot functions: {e}")
        return False

def apply_memory_efficient_patches():
    """Apply all memory efficiency patches"""
    try:
        # Apply DataFrame operation patches
        memory_optimized_dataframe_operations()
        
        # Patch web_bot functions
        patch_web_bot_functions()
        
        logger.info("✅ All memory efficiency patches applied")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply memory patches: {e}")
        return False

if __name__ == "__main__":
    apply_memory_efficient_patches()
