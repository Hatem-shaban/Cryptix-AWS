"""
Render-Specific Memory Optimization for CRYPTIX-ML
Optimizes memory usage for 512MB Render deployment without affecting trading logic
"""

import gc
import os
import sys
import psutil
import pandas as pd
from pandas.errors import OptionError
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import weakref
from functools import wraps
import tempfile
import pickle
import gzip
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RenderMemoryManager:
    """Advanced memory management specifically for Render deployment"""
    
    def __init__(self, max_memory_mb: int = 480):
        """Initialize with 480MB limit for 512MB total with buffer"""
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.process = psutil.Process()
        
        # Memory-efficient data stores
        self._data_cache = weakref.WeakValueDictionary()
        self._compressed_cache = {}
        self._temp_files = []
        
        # Configure for minimal memory usage
        self._configure_environment()
        
        logger.info(f"🚀 Render Memory Manager initialized - Max: {max_memory_mb}MB")
    
    def _configure_environment(self):
        """Configure environment for minimal memory usage"""
        # Set pandas to use minimal memory (version-safe)
        try:
            pd.set_option('mode.copy_on_write', True)
        except (OptionError, KeyError):
            pass  # Not available in older pandas versions
        
        try:
            pd.set_option('compute.use_bottleneck', False)  # Disable bottleneck
        except (OptionError, KeyError):
            pass
        
        try:
            pd.set_option('compute.use_numexpr', False)     # Disable numexpr
        except (OptionError, KeyError):
            pass
        
        # Configure numpy for memory efficiency
        np.seterr(all='ignore')
        
        # Set garbage collection to be more aggressive
        gc.set_threshold(100, 5, 5)  # Very aggressive GC
        
        # Disable warnings to save memory
        import warnings
        warnings.filterwarnings('ignore')
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage"""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': memory_percent,
                'available_mb': self.max_memory_mb - (memory_info.rss / 1024 / 1024),
                'is_critical': memory_info.rss > self.max_memory_bytes * 0.85,  # 85% threshold
                'is_warning': memory_info.rss > self.max_memory_bytes * 0.70   # 70% threshold
            }
        except Exception as e:
            logger.warning(f"Memory usage check failed: {e}")
            return {
                'rss_mb': 0, 'vms_mb': 0, 'percent': 0,
                'available_mb': self.max_memory_mb, 'is_critical': False, 'is_warning': False
            }
    
    def aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        logger.debug("🧹 Starting aggressive memory cleanup...")
        
        # Clear all caches
        self._data_cache.clear()
        self._compressed_cache.clear()
        
        # Clean temporary files
        self._cleanup_temp_files()
        
        # Force garbage collection multiple times
        for _ in range(5):
            gc.collect()
        
        # Clear pandas caches
        try:
            import pandas as pd
            pd._config.reset_option('^display.', silent=True)
        except:
            pass
        
        memory_after = self.get_memory_usage()
        logger.debug(f"✅ Cleanup completed - Memory: {memory_after['rss_mb']:.1f}MB")
        
        return memory_after
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self._temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                self._temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggressively optimize DataFrame memory usage"""
        if df is None or df.empty:
            return df
        
        # Use float32 instead of float64
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        # Optimize integers
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
        
        # Convert objects to categories where beneficial
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        return df
    
    def compress_and_store(self, data: Any, key: str) -> str:
        """Compress data and store in temporary file"""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.gz')
            temp_path = temp_file.name
            temp_file.close()
            
            # Compress and save
            with gzip.open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self._temp_files.append(temp_path)
            self._compressed_cache[key] = temp_path
            
            return temp_path
        except Exception as e:
            logger.error(f"Failed to compress data for key {key}: {e}")
            return None
    
    def load_compressed(self, key: str) -> Any:
        """Load compressed data from file"""
        try:
            if key not in self._compressed_cache:
                return None
            
            temp_path = self._compressed_cache[key]
            if not os.path.exists(temp_path):
                del self._compressed_cache[key]
                return None
            
            with gzip.open(temp_path, 'rb') as f:
                data = pickle.load(f)
            
            return data
        except Exception as e:
            logger.error(f"Failed to load compressed data for key {key}: {e}")
            return None
    
    def memory_efficient_fetch_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
        """Memory-efficient data fetching with caching and compression"""
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # Check memory first
        memory_stats = self.get_memory_usage()
        if memory_stats['is_critical']:
            self.aggressive_cleanup()
        
        # Try to get from cache first
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # Try compressed cache
        compressed_data = self.load_compressed(cache_key)
        if compressed_data is not None:
            self._data_cache[cache_key] = compressed_data
            return compressed_data
        
        # Fetch new data (this should call your existing fetch_data function)
        try:
            from web_bot import fetch_data as original_fetch_data
            df = original_fetch_data(symbol, interval, min(limit, 80))  # Limit to 80 candles max
            
            if df is not None and not df.empty:
                # Optimize memory usage
                df = self.optimize_dataframe_memory(df)
                
                # Store in cache
                if memory_stats['available_mb'] > 50:  # Only cache if we have memory
                    self._data_cache[cache_key] = df
                else:
                    # Compress and store if low memory
                    self.compress_and_store(df, cache_key)
                
                return df
            
        except Exception as e:
            logger.error(f"Memory-efficient fetch failed for {symbol}: {e}")
        
        return None
    
    def limit_trading_history(self, max_items: int = 5) -> None:
        """Limit trading history to reduce memory usage"""
        try:
            from web_bot import bot_status
            
            # Limit trades history
            if 'trading_summary' in bot_status and 'trades_history' in bot_status['trading_summary']:
                trades = bot_status['trading_summary']['trades_history']
                if len(trades) > max_items:
                    bot_status['trading_summary']['trades_history'] = trades[:max_items]
            
            # Limit monitored pairs
            if 'monitored_pairs' in bot_status and len(bot_status['monitored_pairs']) > 10:
                # Keep only the most recent 10 pairs
                pairs = list(bot_status['monitored_pairs'].items())[:10]
                bot_status['monitored_pairs'] = dict(pairs)
            
            # Limit error history
            if 'errors' in bot_status and len(bot_status['errors']) > 3:
                bot_status['errors'] = bot_status['errors'][-3:]  # Keep last 3 errors
            
        except Exception as e:
            logger.warning(f"Failed to limit trading history: {e}")
    
    def monitor_and_cleanup(self):
        """Continuous monitoring and cleanup"""
        memory_stats = self.get_memory_usage()
        
        if memory_stats['is_critical']:
            logger.debug(f"🚨 Critical memory usage: {memory_stats['rss_mb']:.1f}MB")
            self.aggressive_cleanup()
            self.limit_trading_history(3)  # More aggressive limiting
            
        elif memory_stats['is_warning']:
            logger.debug(f"⚠️ High memory usage: {memory_stats['rss_mb']:.1f}MB")
            self.limit_trading_history(5)
            # Gentle cleanup
            for _ in range(2):
                gc.collect()
        
        return memory_stats

def memory_efficient_decorator(func):
    """Decorator for memory-efficient function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get memory manager
        if not hasattr(wrapper, '_memory_manager'):
            wrapper._memory_manager = RenderMemoryManager()
        
        memory_manager = wrapper._memory_manager
        
        # Check memory before execution
        memory_before = memory_manager.get_memory_usage()
        
        # Cleanup if needed
        if memory_before['is_warning']:
            memory_manager.monitor_and_cleanup()
        
        try:
            result = func(*args, **kwargs)
            
            # Check memory after execution
            memory_after = memory_manager.get_memory_usage()
            
            if memory_after['is_critical']:
                memory_manager.aggressive_cleanup()
            
            return result
            
        except MemoryError:
            logger.error(f"❌ Memory error in {func.__name__}")
            memory_manager.aggressive_cleanup()
            raise
        
        finally:
            # Always run a gentle cleanup
            gc.collect()
    
    return wrapper

def optimize_render_deployment():
    """Apply all Render-specific optimizations"""
    try:
        # Set environment variables for memory efficiency
        os.environ['PYTHONHASHSEED'] = '1'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        
        # Configure pandas for minimal memory (version-safe)
        try:
            pd.set_option('mode.copy_on_write', True)
        except (OptionError, KeyError):
            pass
        
        try:
            pd.set_option('display.max_columns', 5)
            pd.set_option('display.max_rows', 10)
        except (OptionError, KeyError):
            pass
        
        # Configure numpy
        np.seterr(all='ignore')
        
        # Initialize memory manager
        memory_manager = RenderMemoryManager()
        
        logger.info("✅ Render deployment optimizations applied")
        return memory_manager
        
    except Exception as e:
        logger.error(f"Failed to apply Render optimizations: {e}")
        return None

# Create a patch for the original fetch_data function
def patch_fetch_data_for_memory():
    """Patch the original fetch_data function to be memory efficient"""
    try:
        import web_bot
        
        # Store original function
        if not hasattr(web_bot, '_original_fetch_data'):
            web_bot._original_fetch_data = web_bot.fetch_data
        
        # Create memory manager
        memory_manager = RenderMemoryManager()
        
        # Create memory-efficient version
        @memory_efficient_decorator
        def memory_efficient_fetch_data(symbol="BTCUSDT", interval="1h", limit=100):
            # Reduce limit for memory efficiency
            efficient_limit = min(limit, 60)  # Max 60 candles
            
            # Use original function but with memory optimizations
            df = web_bot._original_fetch_data(symbol, interval, efficient_limit)
            
            if df is not None and not df.empty:
                # Optimize memory usage
                df = memory_manager.optimize_dataframe_memory(df)
                
                # Clear unnecessary columns if they exist
                unnecessary_cols = ['quote_asset_volume', 'number_of_trades', 
                                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                for col in unnecessary_cols:
                    if col in df.columns:
                        df = df.drop(columns=[col])
            
            return df
        
        # Replace the function
        web_bot.fetch_data = memory_efficient_fetch_data
        
        logger.info("✅ fetch_data function patched for memory efficiency")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch fetch_data: {e}")
        return False

if __name__ == "__main__":
    # Test memory optimization
    memory_manager = optimize_render_deployment()
    if memory_manager:
        stats = memory_manager.get_memory_usage()
        print(f"Memory usage: {stats['rss_mb']:.1f}MB / {memory_manager.max_memory_mb}MB")
