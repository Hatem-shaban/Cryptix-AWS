"""
Memory Optimizer for CRYPTIX-ML Trading Bot
Optimizes memory usage for Render deployment (512MB limit)
"""

import gc
import os
import sys
import pandas as pd
import numpy as np
import logging
import psutil
from typing import Dict, Any, Optional
import warnings
from functools import wraps

# Suppress warnings to reduce memory overhead
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Memory optimization utilities for constrained environments with Singleton pattern"""
    _instance = None
    _initialized = False
    
    def __new__(cls, max_memory_mb: int = 450):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, max_memory_mb: int = 450):
        """
        Initialize memory optimizer
        
        Args:
            max_memory_mb: Maximum memory usage in MB (450MB for 512MB limit with buffer)
        """
        # Only initialize once
        if self._initialized:
            return
        
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.process = psutil.Process()
        
        # Set pandas options for memory efficiency
        self._configure_pandas()
        
        # Set numpy options for memory efficiency
        self._configure_numpy()
        
        logger.info(f"🧠 Memory Optimizer initialized - Max memory: {max_memory_mb}MB")
        
        MemoryOptimizer._initialized = True
    
    def _configure_pandas(self):
        """Configure pandas for memory efficiency"""
        # Reduce memory usage for categorical data
        try:
            pd.set_option('mode.copy_on_write', True)
        except (KeyError, AttributeError):
            pass  # Not available in older pandas versions
        
        # Optimize string memory usage
        try:
            pd.set_option('string_storage', 'python')
        except (KeyError, AttributeError):
            pass
        
        # Reduce display overhead
        try:
            pd.set_option('display.max_columns', 10)
            pd.set_option('display.max_rows', 20)
        except (KeyError, AttributeError):
            pass
    
    def _configure_numpy(self):
        """Configure numpy for memory efficiency"""
        # Use memory-efficient error handling
        np.seterr(all='ignore')
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': memory_percent,
                'available_mb': self.max_memory_mb - (memory_info.rss / 1024 / 1024),
                'is_critical': memory_info.rss > self.max_memory_bytes * 0.9  # 90% threshold
            }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_mb': self.max_memory_mb, 'is_critical': False}
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is approaching limits"""
        memory_stats = self.get_memory_usage()
        return memory_stats['rss_mb'] > self.max_memory_mb * 0.8  # 80% threshold
    
    def force_garbage_collection(self):
        """Force aggressive garbage collection"""
        # Run garbage collection multiple times for effectiveness
        for _ in range(3):
            gc.collect()
        
        # Clear any remaining cycles
        gc.set_debug(0)
    
    def optimize_dataframe(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        if df is None or df.empty:
            return df
        
        if not inplace:
            df = df.copy()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to categorical where beneficial
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        return df
    
    def limit_dataframe_size(self, df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
        """Limit DataFrame size to prevent memory overflow"""
        if df is None or df.empty:
            return df
        
        if len(df) > max_rows:
            # Keep most recent data
            df = df.tail(max_rows).copy()
            logger.info(f"🔄 DataFrame limited to {max_rows} rows for memory efficiency")
        
        return df
    
    def memory_safe_concat(self, dataframes: list, max_total_rows: int = 5000) -> pd.DataFrame:
        """Safely concatenate DataFrames with memory limits"""
        if not dataframes:
            return pd.DataFrame()
        
        # Filter out empty DataFrames
        valid_dfs = [df for df in dataframes if df is not None and not df.empty]
        
        if not valid_dfs:
            return pd.DataFrame()
        
        # Calculate total rows
        total_rows = sum(len(df) for df in valid_dfs)
        
        if total_rows > max_total_rows:
            # Proportionally sample from each DataFrame
            sample_ratio = max_total_rows / total_rows
            sampled_dfs = []
            
            for df in valid_dfs:
                sample_size = max(1, int(len(df) * sample_ratio))
                if sample_size < len(df):
                    sampled_df = df.tail(sample_size).copy()
                else:
                    sampled_df = df.copy()
                sampled_dfs.append(sampled_df)
            
            valid_dfs = sampled_dfs
            logger.info(f"🔄 Sampled DataFrames to {max_total_rows} total rows")
        
        # Concatenate with memory optimization
        result = pd.concat(valid_dfs, ignore_index=True)
        
        # Optimize the result
        result = self.optimize_dataframe(result)
        
        return result
    
    def clear_large_variables(self, variables_dict: Dict[str, Any], size_threshold_mb: float = 10):
        """Clear large variables from memory"""
        cleared_count = 0
        
        for var_name, var_value in list(variables_dict.items()):
            try:
                # Estimate memory usage
                if hasattr(var_value, 'memory_usage'):
                    # DataFrame memory usage
                    memory_mb = var_value.memory_usage(deep=True).sum() / 1024 / 1024
                elif hasattr(var_value, 'nbytes'):
                    # NumPy array
                    memory_mb = var_value.nbytes / 1024 / 1024
                else:
                    # Use sys.getsizeof as approximation
                    memory_mb = sys.getsizeof(var_value) / 1024 / 1024
                
                if memory_mb > size_threshold_mb:
                    del variables_dict[var_name]
                    cleared_count += 1
                    logger.info(f"🗑️ Cleared large variable '{var_name}' ({memory_mb:.1f}MB)")
                    
            except Exception as e:
                logger.warning(f"Could not check size of variable '{var_name}': {e}")
        
        if cleared_count > 0:
            self.force_garbage_collection()
        
        return cleared_count

def memory_monitor(max_memory_mb: int = 450):
    """Decorator to monitor memory usage of functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = MemoryOptimizer(max_memory_mb)
            
            # Check memory before execution
            memory_before = optimizer.get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                # Check memory after execution
                memory_after = optimizer.get_memory_usage()
                memory_diff = memory_after['rss_mb'] - memory_before['rss_mb']
                
                if memory_diff > 50:  # More than 50MB increase
                    logger.warning(f"🚨 Function '{func.__name__}' used {memory_diff:.1f}MB")
                
                # Force cleanup if memory is high
                if memory_after['is_critical']:
                    logger.debug(f"🚨 Critical memory usage: {memory_after['rss_mb']:.1f}MB")
                    optimizer.force_garbage_collection()
                
                return result
                
            except MemoryError:
                logger.error(f"❌ Memory error in function '{func.__name__}'")
                optimizer.force_garbage_collection()
                raise
                
        return wrapper
    return decorator

def optimize_ml_data_loading():
    """Optimize ML data loading for memory efficiency without reducing trading symbols"""
    # Keep original trading symbols - do not reduce
    return {
        'max_rows_per_symbol': 150,  # Reduce data retention per symbol
        'use_incremental': True,
        'optimize_dtypes': True,
        'enable_chunked_processing': True,  # Process data in chunks
        'clear_intermediate_data': True,    # Clear temporary calculations
        'use_float32': True,               # Use 32-bit floats instead of 64-bit
        'compress_historical_data': True   # Compress older data
    }

def get_memory_safe_config():
    """Get memory-safe configuration for Render deployment without reducing trading assets"""
    return {
        'ML_LOOKBACK_DAYS': 10,  # Reduce historical data lookback
        'ML_MIN_TRAINING_SAMPLES': 30,  # Reduce minimum training samples
        'MAX_TRADES_HISTORY': 10,  # Keep only recent trades in memory
        'API_RATE_LIMITS': {
            'calls_per_minute': 800,  # Slightly reduce API calls
            'calls_per_second': 8,
        },
        'CACHE_SIZE_LIMIT': 5,  # Reduce cache entries
        'FORCE_GC_INTERVAL': 180,  # Force GC every 3 minutes
        'DATA_COMPRESSION': True,   # Enable data compression
        'LAZY_LOADING': True,       # Load data only when needed
        'CLEAR_TEMP_FILES': True,   # Auto-clear temporary files
        'USE_MEMORY_MAPPING': True, # Use memory mapping for large data
    }

# Global optimizer instance
_optimizer = None

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = MemoryOptimizer()
    return _optimizer

def log_memory_usage(context: str = ""):
    """Log current memory usage"""
    optimizer = get_memory_optimizer()
    memory_stats = optimizer.get_memory_usage()
    
    status = "🚨 CRITICAL" if memory_stats['is_critical'] else "⚠️ HIGH" if memory_stats['rss_mb'] > 300 else "✅ OK"
    
    logger.debug(f"💾 Memory {context}: {memory_stats['rss_mb']:.1f}MB ({memory_stats['percent']:.1f}%) - {status}")
    
    return memory_stats

def emergency_memory_cleanup():
    """Emergency memory cleanup when approaching limits"""
    optimizer = get_memory_optimizer()
    
    logger.warning("🆘 Emergency memory cleanup initiated")
    
    # Force aggressive garbage collection
    optimizer.force_garbage_collection()
    
    # Clear pandas caches
    try:
        pd.reset_option('^display.', silent=True)
    except:
        pass
    
    # Clear any global caches
    try:
        import gc
        gc.set_threshold(0, 0, 0)  # Disable automatic GC temporarily
        gc.collect()
        gc.set_threshold(700, 10, 10)  # Reset to more aggressive settings
    except:
        pass
    
    memory_after = optimizer.get_memory_usage()
    logger.info(f"✅ Emergency cleanup completed - Memory: {memory_after['rss_mb']:.1f}MB")
    
    return memory_after

if __name__ == "__main__":
    # Test memory optimization
    optimizer = MemoryOptimizer()
    log_memory_usage("Test")
    
    # Test DataFrame optimization
    test_df = pd.DataFrame({
        'int_col': range(1000),
        'float_col': np.random.random(1000),
        'cat_col': ['A', 'B', 'C'] * 334
    })
    
    print(f"Before optimization: {test_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")
    
    optimized_df = optimizer.optimize_dataframe(test_df)
    print(f"After optimization: {optimized_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")
