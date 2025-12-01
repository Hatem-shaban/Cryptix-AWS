"""
Auto Memory Cleanup for CRYPTIX-ML Trading Bot
Automatically manages memory without affecting trading logic
"""

import gc
import os
import sys
import time
import threading
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AutoMemoryManager:
    """Automatic memory management that runs in background"""
    
    def __init__(self, max_memory_mb: int = 480, cleanup_interval: int = 180):
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval = cleanup_interval
        self.running = False
        self.cleanup_thread = None
        
        # Import memory tools
        try:
            import psutil
            self.process = psutil.Process()
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            logger.warning("psutil not available - memory monitoring limited")
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        if self.psutil_available:
            try:
                return self.process.memory_info().rss / 1024 / 1024
            except:
                pass
        
        # Fallback method
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except:
            return 0
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical"""
        current_mb = self.get_memory_usage_mb()
        return current_mb > (self.max_memory_mb * 0.85)  # 85% threshold
    
    def is_memory_warning(self) -> bool:
        """Check if memory usage is in warning zone"""
        current_mb = self.get_memory_usage_mb()
        return current_mb > (self.max_memory_mb * 0.70)  # 70% threshold
    
    def gentle_cleanup(self):
        """Perform gentle memory cleanup"""
        try:
            # Standard garbage collection
            gc.collect()
            
            # Clear trading history if available
            try:
                from web_bot import bot_status
                
                # Limit trading history to last 5 trades
                if 'trading_summary' in bot_status and 'trades_history' in bot_status['trading_summary']:
                    trades = bot_status['trading_summary']['trades_history']
                    if len(trades) > 5:
                        bot_status['trading_summary']['trades_history'] = trades[:5]
                
                # Limit monitored pairs to 8
                if 'monitored_pairs' in bot_status and len(bot_status['monitored_pairs']) > 8:
                    pairs = list(bot_status['monitored_pairs'].items())[:8]
                    bot_status['monitored_pairs'] = dict(pairs)
                
                # Limit error history to 3
                if 'errors' in bot_status and len(bot_status['errors']) > 3:
                    bot_status['errors'] = bot_status['errors'][-3:]
                    
            except ImportError:
                pass  # web_bot not imported yet
            except Exception as e:
                logger.warning(f"Error during trading history cleanup: {e}")
            
        except Exception as e:
            logger.warning(f"Gentle cleanup error: {e}")
    
    def aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        try:
            logger.debug("🧹 Performing aggressive memory cleanup...")
            
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            # Clear caches if available
            try:
                from web_bot import bot_status
                
                # Clear various caches
                if 'exchange_info_cache' in bot_status:
                    bot_status['exchange_info_cache'] = None
                
                if 'coinbase_cache' in bot_status:
                    bot_status['coinbase_cache'] = {}
                
                if 'balance_cache' in bot_status:
                    bot_status['balance_cache'] = {}
                
                # More aggressive trading history limiting
                if 'trading_summary' in bot_status and 'trades_history' in bot_status['trading_summary']:
                    trades = bot_status['trading_summary']['trades_history']
                    if len(trades) > 3:
                        bot_status['trading_summary']['trades_history'] = trades[:3]
                
                # Limit monitored pairs more aggressively
                if 'monitored_pairs' in bot_status and len(bot_status['monitored_pairs']) > 5:
                    pairs = list(bot_status['monitored_pairs'].items())[:5]
                    bot_status['monitored_pairs'] = dict(pairs)
                    
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Error during cache cleanup: {e}")
            
            # Clear pandas caches
            try:
                import pandas as pd
                pd._config.reset_option('^display.', silent=True)
            except:
                pass
            
            # Force final garbage collection
            gc.collect()
            
            memory_after = self.get_memory_usage_mb()
            logger.debug(f"✅ Aggressive cleanup completed - Memory: {memory_after:.1f}MB")
            
        except Exception as e:
            logger.error(f"Aggressive cleanup error: {e}")
    
    def cleanup_cycle(self):
        """Main cleanup cycle that runs in background"""
        while self.running:
            try:
                # Check memory status
                current_mb = self.get_memory_usage_mb()
                
                if self.is_memory_critical():
                    logger.debug(f"🚨 Critical memory usage: {current_mb:.1f}MB - performing aggressive cleanup")
                    self.aggressive_cleanup()
                elif self.is_memory_warning():
                    logger.debug(f"⚠️ High memory usage: {current_mb:.1f}MB - performing gentle cleanup")
                    self.gentle_cleanup()
                else:
                    # Regular gentle cleanup
                    self.gentle_cleanup()
                
                # Wait for next cleanup cycle
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup cycle: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def start(self):
        """Start automatic memory management"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self.cleanup_cycle, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"🤖 Auto Memory Manager started - cleanup every {self.cleanup_interval}s")
    
    def stop(self):
        """Stop automatic memory management"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        logger.info("🛑 Auto Memory Manager stopped")

# Global instance
_auto_memory_manager: Optional[AutoMemoryManager] = None

def start_auto_memory_management(max_memory_mb: int = 480, cleanup_interval: int = 180):
    """Start automatic memory management"""
    global _auto_memory_manager
    
    if _auto_memory_manager is None:
        _auto_memory_manager = AutoMemoryManager(max_memory_mb, cleanup_interval)
        _auto_memory_manager.start()
    
    return _auto_memory_manager

def stop_auto_memory_management():
    """Stop automatic memory management"""
    global _auto_memory_manager
    
    if _auto_memory_manager:
        _auto_memory_manager.stop()
        _auto_memory_manager = None

def get_auto_memory_manager() -> Optional[AutoMemoryManager]:
    """Get the auto memory manager instance"""
    return _auto_memory_manager

def emergency_memory_cleanup():
    """Emergency memory cleanup function"""
    try:
        logger.warning("🆘 Emergency memory cleanup triggered")
        
        # Force aggressive garbage collection
        for _ in range(5):
            gc.collect()
        
        # Clear all possible caches
        try:
            from web_bot import bot_status
            
            # Clear all caches
            bot_status['exchange_info_cache'] = None
            bot_status['coinbase_cache'] = {}
            bot_status['balance_cache'] = {}
            
            # Minimize trading history
            if 'trading_summary' in bot_status:
                if 'trades_history' in bot_status['trading_summary']:
                    bot_status['trading_summary']['trades_history'] = bot_status['trading_summary']['trades_history'][:2]
            
            # Minimize monitored pairs
            if 'monitored_pairs' in bot_status:
                pairs = list(bot_status['monitored_pairs'].items())[:3]
                bot_status['monitored_pairs'] = dict(pairs)
            
            # Clear errors
            bot_status['errors'] = []
            
        except:
            pass
        
        # Clear pandas
        try:
            import pandas as pd
            pd._config.reset_option('^display.', silent=True)
        except:
            pass
        
        # Final garbage collection
        gc.collect()
        
        logger.info("✅ Emergency cleanup completed")
        
    except Exception as e:
        logger.error(f"Emergency cleanup failed: {e}")

if __name__ == "__main__":
    # Test auto memory management
    manager = start_auto_memory_management()
    print(f"Auto memory manager started: {manager is not None}")
    
    # Test for 30 seconds
    time.sleep(30)
    
    stop_auto_memory_management()
    print("Auto memory manager stopped")
