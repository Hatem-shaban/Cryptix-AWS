"""
Oracle Cloud VM Launcher for CRYPTIX-ML Trading Bot
Optimized for Oracle Always Free Tier VM.Standard.A1.Flex
"""

import os
import sys
import gc
import psutil
import time
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

# Import memory optimization utilities
from memory_optimizer import optimize_dataframe_memory
from auto_memory_manager import monitor_memory_usage
from render_memory_optimizer import emergency_cleanup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/oracle_launcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class OracleVMLauncher:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.max_memory_percent = 75.0  # 75% memory threshold
        self.check_interval = 300  # Check every 5 minutes
        self.last_check = time.time()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            return self.process.memory_percent()
        except Exception as e:
            logging.error(f"Error getting memory usage: {e}")
            return 0.0

    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        try:
            memory_percent = self.get_memory_usage()
            if memory_percent > self.max_memory_percent:
                logging.warning(f"High memory usage: {memory_percent:.1f}%")
                self.optimize_memory()
                return False
            return True
        except Exception as e:
            logging.error(f"Memory check failed: {e}")
            return False

    def optimize_memory(self):
        """Optimize memory usage"""
        try:
            logging.info("Running memory optimization...")
            
            # Force garbage collection
            gc.collect()
            
            # Run emergency cleanup if needed
            if self.get_memory_usage() > 85.0:
                emergency_cleanup()
            
            logging.info(f"Memory after optimization: {self.get_memory_usage():.1f}%")
            
        except Exception as e:
            logging.error(f"Memory optimization failed: {e}")

    def monitor_resources(self):
        """Monitor system resources"""
        try:
            current_time = time.time()
            if current_time - self.last_check >= self.check_interval:
                self.last_check = current_time
                
                # Check memory usage
                self.check_memory()
                
                # Log system stats
                vm = psutil.virtual_memory()
                logging.info(f"System Memory: {vm.percent}% used")
                logging.info(f"Process Memory: {self.get_memory_usage():.1f}%")
                
        except Exception as e:
            logging.error(f"Resource monitoring failed: {e}")

    def setup_directories(self):
        """Ensure required directories exist"""
        try:
            dirs = ['logs', 'models', 'data']
            for d in dirs:
                Path(d).mkdir(exist_ok=True)
            logging.info("Directory setup complete")
        except Exception as e:
            logging.error(f"Directory setup failed: {e}")
            sys.exit(1)

    def run(self):
        """Main run loop"""
        try:
            logging.info("Starting Oracle VM Launcher...")
            self.setup_directories()
            
            # Import web_bot only after setup
            from web_bot import app
            
            # Configure Flask for Oracle VM
            app.config['ORACLE_VM'] = True
            app.config['MEMORY_MONITORING'] = True
            
            # Start the trading bot
            logging.info("Starting trading bot...")
            app.run(
                host='127.0.0.1',  # Internal only - Nginx handles external
                port=5000,
                threaded=True
            )
            
        except Exception as e:
            logging.error(f"Launch failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    launcher = OracleVMLauncher()
    launcher.run()