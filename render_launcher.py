"""
Render-Optimized Web Bot Launcher
Applies memory optimizations without changing trading logic
"""

import os
import sys
import gc
import logging
from datetime import datetime

# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def apply_render_optimizations():
    """Apply Render-specific optimizations before importing main modules"""
    try:
        # Memory optimization environment variables
        os.environ['PYTHONHASHSEED'] = '1'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Apply compatibility patches first
        from render_compatibility import apply_render_compatibility
        if apply_render_compatibility():
            logger.info("✅ Render compatibility patches applied")
        
        # Import and apply memory optimizations
        from render_memory_optimizer import optimize_render_deployment, patch_fetch_data_for_memory
        from memory_optimizer import get_memory_optimizer, log_memory_usage
        from memory_patches import apply_memory_efficient_patches
        
        # Initialize Render optimizations
        memory_manager = optimize_render_deployment()
        if memory_manager:
            logger.info("✅ Render memory optimizations applied")
        
        # Apply memory-efficient patches to trading functions
        if apply_memory_efficient_patches():
            logger.info("✅ Memory-efficient patches applied to trading functions")
        
        # Patch data fetching for memory efficiency
        if patch_fetch_data_for_memory():
            logger.info("✅ Data fetching optimized for memory")
        
        # Get global memory optimizer
        optimizer = get_memory_optimizer()
        log_memory_usage("Startup")
        
        # Aggressive garbage collection setup
        gc.set_threshold(100, 5, 5)  # Very aggressive
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply Render optimizations: {e}")
        return False

def optimize_flask_for_render():
    """Optimize Flask settings for Render deployment"""
    try:
        import flask
        
        # Minimize Flask memory usage
        flask_config = {
            'TEMPLATES_AUTO_RELOAD': False,
            'SEND_FILE_MAX_AGE_DEFAULT': 31536000,  # 1 year cache
            'MAX_CONTENT_LENGTH': 1024 * 1024,     # 1MB max upload
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': False,
        }
        
        logger.info("✅ Flask optimized for Render")
        return flask_config
        
    except Exception as e:
        logger.error(f"Failed to optimize Flask: {e}")
        return {}

def setup_render_error_handling():
    """Setup error handling for Render environment"""
    def handle_memory_error(exc_type, exc_value, exc_traceback):
        if exc_type == MemoryError:
            logger.error("🚨 MEMORY ERROR - Attempting emergency cleanup")
            try:
                from memory_optimizer import emergency_memory_cleanup
                emergency_memory_cleanup()
            except:
                pass
            # Force exit to prevent memory corruption
            os._exit(1)
        else:
            # Default exception handling
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = handle_memory_error

def main():
    """Main entry point for Render deployment"""
    logger.info("🚀 Starting CRYPTIX-ML Bot for Render...")
    logger.info(f"📅 Startup Time: {datetime.now().isoformat()}")
    
    # Apply Render optimizations first
    if not apply_render_optimizations():
        logger.error("❌ Failed to apply optimizations - continuing anyway")
    
    # Run startup configuration (Supabase initialization)
    try:
        from render_startup import startup_configuration
        if not startup_configuration():
            logger.warning("⚠️ Startup configuration failed - continuing with fallback")
    except Exception as e:
        logger.warning(f"⚠️ Startup configuration error: {e} - continuing with fallback")
    
    # Setup error handling
    setup_render_error_handling()
    
    # Get Flask optimizations
    flask_config = optimize_flask_for_render()
    
    try:
        # Import main modules AFTER optimizations are applied
        logger.info("📦 Importing main modules...")
        
        # Import the main web bot
        from web_bot import app, bot_status, initialize_client
        
        # Apply Flask configurations
        for key, value in flask_config.items():
            app.config[key] = value
        
        # Initialize API client with emergency check
        logger.info("🔑 Initializing trading client...")
        
        # EMERGENCY CHECK: Skip API initialization if in emergency mode during ban
        try:
            import config
            if hasattr(config, 'EMERGENCY_MODE') and config.EMERGENCY_MODE:
                # Check if ban is still active
                current_time = datetime.now().timestamp() * 1000
                ban_until = 1760962139050  # Latest ban timestamp
                
                if current_time < ban_until:
                    ban_lift_time = datetime.fromtimestamp(ban_until / 1000)
                    minutes_remaining = (ban_until - current_time) / 1000 / 60
                    logger.warning(f"🚨 EMERGENCY: API ban active until {ban_lift_time.strftime('%H:%M:%S')}")
                    logger.warning(f"⏰ {minutes_remaining:.1f} minutes remaining - SKIPPING API initialization")
                    logger.warning("⚠️ Trading client initialization failed - running in demo mode")
                    
                    # Set demo mode in bot_status
                    bot_status['api_connected'] = False
                    bot_status['demo_mode'] = True
                else:
                    logger.info("✅ API ban has been lifted - proceeding with emergency rate limits")
                    if initialize_client():
                        logger.info("✅ Trading client initialized successfully")
                    else:
                        logger.warning("⚠️ Trading client initialization failed - running in demo mode")
            else:
                # Normal initialization if not in emergency mode
                if initialize_client():
                    logger.info("✅ Trading client initialized successfully")
                else:
                    logger.warning("⚠️ Trading client initialization failed - running in demo mode")
        except Exception as e:
            logger.warning(f"⚠️ Emergency check failed: {e} - attempting normal initialization")
            if initialize_client():
                logger.info("✅ Trading client initialized successfully")
            else:
                logger.warning("⚠️ Trading client initialization failed - running in demo mode")
        
        # Apply memory monitoring
        from render_memory_optimizer import RenderMemoryManager
        from auto_memory_manager import start_auto_memory_management
        
        memory_manager = RenderMemoryManager()
        
        # Start automatic memory management
        auto_manager = start_auto_memory_management(max_memory_mb=480, cleanup_interval=180)
        logger.info("✅ Automatic memory management started")
        
        # Add memory monitoring to Flask app
        @app.before_request
        def before_request():
            memory_stats = memory_manager.monitor_and_cleanup()
            if memory_stats['is_critical']:
                logger.debug(f"🚨 Critical memory before request: {memory_stats['rss_mb']:.1f}MB")
        
        @app.after_request
        def after_request(response):
            # Cleanup after each request
            gc.collect()
            return response
        
        # Get port from environment
        port = int(os.environ.get('PORT', 5000))
        
        logger.info(f"🌐 Starting Flask server on port {port}")
        logger.info("✅ CRYPTIX-ML Bot ready for Render deployment")
        
        # ===== AUTOSTART: Start trading bot automatically on deployment =====
        try:
            # Check if autostart is enabled (default: True for Render)
            autostart_enabled = os.getenv('AUTOSTART_BOT', 'True').lower() in ['true', '1', 'yes', 'on']
            
            if autostart_enabled:
                logger.info("🤖 AUTOSTART: Starting trading bot automatically...")
                
                # Import the start function
                from web_bot import start_trading_bot, bot_status
                
                # Check if bot is not already running
                if not bot_status.get('running', False):
                    # Start the bot in a separate thread
                    import threading
                    bot_thread = threading.Thread(target=start_trading_bot, daemon=True)
                    bot_thread.start()
                    
                    logger.info("✅ AUTOSTART: Trading bot started successfully")
                    logger.info("📊 Bot is now monitoring markets and executing trades")
                else:
                    logger.info("ℹ️ AUTOSTART: Bot already running, skipping autostart")
            else:
                logger.info("ℹ️ AUTOSTART: Disabled (set AUTOSTART_BOT=True to enable)")
                logger.info("💡 Use /start endpoint or web interface to start manually")
                
        except Exception as autostart_error:
            logger.error(f"❌ AUTOSTART failed: {autostart_error}")
            logger.warning("⚠️ Bot can still be started manually via /start endpoint")
        # ===== END AUTOSTART =====
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,  # Never use debug in production
            threaded=True,
            use_reloader=False  # Disable reloader to save memory
        )
        
    except Exception as e:
        logger.error(f"❌ Fatal error during startup: {e}")
        # Log memory stats for debugging
        try:
            from memory_optimizer import log_memory_usage
            log_memory_usage("Fatal Error")
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
