"""
Startup Configuration for CRYPTIX-ML on Render
Handles Supabase migration and initialization on deployment
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if Supabase environment variables are set"""
    # Only check Supabase vars for startup - other vars are checked by their respective modules
    required_supabase_vars = [
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY'
    ]
    
    optional_vars = [
        'BINANCE_API_KEY',
        'BINANCE_SECRET_KEY', 
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing_supabase = []
    for var in required_supabase_vars:
        if not os.getenv(var):
            missing_supabase.append(var)
    
    if missing_supabase:
        logger.warning(f"⚠️ Supabase not configured: {', '.join(missing_supabase)}")
        logger.info("📋 Will fall back to file-based position tracking")
        return False
    
    logger.info("✅ Supabase environment variables are set")
    
    # Log status of optional vars (for debugging)
    for var in optional_vars:
        if os.getenv(var):
            logger.info(f"✅ {var}: Configured")
        else:
            logger.info(f"⚠️ {var}: Not set (will be checked by respective modules)")
    
    return True

def initialize_supabase():
    """Initialize Supabase and run migrations if needed"""
    try:
        logger.info("🔄 Initializing Supabase connection...")
        
        from supabase_position_tracker import SupabasePositionTracker
        tracker = SupabasePositionTracker()
        
        # Check health
        health = tracker.health_check()
        if health['status'] != 'healthy':
            logger.error(f"❌ Supabase health check failed: {health.get('error')}")
            return False
        
        logger.info("✅ Supabase connection healthy")
        
        # Check if migrations are needed
        migration_status = tracker.supabase.table('migration_status').select('*').execute()
        
        needs_migration = False
        for migration in migration_status.data:
            if migration['status'] == 'pending':
                logger.info(f"📋 Pending migration: {migration['migration_name']}")
                needs_migration = True
        
        # Auto-migrate positions.json if it exists and migration is pending
        positions_file = Path("logs/positions.json")
        if positions_file.exists():
            json_migration = next(
                (m for m in migration_status.data if m['migration_name'] == 'positions_json_import'),
                None
            )
            
            if json_migration and json_migration['status'] == 'pending':
                logger.info("🔄 Auto-migrating existing positions.json...")
                success = tracker.migrate_from_json(str(positions_file))
                if success:
                    logger.info("✅ Positions JSON migrated successfully")
                else:
                    logger.error("❌ Positions JSON migration failed")
        
        logger.info(f"📊 Current status: {health['trades_count']} trades, {health['positions_count']} positions")
        return True
        
    except Exception as e:
        logger.error(f"❌ Supabase initialization failed: {e}")
        return False

def startup_configuration():
    """Run complete startup configuration"""
    logger.info("🚀 Starting CRYPTIX-ML Supabase initialization...")
    
    # Step 1: Check environment
    supabase_available = check_environment()
    if not supabase_available:
        logger.info("📋 Supabase not available - bot will use file-based tracking")
        return True  # Don't fail startup, just skip Supabase setup
    
    # Step 2: Initialize Supabase
    if not initialize_supabase():
        logger.warning("⚠️ Supabase initialization failed, falling back to file-based tracking")
        # Don't fail startup - let the bot run with file-based fallback
    
    # Step 3: Test position tracker
    try:
        from supabase_position_tracker import get_position_tracker
        tracker = get_position_tracker()
        logger.info("✅ Position tracker initialized successfully")
        
        # Show portfolio summary
        summary = tracker.get_portfolio_summary()
        logger.info(f"💰 Current portfolio: {summary['total_positions']} positions, ${summary['total_cost']:.2f} total cost")
        
    except Exception as e:
        logger.error(f"❌ Position tracker test failed: {e}")
        return False
    
    logger.info("🎉 CRYPTIX-ML startup configuration completed successfully!")
    return True

if __name__ == "__main__":
    success = startup_configuration()
    if not success:
        sys.exit(1)