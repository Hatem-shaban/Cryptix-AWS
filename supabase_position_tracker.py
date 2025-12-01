"""
Supabase Position Tracker for CRYPTIX-ML
Smart replacement for file-based position tracking
"""

import os
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)

class SupabasePositionTracker:
    """
    Cloud-based position tracker using Supabase
    Replaces SmartPositionTracker with persistent database storage
    Implements Singleton pattern to ensure single database connection
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if self._initialized:
            return
        
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for full access
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables")
        
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("✅ Connected to Supabase successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            raise
        
        # Cache positions in memory for performance
        self._positions_cache = {}
        self._cache_last_updated = None
        self.load_positions()
        
        SupabasePositionTracker._initialized = True
    
    def load_positions(self) -> Dict:
        """Load current positions from Supabase"""
        try:
            response = self.supabase.table('positions').select('*').execute()
            
            self._positions_cache = {}
            for pos in response.data:
                self._positions_cache[pos['symbol']] = {
                    'quantity': float(pos['quantity']),
                    'avg_buy_price': float(pos['avg_buy_price']),
                    'total_cost': float(pos['total_cost']),
                    'trades': []  # Will be loaded separately if needed
                }
            
            self._cache_last_updated = datetime.now(timezone.utc)
            logger.info(f"📊 Loaded {len(self._positions_cache)} positions from Supabase")
            return self._positions_cache
            
        except Exception as e:
            logger.error(f"❌ Error loading positions: {e}")
            return {}
    
    @property
    def positions(self) -> Dict:
        """Get current positions (cached)"""
        # Refresh cache if older than 5 minutes
        if (not self._cache_last_updated or 
            (datetime.now(timezone.utc) - self._cache_last_updated).seconds > 300):
            self.load_positions()
        
        return self._positions_cache
    
    def add_trade(self, symbol: str, action: str, quantity: float, price: float, 
                  timestamp: Optional[str] = None, source: str = 'bot') -> bool:
        """
        Add a new trade to Supabase (triggers automatic position update)
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            action: 'BUY' or 'SELL'
            quantity: Trade quantity
            price: Trade price
            timestamp: ISO timestamp (defaults to now)
            source: Trade source ('bot', 'binance_history', 'manual')
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc).isoformat()
            elif isinstance(timestamp, str) and not timestamp.endswith('Z'):
                # Ensure timezone info
                timestamp = f"{timestamp}+00:00" if '+' not in timestamp else timestamp
            
            trade_data = {
                'symbol': symbol,
                'action': action.upper(),
                'quantity': str(Decimal(str(quantity))),
                'price': str(Decimal(str(price))),
                'timestamp': timestamp,
                'source': source
            }
            
            # Insert trade (trigger will automatically update position)
            response = self.supabase.table('trades').insert(trade_data).execute()
            
            if response.data:
                logger.info(f"✅ Trade logged: {action} {quantity} {symbol} @ {price}")
                # Invalidate cache to force reload
                self._cache_last_updated = None
                return True
            else:
                logger.error(f"❌ Failed to log trade: No data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error adding trade: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
    
    def get_trade_history(self, symbol: str = None, limit: int = 100, days: int = 30) -> List[Dict]:
        """Get trade history from Supabase"""
        try:
            query = self.supabase.table('trades').select('*').order('timestamp', desc=True)
            
            if symbol:
                query = query.eq('symbol', symbol)

            if days > 0:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                query = query.gte('timestamp', cutoff_date.isoformat())
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            return response.data
            
        except Exception as e:
            logger.error(f"❌ Error getting trade history: {e}")
            return []
    
    def log_signal(self, signal: str, symbol: str, price: float, indicators: Dict, reason: str = "") -> bool:
        """Log trading signal to Supabase"""
        try:
            signal_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal': signal,
                'symbol': symbol,
                'price': float(price),
                'rsi': float(indicators.get('rsi', 0)),
                'macd': float(indicators.get('macd', 0)),
                'macd_trend': str(indicators.get('macd_trend', '')),
                'sentiment': str(indicators.get('sentiment', '')),
                'sma5': float(indicators.get('sma5', 0)),
                'sma20': float(indicators.get('sma20', 0)),
                'reason': str(reason)
            }
            
            self.supabase.table('signals').insert(signal_data).execute()
            logger.info(f"✅ Signal logged to Supabase: {signal} for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error logging signal to Supabase: {e}")
            return False
    
    def get_signal_history(self, symbol: str = None, limit: int = 100, days: int = 0) -> List[Dict]:
        """Get signal history from Supabase"""
        try:
            query = self.supabase.table('signals').select('*').order('timestamp', desc=True)
            
            if symbol:
                query = query.eq('symbol', symbol)
            
            if days > 0:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                query = query.gte('timestamp', cutoff_date.isoformat())
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            # Format timestamps to Cairo time for display
            signals = []
            for signal in response.data:
                signal_copy = signal.copy()
                if 'timestamp' in signal_copy:
                    try:
                        # Use dateutil parser for more flexible timestamp parsing
                        from dateutil import parser as dateutil_parser
                        utc_time = dateutil_parser.isoparse(signal_copy['timestamp'])
                        cairo_time = utc_time.astimezone(timezone(timedelta(hours=2)))
                        signal_copy['cairo_time'] = cairo_time.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        try:
                            # Fallback: manual parsing
                            ts_str = signal_copy['timestamp'].replace('Z', '+00:00')
                            # Python 3.7+ fromisoformat is strict, try with fixed microseconds
                            if '+' in ts_str:
                                main_part, tz_part = ts_str.rsplit('+', 1)
                                tz_part = '+' + tz_part
                            else:
                                parts = ts_str.rsplit('-', 1)
                                main_part = parts[0]
                                tz_part = '-' + parts[1]
                            
                            if '.' in main_part:
                                dt_part, us_part = main_part.rsplit('.', 1)
                                # Convert fractional seconds to microseconds (6 digits)
                                # If we have 5 digits like .11429, convert to 114290 microseconds
                                us_part = (us_part + '000000')[:6]
                                ts_str = dt_part + '.' + us_part + tz_part
                            
                            utc_time = datetime.fromisoformat(ts_str)
                            cairo_time = utc_time.astimezone(timezone(timedelta(hours=2)))
                            signal_copy['cairo_time'] = cairo_time.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as ts_err:
                            logger.warning(f"Could not parse timestamp {signal_copy['timestamp']}: {ts_err}")
                            signal_copy['cairo_time'] = 'N/A'
                signals.append(signal_copy)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting signal history: {e}")
            return []
    
    def log_error(self, error_message: str, error_type: str = "GENERAL", 
                  function_name: str = "", severity: str = "ERROR", bot_status: bool = False) -> bool:
        """Log error to Supabase"""
        try:
            error_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error_type': str(error_type),
                'error_message': str(error_message),
                'function_name': str(function_name),
                'severity': str(severity),
                'bot_status': bool(bot_status)
            }
            
            self.supabase.table('errors').insert(error_data).execute()
            logger.info(f"✅ Error logged to Supabase: {error_type} - {error_message}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error logging error to Supabase: {e}")
            return False
    
    def get_error_history(self, limit: int = 50, severity: str = None, days: int = 0) -> List[Dict]:
        """Get error history from Supabase"""
        try:
            query = self.supabase.table('errors').select('*').order('timestamp', desc=True)
            
            if severity:
                query = query.eq('severity', severity)
            
            if days > 0:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                query = query.gte('timestamp', cutoff_date.isoformat())
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            # Format timestamps to Cairo time for display
            errors = []
            for error in response.data:
                error_copy = error.copy()
                if 'timestamp' in error_copy:
                    try:
                        # Use dateutil parser for more flexible timestamp parsing
                        from dateutil import parser as dateutil_parser
                        utc_time = dateutil_parser.isoparse(error_copy['timestamp'])
                        cairo_time = utc_time.astimezone(timezone(timedelta(hours=2)))
                        error_copy['cairo_time'] = cairo_time.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        try:
                            # Fallback: manual parsing
                            ts_str = error_copy['timestamp'].replace('Z', '+00:00')
                            # Python 3.7+ fromisoformat is strict, try with fixed microseconds
                            if '+' in ts_str:
                                main_part, tz_part = ts_str.rsplit('+', 1)
                                tz_part = '+' + tz_part
                            else:
                                parts = ts_str.rsplit('-', 1)
                                main_part = parts[0]
                                tz_part = '-' + parts[1]
                            
                            if '.' in main_part:
                                dt_part, us_part = main_part.rsplit('.', 1)
                                # Convert fractional seconds to microseconds (6 digits)
                                us_part = (us_part + '000000')[:6]
                                ts_str = dt_part + '.' + us_part + tz_part
                            
                            utc_time = datetime.fromisoformat(ts_str)
                            cairo_time = utc_time.astimezone(timezone(timedelta(hours=2)))
                            error_copy['cairo_time'] = cairo_time.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as ts_err:
                            logger.warning(f"Could not parse timestamp {error_copy['timestamp']}: {ts_err}")
                            error_copy['cairo_time'] = 'N/A'
                errors.append(error_copy)
            
            return errors
            
        except Exception as e:
            logger.error(f"❌ Error getting error history: {e}")
            return []
    
    def calculate_position_value(self, symbol: str, current_price: float) -> Dict:
        """Calculate current position value and P&L"""
        position = self.get_position(symbol)
        if not position or position['quantity'] <= 0:
            return {'value': 0, 'pnl': 0, 'pnl_percentage': 0}
        
        current_value = position['quantity'] * current_price
        total_cost = position['total_cost']
        pnl = current_value - total_cost
        pnl_percentage = (pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'value': current_value,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'quantity': position['quantity'],
            'avg_buy_price': position['avg_buy_price']
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary of all positions"""
        try:
            response = self.supabase.table('positions').select('*').gt('quantity', 0).execute()
            
            summary = {
                'total_positions': len(response.data),
                'total_cost': sum(float(pos['total_cost']) for pos in response.data),
                'positions': {}
            }
            
            for pos in response.data:
                summary['positions'][pos['symbol']] = {
                    'quantity': float(pos['quantity']),
                    'avg_buy_price': float(pos['avg_buy_price']),
                    'total_cost': float(pos['total_cost'])
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio summary: {e}")
            return {'total_positions': 0, 'total_cost': 0, 'positions': {}}
    
    def migrate_from_json(self, json_file_path: str) -> bool:
        """
        Migrate existing positions.json to Supabase
        
        Args:
            json_file_path: Path to existing positions.json file
        """
        try:
            with open(json_file_path, 'r') as f:
                old_positions = json.load(f)
            
            logger.info(f"🔄 Migrating {len(old_positions)} positions from JSON...")
            
            # Check if already migrated
            migration_check = self.supabase.table('migration_status')\
                .select('*')\
                .eq('migration_name', 'positions_json_import')\
                .execute()
            
            if (migration_check.data and 
                migration_check.data[0]['status'] == 'completed'):
                logger.info("⏭️ JSON migration already completed")
                return True
            
            migrated_count = 0
            for symbol, position_data in old_positions.items():
                trades = position_data.get('trades', [])
                
                for trade in trades:
                    success = self.add_trade(
                        symbol=symbol,
                        action=trade['action'],
                        quantity=float(trade['quantity']),
                        price=float(trade['price']),
                        timestamp=trade['timestamp'],
                        source='positions_json_import'
                    )
                    
                    if success:
                        migrated_count += 1
            
            # Mark migration as completed
            self.supabase.table('migration_status')\
                .update({'status': 'completed', 'completed_at': datetime.now(timezone.utc).isoformat()})\
                .eq('migration_name', 'positions_json_import')\
                .execute()
            
            logger.info(f"✅ Migrated {migrated_count} trades from JSON")
            return True
            
        except Exception as e:
            logger.error(f"❌ JSON migration failed: {e}")
            
            # Mark migration as failed
            self.supabase.table('migration_status')\
                .update({
                    'status': 'failed', 
                    'error_message': str(e),
                    'completed_at': datetime.now(timezone.utc).isoformat()
                })\
                .eq('migration_name', 'positions_json_import')\
                .execute()
            
            return False
    
    def migrate_from_binance_excel(self, excel_file_path: str) -> bool:
        """
        Migrate historical data from Binance Excel export
        
        Args:
            excel_file_path: Path to Binance order history Excel file
        """
        try:
            logger.info(f"🔄 Migrating Binance history from {excel_file_path}...")
            
            # Check if already migrated
            migration_check = self.supabase.table('migration_status')\
                .select('*')\
                .eq('migration_name', 'binance_historical_import')\
                .execute()
            
            if (migration_check.data and 
                migration_check.data[0]['status'] == 'completed'):
                logger.info("⏭️ Binance migration already completed")
                return True
            
            # Read Excel file
            df = pd.read_excel(excel_file_path)
            logger.info(f"📊 Found {len(df)} records in Excel file")
            
            # Expected columns (adjust based on your Excel structure):
            # Date(UTC), Pair, Type, Order Price, Order Amount, AvgTrading Price, Filled, Total
            
            migrated_count = 0
            for _, row in df.iterrows():
                try:
                    # Parse Binance data (adjust column names as needed)
                    symbol = row.get('Pair', '').replace('/', '')  # Convert BTC/USDT to BTCUSDT
                    if not symbol:
                        continue
                    
                    action = 'BUY' if row.get('Type', '').upper() in ['BUY', 'MARKET_BUY'] else 'SELL'
                    quantity = float(row.get('Filled', 0) or row.get('Order Amount', 0))
                    price = float(row.get('AvgTrading Price', 0) or row.get('Order Price', 0))
                    
                    # Parse timestamp
                    timestamp_str = str(row.get('Date(UTC)', ''))
                    if timestamp_str and timestamp_str != 'nan':
                        timestamp = pd.to_datetime(timestamp_str).isoformat() + '+00:00'
                    else:
                        continue  # Skip records without valid timestamp
                    
                    if quantity > 0 and price > 0:
                        success = self.add_trade(
                            symbol=symbol,
                            action=action,
                            quantity=quantity,
                            price=price,
                            timestamp=timestamp,
                            source='binance_history'
                        )
                        
                        if success:
                            migrated_count += 1
                
                except Exception as row_error:
                    logger.warning(f"⚠️ Skipping row due to error: {row_error}")
                    continue
            
            # Mark migration as completed
            self.supabase.table('migration_status')\
                .update({'status': 'completed', 'completed_at': datetime.now(timezone.utc).isoformat()})\
                .eq('migration_name', 'binance_historical_import')\
                .execute()
            
            logger.info(f"✅ Migrated {migrated_count} Binance trades")
            return True
            
        except Exception as e:
            logger.error(f"❌ Binance migration failed: {e}")
            
            # Mark migration as failed
            self.supabase.table('migration_status')\
                .update({
                    'status': 'failed', 
                    'error_message': str(e),
                    'completed_at': datetime.now(timezone.utc).isoformat()
                })\
                .eq('migration_name', 'binance_historical_import')\
                .execute()
            
            return False
    
    def health_check(self) -> Dict:
        """Check Supabase connection and data integrity"""
        try:
            # Test connection
            response = self.supabase.table('configuration').select('key').limit(1).execute()
            
            # Count records
            trades_count = self.supabase.table('trades').select('id', count='exact').execute()
            positions_count = self.supabase.table('positions').select('symbol', count='exact').execute()
            
            return {
                'status': 'healthy',
                'connected': True,
                'trades_count': trades_count.count if hasattr(trades_count, 'count') else 0,
                'positions_count': positions_count.count if hasattr(positions_count, 'count') else 0,
                'last_updated': self._cache_last_updated.isoformat() if self._cache_last_updated else None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'connected': False,
                'error': str(e)
            }
    
    # Compatibility methods for existing code
    def update_position(self, symbol: str, quantity: float, price: float, action: str):
        """Compatibility method - matches old SmartPositionTracker signature"""
        return self.add_trade(symbol, action, quantity, price)
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get current position info for symbol - compatibility method"""
        return self.get_position(symbol)
    
    def calculate_profit_potential(self, symbol: str, current_price: float) -> Dict:
        """Calculate profit potential at current price - compatibility method"""
        position = self.get_position(symbol)
        
        if not position or position['quantity'] <= 0:
            return {
                'has_position': False,
                'profit_pct': 0.0,
                'profit_usd': 0.0,
                'is_profitable': False
            }
        
        avg_buy_price = position['avg_buy_price']
        quantity = position['quantity']
        
        profit_per_unit = current_price - avg_buy_price
        total_profit = profit_per_unit * quantity
        profit_pct = (profit_per_unit / avg_buy_price) * 100
        
        return {
            'has_position': True,
            'avg_buy_price': avg_buy_price,
            'current_price': current_price,
            'quantity': quantity,
            'profit_per_unit': profit_per_unit,
            'profit_usd': total_profit,
            'profit_pct': profit_pct,
            'is_profitable': profit_pct > 0,
            'meets_minimum_profit': profit_pct >= 2.0  # Default minimum profit
        }
    
    def should_allow_partial_sell(self, symbol: str, current_price: float, 
                                 minimum_profit_pct: float = 2.0) -> Tuple[bool, str]:
        """Determine if partial sell should be allowed based on profitability - compatibility method"""
        profit_info = self.calculate_profit_potential(symbol, current_price)
        
        if not profit_info['has_position']:
            return False, f"No tracked position for {symbol}"
        
        if not profit_info['is_profitable']:
            return False, f"Would sell at loss: {profit_info['profit_pct']:.2f}% (avg buy: ${profit_info['avg_buy_price']:.4f}, current: ${current_price:.4f})"
        
        if profit_info['profit_pct'] < minimum_profit_pct:
            return False, f"Profit too low: {profit_info['profit_pct']:.2f}% < {minimum_profit_pct}% minimum"
        
        return True, f"Profitable sell: {profit_info['profit_pct']:.2f}% profit (avg buy: ${profit_info['avg_buy_price']:.4f})"

# Compatibility wrapper - can replace existing SmartPositionTracker
class SmartPositionTracker(SupabasePositionTracker):
    """
    Drop-in replacement for file-based SmartPositionTracker
    Maintains the same interface while using Supabase backend
    """
    
    def __init__(self, positions_file=None):
        # Initialize Supabase (ignore file parameter)
        super().__init__()
        
        # Auto-migrate existing JSON if it exists
        if positions_file and os.path.exists(positions_file):
            logger.info(f"🔄 Found existing positions file: {positions_file}")
            self.migrate_from_json(positions_file)
    
    def save_positions(self):
        """Compatibility method - positions are auto-saved in Supabase"""
        logger.info("💾 Positions automatically saved to Supabase")
        return True
    
    def get_current_positions(self) -> Dict:
        """Compatibility method for existing code"""
        return self.positions
    
    def rebuild_positions_from_history(self):
        """Compatibility method - not needed with Supabase (auto-calculated)"""
        logger.info("🔄 Positions are automatically calculated from trades in Supabase")
        return True


# Factory function for easy integration
def get_position_tracker():
    """Get the appropriate position tracker based on configuration"""
    try:
        # Try Supabase first
        if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
            return SupabasePositionTracker()
        else:
            logger.warning("⚠️ Supabase not configured, falling back to file-based tracker")
            # Fallback to original file-based tracker
            from smart_position_tracker import SmartPositionTracker as FileBasedTracker
            return FileBasedTracker()
    except Exception as e:
        logger.error(f"❌ Failed to initialize position tracker: {e}")
        raise