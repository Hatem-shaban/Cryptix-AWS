#!/usr/bin/env python3
"""
Smart Position Tracker for CRYPTIX Trading Bot
Tracks buy/sell positions and calculates average buy prices for profit optimization
"""

import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import config

class SmartPositionTracker:
    """
    Tracks positions and calculates weighted average buy prices to ensure profitable sells
    """
    
    def __init__(self):
        self.positions = {}  # {symbol: {quantity: float, avg_buy_price: float, trades: []}}
        self.positions_file = Path("logs") / "positions.json"
        self.load_positions()
    
    def load_positions(self):
        """Load existing positions from file"""
        try:
            if self.positions_file.exists():
                with open(self.positions_file, 'r') as f:
                    self.positions = json.load(f)
                print(f"📊 Loaded {len(self.positions)} tracked positions")
            else:
                self.rebuild_positions_from_history()
        except Exception as e:
            print(f"⚠️ Error loading positions: {e}")
            self.rebuild_positions_from_history()
    
    def save_positions(self):
        """Save positions to file"""
        try:
            self.positions_file.parent.mkdir(exist_ok=True)
            with open(self.positions_file, 'w') as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving positions: {e}")
    
    def rebuild_positions_from_history(self):
        """Rebuild position tracking from trade history CSV"""
        print("🔄 Rebuilding positions from trade history...")
        self.positions = {}
        
        try:
            trade_history_file = Path("logs") / "trade_history.csv"
            if not trade_history_file.exists():
                print("📝 No trade history found, starting fresh")
                return
            
            with open(trade_history_file, 'r') as f:
                reader = csv.DictReader(f)
                trades = list(reader)
            
            # Process trades in chronological order (reverse since CSV has newest first)
            trades.reverse()
            
            for trade in trades:
                signal = trade.get('signal', '').upper()
                symbol = trade.get('symbol', '')
                quantity = float(trade.get('quantity', 0))
                price = float(trade.get('price', 0))
                
                if signal == 'BUY' and symbol and quantity > 0 and price > 0:
                    self.update_position(symbol, quantity, price, 'BUY')
                elif signal in ['SELL', 'SELL_PARTIAL'] and symbol and quantity > 0:
                    self.update_position(symbol, quantity, price, 'SELL')
            
            print(f"✅ Rebuilt {len(self.positions)} positions from history")
            self.save_positions()
            
        except Exception as e:
            print(f"❌ Error rebuilding positions: {e}")
    
    def update_position(self, symbol: str, quantity: float, price: float, action: str):
        """Update position based on trade"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0.0,
                'avg_buy_price': 0.0,
                'total_cost': 0.0,
                'trades': []
            }
        
        position = self.positions[symbol]
        
        if action == 'BUY':
            # Add to position
            new_total_cost = position['total_cost'] + (quantity * price)
            new_quantity = position['quantity'] + quantity
            
            if new_quantity > 0:
                position['avg_buy_price'] = new_total_cost / new_quantity
                position['quantity'] = new_quantity
                position['total_cost'] = new_total_cost
            
            position['trades'].append({
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
            
        elif action == 'SELL':
            # Reduce position
            position['quantity'] = max(0, position['quantity'] - quantity)
            
            if position['quantity'] <= 0:
                # Position fully closed
                position['quantity'] = 0.0
                position['avg_buy_price'] = 0.0
                position['total_cost'] = 0.0
            else:
                # Partial sell - reduce total cost proportionally
                position['total_cost'] = position['quantity'] * position['avg_buy_price']
            
            position['trades'].append({
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
        
        # Clean up empty positions
        if position['quantity'] <= 0:
            del self.positions[symbol]
        
        self.save_positions()
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get current position info for symbol"""
        return self.positions.get(symbol)
    
    def calculate_profit_potential(self, symbol: str, current_price: float) -> Dict:
        """Calculate profit potential at current price"""
        position = self.get_position_info(symbol)
        
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
            'meets_minimum_profit': profit_pct >= config.REBALANCING.get('minimum_profit_pct', 2.0)
        }
    
    def should_allow_partial_sell(self, symbol: str, current_price: float, 
                                 minimum_profit_pct: float = 2.0) -> Tuple[bool, str]:
        """Determine if partial sell should be allowed based on profitability"""
        profit_info = self.calculate_profit_potential(symbol, current_price)
        
        if not profit_info['has_position']:
            return False, f"No tracked position for {symbol}"
        
        if not profit_info['is_profitable']:
            return False, f"Would sell at loss: {profit_info['profit_pct']:.2f}% (avg buy: ${profit_info['avg_buy_price']:.4f}, current: ${current_price:.4f})"
        
        if profit_info['profit_pct'] < minimum_profit_pct:
            return False, f"Profit too low: {profit_info['profit_pct']:.2f}% < {minimum_profit_pct}% minimum"
        
        return True, f"Profitable sell: {profit_info['profit_pct']:.2f}% profit (avg buy: ${profit_info['avg_buy_price']:.4f})"
    
    def get_summary(self) -> Dict:
        """Get summary of all positions"""
        summary = {
            'total_positions': len(self.positions),
            'positions': {}
        }
        
        for symbol, position in self.positions.items():
            summary['positions'][symbol] = {
                'quantity': position['quantity'],
                'avg_buy_price': position['avg_buy_price'],
                'total_invested': position['total_cost'],
                'trades_count': len(position['trades'])
            }
        
        return summary
    
    def print_positions_summary(self):
        """Print a formatted summary of all positions"""
        if not self.positions:
            print("📊 No active positions tracked")
            return
        
        print(f"\n📊 Position Summary ({len(self.positions)} active positions):")
        print("=" * 70)
        
        for symbol, position in self.positions.items():
            print(f"🔹 {symbol}:")
            print(f"   Quantity: {position['quantity']:.8f}")
            print(f"   Avg Buy Price: ${position['avg_buy_price']:.4f}")
            print(f"   Total Invested: ${position['total_cost']:.2f}")
            print(f"   Trades: {len(position['trades'])}")
            print()

# Global instance
_position_tracker = None

def get_position_tracker() -> SmartPositionTracker:
    """Get global position tracker instance"""
    global _position_tracker
    if _position_tracker is None:
        _position_tracker = SmartPositionTracker()
    return _position_tracker

if __name__ == "__main__":
    # Test the position tracker
    tracker = SmartPositionTracker()
    tracker.print_positions_summary()