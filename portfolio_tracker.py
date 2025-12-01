#!/usr/bin/env python3
"""
Portfolio Tracker - Track total portfolio value and real P&L
"""

import json
import os
from datetime import datetime

class PortfolioTracker:
    """Portfolio tracking with Singleton pattern to ensure consistent state"""
    _instance = None
    _initialized = False
    
    def __new__(cls, data_file='portfolio_tracking.json'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, data_file='portfolio_tracking.json'):
        # Only initialize once
        if self._initialized:
            return
        
        self.data_file = data_file
        self.data = self._load_data()
        
        PortfolioTracker._initialized = True
    
    def _load_data(self):
        """Load portfolio tracking data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default data structure
        return {
            'starting_balance': None,
            'last_balance': None,
            'start_date': None,
            'last_update': None,
            'total_deposits': 0,
            'total_withdrawals': 0
        }
    
    def _save_data(self):
        """Save portfolio tracking data"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save portfolio data: {e}")
    
    def set_starting_balance(self, balance):
        """Set the starting balance (call once at beginning)"""
        if self.data['starting_balance'] is None:
            self.data['starting_balance'] = balance
            self.data['start_date'] = datetime.now().isoformat()
            self.data['last_balance'] = balance
            self._save_data()
            print(f"🎯 Starting balance set: ${balance:.2f}")
        else:
            print(f"📊 Starting balance already set: ${self.data['starting_balance']:.2f}")
    
    def update_balance(self, current_balance):
        """Update current balance and calculate P&L"""
        if self.data['starting_balance'] is None:
            # Auto-set starting balance if not set
            self.set_starting_balance(current_balance)
            return 0
        
        # Calculate real P&L
        real_pnl = current_balance - self.data['starting_balance'] - self.data['total_deposits'] + self.data['total_withdrawals']
        
        # Update tracking
        self.data['last_balance'] = current_balance
        self.data['last_update'] = datetime.now().isoformat()
        self._save_data()
        
        return real_pnl
    
    def add_deposit(self, amount):
        """Record a deposit to adjust P&L calculation"""
        self.data['total_deposits'] += amount
        self._save_data()
        print(f"💰 Deposit recorded: ${amount:.2f}")
    
    def add_withdrawal(self, amount):
        """Record a withdrawal to adjust P&L calculation"""
        self.data['total_withdrawals'] += amount
        self._save_data()
        print(f"💸 Withdrawal recorded: ${amount:.2f}")
    
    def get_summary(self):
        """Get portfolio summary"""
        if self.data['starting_balance'] is None:
            return "No portfolio data available"
        
        current = self.data['last_balance'] or 0
        starting = self.data['starting_balance']
        deposits = self.data['total_deposits']
        withdrawals = self.data['total_withdrawals']
        
        real_pnl = current - starting - deposits + withdrawals
        pnl_pct = (real_pnl / starting * 100) if starting > 0 else 0
        
        return {
            'starting_balance': starting,
            'current_balance': current,
            'total_deposits': deposits,
            'total_withdrawals': withdrawals,
            'real_pnl': real_pnl,
            'pnl_percentage': pnl_pct,
            'start_date': self.data['start_date'],
            'last_update': self.data['last_update']
        }

# Global instance
_portfolio_tracker = None

def get_portfolio_tracker():
    """Get global portfolio tracker instance"""
    global _portfolio_tracker
    if _portfolio_tracker is None:
        _portfolio_tracker = PortfolioTracker()
    return _portfolio_tracker

if __name__ == "__main__":
    # Test the tracker
    tracker = PortfolioTracker()
    
    # Simulate user's scenario
    tracker.set_starting_balance(1000.0)  # Started with $1000
    
    # Update to current balance
    current_pnl = tracker.update_balance(1014.0)  # Now has $1014 (assuming same $14 profit)
    
    print("\n📊 PORTFOLIO SUMMARY:")
    summary = tracker.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: ${value:.2f}")
        else:
            print(f"{key}: {value}")