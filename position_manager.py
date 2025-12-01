#!/usr/bin/env python3
"""
Advanced Position and Order Size Management for CRYPTIX Trading Bot
Implements optimal position sizing, risk-adjusted order execution, and portfolio balance
"""

import math
import numpy as np
from datetime import datetime, timedelta
import config
from typing import Dict, Tuple, Optional

class AdvancedPositionManager:
    """
    Advanced position sizing and risk management for optimal trading execution
    """
    
    def __init__(self):
        self.volatility_cache = {}
        self.position_history = []
        self.max_position_history = 100
        self.correlation_cache = {}  # Cache for correlation calculations
        
    def check_correlation_risk(self, new_symbol: str, current_positions: list) -> Dict:
        """
        Check correlation risk for new position
        Returns dict with risk assessment and position size multiplier
        """
        if len(current_positions) == 0:
            return {'approved': True, 'multiplier': 1.0, 'reason': 'No existing positions'}
            
        # Count highly correlated pairs
        high_correlation_count = 0
        correlation_threshold = 0.7  # Consider pairs with >70% correlation
        
        for pos in current_positions:
            correlation = self.correlation_cache.get(f"{new_symbol}_{pos['symbol']}", 0.5)
            if correlation > correlation_threshold:
                high_correlation_count += 1
                
        # Risk assessment based on correlation
        if high_correlation_count >= 2:
            return {
                'approved': False,
                'multiplier': 0.0,
                'reason': f'Too many correlated positions ({high_correlation_count})'
            }
        elif high_correlation_count == 1:
            return {
                'approved': True,
                'multiplier': 0.7,  # Reduce size by 30%
                'reason': 'Moderate correlation detected'
            }
            
        return {'approved': True, 'multiplier': 1.0, 'reason': 'Low correlation risk'}
    
    def calculate_optimal_position_size(self, 
                                      symbol: str, 
                                      signal: str, 
                                      current_price: float,
                                      account_balance: float,
                                      volatility: float,
                                      confidence_score: float = 0.5,
                                      market_regime: str = "NORMAL") -> Dict:
        """
        Calculate optimal position size using multiple factors:
        - Kelly Criterion for optimal sizing
        - Volatility-based position scaling
        - Confidence-weighted sizing
        - Market regime adjustments
        - Portfolio heat management
        """
        try:
            # Base risk amount from config
            base_risk_pct = config.RISK_PERCENTAGE / 100
            
            # 1. Volatility Adjustment
            volatility_factor = self._calculate_volatility_factor(volatility, market_regime)
            
            # 2. Confidence-based sizing (0.1 to 1.0 multiplier)
            confidence_factor = max(0.1, min(1.0, confidence_score))
            
            # 3. Market regime adjustment
            regime_factor = self._get_market_regime_factor(market_regime)
            
            # 4. Portfolio heat check (reduce if too many open positions)
            heat_factor = self._calculate_portfolio_heat_factor()
            
            # 5. Kelly Criterion estimate (simplified)
            kelly_factor = self._estimate_kelly_factor(symbol, signal)
            
            # Check correlation risk
            correlation_check = self.check_correlation_risk(symbol, 
                getattr(config, 'ACTIVE_POSITIONS', []))
            correlation_factor = correlation_check['multiplier']
            
            # Enhanced risk factor calculation
            total_risk_factor = (
                base_risk_pct * 
                volatility_factor * 
                confidence_factor * 
                regime_factor * 
                heat_factor * 
                kelly_factor * 
                correlation_factor  # Apply correlation adjustment
            )
            
            # Additional safeguards for bearish markets
            if market_regime == 'BEARISH':
                total_risk_factor *= 0.5  # 50% reduction in bearish markets
                
            # Volatility circuit breaker
            if volatility > 1.2:  # Extreme volatility
                total_risk_factor *= 0.3  # 70% reduction
            
            # Calculate position size
            risk_amount = account_balance * total_risk_factor
            raw_position_size = risk_amount / current_price
            
            # Apply minimum and maximum constraints
            min_trade_usdt = config.MIN_TRADE_USDT
            max_position_pct = getattr(config, 'MAX_POSITION_PCT', 10.0) / 100  # 10% max position
            
            min_quantity = min_trade_usdt / current_price
            max_quantity = (account_balance * max_position_pct) / current_price
            
            # Final position size
            optimal_quantity = max(min_quantity, min(max_quantity, raw_position_size))
            
            return {
                'quantity': optimal_quantity,
                'risk_amount': optimal_quantity * current_price,
                'risk_percentage': (optimal_quantity * current_price / account_balance) * 100,
                'factors': {
                    'volatility_factor': volatility_factor,
                    'confidence_factor': confidence_factor,
                    'regime_factor': regime_factor,
                    'heat_factor': heat_factor,
                    'kelly_factor': kelly_factor
                },
                'constraints': {
                    'min_quantity': min_quantity,
                    'max_quantity': max_quantity,
                    'raw_size': raw_position_size
                }
            }
            
        except Exception as e:
            print(f"Error calculating optimal position size: {e}")
            # Fallback to basic sizing
            fallback_risk = account_balance * (config.RISK_PERCENTAGE / 100)
            return {
                'quantity': fallback_risk / current_price,
                'risk_amount': fallback_risk,
                'risk_percentage': config.RISK_PERCENTAGE,
                'factors': {'error': str(e)},
                'constraints': {}
            }
    
    def _calculate_volatility_factor(self, volatility: float, market_regime: str) -> float:
        """
        Enhanced volatility-based position sizing with market regime consideration
        """
        # Dynamic volatility thresholds
        extreme_vol = 1.2
        high_vol = 0.8
        medium_vol = 0.5
        low_vol = 0.2
        
        # Market regime multipliers
        regime_multipliers = {
            'BEARISH': 0.5,    # 50% reduction in bearish markets
            'VOLATILE': 0.7,   # 30% reduction in volatile markets
            'EXTREME': 0.5,    # 50% reduction in extreme conditions
            'NORMAL': 1.0      # No reduction in normal markets
        }
        
        # Regime-specific adjustments
        regime_multipliers = {
            'QUIET': 1.2,      # Larger positions in quiet markets
            'NORMAL': 1.0,     # Standard sizing
            'VOLATILE': 0.8,   # Smaller positions in volatile markets
            'EXTREME': 0.5     # Much smaller positions in extreme volatility
        }
        
        # Calculate volatility factor (0.3 to 1.5)
        if volatility <= low_vol:
            vol_factor = 1.2  # Low volatility allows larger positions
        elif volatility <= medium_vol:
            vol_factor = 1.0  # Medium volatility uses standard sizing
        elif volatility <= high_vol:
            vol_factor = 0.7  # High volatility reduces position size
        else:
            vol_factor = 0.4  # Extreme volatility heavily reduces size
        
        # Apply market regime adjustment
        regime_mult = regime_multipliers.get(market_regime, 1.0)
        
        return vol_factor * regime_mult
    
    def _get_market_regime_factor(self, market_regime: str) -> float:
        """
        Adjust position size based on market regime
        """
        regime_factors = {
            'QUIET': 0.8,      # Conservative in quiet markets
            'NORMAL': 1.0,     # Standard sizing
            'VOLATILE': 1.1,   # Slightly larger in volatile (more opportunities)
            'EXTREME': 0.6     # Very conservative in extreme conditions
        }
        return regime_factors.get(market_regime, 1.0)
    
    def _calculate_portfolio_heat_factor(self) -> float:
        """
        Reduce position size if portfolio is overheated (too many positions)
        """
        # This would integrate with the rebalancing system
        # For now, return a conservative factor
        max_heat = getattr(config, 'MAX_PORTFOLIO_HEAT', 5)  # Max 5 concurrent positions
        current_heat = len(getattr(config.REBALANCING, 'assets_to_monitor', []))
        
        if current_heat >= max_heat:
            return 0.5  # Reduce to 50% if overheated
        elif current_heat >= max_heat * 0.8:
            return 0.7  # Reduce to 70% if getting hot
        else:
            return 1.0  # Normal sizing
    
    def _estimate_kelly_factor(self, symbol: str, signal: str) -> float:
        """
        Simplified Kelly Criterion estimation for position sizing
        Based on historical win rate and average win/loss ratio
        """
        try:
            # Get recent performance for this symbol/signal type
            win_rate = getattr(config, 'ESTIMATED_WIN_RATE', 0.55)  # 55% default
            avg_win = getattr(config, 'ESTIMATED_AVG_WIN', 1.5)    # 1.5% average win
            avg_loss = getattr(config, 'ESTIMATED_AVG_LOSS', 1.0)  # 1.0% average loss
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss > 0:
                b = avg_win / avg_loss
                p = win_rate
                q = 1 - win_rate
                kelly_fraction = (b * p - q) / b
                
                # Cap Kelly at reasonable levels (0.1 to 1.5)
                kelly_factor = max(0.1, min(1.5, kelly_fraction))
            else:
                kelly_factor = 0.5  # Conservative default
                
            return kelly_factor
            
        except Exception:
            return 0.5  # Safe default
    
    def calculate_dynamic_stop_loss(self, 
                                  entry_price: float,
                                  volatility: float,
                                  atr: float,
                                  signal: str) -> float:
        """
        Calculate dynamic stop loss based on volatility and ATR
        """
        # Base stop loss percentages
        base_stop_pct = getattr(config, 'BASE_STOP_LOSS_PCT', 3.0) / 100  # 3%
        
        # ATR-based stop (2x ATR is common)
        atr_multiplier = getattr(config, 'ATR_STOP_MULTIPLIER', 2.0)
        atr_stop = atr * atr_multiplier
        
        # Volatility adjustment (higher vol = wider stops)
        vol_multiplier = 1.0 + (volatility * 2)  # Scale volatility impact
        
        # Calculate stop distance
        if signal == "BUY":
            # For longs, stop below entry
            pct_stop = entry_price * base_stop_pct * vol_multiplier
            final_stop = entry_price - max(pct_stop, atr_stop)
        else:  # SELL
            # For shorts, stop above entry
            pct_stop = entry_price * base_stop_pct * vol_multiplier
            final_stop = entry_price + max(pct_stop, atr_stop)
        
        return final_stop
    
    def should_scale_out(self, 
                        current_price: float,
                        entry_price: float,
                        unrealized_pnl_pct: float,
                        position_age_minutes: int,
                        market_regime: str = "NORMAL",
                        volatility: float = None) -> Dict:
        """
        Enhanced scale out logic with volatility and market regime consideration
        """
        # Dynamic scale out thresholds based on market conditions
        scale_out_config = {
            'NORMAL': {
                'first_target_pct': 2.0,     # Scale 30% at 2% profit
                'second_target_pct': 4.0,    # Scale 50% at 4% profit
                'time_based_pct': 1.0,       # Scale if >4h with 1% profit
                'max_hold_minutes': 240      # 4 hours
            },
            'VOLATILE': {
                'first_target_pct': 1.0,     # Scale 30% at 1% profit
                'second_target_pct': 2.0,    # Scale 50% at 2% profit
                'time_based_pct': 0.5,       # Scale if >2h with 0.5% profit
                'max_hold_minutes': 120      # 2 hours
            },
            'BEARISH': {
                'first_target_pct': 0.8,     # Scale 30% at 0.8% profit
                'second_target_pct': 1.5,    # Scale 50% at 1.5% profit
                'time_based_pct': 0.4,       # Scale if >1h with 0.4% profit
                'max_hold_minutes': 60       # 1 hour
            }
        }
        
        should_scale = False
        scale_percentage = 0
        reason = ""
        
        # Get regime-specific config
        regime_config = scale_out_config.get(market_regime, scale_out_config['NORMAL'])
        
        # Emergency exit in extreme conditions
        if volatility and volatility > 1.2:  # Extreme volatility
            if unrealized_pnl_pct > 0:
                return {
                    'should_scale': True,
                    'scale_percentage': 100,  # Full exit
                    'reason': f"Emergency exit: Extreme volatility {volatility:.2f}",
                    'targets': regime_config
                }
        
        # Smart profit taking based on market regime
        if unrealized_pnl_pct >= regime_config['second_target_pct']:
            should_scale = True
            scale_percentage = 75 if market_regime == 'BEARISH' else 50
            reason = f"Hit second target: {unrealized_pnl_pct:.1f}% ({market_regime} market)"
        elif unrealized_pnl_pct >= regime_config['first_target_pct']:
            should_scale = True
            scale_percentage = 50 if market_regime == 'BEARISH' else 30
            reason = f"Hit first target: {unrealized_pnl_pct:.1f}% ({market_regime} market)"
        
        # Time-based scaling
        elif (position_age_minutes >= scale_out_config['max_hold_minutes'] and 
              unrealized_pnl_pct >= scale_out_config['time_based_pct']):
            should_scale = True
            scale_percentage = 25  # Scale out 25%
            reason = f"Time-based exit: {position_age_minutes}min old with {unrealized_pnl_pct:.1f}% profit"
        
        return {
            'should_scale': should_scale,
            'scale_percentage': scale_percentage,
            'reason': reason,
            'targets': scale_out_config
        }

def get_position_manager():
    """Get singleton instance of position manager"""
    if not hasattr(get_position_manager, '_instance'):
        get_position_manager._instance = AdvancedPositionManager()
    return get_position_manager._instance
