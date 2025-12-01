#!/usr/bin/env python3
"""
Enhanced Risk Management System for CRYPTIX Trading Bot
Implements advanced risk controls, portfolio protection, and dynamic risk adjustment
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import config

class EnhancedRiskManager:
    """
    Advanced risk management system with multiple layers of protection
    Implements Singleton pattern to maintain consistent risk state across the application
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
        
        self.risk_metrics = {}
        self.drawdown_tracker = {}
        self.exposure_tracker = {}
        self.risk_alerts = []
        self.last_risk_check = None
        self.emergency_stop = False
        
        # Initialize risk limits from config
        self.max_daily_loss = getattr(config, 'MAX_DAILY_LOSS', 50.0)
        self.max_drawdown = getattr(config, 'MAX_DRAWDOWN', 15.0)
        self.max_portfolio_exposure = getattr(config, 'MAX_PORTFOLIO_EXPOSURE', 80.0)
        self.max_consecutive_losses = getattr(config, 'MAX_CONSECUTIVE_LOSSES', 5)
        
        EnhancedRiskManager._initialized = True
        
    def comprehensive_risk_check(self, 
                                account_balance: float,
                                current_positions: Dict,
                                proposed_trade: Dict,
                                market_conditions: Dict) -> Dict:
        """
        Comprehensive risk assessment before executing any trade
        """
        try:
            risk_assessment = {
                'approved': True,
                'risk_score': 0.0,
                'warnings': [],
                'blocks': [],
                'adjustments': {},
                'metrics': {}
            }
            
            # 1. Portfolio Heat Check
            heat_check = self._check_portfolio_heat(current_positions, account_balance)
            risk_assessment['metrics']['portfolio_heat'] = heat_check
            
            # 2. Drawdown Protection
            drawdown_check = self._check_drawdown_limits(account_balance)
            risk_assessment['metrics']['drawdown'] = drawdown_check
            
            # 3. Daily Loss Limits
            daily_loss_check = self._check_daily_loss_limits()
            risk_assessment['metrics']['daily_loss'] = daily_loss_check
            
            # 4. Position Correlation Risk
            correlation_check = self._check_position_correlation(current_positions, proposed_trade)
            risk_assessment['metrics']['correlation'] = correlation_check
            
            # 5. Market Volatility Assessment
            volatility_check = self._assess_market_volatility_risk(market_conditions)
            risk_assessment['metrics']['volatility'] = volatility_check
            
            # 6. Liquidity Risk Assessment
            liquidity_check = self._assess_liquidity_risk(proposed_trade, market_conditions)
            risk_assessment['metrics']['liquidity'] = liquidity_check
            
            # 7. Consecutive Loss Protection
            consecutive_loss_check = self._check_consecutive_losses()
            risk_assessment['metrics']['consecutive_losses'] = consecutive_loss_check
            
            # 8. Emergency Stop Check
            emergency_check = self._check_emergency_conditions(account_balance, market_conditions)
            risk_assessment['metrics']['emergency'] = emergency_check
            
            # Combine all risk assessments
            final_assessment = self._combine_risk_assessments(risk_assessment)
            
            # Log risk assessment
            self._log_risk_assessment(final_assessment)
            
            return final_assessment
            
        except Exception as e:
            print(f"Error in comprehensive risk check: {e}")
            return {
                'approved': False,
                'risk_score': 1.0,
                'warnings': [],
                'blocks': [f"Risk check error: {e}"],
                'adjustments': {},
                'metrics': {}
            }
    
    def _check_portfolio_heat(self, current_positions: Dict, account_balance: float) -> Dict:
        """
        Check portfolio heat (total risk exposure)
        """
        try:
            total_exposure = 0.0
            position_count = len(current_positions)
            
            for symbol, position in current_positions.items():
                position_value = position.get('value', 0)
                total_exposure += position_value
            
            exposure_pct = (total_exposure / account_balance) * 100 if account_balance > 0 else 0
            
            heat_level = "LOW"
            if exposure_pct > self.max_portfolio_exposure:
                heat_level = "CRITICAL"
            elif exposure_pct > self.max_portfolio_exposure * 0.8:
                heat_level = "HIGH"
            elif exposure_pct > self.max_portfolio_exposure * 0.6:
                heat_level = "MEDIUM"
            
            return {
                'exposure_pct': exposure_pct,
                'position_count': position_count,
                'heat_level': heat_level,
                'max_allowed': self.max_portfolio_exposure,
                'risk_score': min(1.0, exposure_pct / self.max_portfolio_exposure)
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
    
    def _check_drawdown_limits(self, current_balance: float) -> Dict:
        """
        Monitor and protect against excessive drawdowns
        """
        try:
            # Get peak balance (this would be stored in persistent storage)
            peak_balance = getattr(self, 'peak_balance', current_balance)
            if current_balance > peak_balance:
                self.peak_balance = current_balance
                peak_balance = current_balance
            
            # Calculate current drawdown
            drawdown_amount = peak_balance - current_balance
            drawdown_pct = (drawdown_amount / peak_balance) * 100 if peak_balance > 0 else 0
            
            # Determine risk level
            if drawdown_pct > self.max_drawdown:
                risk_level = "CRITICAL"
            elif drawdown_pct > self.max_drawdown * 0.8:
                risk_level = "HIGH"
            elif drawdown_pct > self.max_drawdown * 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'current_drawdown_pct': drawdown_pct,
                'max_allowed_pct': self.max_drawdown,
                'peak_balance': peak_balance,
                'current_balance': current_balance,
                'risk_level': risk_level,
                'risk_score': min(1.0, drawdown_pct / self.max_drawdown)
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
    
    def _check_daily_loss_limits(self) -> Dict:
        """
        Check daily loss limits to prevent catastrophic losses
        """
        try:
            # Get today's P&L (this would be calculated from trade history)
            today = datetime.now().date()
            daily_pnl = getattr(self, 'daily_pnl', 0.0)  # This should be tracked elsewhere
            
            # Calculate risk level
            loss_pct = abs(daily_pnl) / self.max_daily_loss if daily_pnl < 0 else 0
            
            if daily_pnl < -self.max_daily_loss:
                risk_level = "CRITICAL"
            elif daily_pnl < -self.max_daily_loss * 0.8:
                risk_level = "HIGH"
            elif daily_pnl < -self.max_daily_loss * 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'daily_pnl': daily_pnl,
                'max_daily_loss': self.max_daily_loss,
                'loss_percentage': loss_pct * 100,
                'risk_level': risk_level,
                'risk_score': loss_pct
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
    
    def _check_position_correlation(self, current_positions: Dict, proposed_trade: Dict) -> Dict:
        """
        Check correlation risk between positions to avoid overexposure to correlated assets
        """
        try:
            # Simplified correlation check based on asset categories
            crypto_majors = ['BTC', 'ETH']
            altcoins = ['BNB', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX']
            defi_tokens = ['UNI', 'AAVE', 'COMP', 'MKR']
            
            categories = {
                'majors': crypto_majors,
                'altcoins': altcoins,
                'defi': defi_tokens
            }
            
            # Count positions by category
            category_exposure = {cat: 0 for cat in categories}
            
            for symbol in current_positions:
                base_asset = symbol.replace('USDT', '').replace('BUSD', '')
                for cat, assets in categories.items():
                    if base_asset in assets:
                        category_exposure[cat] += 1
                        break
            
            # Check proposed trade
            proposed_symbol = proposed_trade.get('symbol', '')
            proposed_base = proposed_symbol.replace('USDT', '').replace('BUSD', '')
            proposed_category = None
            
            for cat, assets in categories.items():
                if proposed_base in assets:
                    proposed_category = cat
                    break
            
            # Calculate correlation risk
            max_per_category = 3  # Max 3 positions per category
            correlation_risk = 0.0
            
            if proposed_category and category_exposure.get(proposed_category, 0) >= max_per_category:
                correlation_risk = 1.0  # High correlation risk
            elif proposed_category and category_exposure.get(proposed_category, 0) >= max_per_category - 1:
                correlation_risk = 0.7  # Medium correlation risk
            
            return {
                'category_exposure': category_exposure,
                'proposed_category': proposed_category,
                'correlation_risk': correlation_risk,
                'max_per_category': max_per_category,
                'risk_score': correlation_risk
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
    
    def _assess_market_volatility_risk(self, market_conditions: Dict) -> Dict:
        """
        Assess market volatility and adjust risk accordingly
        """
        try:
            market_regime = market_conditions.get('regime', 'NORMAL')
            volatility = market_conditions.get('volatility', 0.5)
            volume_surge = market_conditions.get('volume_surge', 1.0)
            
            # Calculate volatility risk score
            vol_risk = 0.0
            
            if market_regime == 'EXTREME':
                vol_risk = 0.9
            elif market_regime == 'VOLATILE':
                vol_risk = 0.7
            elif volatility > 1.0:
                vol_risk = 0.6
            elif volatility > 0.8:
                vol_risk = 0.4
            else:
                vol_risk = 0.2
            
            # Adjust for volume surge
            if volume_surge > 3.0:
                vol_risk = min(1.0, vol_risk + 0.2)
            
            return {
                'market_regime': market_regime,
                'volatility': volatility,
                'volume_surge': volume_surge,
                'volatility_risk': vol_risk,
                'risk_score': vol_risk
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
    
    def _assess_liquidity_risk(self, proposed_trade: Dict, market_conditions: Dict) -> Dict:
        """
        Assess liquidity risk for the proposed trade
        """
        try:
            symbol = proposed_trade.get('symbol', '')
            trade_size = proposed_trade.get('quantity', 0)
            current_price = proposed_trade.get('price', 0)
            trade_value = trade_size * current_price
            
            # Get market data
            avg_volume_24h = market_conditions.get('volume_24h', 0)
            current_spread = market_conditions.get('spread_pct', 0.1)
            
            # Calculate liquidity metrics
            volume_impact = (trade_value / avg_volume_24h) * 100 if avg_volume_24h > 0 else 0
            
            # Liquidity risk assessment
            liquidity_risk = 0.0
            
            if volume_impact > 5.0:  # Trade > 5% of daily volume
                liquidity_risk = 0.9
            elif volume_impact > 2.0:  # Trade > 2% of daily volume
                liquidity_risk = 0.6
            elif volume_impact > 1.0:  # Trade > 1% of daily volume
                liquidity_risk = 0.3
            else:
                liquidity_risk = 0.1
            
            # Adjust for spread
            if current_spread > 0.2:  # Spread > 0.2%
                liquidity_risk = min(1.0, liquidity_risk + 0.3)
            elif current_spread > 0.1:  # Spread > 0.1%
                liquidity_risk = min(1.0, liquidity_risk + 0.1)
            
            return {
                'volume_impact_pct': volume_impact,
                'spread_pct': current_spread,
                'trade_value': trade_value,
                'avg_volume_24h': avg_volume_24h,
                'liquidity_risk': liquidity_risk,
                'risk_score': liquidity_risk
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
    
    def _check_consecutive_losses(self) -> Dict:
        """
        Check consecutive loss protection
        """
        try:
            # Get consecutive losses from bot status (this should be tracked elsewhere)
            consecutive_losses = getattr(self, 'consecutive_losses', 0)
            
            risk_level = "LOW"
            if consecutive_losses >= self.max_consecutive_losses:
                risk_level = "CRITICAL"
            elif consecutive_losses >= self.max_consecutive_losses * 0.8:
                risk_level = "HIGH"
            elif consecutive_losses >= self.max_consecutive_losses * 0.6:
                risk_level = "MEDIUM"
            
            risk_score = min(1.0, consecutive_losses / self.max_consecutive_losses)
            
            return {
                'consecutive_losses': consecutive_losses,
                'max_allowed': self.max_consecutive_losses,
                'risk_level': risk_level,
                'risk_score': risk_score
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
    
    def _check_emergency_conditions(self, account_balance: float, market_conditions: Dict) -> Dict:
        """
        Check for emergency conditions that require immediate action
        """
        try:
            emergency_triggers = []
            
            # Market crash detection
            market_drop_24h = market_conditions.get('market_drop_24h_pct', 0)
            if market_drop_24h > 20:  # Market dropped >20% in 24h
                emergency_triggers.append(f"Market crash: {market_drop_24h:.1f}% drop")
            
            # Extreme volatility
            volatility = market_conditions.get('volatility', 0)
            if volatility > 2.0:  # Extreme volatility
                emergency_triggers.append(f"Extreme volatility: {volatility:.2f}")
            
            # Flash crash detection
            price_drop_1h = market_conditions.get('price_drop_1h_pct', 0)
            if price_drop_1h > 10:  # >10% drop in 1 hour
                emergency_triggers.append(f"Flash crash: {price_drop_1h:.1f}% in 1h")
            
            # System overload
            api_errors = market_conditions.get('api_errors_per_hour', 0)
            if api_errors > 50:  # Too many API errors
                emergency_triggers.append(f"System overload: {api_errors} API errors/hour")
            
            emergency_active = len(emergency_triggers) > 0
            
            return {
                'emergency_active': emergency_active,
                'triggers': emergency_triggers,
                'risk_score': 1.0 if emergency_active else 0.0
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
    
    def _combine_risk_assessments(self, risk_assessment: Dict) -> Dict:
        """
        Combine all risk assessments into final decision
        """
        try:
            metrics = risk_assessment['metrics']
            
            # Calculate overall risk score (weighted average)
            weights = {
                'portfolio_heat': 0.2,
                'drawdown': 0.25,
                'daily_loss': 0.2,
                'correlation': 0.1,
                'volatility': 0.1,
                'liquidity': 0.1,
                'consecutive_losses': 0.05
            }
            
            total_risk_score = 0.0
            for metric, weight in weights.items():
                if metric in metrics and 'risk_score' in metrics[metric]:
                    total_risk_score += metrics[metric]['risk_score'] * weight
            
            # Check for critical blocks
            blocks = []
            warnings = []
            
            # Emergency conditions
            if metrics.get('emergency', {}).get('emergency_active', False):
                blocks.append("Emergency market conditions detected")
            
            # Critical risk levels
            if metrics.get('drawdown', {}).get('risk_level') == 'CRITICAL':
                blocks.append("Maximum drawdown exceeded")
            
            if metrics.get('daily_loss', {}).get('risk_level') == 'CRITICAL':
                blocks.append("Daily loss limit exceeded")
            
            if metrics.get('portfolio_heat', {}).get('heat_level') == 'CRITICAL':
                blocks.append("Portfolio overheated")
            
            if metrics.get('consecutive_losses', {}).get('risk_level') == 'CRITICAL':
                blocks.append("Too many consecutive losses")
            
            # High risk warnings
            if metrics.get('volatility', {}).get('volatility_risk', 0) > 0.7:
                warnings.append("High market volatility")
            
            if metrics.get('liquidity', {}).get('liquidity_risk', 0) > 0.6:
                warnings.append("Liquidity concerns")
            
            if metrics.get('correlation', {}).get('correlation_risk', 0) > 0.7:
                warnings.append("High correlation risk")
            
            # Final approval decision
            approved = len(blocks) == 0 and total_risk_score < 0.8
            
            # Position size adjustments based on risk
            adjustments = {}
            if total_risk_score > 0.6:
                adjustments['position_size_multiplier'] = 0.5  # Reduce position size by 50%
            elif total_risk_score > 0.4:
                adjustments['position_size_multiplier'] = 0.75  # Reduce position size by 25%
            
            risk_assessment.update({
                'approved': approved,
                'risk_score': total_risk_score,
                'warnings': warnings,
                'blocks': blocks,
                'adjustments': adjustments
            })
            
            return risk_assessment
            
        except Exception as e:
            risk_assessment.update({
                'approved': False,
                'risk_score': 1.0,
                'warnings': [],
                'blocks': [f"Risk combination error: {e}"],
                'adjustments': {}
            })
            return risk_assessment
    
    def _log_risk_assessment(self, assessment: Dict):
        """
        Log risk assessment for monitoring and analysis
        """
        try:
            # This would log to a risk management database or file
            timestamp = datetime.now().isoformat()
            risk_log_entry = {
                'timestamp': timestamp,
                'approved': assessment['approved'],
                'risk_score': assessment['risk_score'],
                'warnings_count': len(assessment['warnings']),
                'blocks_count': len(assessment['blocks'])
            }
            
            # Store in memory for now (would be persistent storage in production)
            if not hasattr(self, 'risk_log'):
                self.risk_log = []
            
            self.risk_log.append(risk_log_entry)
            
            # Keep only last 1000 entries
            if len(self.risk_log) > 1000:
                self.risk_log = self.risk_log[-1000:]
                
        except Exception as e:
            print(f"Error logging risk assessment: {e}")
    
    def calculate_dynamic_stop_loss(self, 
                                  entry_price: float,
                                  position_size: float,
                                  account_balance: float,
                                  volatility: float,
                                  atr: float) -> Dict:
        """
        Calculate dynamic stop loss based on multiple risk factors
        """
        try:
            # Base stop loss (2% of account balance max risk)
            max_loss_amount = account_balance * 0.02  # 2% max loss per trade
            
            # Calculate stop based on ATR (2.5x ATR is standard)
            atr_stop_distance = atr * 2.5
            atr_stop_price = entry_price - atr_stop_distance
            atr_loss_amount = position_size * atr_stop_distance
            
            # Calculate stop based on volatility
            vol_multiplier = 1.0 + (volatility * 2)  # Higher vol = wider stops
            vol_stop_distance = entry_price * 0.03 * vol_multiplier  # 3% base * vol multiplier
            vol_stop_price = entry_price - vol_stop_distance
            vol_loss_amount = position_size * vol_stop_distance
            
            # Use the most conservative (closest) stop that doesn't exceed max loss
            stop_options = [
                {'type': 'ATR', 'price': atr_stop_price, 'loss': atr_loss_amount},
                {'type': 'Volatility', 'price': vol_stop_price, 'loss': vol_loss_amount}
            ]
            
            # Filter stops that don't exceed max loss
            valid_stops = [stop for stop in stop_options if stop['loss'] <= max_loss_amount]
            
            if valid_stops:
                # Choose the stop with highest price (most conservative)
                final_stop = max(valid_stops, key=lambda x: x['price'])
            else:
                # If all stops exceed max loss, use max loss constraint
                stop_distance = max_loss_amount / position_size
                final_stop = {
                    'type': 'Max Loss',
                    'price': entry_price - stop_distance,
                    'loss': max_loss_amount
                }
            
            return {
                'stop_price': final_stop['price'],
                'stop_distance': entry_price - final_stop['price'],
                'max_loss_amount': final_stop['loss'],
                'max_loss_pct': (final_stop['loss'] / account_balance) * 100,
                'stop_type': final_stop['type'],
                'all_options': stop_options
            }
            
        except Exception as e:
            # Fallback to simple 3% stop
            fallback_stop = entry_price * 0.97
            return {
                'stop_price': fallback_stop,
                'stop_distance': entry_price - fallback_stop,
                'max_loss_amount': position_size * (entry_price - fallback_stop),
                'max_loss_pct': 3.0,
                'stop_type': 'Fallback',
                'error': str(e)
            }

def get_risk_manager():
    """Get singleton instance of risk manager"""
    if not hasattr(get_risk_manager, '_instance'):
        get_risk_manager._instance = EnhancedRiskManager()
    return get_risk_manager._instance
