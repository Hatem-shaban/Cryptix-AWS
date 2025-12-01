#!/usr/bin/env python3
"""
Smart Profit Taking Strategy for CRYPTIX Trading Bot
Implements dynamic profit-taking based on market conditions, momentum, and volatility
"""

import numpy as np
from typing import Dict, Tuple, Optional
import config

class SmartProfitTaker:
    """
    Smart profit taking strategy that adapts to market conditions
    """
    
    def __init__(self):
        self.profit_levels = config.REBALANCING.get('profit_taking_levels', {
            'conservative': 2.0,
            'moderate': 5.0,
            'aggressive': 10.0
        })
    
    def analyze_profit_opportunity(self, symbol: str, current_price: float, 
                                 avg_buy_price: float, rsi: float = None,
                                 volatility: float = None, momentum: float = None) -> Dict:
        """
        Analyze profit-taking opportunity based on multiple factors
        
        Returns:
            Dict with profit analysis and recommendations
        """
        profit_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100
        
        # Base profit assessment
        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'avg_buy_price': avg_buy_price,
            'profit_pct': profit_pct,
            'profit_usd_per_unit': current_price - avg_buy_price,
            'is_profitable': profit_pct > 0
        }
        
        # Determine profit level category
        if profit_pct >= self.profit_levels['aggressive']:
            analysis['profit_level'] = 'aggressive'
        elif profit_pct >= self.profit_levels['moderate']:
            analysis['profit_level'] = 'moderate'
        elif profit_pct >= self.profit_levels['conservative']:
            analysis['profit_level'] = 'conservative'
        else:
            analysis['profit_level'] = 'minimal'
        
        # Market condition analysis
        market_analysis = self._analyze_market_conditions(rsi, volatility, momentum)
        analysis.update(market_analysis)
        
        # Generate profit-taking recommendation
        recommendation = self._generate_recommendation(analysis)
        analysis['recommendation'] = recommendation
        
        return analysis
    
    def _analyze_market_conditions(self, rsi: float = None, volatility: float = None, 
                                 momentum: float = None) -> Dict:
        """Analyze current market conditions"""
        conditions = {
            'market_pressure': 'neutral',
            'volatility_level': 'normal',
            'momentum_strength': 'neutral'
        }
        
        # RSI-based market pressure
        if rsi is not None:
            if rsi >= 80:
                conditions['market_pressure'] = 'extreme_overbought'
            elif rsi >= 70:
                conditions['market_pressure'] = 'overbought'
            elif rsi >= 60:
                conditions['market_pressure'] = 'bullish'
            elif rsi <= 20:
                conditions['market_pressure'] = 'extreme_oversold'
            elif rsi <= 30:
                conditions['market_pressure'] = 'oversold'
            elif rsi <= 40:
                conditions['market_pressure'] = 'bearish'
        
        # Volatility assessment
        if volatility is not None:
            if volatility >= 0.05:  # 5%+ volatility
                conditions['volatility_level'] = 'high'
            elif volatility >= 0.03:  # 3%+ volatility
                conditions['volatility_level'] = 'elevated'
            elif volatility <= 0.01:  # <1% volatility
                conditions['volatility_level'] = 'low'
        
        # Momentum strength
        if momentum is not None:
            if abs(momentum) >= 0.03:  # Strong momentum
                conditions['momentum_strength'] = 'strong'
            elif abs(momentum) >= 0.01:  # Moderate momentum
                conditions['momentum_strength'] = 'moderate'
            else:
                conditions['momentum_strength'] = 'weak'
        
        return conditions
    
    def _generate_recommendation(self, analysis: Dict) -> Dict:
        """Generate profit-taking recommendation based on analysis"""
        profit_pct = analysis['profit_pct']
        profit_level = analysis['profit_level']
        market_pressure = analysis['market_pressure']
        volatility_level = analysis['volatility_level']
        
        recommendation = {
            'action': 'hold',
            'sell_percentage': 0,
            'confidence': 0.5,
            'reasoning': [],
            'urgency': 'low'
        }
        
        # Base recommendation based on profit level
        if profit_level == 'aggressive':
            recommendation['action'] = 'sell_major'
            recommendation['sell_percentage'] = 60
            recommendation['confidence'] = 0.9
            recommendation['reasoning'].append(f"Aggressive profit level ({profit_pct:.1f}%) - time to take major profits")
            recommendation['urgency'] = 'high'
            
        elif profit_level == 'moderate':
            recommendation['action'] = 'sell_partial'
            recommendation['sell_percentage'] = 40
            recommendation['confidence'] = 0.7
            recommendation['reasoning'].append(f"Moderate profit level ({profit_pct:.1f}%) - partial profit taking recommended")
            recommendation['urgency'] = 'medium'
            
        elif profit_level == 'conservative':
            recommendation['action'] = 'sell_small'
            recommendation['sell_percentage'] = 25
            recommendation['confidence'] = 0.6
            recommendation['reasoning'].append(f"Conservative profit level ({profit_pct:.1f}%) - small profit taking")
            recommendation['urgency'] = 'low'
        
        # Market pressure adjustments
        if market_pressure == 'extreme_overbought':
            recommendation['sell_percentage'] = min(80, recommendation['sell_percentage'] + 20)
            recommendation['confidence'] = min(1.0, recommendation['confidence'] + 0.2)
            recommendation['reasoning'].append("Extreme overbought conditions - increase selling")
            recommendation['urgency'] = 'high'
            
        elif market_pressure == 'overbought':
            recommendation['sell_percentage'] = min(70, recommendation['sell_percentage'] + 10)
            recommendation['confidence'] = min(1.0, recommendation['confidence'] + 0.1)
            recommendation['reasoning'].append("Overbought conditions - moderate selling increase")
            
        elif market_pressure in ['bullish', 'neutral']:
            # Reduce selling in bullish conditions
            if profit_level == 'conservative':
                recommendation['sell_percentage'] = max(0, recommendation['sell_percentage'] - 10)
                recommendation['reasoning'].append("Bullish momentum - reduce profit taking")
        
        # Volatility adjustments
        if volatility_level == 'high':
            recommendation['sell_percentage'] = min(80, recommendation['sell_percentage'] + 15)
            recommendation['reasoning'].append("High volatility - increase profit taking")
            recommendation['urgency'] = 'high'
            
        elif volatility_level == 'low':
            recommendation['sell_percentage'] = max(0, recommendation['sell_percentage'] - 5)
            recommendation['reasoning'].append("Low volatility - reduce urgency")
        
        # Final action determination
        if recommendation['sell_percentage'] <= 10:
            recommendation['action'] = 'hold'
        elif recommendation['sell_percentage'] <= 30:
            recommendation['action'] = 'sell_small'
        elif recommendation['sell_percentage'] <= 50:
            recommendation['action'] = 'sell_partial'
        else:
            recommendation['action'] = 'sell_major'
        
        return recommendation
    
    def should_take_profits(self, symbol: str, current_price: float, avg_buy_price: float,
                          minimum_profit_pct: float = 2.0, **market_data) -> Tuple[bool, Dict]:
        """
        Determine if profits should be taken and provide detailed analysis
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            avg_buy_price: Average buy price of position
            minimum_profit_pct: Minimum profit percentage required
            **market_data: Additional market data (rsi, volatility, momentum)
        
        Returns:
            Tuple of (should_take_profits, analysis_dict)
        """
        analysis = self.analyze_profit_opportunity(
            symbol, current_price, avg_buy_price,
            market_data.get('rsi'), market_data.get('volatility'), market_data.get('momentum')
        )
        
        # Check minimum profit requirement
        if analysis['profit_pct'] < minimum_profit_pct:
            return False, {
                **analysis,
                'decision': 'hold',
                'reason': f"Profit {analysis['profit_pct']:.2f}% below minimum {minimum_profit_pct}%"
            }
        
        recommendation = analysis['recommendation']
        should_sell = recommendation['action'] != 'hold'
        
        return should_sell, {
            **analysis,
            'decision': recommendation['action'],
            'reason': '; '.join(recommendation['reasoning'])
        }

# Global instance
_profit_taker = None

def get_smart_profit_taker() -> SmartProfitTaker:
    """Get global smart profit taker instance"""
    global _profit_taker
    if _profit_taker is None:
        _profit_taker = SmartProfitTaker()
    return _profit_taker