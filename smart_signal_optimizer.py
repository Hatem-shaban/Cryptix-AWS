#!/usr/bin/env python3
"""
Smart Signal Optimizer for CRYPTIX Trading Bot
Enhanced buy/sell logic for maximum profitability optimization
Implements smart entry/exit timing to truly buy low and sell high
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import config

class SmartSignalOptimizer:
    """
    Advanced signal optimization for maximum profitability
    Ensures we buy at local lows and sell at local highs
    """
    
    def __init__(self):
        self.price_history = {}  # Track price patterns for each symbol
        self.position_tracking = {}  # Track our positions and entry prices
        self.profit_targets = {}  # Dynamic profit targets per symbol
        self.support_resistance = {}  # S/R levels for each symbol
        
    def optimize_entry_exit(self, symbol: str, signal: str, df: pd.DataFrame, 
                           indicators: Dict, current_balance: Dict) -> Dict:
        """
        Optimize entry/exit timing for maximum profitability
        Returns enhanced signal with timing optimization
        """
        try:
            current_price = indicators['current_price']
            
            # Update price history and analysis
            self._update_price_analysis(symbol, df, current_price)
            
            # Get profitability analysis
            profitability_score = self._calculate_profitability_score(
                symbol, signal, current_price, indicators, current_balance
            )
            
            # Smart entry/exit timing
            timing_analysis = self._analyze_optimal_timing(symbol, signal, df, current_price)
            
            # Risk-reward assessment
            risk_reward = self._calculate_risk_reward_ratio(symbol, signal, current_price)
            
            # Market momentum analysis
            momentum_score = self._analyze_market_momentum(df, indicators)
            
            # Combine all factors for final decision
            optimization_result = self._combine_optimization_factors(
                signal, profitability_score, timing_analysis, risk_reward, momentum_score
            )
            
            return optimization_result
            
        except Exception as e:
            print(f"Error in signal optimization: {e}")
            return {
                'optimized_signal': signal,
                'confidence': 0.5,
                'reason': f"Optimization error: {e}",
                'should_wait': False,
                'target_price': current_price
            }
    
    def _update_price_analysis(self, symbol: str, df: pd.DataFrame, current_price: float):
        """Update price analysis and support/resistance levels"""
        try:
            # Calculate support and resistance levels
            high_prices = df['high'].tail(50)
            low_prices = df['low'].tail(50)
            close_prices = df['close'].tail(50)
            
            # Dynamic support/resistance calculation
            resistance_levels = []
            support_levels = []
            
            # Find recent swing highs and lows
            for i in range(2, len(high_prices) - 2):
                # Swing high
                if (high_prices.iloc[i] > high_prices.iloc[i-1] and 
                    high_prices.iloc[i] > high_prices.iloc[i-2] and
                    high_prices.iloc[i] > high_prices.iloc[i+1] and 
                    high_prices.iloc[i] > high_prices.iloc[i+2]):
                    resistance_levels.append(high_prices.iloc[i])
                
                # Swing low
                if (low_prices.iloc[i] < low_prices.iloc[i-1] and 
                    low_prices.iloc[i] < low_prices.iloc[i-2] and
                    low_prices.iloc[i] < low_prices.iloc[i+1] and 
                    low_prices.iloc[i] < low_prices.iloc[i+2]):
                    support_levels.append(low_prices.iloc[i])
            
            # Get strongest levels (most recent and most tested)
            resistance_levels = sorted(resistance_levels, reverse=True)[:5]
            support_levels = sorted(support_levels)[:5]
            
            self.support_resistance[symbol] = {
                'resistance': resistance_levels,
                'support': support_levels,
                'current_price': current_price,
                'price_range': {
                    'high_24h': high_prices.max(),
                    'low_24h': low_prices.min(),
                    'avg_price': close_prices.mean()
                },
                'updated': datetime.now()
            }
            
        except Exception as e:
            print(f"Error updating price analysis for {symbol}: {e}")
    
    def _calculate_profitability_score(self, symbol: str, signal: str, current_price: float,
                                     indicators: Dict, current_balance: Dict) -> float:
        """Calculate profitability potential score (0-1)"""
        try:
            score = 0.5  # Base score
            
            sr_data = self.support_resistance.get(symbol, {})
            
            if signal == "BUY":
                # Check if we're buying near support levels (good for profitability)
                support_levels = sr_data.get('support', [])
                if support_levels:
                    closest_support = min(support_levels, key=lambda x: abs(x - current_price))
                    support_distance = abs(current_price - closest_support) / current_price
                    
                    # Closer to support = higher profitability score
                    if support_distance < 0.01:  # Within 1% of support
                        score += 0.3
                    elif support_distance < 0.02:  # Within 2% of support
                        score += 0.2
                    elif support_distance < 0.03:  # Within 3% of support
                        score += 0.1
                
                # Check if price is in lower part of recent range
                price_range = sr_data.get('price_range', {})
                if price_range:
                    range_position = (current_price - price_range.get('low_24h', current_price)) / \
                                   max(0.01, price_range.get('high_24h', current_price) - price_range.get('low_24h', current_price))
                    
                    # Lower position = better buy opportunity
                    if range_position < 0.3:  # In bottom 30% of range
                        score += 0.2
                    elif range_position < 0.5:  # In bottom 50% of range
                        score += 0.1
                
                # RSI oversold analysis for buying
                rsi = indicators.get('rsi', 50)
                if rsi < 25:  # Very oversold - great buy opportunity
                    score += 0.2
                elif rsi < 35:  # Oversold - good buy opportunity
                    score += 0.15
                elif rsi < 45:  # Slightly oversold
                    score += 0.1
                elif rsi > 60:  # Not a good time to buy
                    score -= 0.2
            
            elif signal == "SELL":
                # Check if we have a position to sell
                base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol.split('USDT')[0]
                has_position = current_balance.get(base_asset, 0) > 0
                
                if not has_position:
                    return 0.0  # No point in selling if we don't have the asset
                
                # Check if we're selling near resistance levels (good for profitability)
                resistance_levels = sr_data.get('resistance', [])
                if resistance_levels:
                    closest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                    resistance_distance = abs(current_price - closest_resistance) / current_price
                    
                    # Closer to resistance = higher profitability score
                    if resistance_distance < 0.01:  # Within 1% of resistance
                        score += 0.3
                    elif resistance_distance < 0.02:  # Within 2% of resistance
                        score += 0.2
                    elif resistance_distance < 0.03:  # Within 3% of resistance
                        score += 0.1
                
                # Check if price is in upper part of recent range
                price_range = sr_data.get('price_range', {})
                if price_range:
                    range_position = (current_price - price_range.get('low_24h', current_price)) / \
                                   max(0.01, price_range.get('high_24h', current_price) - price_range.get('low_24h', current_price))
                    
                    # Higher position = better sell opportunity
                    if range_position > 0.7:  # In top 30% of range
                        score += 0.2
                    elif range_position > 0.5:  # In top 50% of range
                        score += 0.1
                
                # RSI overbought analysis for selling
                rsi = indicators.get('rsi', 50)
                if rsi > 75:  # Very overbought - great sell opportunity
                    score += 0.2
                elif rsi > 65:  # Overbought - good sell opportunity
                    score += 0.15
                elif rsi > 55:  # Slightly overbought
                    score += 0.1
                elif rsi < 40:  # Not a good time to sell
                    score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"Error calculating profitability score: {e}")
            return 0.5
    
    def _analyze_optimal_timing(self, symbol: str, signal: str, df: pd.DataFrame, current_price: float) -> Dict:
        """Analyze if current timing is optimal for entry/exit"""
        try:
            # Recent price momentum
            recent_prices = df['close'].tail(10)
            price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[-5]) / recent_prices.iloc[-5]
            
            # Volume confirmation
            recent_volumes = df['volume'].tail(10) if 'volume' in df.columns else pd.Series([1] * 10)
            avg_volume = recent_volumes.mean()
            current_volume = recent_volumes.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility analysis
            price_changes = recent_prices.pct_change().dropna()
            volatility = price_changes.std()
            
            timing_score = 0.5
            wait_reason = ""
            should_wait = False
            
            if signal == "BUY":
                # For buying, we want to catch falling knives at support or oversold bounces
                if price_momentum < -0.02 and volume_ratio > 1.5:  # Strong selling with volume - good for buying
                    timing_score += 0.3
                elif price_momentum < -0.01:  # Moderate selling - still good for buying dips
                    timing_score += 0.2
                elif price_momentum > 0.03:  # Price rising too fast
                    timing_score -= 0.2
                    should_wait = True
                    wait_reason = "Price rising too fast - wait for pullback"
                elif price_momentum > 0.015:  # Moderate rise - not ideal but not terrible
                    timing_score -= 0.1
                
                # High volatility can be good for buying dips if we catch the bottom
                if volatility > 0.03:  # High volatility
                    timing_score += 0.15
                elif volatility > 0.02:  # Moderate volatility
                    timing_score += 0.1
                    
            elif signal == "SELL":
                # For selling, we want momentum and volume confirmation
                if price_momentum > 0.02 and volume_ratio > 1.5:  # Strong buying with volume - good for selling
                    timing_score += 0.3
                elif price_momentum > 0.01:  # Moderate buying - still decent for selling
                    timing_score += 0.2
                elif price_momentum < -0.02:  # Strong decline - not good for selling
                    timing_score -= 0.2
                    should_wait = True
                    wait_reason = "Price declining - wait for bounce before selling"
                elif price_momentum < -0.01:  # Moderate decline
                    timing_score -= 0.1
                
                # Lower volatility preferred for selling (more predictable exits)
                if volatility < 0.01:  # Low volatility
                    timing_score += 0.1
            
            # For HOLD signals, check if conditions suggest we should consider trading
            elif signal == "HOLD":
                # Check for oversold bounce opportunities (HOLD -> BUY)
                if price_momentum < -0.03 and volume_ratio > 2.0:  # Severe oversold with volume
                    timing_score += 0.4  # Strong signal to consider buying
                elif price_momentum < -0.015 and volume_ratio > 1.3:  # Moderate oversold
                    timing_score += 0.2
                
                # Check for overbought selling opportunities (HOLD -> SELL)  
                elif price_momentum > 0.03 and volume_ratio > 2.0:  # Strong rally with volume
                    timing_score += 0.3  # Consider selling into strength
                elif price_momentum > 0.02 and volume_ratio > 1.3:  # Moderate rally
                    timing_score += 0.15
                elif price_momentum < -0.03:  # Price falling too fast
                    timing_score -= 0.3
                    should_wait = True
                    wait_reason = "Price falling too fast - wait for bounce"
            
            return {
                'timing_score': max(0.0, min(1.0, timing_score)),
                'should_wait': should_wait,
                'wait_reason': wait_reason,
                'price_momentum': price_momentum,
                'volume_ratio': volume_ratio,
                'volatility': volatility
            }
            
        except Exception as e:
            print(f"Error analyzing timing: {e}")
            return {
                'timing_score': 0.5,
                'should_wait': False,
                'wait_reason': "",
                'price_momentum': 0,
                'volume_ratio': 1,
                'volatility': 0
            }
    
    def _calculate_risk_reward_ratio(self, symbol: str, signal: str, current_price: float) -> float:
        """Calculate risk/reward ratio for the trade"""
        try:
            sr_data = self.support_resistance.get(symbol, {})
            
            if signal == "BUY":
                # For buy signals, calculate potential upside vs downside
                resistance_levels = sr_data.get('resistance', [])
                support_levels = sr_data.get('support', [])
                
                # Find nearest resistance (target) and support (stop loss)
                upside_target = None
                downside_risk = None
                
                if resistance_levels:
                    upside_targets = [r for r in resistance_levels if r > current_price]
                    if upside_targets:
                        upside_target = min(upside_targets)
                
                if support_levels:
                    downside_risks = [s for s in support_levels if s < current_price]
                    if downside_risks:
                        downside_risk = max(downside_risks)
                
                if upside_target and downside_risk:
                    potential_gain = (upside_target - current_price) / current_price
                    potential_loss = (current_price - downside_risk) / current_price
                    
                    if potential_loss > 0:
                        risk_reward = potential_gain / potential_loss
                        return min(5.0, risk_reward)  # Cap at 5:1
                
            elif signal == "SELL":
                # For sell signals, we're already in profit territory
                # Calculate if we should wait for higher prices or sell now
                resistance_levels = sr_data.get('resistance', [])
                
                if resistance_levels:
                    higher_resistance = [r for r in resistance_levels if r > current_price]
                    if higher_resistance:
                        next_target = min(higher_resistance)
                        potential_additional_gain = (next_target - current_price) / current_price
                        
                        # If very close to resistance, good to sell
                        if potential_additional_gain < 0.02:  # Less than 2% upside
                            return 3.0  # Good risk/reward to sell now
                        else:
                            return 1.5  # Moderate risk/reward
            
            return 2.0  # Default decent risk/reward
            
        except Exception as e:
            print(f"Error calculating risk/reward: {e}")
            return 2.0
    
    def _analyze_market_momentum(self, df: pd.DataFrame, indicators: Dict) -> float:
        """Analyze overall market momentum"""
        try:
            # Multiple timeframe momentum
            momentum_score = 0.5
            
            # Short-term momentum (SMA5 vs SMA20)
            sma5 = indicators.get('sma5', 0)
            sma20 = indicators.get('sma20', 0)
            
            if sma5 and sma20:
                sma_momentum = (sma5 - sma20) / sma20
                if abs(sma_momentum) > 0.02:  # Strong momentum
                    momentum_score += 0.2 if sma_momentum > 0 else -0.2
                elif abs(sma_momentum) > 0.01:  # Moderate momentum
                    momentum_score += 0.1 if sma_momentum > 0 else -0.1
            
            # MACD momentum
            macd_trend = indicators.get('macd_trend', 'NEUTRAL')
            if macd_trend == 'BULLISH':
                momentum_score += 0.15
            elif macd_trend == 'BEARISH':
                momentum_score -= 0.15
            
            # VWAP momentum
            current_price = indicators.get('current_price', 0)
            vwap = indicators.get('vwap', 0)
            
            if current_price and vwap:
                vwap_momentum = (current_price - vwap) / vwap
                if abs(vwap_momentum) > 0.01:
                    momentum_score += 0.1 if vwap_momentum > 0 else -0.1
            
            return max(0.0, min(1.0, momentum_score))
            
        except Exception as e:
            print(f"Error analyzing momentum: {e}")
            return 0.5
    
    def _combine_optimization_factors(self, signal: str, profitability_score: float,
                                    timing_analysis: Dict, risk_reward: float, momentum_score: float) -> Dict:
        """Combine all optimization factors for final decision"""
        try:
            # Weighted combination of factors
            weights = {
                'profitability': 0.4,  # Most important - are we buying low/selling high?
                'timing': 0.25,        # Is timing optimal?
                'risk_reward': 0.2,    # Is the risk/reward favorable?
                'momentum': 0.15       # Does momentum support the move?
            }
            
            # Normalize risk_reward to 0-1 scale
            risk_reward_normalized = min(1.0, risk_reward / 3.0)
            
            # Calculate composite confidence score
            confidence = (
                profitability_score * weights['profitability'] +
                timing_analysis['timing_score'] * weights['timing'] +
                risk_reward_normalized * weights['risk_reward'] +
                momentum_score * weights['momentum']
            )
            
            # Determine if we should execute, wait, or skip
            should_wait = timing_analysis['should_wait']
            optimized_signal = signal
            
            # Adaptive confidence threshold - lower for better opportunities
            base_confidence = 0.55  # Lowered from 0.65 for more opportunities
            
            # Even if original signal is HOLD, check if we can find a trading opportunity
            if signal == "HOLD":
                # For HOLD signals, check if we can upgrade to BUY/SELL based on strong factors
                if profitability_score > 0.7 and timing_analysis['timing_score'] > 0.6:
                    if momentum_score > 0.6:  # Strong bullish momentum
                        optimized_signal = "BUY"
                        reason = f"Smart Optimizer found BUY opportunity despite HOLD signal (Profit: {profitability_score:.2f}, Timing: {timing_analysis['timing_score']:.2f})"
                    elif momentum_score < 0.4:  # Strong bearish momentum and we have assets
                        optimized_signal = "SELL"
                        reason = f"Smart Optimizer found SELL opportunity despite HOLD signal (Profit: {profitability_score:.2f}, Timing: {timing_analysis['timing_score']:.2f})"
                    else:
                        reason = f"Good setup found but momentum unclear (Score: {confidence:.2f})"
                elif profitability_score > 0.8:  # Exceptional profitability even with lower timing
                    if momentum_score > 0.5:
                        optimized_signal = "BUY" 
                        reason = f"Exceptional profitability BUY opportunity (Score: {profitability_score:.2f})"
                    else:
                        reason = f"High profitability but waiting for better momentum (Score: {confidence:.2f})"
                else:
                    reason = f"HOLD confirmed - no strong trading opportunity (Score: {confidence:.2f})"
            else:
                # For non-HOLD signals, apply confidence threshold
                min_confidence = base_confidence
                
                if confidence < min_confidence:
                    optimized_signal = "HOLD"
                    reason = f"Low confidence - downgraded to HOLD ({confidence:.2f} < {min_confidence})"
                elif should_wait:
                    optimized_signal = "HOLD"
                    reason = f"Good setup but timing suggests waiting: {timing_analysis['wait_reason']}"
                else:
                    if signal == "BUY":
                        reason = f"Optimized BUY - High confidence buying opportunity (Score: {confidence:.2f})"
                    elif signal == "SELL":
                        reason = f"Optimized SELL - High confidence selling opportunity (Score: {confidence:.2f})"
                    else:
                        reason = f"Optimized {signal} - Confidence: {confidence:.2f}"
            
            return {
                'optimized_signal': optimized_signal,
                'confidence': confidence,
                'reason': reason,
                'should_wait': should_wait,
                'factors': {
                    'profitability_score': profitability_score,
                    'timing_score': timing_analysis['timing_score'],
                    'risk_reward_ratio': risk_reward,
                    'momentum_score': momentum_score
                },
                'timing_details': timing_analysis
            }
            
        except Exception as e:
            print(f"Error combining optimization factors: {e}")
            return {
                'optimized_signal': signal,
                'confidence': 0.5,
                'reason': f"Optimization error: {e}",
                'should_wait': False,
                'factors': {}
            }

# Global instance
_signal_optimizer = None

def get_signal_optimizer():
    """Get the global signal optimizer instance"""
    global _signal_optimizer
    if _signal_optimizer is None:
        _signal_optimizer = SmartSignalOptimizer()
    return _signal_optimizer
