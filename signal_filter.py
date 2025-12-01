#!/usr/bin/env python3
"""
Enhanced Signal Filtering System for CRYPTIX Trading Bot
Implements advanced signal validation, noise reduction, and quality scoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import config

class EnhancedSignalFilter:
    """
    Advanced signal filtering to reduce false signals and improve trade quality
    """
    
    def __init__(self):
        self.signal_history = {}
        self.false_signal_count = {}
        self.signal_quality_scores = {}
        self.market_condition_cache = {}
        self.noise_threshold = getattr(config, 'SIGNAL_NOISE_THRESHOLD', 0.7)
    
    def filter_and_validate_signal(self, 
                                 symbol: str,
                                 raw_signal: str,
                                 indicators: Dict,
                                 market_data: Dict,
                                 confidence_threshold: float = 0.65) -> Dict:
        """
        Comprehensive signal filtering and validation
        Returns enhanced signal with quality score and reasoning
        """
        try:
            # 1. Signal Quality Assessment
            quality_score = self._calculate_signal_quality(symbol, raw_signal, indicators, market_data)
            
            # 2. Market Context Validation
            context_valid = self._validate_market_context(symbol, raw_signal, market_data)
            
            # 3. Noise Filtering
            is_noise = self._detect_signal_noise(symbol, raw_signal, indicators)
            
            # 4. Momentum Confirmation
            momentum_confirmed = self._confirm_signal_momentum(indicators, raw_signal)
            
            # 5. Volume Validation
            volume_valid = self._validate_volume_support(market_data, raw_signal)
            
            # 6. Multi-timeframe Alignment
            timeframe_aligned = self._check_timeframe_alignment(symbol, raw_signal)
            
            # 7. Risk-Reward Assessment
            risk_reward = self._assess_risk_reward_ratio(indicators, raw_signal)
            
            # Combine all filters
            filtered_signal = self._combine_filter_results(
                raw_signal, quality_score, context_valid, is_noise,
                momentum_confirmed, volume_valid, timeframe_aligned,
                risk_reward, confidence_threshold
            )
            
            # Store signal for historical analysis
            self._store_signal_result(symbol, filtered_signal)
            
            return filtered_signal
            
        except Exception as e:
            print(f"Error in signal filtering: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'quality_score': 0.0,
                'filters_passed': 0,
                'reason': f"Filter error: {e}",
                'raw_signal': raw_signal
            }
    
    def _calculate_signal_quality(self, symbol: str, signal: str, indicators: Dict, market_data: Dict) -> float:
        """
        Calculate comprehensive signal quality score (0-1)
        """
        quality_factors = []
        
        # Factor 1: Indicator Convergence (0-0.3)
        convergence_score = self._assess_indicator_convergence(indicators)
        quality_factors.append(('convergence', convergence_score, 0.3))
        
        # Factor 2: Signal Strength (0-0.25)
        strength_score = self._assess_signal_strength(indicators, signal)
        quality_factors.append(('strength', strength_score, 0.25))
        
        # Factor 3: Market Structure (0-0.2)
        structure_score = self._assess_market_structure(market_data)
        quality_factors.append(('structure', structure_score, 0.2))
        
        # Factor 4: Volatility Context (0-0.15)
        volatility_score = self._assess_volatility_context(indicators)
        quality_factors.append(('volatility', volatility_score, 0.15))
        
        # Factor 5: Historical Performance (0-0.1)
        history_score = self._assess_historical_performance(symbol, signal)
        quality_factors.append(('history', history_score, 0.1))
        
        # Calculate weighted quality score
        total_score = sum(score * weight for _, score, weight in quality_factors)
        
        return min(1.0, max(0.0, total_score))
    
    def _assess_indicator_convergence(self, indicators: Dict) -> float:
        """
        Assess how well technical indicators agree with each other
        """
        try:
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # RSI assessment
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                bullish_signals += 2  # Strong oversold
            elif rsi < 40:
                bullish_signals += 1  # Mild oversold
            elif rsi > 70:
                bearish_signals += 2  # Strong overbought
            elif rsi > 60:
                bearish_signals += 1  # Mild overbought
            total_signals += 2
            
            # MACD assessment
            macd_trend = indicators.get('macd_trend', 'NEUTRAL')
            if macd_trend == 'BULLISH':
                bullish_signals += 2
            elif macd_trend == 'BEARISH':
                bearish_signals += 2
            total_signals += 2
            
            # EMA alignment
            current_price = indicators.get('current_price', 0)
            ema50 = indicators.get('ema50')
            ema200 = indicators.get('ema200')
            
            if ema50 and ema200 and current_price:
                if current_price > ema50 > ema200:
                    bullish_signals += 1
                elif current_price < ema50 < ema200:
                    bearish_signals += 1
                total_signals += 1
            
            # Stochastic assessment
            stoch_k = indicators.get('stoch_k')
            if stoch_k:
                if stoch_k < 20:
                    bullish_signals += 1
                elif stoch_k > 80:
                    bearish_signals += 1
                total_signals += 1
            
            # Calculate convergence (higher = more agreement)
            if total_signals == 0:
                return 0.5
            
            max_directional = max(bullish_signals, bearish_signals)
            convergence = max_directional / total_signals
            
            return convergence
            
        except Exception:
            return 0.5  # Neutral if error
    
    def _assess_signal_strength(self, indicators: Dict, signal: str) -> float:
        """
        Assess the strength of the trading signal
        """
        try:
            if signal == 'HOLD':
                return 0.3  # Neutral strength for hold
            
            strength_score = 0.0
            
            # RSI strength contribution
            rsi = indicators.get('rsi', 50)
            if signal == 'BUY':
                if rsi < 25:
                    strength_score += 0.4  # Very oversold
                elif rsi < 35:
                    strength_score += 0.3  # Oversold
                elif rsi < 45:
                    strength_score += 0.2  # Slightly oversold
            elif signal == 'SELL':
                if rsi > 75:
                    strength_score += 0.4  # Very overbought
                elif rsi > 65:
                    strength_score += 0.3  # Overbought
                elif rsi > 55:
                    strength_score += 0.2  # Slightly overbought
            
            # MACD histogram strength
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            histogram = macd - macd_signal
            
            if signal == 'BUY' and histogram > 0:
                strength_score += min(0.3, abs(histogram) * 100)  # Scale histogram
            elif signal == 'SELL' and histogram < 0:
                strength_score += min(0.3, abs(histogram) * 100)
            
            # ADX trend strength
            adx = indicators.get('adx')
            if adx:
                if adx > 25:
                    strength_score += 0.3  # Strong trend
                elif adx > 20:
                    strength_score += 0.2  # Moderate trend
                elif adx > 15:
                    strength_score += 0.1  # Weak trend
            
            return min(1.0, strength_score)
            
        except Exception:
            return 0.5
    
    def _validate_market_context(self, symbol: str, signal: str, market_data: Dict) -> bool:
        """
        Validate signal against current market context
        """
        try:
            # Check market hours (avoid low liquidity times)
            current_hour = datetime.now().hour
            avoid_hours = getattr(config, 'AVOID_TRADING_HOURS', [0, 1, 2, 3])
            
            if current_hour in avoid_hours:
                return False
            
            # Volume validation
            volume_24h = market_data.get('volume_24h', 0)
            min_volume = getattr(config, 'MIN_VOLUME_FILTER', 1000000)
            
            if volume_24h < min_volume:
                return False
            
            # Price change validation (avoid extreme moves that might reverse)
            price_change_24h = abs(market_data.get('price_change_24h_pct', 0))
            max_price_change = getattr(config, 'MAX_PRICE_CHANGE_FILTER', 15.0)  # 15%
            
            if price_change_24h > max_price_change:
                return False  # Too volatile
            
            return True
            
        except Exception:
            return True  # Default to valid if error
    
    def _detect_signal_noise(self, symbol: str, signal: str, indicators: Dict) -> bool:
        """
        Detect if signal is likely noise/false signal
        """
        try:
            # Check for rapid signal changes (whipsaws)
            if symbol in self.signal_history:
                recent_signals = self.signal_history[symbol][-5:]  # Last 5 signals
                if len(recent_signals) >= 3:
                    # Count signal changes in recent history
                    changes = sum(1 for i in range(1, len(recent_signals)) 
                                if recent_signals[i] != recent_signals[i-1])
                    
                    # If more than 2 changes in last 5 signals, likely noise
                    if changes > 2:
                        return True
            
            # Check indicator near neutral zones (RSI 45-55, MACD near zero)
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            
            # RSI in neutral zone
            if 45 <= rsi <= 55:
                return True
            
            # MACD very close to zero (weak signal)
            if abs(macd) < 0.001:
                return True
            
            return False
            
        except Exception:
            return False  # Default to not noise if error
    
    def _confirm_signal_momentum(self, indicators: Dict, signal: str) -> bool:
        """
        Confirm signal has momentum support
        """
        try:
            # Price vs VWAP
            current_price = indicators.get('current_price', 0)
            vwap = indicators.get('vwap')
            
            if vwap and current_price:
                if signal == 'BUY' and current_price < vwap * 0.995:  # Price below VWAP
                    return False
                elif signal == 'SELL' and current_price > vwap * 1.005:  # Price above VWAP
                    return False
            
            # SMA momentum
            sma5 = indicators.get('sma5', 0)
            sma20 = indicators.get('sma20', 0)
            
            if sma5 and sma20:
                if signal == 'BUY' and sma5 <= sma20:  # No upward momentum
                    return False
                elif signal == 'SELL' and sma5 >= sma20:  # No downward momentum
                    return False
            
            return True
            
        except Exception:
            return True  # Default to confirmed if error
    
    def _validate_volume_support(self, market_data: Dict, signal: str) -> bool:
        """
        Validate that volume supports the signal
        """
        try:
            volume_ratio = market_data.get('volume_ratio', 1.0)  # Current vs average volume
            
            # Require above-average volume for signals
            min_volume_ratio = getattr(config, 'MIN_VOLUME_RATIO', 1.2)  # 20% above average
            
            return volume_ratio >= min_volume_ratio
            
        except Exception:
            return True  # Default to valid if error
    
    def _check_timeframe_alignment(self, symbol: str, signal: str) -> bool:
        """
        Check if signal aligns across multiple timeframes
        This is a simplified version - in practice would fetch multi-timeframe data
        """
        # For now, return True as this requires additional data fetching
        # In a full implementation, this would check 5m, 15m, 1h alignment
        return True
    
    def _assess_risk_reward_ratio(self, indicators: Dict, signal: str) -> float:
        """
        Assess potential risk/reward ratio for the signal
        """
        try:
            current_price = indicators.get('current_price', 0)
            atr = indicators.get('atr', 0)
            
            if not current_price or not atr:
                return 0.5  # Neutral if no data
            
            # Estimate stop loss and target based on ATR
            stop_distance = atr * 2  # 2x ATR stop
            target_distance = atr * 3  # 3x ATR target (1.5:1 ratio)
            
            if stop_distance <= 0:
                return 0.5
            
            risk_reward_ratio = target_distance / stop_distance
            
            # Normalize to 0-1 scale (ratios above 2:1 get max score)
            return min(1.0, risk_reward_ratio / 2.0)
            
        except Exception:
            return 0.5
    
    def _combine_filter_results(self, raw_signal: str, quality_score: float, 
                              context_valid: bool, is_noise: bool,
                              momentum_confirmed: bool, volume_valid: bool,
                              timeframe_aligned: bool, risk_reward: float,
                              confidence_threshold: float) -> Dict:
        """
        Combine all filter results into final signal decision
        """
        # Count passed filters
        filters_passed = sum([
            quality_score > 0.5,
            context_valid,
            not is_noise,
            momentum_confirmed,
            volume_valid,
            timeframe_aligned,
            risk_reward > 0.4
        ])
        
        # Calculate overall confidence
        confidence = (
            quality_score * 0.4 +
            (1.0 if context_valid else 0.0) * 0.15 +
            (0.0 if is_noise else 1.0) * 0.15 +
            (1.0 if momentum_confirmed else 0.0) * 0.1 +
            (1.0 if volume_valid else 0.0) * 0.1 +
            risk_reward * 0.1
        )
        
        # Make final signal decision
        if confidence >= confidence_threshold and filters_passed >= 5:
            final_signal = raw_signal
        else:
            final_signal = 'HOLD'
        
        # Build reason string
        reasons = []
        if quality_score <= 0.5:
            reasons.append(f"Low quality ({quality_score:.2f})")
        if not context_valid:
            reasons.append("Invalid market context")
        if is_noise:
            reasons.append("Signal noise detected")
        if not momentum_confirmed:
            reasons.append("No momentum confirmation")
        if not volume_valid:
            reasons.append("Insufficient volume")
        if risk_reward <= 0.4:
            reasons.append(f"Poor risk/reward ({risk_reward:.2f})")
        
        reason = f"Filters passed: {filters_passed}/7. " + ("; ".join(reasons) if reasons else "All filters passed")
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'quality_score': quality_score,
            'filters_passed': filters_passed,
            'reason': reason,
            'raw_signal': raw_signal,
            'filter_details': {
                'context_valid': context_valid,
                'is_noise': is_noise,
                'momentum_confirmed': momentum_confirmed,
                'volume_valid': volume_valid,
                'timeframe_aligned': timeframe_aligned,
                'risk_reward': risk_reward
            }
        }
    
    def _store_signal_result(self, symbol: str, signal_result: Dict):
        """
        Store signal result for historical analysis
        """
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append({
            'timestamp': datetime.now(),
            'signal': signal_result['signal'],
            'confidence': signal_result['confidence'],
            'quality_score': signal_result['quality_score']
        })
        
        # Keep only last 50 signals per symbol
        if len(self.signal_history[symbol]) > 50:
            self.signal_history[symbol] = self.signal_history[symbol][-50:]
    
    def _assess_historical_performance(self, symbol: str, signal: str) -> float:
        """
        Assess historical performance of similar signals for this symbol
        """
        # This would analyze past signal performance
        # For now, return neutral score
        return 0.5
    
    def _assess_market_structure(self, market_data: Dict) -> float:
        """
        Assess current market structure quality
        """
        try:
            # Check bid-ask spread (tighter = better structure)
            spread_pct = market_data.get('spread_pct', 0.1)
            if spread_pct < 0.05:
                spread_score = 1.0
            elif spread_pct < 0.1:
                spread_score = 0.7
            else:
                spread_score = 0.3
            
            # Check volume consistency
            volume_consistency = market_data.get('volume_consistency', 0.5)
            
            return (spread_score + volume_consistency) / 2
            
        except Exception:
            return 0.5
    
    def _assess_volatility_context(self, indicators: Dict) -> float:
        """
        Assess if volatility context is favorable for trading
        """
        try:
            volatility = indicators.get('volatility', 0.5)
            
            # Optimal volatility range (not too low, not too high)
            if 0.3 <= volatility <= 0.8:
                return 1.0  # Optimal range
            elif 0.2 <= volatility <= 1.2:
                return 0.7  # Acceptable range
            else:
                return 0.3  # Suboptimal range
                
        except Exception:
            return 0.5

def get_signal_filter():
    """Get singleton instance of signal filter"""
    if not hasattr(get_signal_filter, '_instance'):
        get_signal_filter._instance = EnhancedSignalFilter()
    return get_signal_filter._instance
