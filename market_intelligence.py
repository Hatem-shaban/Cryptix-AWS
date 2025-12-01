"""
Market Intelligence Module for CRYPTIX Trading Bot
Implements advanced pattern recognition, market regime detection, and adaptive analytics
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class MarketIntelligence:
    """
    Advanced Market Intelligence Engine
    - Pattern Recognition System
    - Market Regime Detection
    - Adaptive Threshold Management
    - Historical Performance Analytics
    """
    
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        self.market_patterns = {}
        self.regime_history = []
        self.adaptive_params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_threshold': 0.001,
            'volume_spike_threshold': 1.5,
            'volatility_threshold': 0.03
        }
        
        # Initialize pattern storage
        self.pattern_database = {}
        self.signal_success_history = []
        self.market_context_cache = {}
        
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical market intelligence data"""
        try:
            # Load pattern database
            if os.path.exists('market_patterns.json'):
                with open('market_patterns.json', 'r') as f:
                    self.market_patterns = json.load(f)
            
            # Load signal history
            if os.path.exists('logs/signal_history.csv'):
                signal_df = pd.read_csv('logs/signal_history.csv')
                self.signal_success_history = signal_df.to_dict('records')
                
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")

    def analyze_market_regime(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive market regime analysis using multiple indicators
        """
        try:
            if len(df) < 20:
                return self._default_regime()
                
            # Calculate regime indicators
            volatility_regime = self._analyze_volatility_regime(df)
            trend_regime = self._analyze_trend_regime(df)
            volume_regime = self._analyze_volume_regime(df)
            momentum_regime = self._analyze_momentum_regime(df)
            
            # Combine regime signals
            regime_score = self._calculate_regime_score(
                volatility_regime, trend_regime, volume_regime, momentum_regime
            )
            
            # Determine primary regime
            regime_classification = self._classify_market_regime(regime_score)
            
            # Calculate confidence and sub-regimes
            regime_analysis = {
                'primary_regime': regime_classification['regime'],
                'confidence': regime_classification['confidence'],
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'volume_regime': volume_regime,
                'momentum_regime': momentum_regime,
                'regime_score': regime_score,
                'timestamp': datetime.now().isoformat(),
                'adaptive_recommendations': self._get_regime_recommendations(regime_classification)
            }
            
            # Store for history
            self.regime_history.append(regime_analysis)
            
            return regime_analysis
            
        except Exception as e:
            print(f"Error in market regime analysis: {e}")
            return self._default_regime()

    def detect_market_patterns(self, df: pd.DataFrame, signal_data: Dict) -> Dict:
        """
        Advanced pattern recognition for trading signals
        """
        try:
            patterns = {
                'technical_patterns': self._detect_technical_patterns(df),
                'volume_patterns': self._detect_volume_patterns(df),
                'volatility_patterns': self._detect_volatility_patterns(df),
                'momentum_patterns': self._detect_momentum_patterns(df),
                'support_resistance_patterns': self._detect_support_resistance(df),
                'signal_quality_score': self._calculate_signal_quality(df, signal_data)
            }
            
            # Analyze historical success for similar patterns
            patterns['historical_success_rate'] = self._get_pattern_success_rate(patterns)
            
            # Pattern confidence scoring
            patterns['pattern_confidence'] = self._calculate_pattern_confidence(patterns)
            
            # Save pattern for future reference
            self._store_pattern(patterns, signal_data)
            
            return patterns
            
        except Exception as e:
            print(f"Error in pattern detection: {e}")
            return {'pattern_confidence': 0.5, 'historical_success_rate': 0.5}

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, market_regime: str) -> Dict:
        """
        Calculate adaptive thresholds based on market conditions and historical performance
        """
        try:
            # Base thresholds
            base_thresholds = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_threshold': 0.001,
                'volume_spike': 1.5,
                'volatility_alert': 0.03
            }
            
            # Market condition adjustments
            market_adjustments = self._get_market_adjustments(df, market_regime)
            
            # Historical performance adjustments
            performance_adjustments = self._get_performance_adjustments()
            
            # Volatility-based adjustments
            volatility_adjustments = self._get_volatility_adjustments(df)
            
            # Calculate final adaptive thresholds
            adaptive_thresholds = {}
            for key, base_value in base_thresholds.items():
                market_adj = market_adjustments.get(key, 1.0)
                perf_adj = performance_adjustments.get(key, 1.0)
                vol_adj = volatility_adjustments.get(key, 1.0)
                
                # Combine adjustments
                final_adjustment = (market_adj + perf_adj + vol_adj) / 3
                adaptive_thresholds[key] = base_value * final_adjustment
                
            # Add context information
            adaptive_thresholds.update({
                'market_regime': market_regime,
                'volatility_factor': volatility_adjustments.get('volatility_factor', 1.0),
                'trend_strength': self._calculate_trend_strength(df),
                'market_stress_level': self._calculate_market_stress(df),
                'timestamp': datetime.now().isoformat()
            })
            
            # Store for future use
            self.adaptive_params.update(adaptive_thresholds)
            
            return adaptive_thresholds
            
        except Exception as e:
            print(f"Error calculating adaptive thresholds: {e}")
            return self.adaptive_params

    def analyze_signal_success_probability(self, signal_data: Dict, market_context: Dict) -> float:
        """
        Analyze probability of signal success based on historical patterns
        """
        try:
            if not self.signal_success_history:
                return 0.5  # Neutral probability
                
            # Create feature vector for current signal
            current_features = self._extract_signal_features(signal_data, market_context)
            
            # Find similar historical signals
            similar_signals = self._find_similar_signals(current_features)
            
            if not similar_signals:
                return 0.5
                
            # Calculate success rate of similar signals
            success_rate = sum(1 for s in similar_signals if s.get('profitable', False)) / len(similar_signals)
            
            # Apply confidence weighting based on sample size
            confidence_weight = min(1.0, len(similar_signals) / 10)
            
            # Blend with overall success rate
            overall_success = sum(1 for s in self.signal_success_history if s.get('profitable', False)) / len(self.signal_success_history)
            
            weighted_probability = (success_rate * confidence_weight + 
                                 overall_success * (1 - confidence_weight))
            
            return max(0.1, min(0.9, weighted_probability))
            
        except Exception as e:
            print(f"Error analyzing signal success probability: {e}")
            return 0.5

    def get_market_intelligence_summary(self, df: pd.DataFrame, signal_data: Dict) -> Dict:
        """
        Generate comprehensive market intelligence summary
        """
        try:
            # Analyze current market regime
            regime_analysis = self.analyze_market_regime(df)
            
            # Detect patterns
            pattern_analysis = self.detect_market_patterns(df, signal_data)
            
            # Calculate adaptive thresholds
            adaptive_thresholds = self.calculate_adaptive_thresholds(df, regime_analysis['primary_regime'])
            
            # Analyze signal probability
            signal_probability = self.analyze_signal_success_probability(signal_data, regime_analysis)
            
            # Market stress assessment
            market_stress = self._assess_market_stress(df)
            
            # Generate recommendations
            recommendations = self._generate_trading_recommendations(
                regime_analysis, pattern_analysis, adaptive_thresholds, signal_probability
            )
            
            summary = {
                'market_regime': regime_analysis,
                'pattern_analysis': pattern_analysis,
                'adaptive_thresholds': adaptive_thresholds,
                'signal_probability': signal_probability,
                'market_stress': market_stress,
                'recommendations': recommendations,
                'intelligence_score': self._calculate_intelligence_score(
                    regime_analysis, pattern_analysis, signal_probability
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error generating market intelligence summary: {e}")
            return self._default_intelligence_summary()

    # Private methods for detailed analysis

    def _analyze_volatility_regime(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility-based market regime"""
        try:
            # Calculate rolling volatility
            returns = df['close'].pct_change()
            vol_short = returns.rolling(5).std()
            vol_medium = returns.rolling(20).std()
            vol_long = returns.rolling(50).std()
            
            current_vol = vol_short.iloc[-1]
            
            # Classify volatility regime
            if current_vol > vol_long.quantile(0.8):
                regime = 'HIGH'
            elif current_vol < vol_long.quantile(0.2):
                regime = 'LOW'
            else:
                regime = 'NORMAL'
                
            return {
                'regime': regime,
                'current_volatility': current_vol,
                'volatility_percentile': vol_long.rank(pct=True).iloc[-1],
                'volatility_trend': 'INCREASING' if vol_short.iloc[-1] > vol_medium.iloc[-1] else 'DECREASING'
            }
            
        except Exception:
            return {'regime': 'NORMAL', 'current_volatility': 0.02}

    def _analyze_trend_regime(self, df: pd.DataFrame) -> Dict:
        """Analyze trend-based market regime"""
        try:
            # Calculate trend indicators
            sma_short = df['close'].rolling(5).mean()
            sma_medium = df['close'].rolling(20).mean()
            sma_long = df['close'].rolling(50).mean()
            
            current_price = df['close'].iloc[-1]
            
            # Determine trend strength and direction
            if current_price > sma_short.iloc[-1] > sma_medium.iloc[-1] > sma_long.iloc[-1]:
                trend = 'STRONG_UPTREND'
            elif current_price < sma_short.iloc[-1] < sma_medium.iloc[-1] < sma_long.iloc[-1]:
                trend = 'STRONG_DOWNTREND'
            elif current_price > sma_medium.iloc[-1]:
                trend = 'WEAK_UPTREND'
            elif current_price < sma_medium.iloc[-1]:
                trend = 'WEAK_DOWNTREND'
            else:
                trend = 'SIDEWAYS'
                
            return {
                'regime': trend,
                'trend_strength': abs(current_price - sma_medium.iloc[-1]) / sma_medium.iloc[-1],
                'momentum': (current_price / df['close'].iloc[-10] - 1) if len(df) >= 10 else 0
            }
            
        except Exception:
            return {'regime': 'SIDEWAYS', 'trend_strength': 0}

    def _analyze_volume_regime(self, df: pd.DataFrame) -> Dict:
        """Analyze volume-based market regime"""
        try:
            if 'volume' not in df.columns:
                return {'regime': 'NORMAL', 'volume_trend': 'STABLE'}
                
            volume_ma = df['volume'].rolling(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma.iloc[-1]
            
            if volume_ratio > 2.0:
                regime = 'SPIKE'
            elif volume_ratio > 1.5:
                regime = 'HIGH'
            elif volume_ratio < 0.7:
                regime = 'LOW'
            else:
                regime = 'NORMAL'
                
            return {
                'regime': regime,
                'volume_ratio': volume_ratio,
                'volume_trend': 'INCREASING' if df['volume'].rolling(5).mean().iloc[-1] > volume_ma.iloc[-1] else 'DECREASING'
            }
            
        except Exception:
            return {'regime': 'NORMAL', 'volume_trend': 'STABLE'}

    def _analyze_momentum_regime(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum-based market regime"""
        try:
            # RSI momentum
            rsi = df.get('rsi', pd.Series([50] * len(df)))
            
            # MACD momentum
            macd = df.get('macd', pd.Series([0] * len(df)))
            
            current_rsi = rsi.iloc[-1]
            current_macd = macd.iloc[-1]
            
            # Classify momentum regime
            if current_rsi > 70 and current_macd > 0:
                regime = 'STRONG_BULLISH'
            elif current_rsi < 30 and current_macd < 0:
                regime = 'STRONG_BEARISH'
            elif current_rsi > 50 and current_macd > 0:
                regime = 'BULLISH'
            elif current_rsi < 50 and current_macd < 0:
                regime = 'BEARISH'
            else:
                regime = 'NEUTRAL'
                
            return {
                'regime': regime,
                'rsi_level': current_rsi,
                'macd_level': current_macd,
                'momentum_score': (current_rsi - 50) / 50 + np.sign(current_macd) * min(abs(current_macd) * 1000, 1)
            }
            
        except Exception:
            return {'regime': 'NEUTRAL', 'momentum_score': 0}

    def _calculate_regime_score(self, vol_regime, trend_regime, volume_regime, momentum_regime) -> float:
        """Calculate composite regime score"""
        try:
            # Scoring system for different regimes
            scores = {
                'volatility': {'LOW': 0.2, 'NORMAL': 0.5, 'HIGH': 0.8},
                'trend': {'STRONG_DOWNTREND': 0.1, 'WEAK_DOWNTREND': 0.3, 'SIDEWAYS': 0.5, 'WEAK_UPTREND': 0.7, 'STRONG_UPTREND': 0.9},
                'volume': {'LOW': 0.3, 'NORMAL': 0.5, 'HIGH': 0.7, 'SPIKE': 0.9},
                'momentum': {'STRONG_BEARISH': 0.1, 'BEARISH': 0.3, 'NEUTRAL': 0.5, 'BULLISH': 0.7, 'STRONG_BULLISH': 0.9}
            }
            
            vol_score = scores['volatility'].get(vol_regime['regime'], 0.5)
            trend_score = scores['trend'].get(trend_regime['regime'], 0.5)
            volume_score = scores['volume'].get(volume_regime['regime'], 0.5)
            momentum_score = scores['momentum'].get(momentum_regime['regime'], 0.5)
            
            # Weighted average
            composite_score = (vol_score * 0.3 + trend_score * 0.3 + volume_score * 0.2 + momentum_score * 0.2)
            
            return composite_score
            
        except Exception:
            return 0.5

    def _classify_market_regime(self, regime_score: float) -> Dict:
        """Classify overall market regime based on composite score"""
        if regime_score > 0.75:
            return {'regime': 'BULLISH_EXTREME', 'confidence': 0.9}
        elif regime_score > 0.6:
            return {'regime': 'BULLISH', 'confidence': 0.7}
        elif regime_score > 0.4:
            return {'regime': 'NEUTRAL', 'confidence': 0.6}
        elif regime_score > 0.25:
            return {'regime': 'BEARISH', 'confidence': 0.7}
        else:
            return {'regime': 'BEARISH_EXTREME', 'confidence': 0.9}

    def _detect_technical_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect technical chart patterns"""
        patterns = {}
        
        try:
            # Simple pattern detection
            if len(df) >= 10:
                recent_highs = df['high'].tail(10) if 'high' in df.columns else df['close'].tail(10)
                recent_lows = df['low'].tail(10) if 'low' in df.columns else df['close'].tail(10)
                
                # Double top/bottom detection
                patterns['double_top'] = self._detect_double_top(recent_highs)
                patterns['double_bottom'] = self._detect_double_bottom(recent_lows)
                
                # Support/resistance levels
                patterns['support_level'] = recent_lows.min()
                patterns['resistance_level'] = recent_highs.max()
                
                # Breakout detection
                current_price = df['close'].iloc[-1]
                patterns['breakout_above_resistance'] = current_price > patterns['resistance_level'] * 0.995
                patterns['breakdown_below_support'] = current_price < patterns['support_level'] * 1.005
                
        except Exception:
            pass
            
        return patterns

    def _detect_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect volume-based patterns"""
        patterns = {}
        
        try:
            if 'volume' in df.columns and len(df) >= 5:
                volume = df['volume']
                volume_ma = volume.rolling(10).mean()
                
                # Volume patterns
                patterns['volume_spike'] = volume.iloc[-1] > volume_ma.iloc[-1] * 1.5
                patterns['volume_dry_up'] = volume.iloc[-1] < volume_ma.iloc[-1] * 0.7
                patterns['volume_increasing'] = volume.tail(3).mean() > volume.tail(6).mean()
                
        except Exception:
            pass
            
        return patterns

    def _detect_volatility_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect volatility-based patterns"""
        patterns = {}
        
        try:
            returns = df['close'].pct_change()
            vol = returns.rolling(10).std()
            
            patterns['volatility_expansion'] = vol.iloc[-1] > vol.rolling(20).mean().iloc[-1] * 1.2
            patterns['volatility_contraction'] = vol.iloc[-1] < vol.rolling(20).mean().iloc[-1] * 0.8
            patterns['low_volatility_breakout'] = (vol.iloc[-5:].mean() < vol.rolling(50).mean().iloc[-1] * 0.7 and 
                                                 abs(returns.iloc[-1]) > vol.rolling(20).mean().iloc[-1] * 2)
            
        except Exception:
            pass
            
        return patterns

    def _detect_momentum_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect momentum-based patterns"""
        patterns = {}
        
        try:
            if 'rsi' in df.columns:
                rsi = df['rsi']
                patterns['rsi_divergence'] = self._detect_rsi_divergence(df, rsi)
                patterns['rsi_oversold_bounce'] = rsi.iloc[-1] < 30 and rsi.iloc[-1] > rsi.iloc[-2]
                patterns['rsi_overbought_pullback'] = rsi.iloc[-1] > 70 and rsi.iloc[-1] < rsi.iloc[-2]
                
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = df['macd']
                macd_signal = df['macd_signal']
                patterns['macd_bullish_crossover'] = (macd.iloc[-1] > macd_signal.iloc[-1] and 
                                                    macd.iloc[-2] <= macd_signal.iloc[-2])
                patterns['macd_bearish_crossover'] = (macd.iloc[-1] < macd_signal.iloc[-1] and 
                                                    macd.iloc[-2] >= macd_signal.iloc[-2])
                
        except Exception:
            pass
            
        return patterns

    def _detect_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Detect support and resistance levels"""
        levels = {}
        
        try:
            if len(df) >= 20:
                recent_data = df.tail(20)
                
                # Support levels (recent lows)
                lows = recent_data['low'] if 'low' in recent_data.columns else recent_data['close']
                support_levels = lows.nsmallest(3).values
                
                # Resistance levels (recent highs)
                highs = recent_data['high'] if 'high' in recent_data.columns else recent_data['close']
                resistance_levels = highs.nlargest(3).values
                
                current_price = df['close'].iloc[-1]
                
                levels = {
                    'support_levels': support_levels.tolist(),
                    'resistance_levels': resistance_levels.tolist(),
                    'nearest_support': max([s for s in support_levels if s < current_price], default=current_price * 0.95),
                    'nearest_resistance': min([r for r in resistance_levels if r > current_price], default=current_price * 1.05),
                    'support_strength': len([s for s in support_levels if abs(s - current_price) / current_price < 0.02]),
                    'resistance_strength': len([r for r in resistance_levels if abs(r - current_price) / current_price < 0.02])
                }
                
        except Exception:
            levels = {'support_levels': [], 'resistance_levels': []}
            
        return levels

    def _calculate_signal_quality(self, df: pd.DataFrame, signal_data: Dict) -> float:
        """Calculate overall signal quality score"""
        try:
            quality_factors = []
            
            # Technical alignment
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if signal_data.get('action') == 'BUY' and rsi < 40:
                    quality_factors.append(0.8)
                elif signal_data.get('action') == 'SELL' and rsi > 60:
                    quality_factors.append(0.8)
                else:
                    quality_factors.append(0.4)
                    
            # Volume confirmation
            if 'volume' in df.columns:
                volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(10).mean().iloc[-1]
                if volume_ratio > 1.2:
                    quality_factors.append(0.7)
                else:
                    quality_factors.append(0.3)
                    
            # Trend alignment
            if len(df) >= 20:
                price_trend = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
                if (signal_data.get('action') == 'BUY' and price_trend > 0) or \
                   (signal_data.get('action') == 'SELL' and price_trend < 0):
                    quality_factors.append(0.7)
                else:
                    quality_factors.append(0.3)
                    
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
            
        except Exception:
            return 0.5

    def _get_pattern_success_rate(self, patterns: Dict) -> float:
        """Get historical success rate for similar patterns"""
        try:
            if not self.signal_success_history:
                return 0.5
                
            # Simple pattern matching based on key indicators
            similar_count = 0
            successful_count = 0
            
            for historical_signal in self.signal_success_history:
                similarity_score = 0
                
                # Compare technical patterns
                if patterns.get('technical_patterns', {}).get('breakout_above_resistance') == \
                   historical_signal.get('patterns', {}).get('breakout_above_resistance'):
                    similarity_score += 1
                    
                if patterns.get('volume_patterns', {}).get('volume_spike') == \
                   historical_signal.get('patterns', {}).get('volume_spike'):
                    similarity_score += 1
                    
                # If patterns are similar enough
                if similarity_score >= 1:
                    similar_count += 1
                    if historical_signal.get('profitable', False):
                        successful_count += 1
                        
            if similar_count >= 3:
                return successful_count / similar_count
            else:
                return 0.5
                
        except Exception:
            return 0.5

    def _calculate_pattern_confidence(self, patterns: Dict) -> float:
        """Calculate confidence in pattern detection"""
        try:
            confidence_factors = []
            
            # Technical pattern confidence
            tech_patterns = patterns.get('technical_patterns', {})
            if any(tech_patterns.values()):
                confidence_factors.append(0.7)
                
            # Volume pattern confidence
            vol_patterns = patterns.get('volume_patterns', {})
            if vol_patterns.get('volume_spike', False):
                confidence_factors.append(0.8)
                
            # Signal quality confidence
            signal_quality = patterns.get('signal_quality_score', 0.5)
            confidence_factors.append(signal_quality)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5

    def _store_pattern(self, patterns: Dict, signal_data: Dict):
        """Store pattern for future analysis"""
        try:
            pattern_entry = {
                'timestamp': datetime.now().isoformat(),
                'patterns': patterns,
                'signal_data': signal_data,
                'market_conditions': self.market_context_cache
            }
            
            # Store in memory
            pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.pattern_database[pattern_id] = pattern_entry
            
            # Save to file periodically
            if len(self.pattern_database) % 10 == 0:
                with open('market_patterns.json', 'w') as f:
                    json.dump(self.market_patterns, f, indent=2)
                    
        except Exception as e:
            print(f"Error storing pattern: {e}")

    def _default_regime(self) -> Dict:
        """Default market regime when analysis fails"""
        return {
            'primary_regime': 'NEUTRAL',
            'confidence': 0.5,
            'volatility_regime': {'regime': 'NORMAL'},
            'trend_regime': {'regime': 'SIDEWAYS'},
            'volume_regime': {'regime': 'NORMAL'},
            'momentum_regime': {'regime': 'NEUTRAL'},
            'regime_score': 0.5,
            'timestamp': datetime.now().isoformat()
        }

    def _default_intelligence_summary(self) -> Dict:
        """Default intelligence summary when analysis fails"""
        return {
            'market_regime': self._default_regime(),
            'pattern_analysis': {'pattern_confidence': 0.5},
            'adaptive_thresholds': self.adaptive_params,
            'signal_probability': 0.5,
            'market_stress': {'stress_level': 'NORMAL'},
            'recommendations': {'position_size_adjustment': 1.0},
            'intelligence_score': 0.5,
            'timestamp': datetime.now().isoformat()
        }

    # Additional helper methods would continue here...
    # (Implementing remaining private methods for completeness)

    def _get_market_adjustments(self, df: pd.DataFrame, regime: str) -> Dict:
        """Get market condition based threshold adjustments"""
        adjustments = {
            'BULLISH_EXTREME': {'rsi_oversold': 0.8, 'rsi_overbought': 1.1, 'macd_threshold': 1.5},
            'BULLISH': {'rsi_oversold': 0.9, 'rsi_overbought': 1.05, 'macd_threshold': 1.2},
            'NEUTRAL': {'rsi_oversold': 1.0, 'rsi_overbought': 1.0, 'macd_threshold': 1.0},
            'BEARISH': {'rsi_oversold': 1.05, 'rsi_overbought': 0.9, 'macd_threshold': 1.2},
            'BEARISH_EXTREME': {'rsi_oversold': 1.1, 'rsi_overbought': 0.8, 'macd_threshold': 1.5}
        }
        return adjustments.get(regime, adjustments['NEUTRAL'])

    def _get_performance_adjustments(self) -> Dict:
        """Get performance-based threshold adjustments"""
        # Analyze recent performance and adjust accordingly
        if not self.signal_success_history:
            return {'rsi_oversold': 1.0, 'rsi_overbought': 1.0, 'macd_threshold': 1.0}
            
        recent_success_rate = sum(1 for s in self.signal_success_history[-20:] if s.get('profitable', False)) / min(20, len(self.signal_success_history))
        
        if recent_success_rate > 0.7:
            return {'rsi_oversold': 0.95, 'rsi_overbought': 1.05, 'macd_threshold': 0.9}
        elif recent_success_rate < 0.4:
            return {'rsi_oversold': 1.05, 'rsi_overbought': 0.95, 'macd_threshold': 1.1}
        else:
            return {'rsi_oversold': 1.0, 'rsi_overbought': 1.0, 'macd_threshold': 1.0}

    def _get_volatility_adjustments(self, df: pd.DataFrame) -> Dict:
        """Get volatility-based threshold adjustments"""
        try:
            current_vol = df['close'].pct_change().rolling(10).std().iloc[-1]
            historical_vol = df['close'].pct_change().rolling(50).std().mean()
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            return {
                'rsi_oversold': 1.0 + (vol_ratio - 1) * 0.1,
                'rsi_overbought': 1.0 - (vol_ratio - 1) * 0.1,
                'macd_threshold': 1.0 + (vol_ratio - 1) * 0.2,
                'volatility_factor': vol_ratio
            }
        except Exception:
            return {'rsi_oversold': 1.0, 'rsi_overbought': 1.0, 'macd_threshold': 1.0, 'volatility_factor': 1.0}

    def _extract_signal_features(self, signal_data: Dict, market_context: Dict) -> Dict:
        """Extract features from current signal for pattern matching"""
        return {
            'action': signal_data.get('action'),
            'rsi': signal_data.get('rsi'),
            'macd_trend': signal_data.get('macd_trend'),
            'volume_ratio': signal_data.get('volume_ratio'),
            'market_regime': market_context.get('primary_regime'),
            'volatility_level': market_context.get('volatility_regime', {}).get('regime'),
            'trend_strength': market_context.get('trend_regime', {}).get('trend_strength', 0)
        }

    def _find_similar_signals(self, current_features: Dict) -> List[Dict]:
        """Find historically similar signals"""
        similar_signals = []
        
        for historical_signal in self.signal_success_history:
            similarity_score = 0
            total_features = 0
            
            for feature, value in current_features.items():
                if feature in historical_signal:
                    total_features += 1
                    if feature == 'action' and historical_signal[feature] == value:
                        similarity_score += 1
                    elif isinstance(value, (int, float)) and isinstance(historical_signal[feature], (int, float)):
                        # Numeric similarity
                        diff = abs(value - historical_signal[feature])
                        if diff <= abs(value) * 0.2:  # Within 20%
                            similarity_score += 1
                    elif str(historical_signal[feature]) == str(value):
                        similarity_score += 1
                        
            if total_features > 0 and similarity_score / total_features >= 0.6:  # 60% similarity
                similar_signals.append(historical_signal)
                
        return similar_signals

    def _assess_market_stress(self, df: pd.DataFrame) -> Dict:
        """Assess current market stress levels"""
        try:
            stress_indicators = []
            
            # Volatility stress
            vol = df['close'].pct_change().rolling(10).std().iloc[-1]
            vol_percentile = df['close'].pct_change().rolling(50).std().rank(pct=True).iloc[-1]
            if vol_percentile > 0.8:
                stress_indicators.append('HIGH_VOLATILITY')
                
            # Volume stress
            if 'volume' in df.columns:
                vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                if vol_ratio > 2.0:
                    stress_indicators.append('VOLUME_SPIKE')
                    
            # Price movement stress
            price_change = abs(df['close'].pct_change().iloc[-1])
            if price_change > vol * 2:
                stress_indicators.append('EXTREME_MOVEMENT')
                
            # Determine overall stress level
            if len(stress_indicators) >= 2:
                stress_level = 'HIGH'
            elif len(stress_indicators) == 1:
                stress_level = 'MEDIUM'
            else:
                stress_level = 'LOW'
                
            return {
                'stress_level': stress_level,
                'stress_indicators': stress_indicators,
                'volatility_stress': vol_percentile,
                'market_fear_index': min(1.0, len(stress_indicators) / 3)
            }
            
        except Exception:
            return {'stress_level': 'NORMAL', 'stress_indicators': []}

    def _generate_trading_recommendations(self, regime_analysis, pattern_analysis, adaptive_thresholds, signal_probability) -> Dict:
        """Generate intelligent trading recommendations"""
        recommendations = {
            'position_size_adjustment': 1.0,
            'risk_adjustment': 1.0,
            'signal_filter_adjustment': 1.0,
            'trading_advice': []
        }
        
        try:
            # Regime-based recommendations
            regime = regime_analysis.get('primary_regime', 'NEUTRAL')
            
            if regime in ['BEARISH_EXTREME', 'BULLISH_EXTREME']:
                recommendations['position_size_adjustment'] = 0.7
                recommendations['risk_adjustment'] = 1.3
                recommendations['trading_advice'].append(f"Extreme {regime.split('_')[0].lower()} regime detected - reduce position sizes")
                
            # Pattern-based recommendations
            pattern_confidence = pattern_analysis.get('pattern_confidence', 0.5)
            if pattern_confidence > 0.8:
                recommendations['signal_filter_adjustment'] = 0.8
                recommendations['trading_advice'].append("High pattern confidence - more aggressive signal filtering")
            elif pattern_confidence < 0.3:
                recommendations['signal_filter_adjustment'] = 1.2
                recommendations['trading_advice'].append("Low pattern confidence - conservative signal filtering")
                
            # Signal probability recommendations
            if signal_probability > 0.7:
                recommendations['position_size_adjustment'] *= 1.2
                recommendations['trading_advice'].append("High signal success probability - consider larger positions")
            elif signal_probability < 0.3:
                recommendations['position_size_adjustment'] *= 0.6
                recommendations['trading_advice'].append("Low signal success probability - reduce position sizes")
                
        except Exception:
            pass
            
        return recommendations

    def _calculate_intelligence_score(self, regime_analysis, pattern_analysis, signal_probability) -> float:
        """Calculate overall market intelligence confidence score"""
        try:
            regime_confidence = regime_analysis.get('confidence', 0.5)
            pattern_confidence = pattern_analysis.get('pattern_confidence', 0.5)
            signal_confidence = abs(signal_probability - 0.5) * 2  # Convert to 0-1 confidence
            
            intelligence_score = (regime_confidence * 0.4 + pattern_confidence * 0.4 + signal_confidence * 0.2)
            return max(0.1, min(0.95, intelligence_score))
            
        except Exception:
            return 0.5

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength indicator"""
        try:
            if len(df) < 20:
                return 0.0
                
            sma_fast = df['close'].rolling(5).mean()
            sma_slow = df['close'].rolling(20).mean()
            
            trend_strength = abs(sma_fast.iloc[-1] - sma_slow.iloc[-1]) / sma_slow.iloc[-1]
            return min(1.0, trend_strength * 10)  # Scale to 0-1
            
        except Exception:
            return 0.0

    def _calculate_market_stress(self, df: pd.DataFrame) -> float:
        """Calculate market stress indicator"""
        try:
            vol = df['close'].pct_change().rolling(10).std().iloc[-1]
            vol_ma = df['close'].pct_change().rolling(50).std().mean()
            
            stress = vol / vol_ma if vol_ma > 0 else 1.0
            return min(2.0, max(0.0, stress))
            
        except Exception:
            return 1.0

    def _get_regime_recommendations(self, regime_classification: Dict) -> Dict:
        """Get specific recommendations based on market regime"""
        regime = regime_classification.get('regime', 'NEUTRAL')
        
        recommendations = {
            'BULLISH_EXTREME': {
                'position_sizing': 'REDUCED',
                'risk_management': 'ENHANCED',
                'signal_filtering': 'STRICT',
                'rebalancing_frequency': 'INCREASED'
            },
            'BULLISH': {
                'position_sizing': 'NORMAL',
                'risk_management': 'NORMAL',
                'signal_filtering': 'NORMAL',
                'rebalancing_frequency': 'NORMAL'
            },
            'NEUTRAL': {
                'position_sizing': 'CONSERVATIVE',
                'risk_management': 'BALANCED',
                'signal_filtering': 'BALANCED',
                'rebalancing_frequency': 'NORMAL'
            },
            'BEARISH': {
                'position_sizing': 'REDUCED',
                'risk_management': 'ENHANCED',
                'signal_filtering': 'STRICT',
                'rebalancing_frequency': 'REDUCED'
            },
            'BEARISH_EXTREME': {
                'position_sizing': 'MINIMAL',
                'risk_management': 'MAXIMUM',
                'signal_filtering': 'VERY_STRICT',
                'rebalancing_frequency': 'MINIMAL'
            }
        }
        
        return recommendations.get(regime, recommendations['NEUTRAL'])

    # Simple pattern detection methods
    def _detect_double_top(self, highs) -> bool:
        """Simple double top pattern detection"""
        try:
            if len(highs) < 7:
                return False
                
            # Look for two peaks with similar heights
            peaks = []
            for i in range(1, len(highs) - 1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    peaks.append((i, highs.iloc[i]))
                    
            if len(peaks) >= 2:
                # Check if last two peaks are similar in height
                last_two = peaks[-2:]
                height_diff = abs(last_two[0][1] - last_two[1][1]) / max(last_two[0][1], last_two[1][1])
                return height_diff < 0.02  # Within 2%
                
            return False
            
        except Exception:
            return False

    def _detect_double_bottom(self, lows) -> bool:
        """Simple double bottom pattern detection"""
        try:
            if len(lows) < 7:
                return False
                
            # Look for two troughs with similar depths
            troughs = []
            for i in range(1, len(lows) - 1):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    troughs.append((i, lows.iloc[i]))
                    
            if len(troughs) >= 2:
                # Check if last two troughs are similar in depth
                last_two = troughs[-2:]
                depth_diff = abs(last_two[0][1] - last_two[1][1]) / max(last_two[0][1], last_two[1][1])
                return depth_diff < 0.02  # Within 2%
                
            return False
            
        except Exception:
            return False

    def _detect_rsi_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """Simple RSI divergence detection"""
        try:
            if len(df) < 10:
                return False
                
            # Look for price making new highs/lows while RSI doesn't
            recent_price = df['close'].tail(5)
            recent_rsi = rsi.tail(5)
            
            prev_price = df['close'].iloc[-10:-5]
            prev_rsi = rsi.iloc[-10:-5]
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            if recent_price.min() < prev_price.min() and recent_rsi.min() > prev_rsi.min():
                return True
                
            # Bearish divergence: price makes higher high, RSI makes lower high
            if recent_price.max() > prev_price.max() and recent_rsi.max() < prev_rsi.max():
                return True
                
            return False
            
        except Exception:
            return False


# Factory function for easy integration
def get_market_intelligence(lookback_days=30):
    """Factory function to get MarketIntelligence instance"""
    return MarketIntelligence(lookback_days=lookback_days)
