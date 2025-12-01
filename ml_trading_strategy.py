"""
ML-Only Trading Strategy for CRYPTIX Trading Bot
====================================================

This strategy relies exclusively on Machine Learning capabilities for signal generation
and trade decisions. It aims for profitable trading (targeting 10%+ daily returns) by:

1. Using ML pattern recognition for signal validation
2. Market regime detection for adaptive trading
3. Intelligent market timing and entry/exit optimization
4. Buy low/sell high logic with ML predictions
5. Smart risk management based on ML confidence scores

Features:
- Pure ML-driven decision making
- Adaptive position sizing based on ML confidence
- Market regime-aware trading adjustments
- Pattern recognition for optimal entry/exit timing
- Smart profit optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging

# Import ML modules
try:
    from ml_predictor import EnhancedMLPredictor
    from market_intelligence import MarketIntelligence
    ML_MODULES_AVAILABLE = True
except ImportError:
    ML_MODULES_AVAILABLE = False

import config

class MLTradingStrategy:
    """
    Pure ML-driven trading strategy for maximum profitability
    """
    
    def __init__(self):
        """Initialize ML Trading Strategy"""
        self.name = "ML_PURE"
        self.description = "Pure Machine Learning Strategy - Smart, Profitable, Non-Conservative"
        
        # Initialize ML components
        if ML_MODULES_AVAILABLE:
            self.ml_predictor = EnhancedMLPredictor()
            self.market_intelligence = MarketIntelligence(lookback_days=30)
        else:
            self.ml_predictor = None
            self.market_intelligence = None
            
        # Strategy parameters (optimized for frequent profitable trading)
        self.min_confidence_threshold = 0.4   # Reduced for more aggressive trading
        self.high_confidence_threshold = 0.65  # Reduced for more signals
        self.profit_target_multiplier = 1.5    # Target profit multiplier
        self.max_position_size_multiplier = 2.0  # Max position size multiplier for high confidence
        
        # Trading performance tracking
        self.trade_history = []
        self.performance_metrics = {
            'daily_return': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # Market regime awareness (more aggressive thresholds)
        self.current_regime = 'NORMAL'
        self.regime_adjustments = {
            'BULLISH_EXTREME': {'position_multiplier': 1.5, 'confidence_threshold': 0.5},
            'BULLISH': {'position_multiplier': 1.2, 'confidence_threshold': 0.45},
            'NEUTRAL': {'position_multiplier': 1.0, 'confidence_threshold': 0.4},
            'BEARISH': {'position_multiplier': 0.9, 'confidence_threshold': 0.45},
            'BEARISH_EXTREME': {'position_multiplier': 0.7, 'confidence_threshold': 0.5}
        }
        
        logging.info(f"🧠 ML Trading Strategy initialized - Target: Smart & Profitable")
    
    def generate_signal(self, df: pd.DataFrame, symbol: str, indicators: Dict) -> Tuple[str, str, Dict]:
        """
        Generate trading signal using pure ML analysis
        
        Returns:
            Tuple[signal, reason, analysis_data]
        """
        if not ML_MODULES_AVAILABLE:
            return "HOLD", "ML modules not available", {}
            
        try:
            # Step 1: Comprehensive Market Intelligence Analysis
            market_analysis = self._analyze_market_conditions(df, symbol, indicators)
            
            # Step 2: ML Pattern Recognition & Prediction
            ml_prediction = self._get_ml_predictions(df, symbol, indicators, market_analysis)
            
            # Step 3: Smart Entry/Exit Timing Analysis
            timing_analysis = self._analyze_timing_opportunities(df, indicators, market_analysis)
            
            # Step 4: Buy Low / Sell High Logic
            price_position_analysis = self._analyze_price_position(df, indicators)
            
            # Step 5: Generate final signal with confidence scoring
            final_signal, reason, analysis_data = self._generate_final_signal(
                market_analysis, ml_prediction, timing_analysis, price_position_analysis, 
                symbol, indicators
            )
            
            # Step 6: Validate signal profitability potential
            validated_signal, final_reason = self._validate_profitability_potential(
                final_signal, reason, analysis_data, df, indicators
            )
            
            # Update performance tracking
            self._update_strategy_metrics(validated_signal, analysis_data)
            
            return validated_signal, final_reason, analysis_data
            
        except Exception as e:
            logging.error(f"Error in ML strategy signal generation: {e}")
            return "HOLD", f"ML Strategy Error: {e}", {}
    
    def _analyze_market_conditions(self, df: pd.DataFrame, symbol: str, indicators: Dict) -> Dict:
        """Comprehensive market analysis using ML intelligence"""
        try:
            # Get comprehensive market intelligence
            market_intel = self.market_intelligence.get_market_intelligence_summary(
                df, {
                    'symbol': symbol,
                    'action': 'ANALYZE',
                    **indicators
                }
            )
            
            # Extract key market conditions
            market_regime = market_intel['market_regime']['primary_regime']
            regime_confidence = market_intel['market_regime']['confidence']
            
            # Update current regime for strategy adjustments
            self.current_regime = market_regime
            
            # Calculate market stress and opportunity scores
            market_stress = market_intel.get('market_stress', {})
            stress_level = market_stress.get('stress_level', 'NORMAL')
            
            # Determine market opportunity score (0-100)
            opportunity_score = self._calculate_market_opportunity_score(market_intel)
            
            return {
                'regime': market_regime,
                'regime_confidence': regime_confidence,
                'stress_level': stress_level,
                'opportunity_score': opportunity_score,
                'intelligence_score': market_intel.get('intelligence_score', 0.5),
                'pattern_confidence': market_intel.get('pattern_analysis', {}).get('pattern_confidence', 0.5),
                'adaptive_thresholds': market_intel.get('adaptive_thresholds', {}),
                'full_analysis': market_intel
            }
            
        except Exception as e:
            logging.error(f"Error in market analysis: {e}")
            return {
                'regime': 'NORMAL',
                'regime_confidence': 0.5,
                'stress_level': 'NORMAL',
                'opportunity_score': 50,
                'intelligence_score': 0.5,
                'pattern_confidence': 0.5
            }
    
    def _get_ml_predictions(self, df: pd.DataFrame, symbol: str, indicators: Dict, market_analysis: Dict) -> Dict:
        """Get ML predictions for price direction and signal success with robust error handling"""
        try:
            # Initialize default values
            signal_success_prob_buy = 0.5
            signal_success_prob_sell = 0.5
            regime_prediction = {'regime': 'NORMAL', 'confidence': 0.5}
            ml_confidence = 0.5
            
            # Try to get ML predictions with graceful error handling
            if self.ml_predictor:
                try:
                    # Create standardized feature vector for signal success prediction
                    # Use only the core features that match the trained model
                    standardized_indicators = {
                        'rsi': indicators.get('rsi', 50),
                        'macd': indicators.get('macd', {}).get('macd', 0) if isinstance(indicators.get('macd'), dict) else indicators.get('macd', 0),
                        'volatility': indicators.get('volatility', 0.02),
                        'volume_ratio': getattr(df, 'volume', pd.Series([1.0])).iloc[-1] / getattr(df, 'volume', pd.Series([1.0])).rolling(20).mean().iloc[-1] if len(df) > 20 else 1.0
                    }
                    
                    # Predict signal success probability with error handling
                    try:
                        signal_success_prob_buy = self.ml_predictor.predict_signal_success(
                            {'action': 'BUY', 'symbol': symbol}, standardized_indicators
                        )
                    except Exception as e:
                        logging.warning(f"Buy signal prediction failed: {e}")
                        signal_success_prob_buy = 0.5
                    
                    try:
                        signal_success_prob_sell = self.ml_predictor.predict_signal_success(
                            {'action': 'SELL', 'symbol': symbol}, standardized_indicators
                        )
                    except Exception as e:
                        logging.warning(f"Sell signal prediction failed: {e}")
                        signal_success_prob_sell = 0.5
                    
                    # Get market regime prediction with error handling
                    try:
                        regime_prediction = self.ml_predictor.predict_market_regime(df)
                    except Exception as e:
                        logging.warning(f"Regime prediction failed: {e}")
                        regime_prediction = {'regime': 'NORMAL', 'confidence': 0.5}
                    
                except Exception as e:
                    logging.warning(f"ML predictor error: {e}")
            
            # Calculate ML confidence score
            ml_confidence = self._calculate_ml_confidence_score(
                signal_success_prob_buy, signal_success_prob_sell, regime_prediction, market_analysis
            )
            
            return {
                'buy_success_probability': signal_success_prob_buy,
                'sell_success_probability': signal_success_prob_sell,
                'regime_prediction': regime_prediction,
                'ml_confidence': ml_confidence,
                'trend_prediction': self._predict_short_term_trend(df, indicators),
                'volatility_forecast': self._forecast_volatility(df)
            }
            
        except Exception as e:
            logging.error(f"Error in ML predictions: {e}")
            # Return default safe values
            return {
                'buy_success_probability': 0.5,
                'sell_success_probability': 0.5,
                'regime_prediction': {'regime': 'NORMAL', 'confidence': 0.5},
                'ml_confidence': 0.5,
                'trend_prediction': {'direction': 'NEUTRAL', 'strength': 0.5},
                'volatility_forecast': 0.02
            }
            return {
                'buy_success_probability': 0.5,
                'sell_success_probability': 0.5,
                'ml_confidence': 0.5,
                'trend_prediction': 'NEUTRAL'
            }
    
    def _analyze_timing_opportunities(self, df: pd.DataFrame, indicators: Dict, market_analysis: Dict) -> Dict:
        """Analyze optimal timing for entries and exits"""
        try:
            current_price = indicators['current_price']
            
            # Momentum analysis
            momentum_score = self._calculate_momentum_score(df, indicators)
            
            # Volume analysis
            volume_score = self._calculate_volume_score(df, indicators)
            
            # Volatility timing
            volatility_timing = self._analyze_volatility_timing(df)
            
            # Market hour analysis
            market_hour_score = self._calculate_market_hour_score()
            
            # Support/Resistance proximity
            sr_analysis = self._analyze_support_resistance_proximity(df, current_price)
            
            # Overall timing score (0-100)
            timing_score = (momentum_score * 0.3 + volume_score * 0.25 + 
                          volatility_timing * 0.2 + market_hour_score * 0.15 + 
                          sr_analysis['timing_score'] * 0.1)
            
            return {
                'timing_score': timing_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'volatility_timing': volatility_timing,
                'market_hour_score': market_hour_score,
                'support_resistance': sr_analysis,
                'optimal_entry': timing_score > 65,
                'should_wait': timing_score < 40
            }
            
        except Exception as e:
            logging.error(f"Error in timing analysis: {e}")
            return {'timing_score': 50, 'optimal_entry': False, 'should_wait': False}
    
    def _analyze_price_position(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze current price position for buy low/sell high strategy"""
        try:
            current_price = indicators['current_price']
            
            # Calculate recent price ranges (multiple timeframes)
            ranges = {
                'short_term': self._calculate_price_range(df, 24),  # 24 periods
                'medium_term': self._calculate_price_range(df, 72), # 72 periods 
                'long_term': self._calculate_price_range(df, 168)   # 168 periods
            }
            
            # Calculate position in each range (0 = bottom, 1 = top)
            position_scores = {}
            for timeframe, range_data in ranges.items():
                if range_data['range'] > 0:
                    position_scores[timeframe] = (current_price - range_data['low']) / range_data['range']
                else:
                    position_scores[timeframe] = 0.5
            
            # Weighted average position (favor shorter timeframes for entry timing)
            avg_position = (position_scores['short_term'] * 0.5 + 
                          position_scores['medium_term'] * 0.3 + 
                          position_scores['long_term'] * 0.2)
            
            # Determine buy/sell opportunity based on position
            buy_opportunity = avg_position < 0.3  # In bottom 30% - good to buy low
            sell_opportunity = avg_position > 0.7  # In top 30% - good to sell high
            
            # Calculate confidence in position analysis
            position_confidence = self._calculate_position_confidence(position_scores, ranges)
            
            return {
                'average_position': avg_position,
                'position_scores': position_scores,
                'ranges': ranges,
                'buy_opportunity': buy_opportunity,
                'sell_opportunity': sell_opportunity,
                'position_confidence': position_confidence,
                'buy_low_score': max(0, (0.4 - avg_position) * 2.5),  # Higher when lower in range
                'sell_high_score': max(0, (avg_position - 0.6) * 2.5)  # Higher when higher in range
            }
            
        except Exception as e:
            logging.error(f"Error in price position analysis: {e}")
            return {
                'average_position': 0.5,
                'buy_opportunity': False,
                'sell_opportunity': False,
                'position_confidence': 0.5
            }
    
    def _generate_final_signal(self, market_analysis: Dict, ml_prediction: Dict, 
                             timing_analysis: Dict, price_position: Dict, 
                             symbol: str, indicators: Dict) -> Tuple[str, str, Dict]:
        """Generate final trading signal by combining all ML analyses"""
        try:
            # Extract key metrics
            buy_success_prob = ml_prediction['buy_success_probability']
            sell_success_prob = ml_prediction['sell_success_probability']
            ml_confidence = ml_prediction['ml_confidence']
            timing_score = timing_analysis['timing_score']
            buy_opportunity = price_position['buy_opportunity']
            sell_opportunity = price_position['sell_opportunity']
            market_opportunity = market_analysis['opportunity_score']
            
            # Get regime adjustments
            regime = market_analysis['regime']
            regime_adj = self.regime_adjustments.get(regime, self.regime_adjustments['NEUTRAL'])
            
            # Calculate composite scores for BUY and SELL
            buy_score = self._calculate_buy_score(
                buy_success_prob, ml_confidence, timing_score, buy_opportunity, 
                market_opportunity, price_position, indicators
            )
            
            sell_score = self._calculate_sell_score(
                sell_success_prob, ml_confidence, timing_score, sell_opportunity,
                market_opportunity, price_position, indicators
            )
            
            # Apply regime adjustments
            buy_score *= regime_adj['position_multiplier']
            sell_score *= regime_adj['position_multiplier']
            
            # Minimum confidence threshold
            min_confidence = regime_adj['confidence_threshold']
            
            # Generate signal based on scores and confidence (more aggressive thresholds)
            if buy_score > sell_score and buy_score > 55 and ml_confidence > min_confidence:
                signal = "BUY"
                confidence = buy_score / 100
                reason = f"ML Buy Signal: Score={buy_score:.1f}, Confidence={ml_confidence:.2f}, Regime={regime}"
                
            elif sell_score > buy_score and sell_score > 55 and ml_confidence > min_confidence:
                signal = "SELL"
                confidence = sell_score / 100
                reason = f"ML Sell Signal: Score={sell_score:.1f}, Confidence={ml_confidence:.2f}, Regime={regime}"
                
            # Alternative: if scores are close but one is strong enough, trade it
            elif buy_score > 65 and ml_confidence > (min_confidence * 0.8):
                signal = "BUY"
                confidence = buy_score / 100
                reason = f"ML Buy Signal (Strong): Score={buy_score:.1f}, Confidence={ml_confidence:.2f}, Regime={regime}"
                
            elif sell_score > 65 and ml_confidence > (min_confidence * 0.8):
                signal = "SELL"
                confidence = sell_score / 100
                reason = f"ML Sell Signal (Strong): Score={sell_score:.1f}, Confidence={ml_confidence:.2f}, Regime={regime}"
                
            else:
                signal = "HOLD"
                confidence = max(buy_score, sell_score) / 100
                reason = f"ML Hold: Buy={buy_score:.1f}, Sell={sell_score:.1f}, Confidence={ml_confidence:.2f}"
            
            # Compile analysis data
            analysis_data = {
                'signal_confidence': confidence,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'ml_confidence': ml_confidence,
                'timing_score': timing_score,
                'market_regime': regime,
                'market_opportunity': market_opportunity,
                'buy_opportunity': buy_opportunity,
                'sell_opportunity': sell_opportunity,
                'position_analysis': price_position,
                'timing_analysis': timing_analysis,
                'market_analysis': market_analysis,
                'ml_prediction': ml_prediction,
                'recommended_position_size': self._calculate_position_size_multiplier(confidence, regime)
            }
            
            return signal, reason, analysis_data
            
        except Exception as e:
            logging.error(f"Error generating final signal: {e}")
            return "HOLD", f"Signal generation error: {e}", {}
    
    def _validate_profitability_potential(self, signal: str, reason: str, analysis_data: Dict,
                                        df: pd.DataFrame, indicators: Dict) -> Tuple[str, str]:
        """Validate if the signal has good profitability potential"""
        try:
            if signal == "HOLD":
                return signal, reason
                
            # Calculate expected profitability
            confidence = analysis_data.get('signal_confidence', 0.5)
            market_opportunity = analysis_data.get('market_opportunity', 50)
            timing_score = analysis_data.get('timing_score', 50)
            
            # Expected profit calculation (more lenient for frequent trading)
            base_profit_expectation = 1.0  # Reduced to 1% base expectation
            confidence_multiplier = confidence * 2  # Up to 2x multiplier
            timing_multiplier = timing_score / 50  # 0.5x to 2x based on timing
            opportunity_multiplier = market_opportunity / 50  # 0.5x to 2x based on opportunity
            
            expected_profit = (base_profit_expectation * confidence_multiplier * 
                             timing_multiplier * opportunity_multiplier)
            
            # More lenient profitability threshold for aggressive trading
            profitability_threshold = 0.5  # Reduced from 1.0% to 0.5%
            
            if expected_profit < profitability_threshold:
                # Signal doesn't meet profitability requirements
                return "HOLD", f"Low Profit Potential: {expected_profit:.1f}% < {profitability_threshold}% threshold"
            
            # Disable position range validation for more aggressive trading
            # (Buy low/sell high logic can be re-enabled later if needed)
            # if signal == "BUY":
            #     position_score = analysis_data.get('position_analysis', {}).get('average_position', 0.5)
            #     if position_score > 0.75:  # Only reject if buying very high in range
            #         return "HOLD", f"Buy price too high in range ({position_score:.1%}) - waiting for better entry"
            #         
            # elif signal == "SELL":
            #     position_score = analysis_data.get('position_analysis', {}).get('average_position', 0.5)
            #     if position_score < 0.25:  # Only reject if selling very low in range
            #         return "HOLD", f"Sell price too low in range ({position_score:.1%}) - waiting for higher price"
            
            # Signal passed profitability validation
            enhanced_reason = f"{reason} | Expected Profit: {expected_profit:.1f}%"
            return signal, enhanced_reason
            
        except Exception as e:
            logging.error(f"Error in profitability validation: {e}")
            return signal, reason
    
    # Helper methods for detailed analysis
    
    def _calculate_market_opportunity_score(self, market_intel: Dict) -> float:
        """Calculate market opportunity score (0-100)"""
        try:
            intelligence_score = market_intel.get('intelligence_score', 0.5)
            pattern_confidence = market_intel.get('pattern_analysis', {}).get('pattern_confidence', 0.5)
            regime_confidence = market_intel.get('market_regime', {}).get('confidence', 0.5)
            
            # Weight different factors
            opportunity_score = (intelligence_score * 40 + pattern_confidence * 35 + regime_confidence * 25) * 100
            
            return min(100, max(0, opportunity_score))
            
        except Exception:
            return 50.0
    
    def _calculate_ml_confidence_score(self, buy_prob: float, sell_prob: float, 
                                     regime_pred: Dict, market_analysis: Dict) -> float:
        """Calculate overall ML confidence score (more generous for frequent trading)"""
        try:
            # Signal probabilities confidence (boosted)
            signal_confidence = max(abs(buy_prob - 0.5), abs(sell_prob - 0.5)) * 2.5  # Increased multiplier
            
            # Regime prediction confidence (boosted)
            regime_confidence = min(0.9, regime_pred.get('confidence', 0.5) * 1.2)  # 20% boost
            
            # Market intelligence confidence (boosted)
            market_confidence = min(0.9, market_analysis.get('intelligence_score', 0.5) * 1.3)  # 30% boost
            
            # Weighted average with base boost for aggressive trading
            base_boost = 0.15  # 15% base confidence boost
            overall_confidence = (signal_confidence * 0.4 + regime_confidence * 0.3 + market_confidence * 0.3) + base_boost
            
            return min(0.95, max(0.2, overall_confidence))
            
        except Exception:
            return 0.5
    
    def _predict_short_term_trend(self, df: pd.DataFrame, indicators: Dict) -> str:
        """Predict short-term trend direction"""
        try:
            # Use multiple indicators for trend prediction
            rsi = indicators.get('rsi', 50)
            macd_trend = indicators.get('macd_trend', 'NEUTRAL')
            current_price = indicators['current_price']
            
            # Simple trend scoring
            trend_score = 0
            
            if rsi < 40:
                trend_score += 1  # Oversold, potential upward trend
            elif rsi > 60:
                trend_score -= 1  # Overbought, potential downward trend
                
            if macd_trend == 'BULLISH':
                trend_score += 1
            elif macd_trend == 'BEARISH':
                trend_score -= 1
                
            # Price vs moving averages
            sma5 = indicators.get('sma5', current_price)
            sma20 = indicators.get('sma20', current_price)
            
            if current_price > sma5 > sma20:
                trend_score += 1
            elif current_price < sma5 < sma20:
                trend_score -= 1
                
            if trend_score > 0:
                return 'BULLISH'
            elif trend_score < 0:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'NEUTRAL'
    
    def _forecast_volatility(self, df: pd.DataFrame) -> float:
        """Forecast near-term volatility"""
        try:
            # Calculate recent volatility
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 10:
                recent_vol = returns.tail(10).std()
                historical_vol = returns.std()
                return min(2.0, recent_vol / historical_vol) if historical_vol > 0 else 1.0
            return 1.0
        except Exception:
            return 1.0
    
    def _calculate_momentum_score(self, df: pd.DataFrame, indicators: Dict) -> float:
        """Calculate momentum score (0-100)"""
        try:
            current_price = indicators['current_price']
            
            # Price momentum over different periods
            if len(df) >= 10:
                price_10 = df['close'].iloc[-10] if len(df) >= 10 else current_price
                momentum_10 = (current_price / price_10 - 1) * 100
            else:
                momentum_10 = 0
                
            # RSI momentum
            rsi = indicators.get('rsi', 50)
            rsi_momentum = (rsi - 50) / 50 * 100
            
            # MACD momentum
            macd_trend = indicators.get('macd_trend', 'NEUTRAL')
            macd_momentum = {'BULLISH': 20, 'BEARISH': -20, 'NEUTRAL': 0}[macd_trend]
            
            # Combined momentum score
            momentum_score = (momentum_10 * 0.4 + rsi_momentum * 0.3 + macd_momentum * 0.3)
            
            # Normalize to 0-100
            return max(0, min(100, 50 + momentum_score))
            
        except Exception:
            return 50.0
    
    def _calculate_volume_score(self, df: pd.DataFrame, indicators: Dict) -> float:
        """Calculate volume score (0-100)"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 50.0
                
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Score based on volume ratio
            if volume_ratio > 2.0:
                return 90.0  # Very high volume
            elif volume_ratio > 1.5:
                return 75.0  # High volume
            elif volume_ratio > 1.2:
                return 60.0  # Above average
            elif volume_ratio < 0.7:
                return 30.0  # Low volume
            else:
                return 50.0  # Normal volume
                
        except Exception:
            return 50.0
    
    def _analyze_volatility_timing(self, df: pd.DataFrame) -> float:
        """Analyze volatility for timing opportunities"""
        try:
            if len(df) < 20:
                return 50.0
                
            # Calculate volatility expansion/contraction
            short_vol = df['close'].pct_change().rolling(5).std().iloc[-1]
            long_vol = df['close'].pct_change().rolling(20).std().iloc[-1]
            
            vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
            
            # Volatility timing score
            if 0.8 <= vol_ratio <= 1.2:
                return 70.0  # Stable volatility - good for trading
            elif vol_ratio < 0.5:
                return 40.0  # Very low volatility - may be building energy
            elif vol_ratio > 2.0:
                return 30.0  # Very high volatility - risky
            else:
                return 50.0  # Normal volatility
                
        except Exception:
            return 50.0
    
    def _calculate_market_hour_score(self) -> float:
        """Calculate score based on current market hours"""
        try:
            current_hour = datetime.utcnow().hour
            
            # High activity hours get higher scores
            if current_hour in config.MARKET_HOURS.get('high_activity_hours', []):
                return 80.0
            elif current_hour in config.MARKET_HOURS.get('us_market', []):
                return 70.0
            elif current_hour in config.MARKET_HOURS.get('european_market', []):
                return 60.0
            else:
                return 40.0  # Low activity hours
                
        except Exception:
            return 50.0
    
    def _analyze_support_resistance_proximity(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Analyze proximity to support/resistance levels"""
        try:
            if len(df) < 50:
                return {'timing_score': 50.0, 'near_support': False, 'near_resistance': False}
                
            # Calculate recent highs and lows
            recent_data = df.tail(50)
            
            # Support levels (recent lows)
            support_levels = []
            for i in range(2, len(recent_data) - 2):
                if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and 
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                    support_levels.append(recent_data['low'].iloc[i])
            
            # Resistance levels (recent highs)
            resistance_levels = []
            for i in range(2, len(recent_data) - 2):
                if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and 
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                    resistance_levels.append(recent_data['high'].iloc[i])
            
            # Find nearest levels
            nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            
            # Calculate proximity scores
            support_distance = (current_price - nearest_support) / current_price
            resistance_distance = (nearest_resistance - current_price) / current_price
            
            near_support = support_distance < 0.02  # Within 2%
            near_resistance = resistance_distance < 0.02  # Within 2%
            
            # Timing score based on proximity
            if near_support:
                timing_score = 80.0  # Good for buying near support
            elif near_resistance:
                timing_score = 75.0  # Good for selling near resistance
            else:
                timing_score = 50.0  # No significant levels nearby
                
            return {
                'timing_score': timing_score,
                'near_support': near_support,
                'near_resistance': near_resistance,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance
            }
            
        except Exception:
            return {'timing_score': 50.0, 'near_support': False, 'near_resistance': False}
    
    def _calculate_price_range(self, df: pd.DataFrame, periods: int) -> Dict:
        """Calculate price range for given number of periods"""
        try:
            if len(df) < periods:
                periods = len(df)
                
            recent_data = df.tail(periods)
            high_price = recent_data['high'].max() if 'high' in recent_data.columns else recent_data['close'].max()
            low_price = recent_data['low'].min() if 'low' in recent_data.columns else recent_data['close'].min()
            
            return {
                'high': high_price,
                'low': low_price,
                'range': high_price - low_price,
                'periods': periods
            }
            
        except Exception:
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0
            return {'high': current_price, 'low': current_price, 'range': 0, 'periods': 0}
    
    def _calculate_position_confidence(self, position_scores: Dict, ranges: Dict) -> float:
        """Calculate confidence in position analysis"""
        try:
            # Check consistency across timeframes
            scores = list(position_scores.values())
            consistency = 1.0 - np.std(scores) if len(scores) > 1 else 1.0
            
            # Check range validity
            range_validity = sum(1 for r in ranges.values() if r['range'] > 0) / len(ranges)
            
            # Combined confidence
            confidence = (consistency * 0.6 + range_validity * 0.4)
            
            return max(0.1, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _calculate_buy_score(self, buy_prob: float, ml_confidence: float, timing_score: float,
                           buy_opportunity: bool, market_opportunity: float, price_position: Dict,
                           indicators: Dict) -> float:
        """Calculate composite buy score"""
        try:
            # Base score from ML prediction (scale to 0-80 instead of 0-100)
            base_score = buy_prob * 80
            
            # ML confidence multiplier (reduced range)
            confidence_multiplier = 0.7 + (ml_confidence * 0.8)  # 0.7x to 1.5x
            
            # Timing bonus (reduced)
            timing_bonus = (timing_score - 50) * 0.2  # -10 to +10 points
            
            # Buy low opportunity bonus (reduced)
            buy_low_bonus = price_position.get('buy_low_score', 0) * 0.3  # Up to +30 points
            
            # Market opportunity factor (reduced range)
            market_factor = 0.8 + (market_opportunity / 100)  # 0.8x to 1.8x multiplier
            
            # RSI oversold bonus (reduced)
            rsi = indicators.get('rsi', 50)
            rsi_bonus = max(0, (35 - rsi) * 0.5) if rsi < 35 else 0  # Bonus for oversold
            
            # Calculate final score
            buy_score = (base_score * confidence_multiplier * market_factor + 
                        timing_bonus + buy_low_bonus + rsi_bonus)
            
            return max(0, min(100, buy_score))
            
        except Exception:
            return 50.0
    
    def _calculate_sell_score(self, sell_prob: float, ml_confidence: float, timing_score: float,
                            sell_opportunity: bool, market_opportunity: float, price_position: Dict,
                            indicators: Dict) -> float:
        """Calculate composite sell score"""
        try:
            # Base score from ML prediction (scale to 0-80 instead of 0-100)
            base_score = sell_prob * 80
            
            # ML confidence multiplier (reduced range)
            confidence_multiplier = 0.7 + (ml_confidence * 0.8)  # 0.7x to 1.5x
            
            # Timing bonus (reduced)
            timing_bonus = (timing_score - 50) * 0.2  # -10 to +10 points
            
            # Sell high opportunity bonus (reduced)
            sell_high_bonus = price_position.get('sell_high_score', 0) * 0.3  # Up to +30 points
            
            # Market opportunity factor (reduced range)
            market_factor = 0.8 + (market_opportunity / 100)  # 0.8x to 1.8x multiplier
            
            # RSI overbought bonus (reduced)
            rsi = indicators.get('rsi', 50)
            rsi_bonus = max(0, (rsi - 65) * 0.5) if rsi > 65 else 0  # Bonus for overbought
            
            # Calculate final score
            sell_score = (base_score * confidence_multiplier * market_factor + 
                         timing_bonus + sell_high_bonus + rsi_bonus)
            
            return max(0, min(100, sell_score))
            
        except Exception:
            return 50.0
    
    def _calculate_position_size_multiplier(self, confidence: float, regime: str) -> float:
        """Calculate position size multiplier based on confidence and market regime"""
        try:
            # Base multiplier from confidence
            confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x to 2.0x
            
            # Regime adjustment
            regime_multiplier = self.regime_adjustments.get(regime, {}).get('position_multiplier', 1.0)
            
            # Combined multiplier
            final_multiplier = confidence_multiplier * regime_multiplier
            
            # Cap the multiplier
            return max(0.2, min(self.max_position_size_multiplier, final_multiplier))
            
        except Exception:
            return 1.0
    
    def _update_strategy_metrics(self, signal: str, analysis_data: Dict):
        """Update strategy performance metrics"""
        try:
            # Store signal in history for performance tracking
            self.trade_history.append({
                'timestamp': datetime.now(),
                'signal': signal,
                'confidence': analysis_data.get('signal_confidence', 0.5),
                'market_regime': analysis_data.get('market_regime', 'NORMAL'),
                'timing_score': analysis_data.get('timing_score', 50)
            })
            
            # Keep only recent history
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
                
        except Exception as e:
            logging.error(f"Error updating strategy metrics: {e}")
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and current status"""
        return {
            'name': self.name,
            'description': self.description,
            'ml_modules_available': ML_MODULES_AVAILABLE,
            'current_regime': self.current_regime,
            'performance_metrics': self.performance_metrics,
            'trade_history_count': len(self.trade_history),
            'confidence_thresholds': {
                'minimum': self.min_confidence_threshold,
                'high_confidence': self.high_confidence_threshold
            },
            'status': 'ACTIVE' if ML_MODULES_AVAILABLE else 'DISABLED'
        }


# Factory function for easy integration
def get_ml_trading_strategy():
    """Factory function to get ML Trading Strategy instance"""
    return MLTradingStrategy()


# Integration function for web_bot.py
def ml_pure_strategy(df: pd.DataFrame, symbol: str, indicators: Dict) -> Tuple[str, str]:
    """
    ML Pure Strategy function for integration with existing signal_generator
    
    This function provides a clean interface for the existing bot to use the new ML strategy
    """
    if not ML_MODULES_AVAILABLE:
        return "HOLD", "ML modules not available for pure ML strategy"
    
    try:
        # Initialize strategy (cached instance could be used for better performance)
        ml_strategy = get_ml_trading_strategy()
        
        # Generate signal using ML strategy
        signal, reason, analysis_data = ml_strategy.generate_signal(df, symbol, indicators)
        
        # Log additional ML insights for debugging
        if 'signal_confidence' in analysis_data:
            confidence = analysis_data['signal_confidence']
            regime = analysis_data.get('market_regime', 'UNKNOWN')
            logging.info(f"ML Strategy: {signal} | Confidence: {confidence:.2f} | Regime: {regime}")
        
        return signal, reason
        
    except Exception as e:
        logging.error(f"Error in ML Pure Strategy: {e}")
        return "HOLD", f"ML Strategy Error: {e}"
