import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os

# Import centralized model paths
from model_paths import MODEL_PATHS

# Minimal import guard to avoid crashing if sklearn is missing
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.cluster import KMeans
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    RandomForestRegressor = None
    StandardScaler = None
    LabelEncoder = None
    joblib = None

class EnhancedMLPredictor:
    """
    Enhanced ML Predictor with Pattern Recognition, Market Regime Prediction, and Adaptive Thresholds
    """
    
    def __init__(self, model_path=None):
        self.trend_model = None
        self.regime_model = None
        self.pattern_model = None
        self.scaler = None
        self.regime_scaler = None
        self.pattern_scaler = None
        
        # Model paths using centralized configuration
        self.trend_model_path = MODEL_PATHS['trend_model'] if model_path is None else model_path
        self.regime_model_path = MODEL_PATHS['regime_model']
        self.pattern_model_path = MODEL_PATHS['pattern_model']
        self.scaler_path = MODEL_PATHS['trend_scaler']
        self.regime_scaler_path = MODEL_PATHS['regime_scaler']
        self.pattern_scaler_path = MODEL_PATHS['pattern_scaler']
        
        # Pattern recognition data
        self.signal_history = []
        self.market_patterns = {}
        self.adaptive_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_threshold': 0.001
        }
        
        # Market regime categories
        self.regime_labels = ['QUIET', 'NORMAL', 'VOLATILE', 'EXTREME']
        
        self._load_models()
        self._initialize_pattern_recognition()

    def _load_models(self):
        """Load all ML models"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            # Load trend prediction model
            if os.path.exists(self.trend_model_path):
                self.trend_model = joblib.load(self.trend_model_path)
                
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                
            # Load market regime model
            if os.path.exists(self.regime_model_path):
                self.regime_model = joblib.load(self.regime_model_path)
                
            if os.path.exists(self.regime_scaler_path):
                self.regime_scaler = joblib.load(self.regime_scaler_path)
                
            # Load pattern recognition model
            if os.path.exists(self.pattern_model_path):
                self.pattern_model = joblib.load(self.pattern_model_path)
                
            if os.path.exists(self.pattern_scaler_path):
                self.pattern_scaler = joblib.load(self.pattern_scaler_path)
                
        except Exception as e:
            print(f"Warning: Could not load ML models: {e}")
            
    def _initialize_pattern_recognition(self):
        """Initialize pattern recognition system"""
        self.pattern_features = [
            'rsi_pattern', 'macd_pattern', 'volume_pattern', 'volatility_pattern',
            'trend_strength', 'momentum_pattern', 'support_resistance_pattern'
        ]
        
        # Load historical patterns if available
        pattern_file = MODEL_PATHS['market_patterns']
        if os.path.exists(pattern_file):
            try:
                with open(pattern_file, 'rb') as f:
                    self.market_patterns = pickle.load(f)
            except Exception:
                self.market_patterns = {}

    def train_market_regime_model(self, df, regime_col='market_regime'):
        """Train market regime prediction model"""
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn not available for regime prediction")
            return 0.0
            
        try:
            # Prepare features for regime prediction
            regime_features = self._extract_regime_features(df)
            
            if regime_col not in df.columns:
                # Create regime labels based on volatility
                df = self._create_regime_labels(df)
                
            X = regime_features.values
            y = df[regime_col].values
            
            # Remove any NaN values
            mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                print("Insufficient data for regime model training")
                return 0.0
            
            # Scale features
            self.regime_scaler = StandardScaler()
            X_scaled = self.regime_scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.regime_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            )
            self.regime_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.regime_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save models
            joblib.dump(self.regime_model, self.regime_model_path)
            joblib.dump(self.regime_scaler, self.regime_scaler_path)
            
            print(f"Market Regime Model trained with accuracy: {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"Error training regime model: {e}")
            return 0.0

    def train_pattern_recognition_model(self, signal_data):
        """Train pattern recognition model for signal success prediction"""
        if not SKLEARN_AVAILABLE or len(signal_data) < 50:
            return 0.0
            
        try:
            # Convert signal data to DataFrame if needed
            if isinstance(signal_data, list):
                df = pd.DataFrame(signal_data)
            else:
                df = signal_data.copy()
                
            # Extract pattern features
            pattern_features = self._extract_pattern_features(df)
            
            # Create success labels (assuming we have outcome data)
            if 'success' not in df.columns:
                # Create synthetic success labels based on profit_loss
                df['success'] = (df.get('profit_loss', 0) > 0).astype(int)
            
            X = pattern_features.values
            y = df['success'].values
            
            # Remove NaN values
            mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 20:
                print("Insufficient data for pattern recognition training")
                return 0.0
            
            # Scale features
            self.pattern_scaler = StandardScaler()
            X_scaled = self.pattern_scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.pattern_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            self.pattern_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.pattern_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            joblib.dump(self.pattern_model, self.pattern_model_path)
            joblib.dump(self.pattern_scaler, self.pattern_scaler_path)
            
            print(f"Pattern Recognition Model trained with accuracy: {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"Error training pattern model: {e}")
            return 0.0

    def predict_market_regime(self, df):
        """Predict market regime using ML model with smart feature alignment"""
        if not SKLEARN_AVAILABLE or self.regime_model is None or self.regime_scaler is None:
            return self._fallback_regime_prediction(df)
            
        try:
            # Extract features using smart alignment
            regime_features = self._extract_regime_features_smart(df)
            
            if regime_features is None or len(regime_features) == 0:
                return self._fallback_regime_prediction(df)
            
            # Use only the latest data point
            X = regime_features.iloc[-1:].values
            
            # Handle feature dimension mismatch gracefully
            try:
                # Scale features
                X_scaled = self.regime_scaler.transform(X)
                
                # Predict
                regime_pred = self.regime_model.predict(X_scaled)[0]
                regime_proba = self.regime_model.predict_proba(X_scaled)[0]
                
                # Get confidence
                confidence = max(regime_proba)
                
                return {
                    'regime': regime_pred,
                    'confidence': confidence,
                    'probabilities': {
                        label: prob for label, prob in zip(getattr(self, 'regime_labels', ['NORMAL']), regime_proba)
                    }
                }
                
            except ValueError as ve:
                if "features" in str(ve).lower():
                    # Feature mismatch - use smart feature alignment
                    aligned_features = self._align_regime_features(X)
                    if aligned_features is not None:
                        X_scaled = self.regime_scaler.transform([aligned_features])
                        regime_pred = self.regime_model.predict(X_scaled)[0]
                        regime_proba = self.regime_model.predict_proba(X_scaled)[0]
                        confidence = max(regime_proba)
                        
                        return {
                            'regime': regime_pred,
                            'confidence': confidence,
                            'probabilities': {
                                label: prob for label, prob in zip(getattr(self, 'regime_labels', ['NORMAL']), regime_proba)
                            }
                        }
                    else:
                        return self._fallback_regime_prediction(df)
                else:
                    raise ve
            
        except Exception as e:
            print(f"Error in regime prediction: {e}")
            return self._fallback_regime_prediction(df)

    def _extract_regime_features_smart(self, df):
        """Extract regime features with smart model alignment"""
        try:
            # Get expected feature count from the trained model
            expected_features = getattr(self.regime_scaler, 'n_features_in_', 4)
            
            # Create core features that are most important for regime detection
            features = pd.DataFrame()
            
            # Feature 1: Price volatility (most important)
            features['volatility'] = df['close'].pct_change().rolling(20).std()
            
            # Feature 2: Volume surge (second most important) 
            if 'volume' in df.columns:
                features['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()
            else:
                features['volume_surge'] = 1.0
            
            # Feature 3: Price change magnitude (third most important)
            features['price_change'] = df['close'].pct_change().abs()
            
            # Feature 4: Trend strength (fourth most important)
            if 'sma5' in df.columns and 'sma20' in df.columns:
                features['trend_strength'] = abs(df['sma5'] - df['sma20']) / df['sma20']
            else:
                # Calculate simple trend from price
                sma5 = df['close'].rolling(5).mean()
                sma20 = df['close'].rolling(20).mean()
                features['trend_strength'] = abs(sma5 - sma20) / sma20
            
            # If model expects more features, add supplementary ones
            if expected_features > 4:
                additional_features = [
                    df['close'].pct_change(periods=4).abs(),  # 4h price change
                    df['close'].pct_change(periods=24).abs() if len(df) > 24 else df['close'].pct_change().abs(),  # 24h price change
                    df.get('rsi', pd.Series([50] * len(df))).rolling(10).std() / 10,  # RSI volatility normalized
                    (df.get('high', df['close']) - df.get('low', df['close'])) / df['close']  # Price range
                ]
                
                for i, add_feature in enumerate(additional_features):
                    if len(features.columns) < expected_features:
                        features[f'feature_{i+5}'] = add_feature
            
            # Ensure we have exactly the right number of features
            while len(features.columns) < expected_features:
                features[f'default_{len(features.columns)}'] = 0.5
                
            # Truncate if we have too many
            if len(features.columns) > expected_features:
                features = features.iloc[:, :expected_features]
            
            return features.ffill().fillna(0.5)
            
        except Exception as e:
            print(f"Error in smart regime feature extraction: {e}")
            return None

    def _align_regime_features(self, features_array):
        """Align features to match trained model expectations"""
        try:
            expected_features = getattr(self.regime_scaler, 'n_features_in_', 4)
            
            if len(features_array) == 0:
                return None
                
            # Take first row if it's 2D
            if len(features_array.shape) > 1:
                features = features_array[0]
            else:
                features = features_array
            
            # Align to expected feature count
            if len(features) > expected_features:
                # Use most important features (first ones)
                aligned = features[:expected_features]
            elif len(features) < expected_features:
                # Pad with neutral values
                aligned = list(features) + [0.5] * (expected_features - len(features))
            else:
                aligned = features
                
            return aligned
            
        except Exception as e:
            print(f"Error aligning regime features: {e}")
            return None

    def predict_signal_success(self, signal_data, current_indicators):
        """Predict success probability of a trading signal with robust error handling"""
        if not SKLEARN_AVAILABLE or self.pattern_model is None or self.pattern_scaler is None:
            return self._fallback_signal_prediction(current_indicators, signal_data.get('action', 'BUY'))
            
        try:
            # Extract signal action
            action = signal_data.get('action', 'BUY')
            
            # Create feature vector from current indicators with standardized features
            features = self._create_pattern_vector_robust(current_indicators, action)
            
            if features is None:
                return self._fallback_signal_prediction(current_indicators, action)
                
            # Handle feature dimension mismatch gracefully
            try:
                # Scale features
                X_scaled = self.pattern_scaler.transform([features])
                
                # Predict success probability
                success_proba = self.pattern_model.predict_proba(X_scaled)[0]
                
                # Get base probability (class 1)
                base_probability = success_proba[1] if len(success_proba) > 1 else 0.5
                
                # Apply action-specific adjustments to differentiate BUY vs SELL
                adjusted_probability = self._adjust_probability_for_action(
                    base_probability, action, current_indicators
                )
                
                return adjusted_probability
                
            except ValueError as ve:
                if "features" in str(ve).lower():
                    # Feature mismatch - use fallback prediction
                    return self._fallback_signal_prediction(current_indicators, action)
                else:
                    raise ve
            
        except Exception as e:
            print(f"Error predicting signal success: {e}")
            return self._fallback_signal_prediction(current_indicators, signal_data.get('action', 'BUY'))

    def _create_pattern_vector_robust(self, indicators, action='BUY'):
        """Create robust feature vector that handles missing indicators gracefully"""
        try:
            # Core features that are always available
            features = [
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('volatility', 0.02),
                indicators.get('volume_ratio', 1.0)
            ]
            
            # Add action-specific feature (1 for BUY, 0 for SELL)
            action_feature = 1.0 if action == 'BUY' else 0.0
            features.append(action_feature)
            
            # Try to match the expected feature count of the trained model
            # This handles the case where models were trained with different feature sets
            expected_features = getattr(self.pattern_scaler, 'n_features_in_', 5)
            
            if expected_features > len(features):
                # Add additional features to match expected count
                additional_features = [
                    indicators.get('current_price', 0) / max(indicators.get('sma20', 1), 1),
                    indicators.get('adx', 25),
                    indicators.get('stoch_k', 50),
                    indicators.get('bb_position', 0.5),  # Bollinger band position
                    indicators.get('momentum', 0),       # Price momentum
                    indicators.get('trend_strength', 0.5) # Trend strength
                ]
                
                # Add only as many as needed
                features.extend(additional_features[:expected_features - len(features)])
            
            # Ensure we have exactly the right number of features
            while len(features) < expected_features:
                features.append(0.5)  # Neutral default values
                
            # Truncate if we have too many
            features = features[:expected_features]
            
            return features
            
        except Exception:
            return None

    def _adjust_probability_for_action(self, base_probability, action, indicators):
        """Adjust probability based on action and market conditions"""
        try:
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            
            if action == 'BUY':
                # For BUY signals, increase probability when oversold
                if rsi < 30:
                    base_probability *= 1.2  # Boost for oversold conditions
                elif rsi > 70:
                    base_probability *= 0.8  # Reduce for overbought conditions
                    
                # Boost for bullish MACD
                if macd > 0:
                    base_probability *= 1.1
                    
            elif action == 'SELL':
                # For SELL signals, increase probability when overbought
                if rsi > 70:
                    base_probability *= 1.2  # Boost for overbought conditions
                elif rsi < 30:
                    base_probability *= 0.8  # Reduce for oversold conditions
                    
                # Boost for bearish MACD
                if macd < 0:
                    base_probability *= 1.1
            
            # Ensure probability stays within bounds
            return max(0.1, min(0.9, base_probability))
            
        except Exception:
            return base_probability

    def _fallback_signal_prediction(self, indicators, action='BUY'):
        """Fallback signal prediction when ML models fail"""
        try:
            # Simple rule-based prediction that considers action
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            
            if action == 'BUY':
                # BUY signal success probability
                if rsi < 30 and macd > 0:  # Oversold with bullish MACD
                    return 0.75
                elif rsi < 40 and macd > 0:  # Slightly oversold with bullish MACD
                    return 0.65
                elif 40 < rsi < 60:  # Neutral RSI
                    return 0.5
                else:
                    return 0.35  # Less favorable conditions for buying
                    
            elif action == 'SELL':
                # SELL signal success probability
                if rsi > 70 and macd < 0:  # Overbought with bearish MACD
                    return 0.75
                elif rsi > 60 and macd < 0:  # Slightly overbought with bearish MACD
                    return 0.65
                elif 40 < rsi < 60:  # Neutral RSI
                    return 0.5
                else:
                    return 0.35  # Less favorable conditions for selling
            
            return 0.5  # Default neutral
                
        except Exception:
            return 0.5

    def calculate_adaptive_thresholds(self, df, market_regime='NORMAL'):
        """Calculate adaptive RSI/MACD thresholds based on market conditions"""
        try:
            # Base thresholds
            base_rsi_oversold = 30
            base_rsi_overbought = 70
            base_macd_threshold = 0.001
            
            # Calculate recent volatility
            if 'volatility' in df.columns:
                recent_vol = df['volatility'].tail(20).mean()
            else:
                recent_vol = df['close'].pct_change().rolling(20).std().iloc[-1]
            
            # Calculate market momentum
            if len(df) >= 10:
                momentum = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100
            else:
                momentum = 0
                
            # Regime-based adjustments
            regime_adjustments = {
                'QUIET': {'rsi_range': 5, 'macd_mult': 0.5},
                'NORMAL': {'rsi_range': 0, 'macd_mult': 1.0},
                'VOLATILE': {'rsi_range': -5, 'macd_mult': 1.5},
                'EXTREME': {'rsi_range': -10, 'macd_mult': 2.0}
            }
            
            adj = regime_adjustments.get(market_regime, regime_adjustments['NORMAL'])
            
            # Volatility adjustments
            vol_factor = max(0.5, min(2.0, recent_vol / 0.02))  # Normalize around 2% daily vol
            
            # Calculate adaptive thresholds
            adaptive_thresholds = {
                'rsi_oversold': max(15, min(40, base_rsi_oversold + adj['rsi_range'] - (vol_factor - 1) * 5)),
                'rsi_overbought': max(60, min(85, base_rsi_overbought - adj['rsi_range'] + (vol_factor - 1) * 5)),
                'macd_threshold': base_macd_threshold * adj['macd_mult'] * vol_factor,
                'volatility_factor': vol_factor,
                'momentum_factor': momentum / 10  # Scale momentum
            }
            
            # Store for future use
            self.adaptive_thresholds = adaptive_thresholds
            
            return adaptive_thresholds
            
        except Exception as e:
            print(f"Error calculating adaptive thresholds: {e}")
            return self.adaptive_thresholds

    def analyze_historical_patterns(self, signal_history):
        """Analyze historical signal success patterns"""
        try:
            if len(signal_history) < 10:
                return {}
                
            df = pd.DataFrame(signal_history)
            
            patterns = {
                'overall_success_rate': 0.0,
                'rsi_patterns': {},
                'macd_patterns': {},
                'volume_patterns': {},
                'time_patterns': {},
                'market_condition_patterns': {}
            }
            
            # Overall success rate
            if 'success' in df.columns:
                patterns['overall_success_rate'] = df['success'].mean()
            elif 'profit_loss' in df.columns:
                patterns['overall_success_rate'] = (df['profit_loss'] > 0).mean()
            
            # RSI pattern analysis
            if 'rsi' in df.columns:
                patterns['rsi_patterns'] = self._analyze_rsi_patterns(df)
            
            # MACD pattern analysis
            if 'macd_trend' in df.columns:
                patterns['macd_patterns'] = self._analyze_macd_patterns(df)
            
            # Volume pattern analysis
            if 'volume_ratio' in df.columns:
                patterns['volume_patterns'] = self._analyze_volume_patterns(df)
            
            # Time-based patterns
            if 'timestamp' in df.columns:
                patterns['time_patterns'] = self._analyze_time_patterns(df)
            
            # Store patterns
            self.market_patterns = patterns
            
            # Save patterns to file
            with open(MODEL_PATHS['market_patterns'], 'wb') as f:
                pickle.dump(patterns, f)
                
            return patterns
            
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return {}

    def _extract_regime_features(self, df):
        """Extract features for market regime prediction"""
        features = pd.DataFrame()
        
        try:
            # Volatility features
            features['volatility'] = df['close'].pct_change().rolling(20).std()
            features['volatility_ma'] = features['volatility'].rolling(5).mean()
            
            # Volume features
            if 'volume' in df.columns:
                features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
                features['volume_trend'] = features['volume_ratio'].rolling(5).mean()
            else:
                features['volume_ratio'] = 1.0
                features['volume_trend'] = 1.0
            
            # Price movement features
            features['price_change_1h'] = df['close'].pct_change(periods=1).abs()
            features['price_change_4h'] = df['close'].pct_change(periods=4).abs()
            features['price_change_24h'] = df['close'].pct_change(periods=24).abs()
            
            # Technical indicator volatility
            if 'rsi' in df.columns:
                features['rsi_volatility'] = df['rsi'].rolling(10).std()
            else:
                features['rsi_volatility'] = 5.0
                
            # Trend strength
            if 'sma5' in df.columns and 'sma20' in df.columns:
                features['trend_strength'] = abs(df['sma5'] - df['sma20']) / df['sma20']
            else:
                features['trend_strength'] = 0.01
                
            # ATR-based volatility
            if 'atr' in df.columns:
                features['atr_ratio'] = df['atr'] / df['close']
            else:
                high_low = df.get('high', df['close']) - df.get('low', df['close'])
                features['atr_ratio'] = high_low.rolling(14).mean() / df['close']
            
            return features.ffill().fillna(0)
            
        except Exception as e:
            print(f"Error extracting regime features: {e}")
            return pd.DataFrame()

    def _extract_pattern_features(self, df):
        """Extract features for pattern recognition"""
        features = pd.DataFrame()
        
        try:
            # RSI patterns
            if 'rsi' in df.columns:
                features['rsi_level'] = df['rsi']
                features['rsi_momentum'] = df['rsi'].diff()
                features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
                features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            
            # MACD patterns
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                features['macd_histogram'] = df['macd'] - df['macd_signal']
                features['macd_cross_up'] = (features['macd_histogram'] > 0).astype(int)
                features['macd_momentum'] = features['macd_histogram'].diff()
            
            # Volume patterns
            if 'volume' in df.columns:
                features['volume_spike'] = (df['volume'] > df['volume'].rolling(10).mean() * 1.5).astype(int)
                features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
            
            # Price patterns
            features['price_momentum'] = df['close'].pct_change(5)
            features['volatility'] = df['close'].pct_change().rolling(10).std()
            
            # Support/Resistance patterns
            features['near_support'] = self._detect_support_resistance(df, 'support')
            features['near_resistance'] = self._detect_support_resistance(df, 'resistance')
            
            return features.fillna(0)
            
        except Exception as e:
            print(f"Error extracting pattern features: {e}")
            return pd.DataFrame()

    def _create_regime_labels(self, df):
        """Create market regime labels based on volatility"""
        try:
            # Calculate volatility
            volatility = df['close'].pct_change().rolling(20).std()
            
            # Define regime thresholds (could be learned from data)
            vol_25 = volatility.quantile(0.25)
            vol_50 = volatility.quantile(0.50)
            vol_75 = volatility.quantile(0.75)
            
            conditions = [
                volatility <= vol_25,
                (volatility > vol_25) & (volatility <= vol_50),
                (volatility > vol_50) & (volatility <= vol_75),
                volatility > vol_75
            ]
            
            df['market_regime'] = np.select(conditions, self.regime_labels, default='NORMAL')
            
            return df
            
        except Exception as e:
            print(f"Error creating regime labels: {e}")
            df['market_regime'] = 'NORMAL'
            return df

    def _fallback_regime_prediction(self, df):
        """Fallback regime prediction without ML"""
        try:
            if len(df) < 20:
                return {'regime': 'NORMAL', 'confidence': 0.5}
                
            # Simple volatility-based regime detection
            recent_vol = df['close'].pct_change().tail(20).std()
            
            if recent_vol > 0.05:
                regime = 'EXTREME'
            elif recent_vol > 0.03:
                regime = 'VOLATILE'
            elif recent_vol < 0.01:
                regime = 'QUIET'
            else:
                regime = 'NORMAL'
                
            return {'regime': regime, 'confidence': 0.6}
            
        except Exception:
            return {'regime': 'NORMAL', 'confidence': 0.5}

    def _create_pattern_vector(self, indicators):
        """Create feature vector from current indicators"""
        try:
            features = [
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('volatility', 0.02),
                indicators.get('volume_ratio', 1.0),
                indicators.get('current_price', 0) / max(indicators.get('sma20', 1), 1),
                indicators.get('adx', 25),
                indicators.get('stoch_k', 50)
            ]
            
            return features
            
        except Exception:
            return None

    def _analyze_rsi_patterns(self, df):
        """Analyze RSI-based success patterns"""
        patterns = {}
        
        try:
            if 'rsi' in df.columns and 'success' in df.columns:
                # RSI level analysis
                rsi_bins = [0, 20, 30, 40, 60, 70, 80, 100]
                df['rsi_bin'] = pd.cut(df['rsi'], bins=rsi_bins, labels=[
                    'very_oversold', 'oversold', 'low', 'neutral', 'high', 'overbought', 'very_overbought'
                ])
                
                patterns = df.groupby('rsi_bin')['success'].agg(['mean', 'count']).to_dict()
                
        except Exception as e:
            print(f"Error analyzing RSI patterns: {e}")
            
        return patterns

    def _analyze_macd_patterns(self, df):
        """Analyze MACD-based success patterns"""
        patterns = {}
        
        try:
            if 'macd_trend' in df.columns and 'success' in df.columns:
                patterns = df.groupby('macd_trend')['success'].agg(['mean', 'count']).to_dict()
                
        except Exception as e:
            print(f"Error analyzing MACD patterns: {e}")
            
        return patterns

    def _analyze_volume_patterns(self, df):
        """Analyze volume-based success patterns"""
        patterns = {}
        
        try:
            if 'volume_ratio' in df.columns and 'success' in df.columns:
                # Volume level analysis
                volume_bins = [0, 0.8, 1.2, 1.5, 2.0, float('inf')]
                df['volume_bin'] = pd.cut(df['volume_ratio'], bins=volume_bins, labels=[
                    'low', 'normal', 'high', 'spike', 'extreme'
                ])
                
                patterns = df.groupby('volume_bin')['success'].agg(['mean', 'count']).to_dict()
                
        except Exception as e:
            print(f"Error analyzing volume patterns: {e}")
            
        return patterns

    def _analyze_time_patterns(self, df):
        """Analyze time-based success patterns"""
        patterns = {}
        
        try:
            if 'timestamp' in df.columns and 'success' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                patterns = df.groupby('hour')['success'].agg(['mean', 'count']).to_dict()
                
        except Exception as e:
            print(f"Error analyzing time patterns: {e}")
            
        return patterns

    def _detect_support_resistance(self, df, level_type='support'):
        """Simple support/resistance detection"""
        try:
            if len(df) < 20:
                return 0
                
            current_price = df['close'].iloc[-1]
            recent_prices = df['close'].tail(20)
            
            if level_type == 'support':
                support_level = recent_prices.min()
                return 1 if current_price <= support_level * 1.02 else 0
            else:  # resistance
                resistance_level = recent_prices.max()
                return 1 if current_price >= resistance_level * 0.98 else 0
                
        except Exception:
            return 0

    # Legacy methods for compatibility
    def train(self, df, feature_cols, target_col):
        """Legacy train method for backward compatibility"""
        if not SKLEARN_AVAILABLE:
            return 0.0
            
        try:
            X = df[feature_cols].values
            y = df[target_col].values
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            self.trend_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.trend_model.fit(X_train, y_train)
            joblib.dump(self.trend_model, self.trend_model_path)
            joblib.dump(self.scaler, self.scaler_path)
            return self.trend_model.score(X_test, y_test)
        except Exception:
            return 0.0

    def predict(self, df, feature_cols):
        """Legacy predict method for backward compatibility"""
        if not SKLEARN_AVAILABLE or self.trend_model is None or self.scaler is None:
            return None
        try:
            X = df[feature_cols].values
            X_scaled = self.scaler.transform(X)
            return self.trend_model.predict(X_scaled)
        except Exception:
            return None

    def predict_proba(self, df, feature_cols):
        """Legacy predict_proba method for backward compatibility"""
        if not SKLEARN_AVAILABLE or self.trend_model is None or self.scaler is None:
            return None
        try:
            X = df[feature_cols].values
            X_scaled = self.scaler.transform(X)
            return self.trend_model.predict_proba(X_scaled)
        except Exception:
            return None

# Legacy class for backward compatibility
class PriceTrendPredictor(EnhancedMLPredictor):
    """Legacy class name for backward compatibility"""
    pass
