"""
Enhanced ML Training Module for CRYPTIX Trading Bot
Handles training with comprehensive historical data from Binance
Supports both batch and incremental/cumulative learning
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import centralized model paths
from model_paths import MODEL_PATHS

# Import incremental learning system
try:
    from incremental_learning import IncrementalMLTrainer, HybridTrainingManager
    INCREMENTAL_AVAILABLE = True
except ImportError:
    INCREMENTAL_AVAILABLE = False
    print("⚠️ Incremental learning not available")

# Safe ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available. ML training will be limited.")

# Import our enhanced data fetcher
try:
    from enhanced_historical_data import EnhancedHistoricalDataFetcher
    ENHANCED_DATA_AVAILABLE = True
except ImportError:
    ENHANCED_DATA_AVAILABLE = False
    print("⚠️ Enhanced data fetcher not available")

# Import data cleaner
try:
    from data_cleaner import DataCleaner
    DATA_CLEANER_AVAILABLE = True
except ImportError:
    DATA_CLEANER_AVAILABLE = False
    print("⚠️ Data cleaner not available")

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """
    Enhanced ML Training Manager using comprehensive historical data
    Supports both batch and incremental/cumulative learning
    """
    
    def __init__(self, use_incremental: bool = True):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.training_history = []
        self.model_performance = {}
        self.use_incremental = use_incremental and INCREMENTAL_AVAILABLE
        
        # Initialize incremental learning system
        if self.use_incremental:
            self.incremental_trainer = IncrementalMLTrainer()
            self.hybrid_manager = HybridTrainingManager()
            logger.info("✅ Incremental learning system enabled")
        else:
            self.incremental_trainer = None
            self.hybrid_manager = None
            if use_incremental:
                logger.warning("⚠️ Incremental learning requested but not available")
        
        # Initialize data cleaner
        if DATA_CLEANER_AVAILABLE:
            self.data_cleaner = DataCleaner()
        else:
            self.data_cleaner = None
            logger.warning("⚠️ Data cleaner not available - training may fail with dirty data")
        
        # Enhanced model paths - using centralized configuration
        self.model_paths = MODEL_PATHS.copy()
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
    
    def fetch_fresh_training_data(self, days_back: int = 90, force_refresh: bool = False, 
                                incremental: bool = True) -> pd.DataFrame:
        """
        Fetch training data with smart incremental loading and better freshness detection
        
        Args:
            days_back: Number of days of historical data to fetch (reduced default)
            force_refresh: Force fresh download
            incremental: Use incremental loading for efficiency
            
        Returns:
            DataFrame with training data
        """
        # Check for existing recent data
        data_files = [f for f in os.listdir('logs') if f.startswith('ml_training_data_') and f.endswith('.csv')]
        
        if not force_refresh and data_files:
            # Use most recent file
            latest_file = max(data_files)
            file_path = os.path.join('logs', latest_file)
            file_time = os.path.getmtime(file_path)
            hours_old = (datetime.now().timestamp() - file_time) / 3600
            
            # If file is less than 2 hours old, use it
            if hours_old < 2:
                logger.info(f"📥 Using recent training data: {latest_file} ({hours_old:.1f}h old)")
                return pd.read_csv(file_path)
            
            # If file is less than 24 hours old and incremental mode, try incremental update
            elif hours_old < 24 and incremental:
                logger.info(f"🔄 Attempting incremental update from: {latest_file} ({hours_old:.1f}h old)")
                existing_df = pd.read_csv(file_path)
                
                # Check if data is actually stale by looking at the data period
                if 'timestamp' in existing_df.columns:
                    try:
                        latest_data_time = pd.to_datetime(existing_df['timestamp']).max()
                        hours_since_data = (datetime.now() - latest_data_time).total_seconds() / 3600
                        
                        # If data is more than 12 hours old, force fresh fetch
                        if hours_since_data > 12:
                            logger.info(f"⚠️ Data is {hours_since_data:.1f}h old, forcing fresh fetch")
                            force_refresh = True
                        else:
                            return self._fetch_incremental_update(existing_df, days_back)
                    except Exception as e:
                        logger.warning(f"Error checking data freshness: {e}")
                        force_refresh = True
            else:
                # File is too old, force fresh fetch
                logger.info(f"⚠️ File is {hours_old:.1f}h old, forcing fresh fetch")
                force_refresh = True
        
        # Fetch fresh data
        if ENHANCED_DATA_AVAILABLE:
            logger.info("🔄 Fetching fresh training data from Binance...")
            fetcher = EnhancedHistoricalDataFetcher()
            df = fetcher.fetch_comprehensive_data(days_back=days_back)
            
            if not df.empty:
                # Save the data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs/ml_training_data_{timestamp}.csv"
                fetcher.save_training_data(df, filename)
                logger.info(f"✅ Fresh training data saved: {filename}")
                
                # Clean up old data files (keep last 3)
                self._cleanup_old_data_files()
                
                return df
            else:
                logger.error("❌ Failed to fetch fresh data")
                return pd.DataFrame()
        else:
            logger.warning("⚠️ Enhanced data fetcher not available, using fallback")
            return self.load_fallback_data()

    def _cleanup_old_data_files(self):
        """Clean up old training data files, keeping only the last 3"""
        try:
            data_files = [f for f in os.listdir('logs') if f.startswith('ml_training_data_') and f.endswith('.csv')]
            if len(data_files) > 3:
                # Sort by modification time and remove oldest
                file_times = [(f, os.path.getmtime(os.path.join('logs', f))) for f in data_files]
                file_times.sort(key=lambda x: x[1], reverse=True)
                
                for file_name, _ in file_times[3:]:  # Keep only the newest 3
                    try:
                        os.remove(os.path.join('logs', file_name))
                        logger.info(f"🗑️ Cleaned up old data file: {file_name}")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_name}: {e}")
        except Exception as e:
            logger.warning(f"Error cleaning up old files: {e}")

    def _fetch_incremental_update(self, existing_df: pd.DataFrame, days_back: int) -> pd.DataFrame:
        """Fetch incremental updates to existing training data with better error handling"""
        try:
            if ENHANCED_DATA_AVAILABLE:
                fetcher = EnhancedHistoricalDataFetcher()
                
                # Check if we have timestamp column
                if 'timestamp' not in existing_df.columns:
                    logger.warning("⚠️ No timestamp column in existing data, fetching fresh")
                    return fetcher.fetch_comprehensive_data(days_back=days_back)
                
                # Check how old the existing data is
                latest_data_time = pd.to_datetime(existing_df['timestamp']).max()
                hours_since_data = (datetime.now() - latest_data_time).total_seconds() / 3600
                
                # If data is more than 7 days old, fetch fresh
                if hours_since_data > 168:  # 7 days
                    logger.info(f"🔄 Data is {hours_since_data/24:.1f} days old, fetching fresh data")
                    return fetcher.fetch_comprehensive_data(days_back=days_back)
                
                # Use incremental fetch for each symbol/timeframe combination
                updated_data = []
                symbols_processed = set()
                
                # Process unique symbol/timeframe combinations
                if 'symbol' in existing_df.columns and 'timeframe' in existing_df.columns:
                    for (symbol, timeframe), group in existing_df.groupby(['symbol', 'timeframe']):
                        combo_key = f"{symbol}_{timeframe}"
                        if combo_key not in symbols_processed:
                            # Fetch incremental data for this combination
                            updated_subset = fetcher.fetch_incremental_data(
                                symbol, timeframe, group
                            )
                            
                            if not updated_subset.empty and len(updated_subset) > len(group):
                                updated_data.append(updated_subset)
                                logger.info(f"📈 Updated {symbol} {timeframe}: {len(updated_subset)} records")
                            else:
                                updated_data.append(group)  # Use existing if no new data
                            
                            symbols_processed.add(combo_key)
                else:
                    # Fallback: fetch fresh data if structure is unclear
                    logger.warning("⚠️ Unclear data structure, fetching fresh")
                    return fetcher.fetch_comprehensive_data(days_back=days_back)
                
                if updated_data:
                    combined_df = pd.concat(updated_data, ignore_index=True)
                    
                    # Save incremental update
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"logs/ml_training_data_{timestamp}.csv"
                    fetcher.save_training_data(combined_df, filename)
                    logger.info(f"✅ Incremental training data saved: {filename}")
                    
                    # Clean up old files
                    self._cleanup_old_data_files()
                    
                    return combined_df
                
            # Fallback to existing data if incremental fails
            logger.warning("⚠️ Incremental update failed, using existing data")
            return existing_df
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            logger.info("🔄 Falling back to fresh data fetch")
            
            # If incremental fails, try fresh fetch
            if ENHANCED_DATA_AVAILABLE:
                try:
                    fetcher = EnhancedHistoricalDataFetcher()
                    return fetcher.fetch_comprehensive_data(days_back=days_back)
                except Exception as e2:
                    logger.error(f"Fresh fetch also failed: {e2}")
            
            return existing_df
    
    def load_fallback_data(self) -> pd.DataFrame:
        """Load fallback data from existing CSV files"""
        try:
            # Try to load the most comprehensive existing data
            if os.path.exists('logs/trade_history_combined.csv'):
                df = pd.read_csv('logs/trade_history_combined.csv')
                logger.info(f"📥 Loaded fallback data: {len(df)} records")
                return df
            else:
                logger.error("❌ No fallback data available")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading fallback data: {e}")
            return pd.DataFrame()
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare comprehensive ML features from the dataset
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Tuple of (feature_df, feature_names)
        """
        logger.info("🔧 Preparing ML features...")
        
        # Clean data first if cleaner is available
        if self.data_cleaner:
            df_clean = self.data_cleaner.clean_data(df)
        else:
            df_clean = df.copy()
            # Basic cleaning without data cleaner
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        # Core feature columns (technical indicators)
        feature_columns = [
            # Price and returns
            'returns', 'log_returns', 'high_low_ratio', 'open_close_ratio',
            
            # Volume features
            'volume_ratio', 'price_volume',
            
            # Technical indicators
            'rsi', 'rsi_oversold', 'rsi_overbought',
            'macd', 'macd_signal', 'macd_histogram', 'macd_trend', 'macd_crossover',
            
            # Moving averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            
            # Moving average signals
            'price_above_sma_20', 'price_above_sma_50', 'price_above_sma_100', 'price_above_sma_200',
            'price_above_ema_20', 'price_above_ema_50', 'price_above_ema_100', 'price_above_ema_200',
            
            # Bollinger Bands
            'bb_width', 'bb_position', 'bb_squeeze',
            
            # Volatility
            'atr', 'volatility', 'volatility_ratio',
            
            # Momentum
            'stoch_k', 'stoch_d', 'williams_r', 'roc', 'momentum',
            
            # Trend strength
            'adx', 'plus_di', 'minus_di', 'di_diff', 'trend_strength',
            
            # VWAP
            'vwap_distance',
            
            # Pattern recognition
            'doji', 'hammer', 'shooting_star', 'engulfing',
            
            # Support/Resistance
            'resistance_distance', 'support_distance'
        ]
        
        # Filter columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in df_clean.columns]
        
        if not available_features:
            logger.error("❌ No feature columns found in data!")
            return pd.DataFrame(), []
        
        # Create feature DataFrame
        feature_df = df_clean[available_features + ['symbol', 'timeframe']].copy()
        
        # Handle categorical variables
        if 'symbol' in feature_df.columns:
            # Create symbol dummy variables
            symbol_dummies = pd.get_dummies(feature_df['symbol'], prefix='symbol')
            feature_df = pd.concat([feature_df, symbol_dummies], axis=1)
            feature_df = feature_df.drop('symbol', axis=1)
        
        if 'timeframe' in feature_df.columns:
            # Create timeframe dummy variables
            timeframe_dummies = pd.get_dummies(feature_df['timeframe'], prefix='timeframe')
            feature_df = pd.concat([feature_df, timeframe_dummies], axis=1)
            feature_df = feature_df.drop('timeframe', axis=1)
        
        # Final cleaning of feature DataFrame
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[numeric_cols] = feature_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)
        
        logger.info(f"✅ Prepared {len(feature_df.columns)} features for {len(feature_df)} samples")
        return feature_df, list(feature_df.columns)
    
    def train_trend_prediction_model(self, df: pd.DataFrame, 
                                     force_batch: bool = False) -> Dict:
        """Train price trend prediction model with incremental learning support"""
        logger.info("🎯 Training trend prediction model...")
        
        try:
            # Prepare features
            features_df, feature_names = self.prepare_ml_features(df)
            
            if features_df.empty:
                return {'success': False, 'error': 'No features available'}
            
            # Create REAL trend target using ADX + directional movement
            # Only label as trending when ADX > 25 (confirmed trend strength)
            if 'adx' in df.columns and 'plus_di' in df.columns and 'minus_di' in df.columns:
                y = np.where(
                    (df['adx'] > 25) & (df['plus_di'] > df['minus_di']),
                    1,  # Confirmed uptrend (ADX strong + bullish DI)
                    np.where(
                        (df['adx'] > 25) & (df['plus_di'] < df['minus_di']),
                        0,  # Confirmed downtrend (ADX strong + bearish DI)
                        0  # Default to 0 for no-trend (helps balance classes)
                    )
                )
                
                # Convert to pandas Series to maintain index alignment
                y = pd.Series(y, index=df.index, dtype=int)
                
                logger.info(f"📊 Trend target distribution: "
                            f"Uptrend={y.sum():,}, Downtrend/Neutral={(len(y) - y.sum()):,}, "
                            f"Ratio={y.sum()/len(y):.1%}")
            else:
                # Fallback: use improved threshold-based approach (2% minimum meaningful move)
                if 'future_return_4h' in df.columns:
                    y = (df['future_return_4h'] > 0.02).astype(int)
                    logger.warning("⚠️ Using fallback trend target with 2% threshold (ADX features not available)")
                else:
                    # Last resort: next period return with threshold
                    future_return = df['close'].shift(-1) / df['close'] - 1
                    y = (future_return > 0.02).astype(int)
                    logger.warning("⚠️ Using basic fallback trend target")
            
            # Remove rows with NaN targets
            valid_mask = ~y.isna()
            X = features_df[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Validate and clean features if data cleaner is available
            if self.data_cleaner:
                X, y = self.data_cleaner.validate_features(X, y)
            
            # Use incremental learning if enabled and appropriate
            if self.use_incremental and not force_batch:
                logger.info("📈 Using incremental learning approach")
                
                # Prepare data for incremental training
                train_df = pd.concat([X, y.rename('target')], axis=1)
                
                result = self.hybrid_manager.train_smart(
                    train_df, 'trend', 
                    feature_cols=list(X.columns),
                    target_col='target',
                    force_batch=force_batch
                )
                
                if result['success']:
                    logger.info(f"✅ Incremental trend model trained - Samples: {result['total_samples_seen']:,}")
                    return result
                else:
                    logger.warning("⚠️ Incremental training failed, falling back to batch")
                    # Fall through to batch training
            
            # BATCH TRAINING (original implementation)
            logger.info("📊 Using batch learning approach")
            
            # Split data (time-aware split)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=min(50, len(feature_names)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Save model components
            joblib.dump(model, self.model_paths['trend_model'])
            joblib.dump(scaler, self.model_paths['trend_scaler'])
            joblib.dump(selector, self.model_paths['trend_selector'])
            
            self.models['trend'] = model
            self.scalers['trend'] = scaler
            self.feature_selectors['trend'] = selector
            
            result = {
                'success': True,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'features_used': selector.get_support().sum(),
                'training_samples': len(X_train),
                'training_mode': 'batch'
            }
            
            logger.info(f"✅ Trend model trained (batch) - Test accuracy: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training trend model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_signal_success_model(self, df: pd.DataFrame, 
                                   force_batch: bool = False) -> Dict:
        """Train signal success prediction model with incremental learning support"""
        logger.info("🎯 Training signal success model...")
        
        try:
            # Prepare features
            features_df, feature_names = self.prepare_ml_features(df)
            
            if features_df.empty:
                return {'success': False, 'error': 'No features available'}
            
            # Use signal_success target if available
            if 'signal_success' in df.columns:
                y = df['signal_success']
            else:
                # Create signal success target (2% gain in next 4 periods)
                if 'future_return_4h' in df.columns:
                    y = (df['future_return_4h'] > 0.02).astype(int)
                else:
                    y = (df['close'].shift(-4) / df['close'] > 1.02).astype(int)
            
            # Remove rows with NaN targets
            valid_mask = ~y.isna()
            X = features_df[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Validate and clean features if data cleaner is available
            if self.data_cleaner:
                X, y = self.data_cleaner.validate_features(X, y)
            
            # Use incremental learning if enabled and appropriate
            if self.use_incremental and not force_batch:
                logger.info("📈 Using incremental learning approach")
                
                # Prepare data for incremental training
                train_df = pd.concat([X, y.rename('target')], axis=1)
                
                result = self.hybrid_manager.train_smart(
                    train_df, 'signal',
                    feature_cols=list(X.columns),
                    target_col='target',
                    force_batch=force_batch
                )
                
                if result['success']:
                    logger.info(f"✅ Incremental signal model trained - Samples: {result['total_samples_seen']:,}")
                    return result
                else:
                    logger.warning("⚠️ Incremental training failed, falling back to batch")
                    # Fall through to batch training
            
            # BATCH TRAINING (original implementation)
            logger.info("📊 Using batch learning approach")
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(40, len(feature_names)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train model with class balancing
            model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=30,
                min_samples_leaf=15,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Calculate AUC
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Save model components
            joblib.dump(model, self.model_paths['signal_model'])
            joblib.dump(scaler, self.model_paths['signal_scaler'])
            joblib.dump(selector, self.model_paths['signal_selector'])
            
            self.models['signal'] = model
            self.scalers['signal'] = scaler
            self.feature_selectors['signal'] = selector
            
            result = {
                'success': True,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'auc_score': auc_score,
                'features_used': selector.get_support().sum(),
                'training_samples': len(X_train),
                'positive_rate': y_train.mean(),
                'training_mode': 'batch'
            }
            
            logger.info(f"✅ Signal model trained (batch) - Test AUC: {auc_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training signal model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_market_regime_model(self, df: pd.DataFrame,
                                  force_batch: bool = False) -> Dict:
        """Train market regime classification model with incremental learning support"""
        logger.info("🎯 Training market regime model...")
        
        try:
            # Prepare features
            features_df, feature_names = self.prepare_ml_features(df)
            
            if features_df.empty:
                return {'success': False, 'error': 'No features available'}
            
            # Use market_regime target if available
            if 'market_regime' in df.columns:
                y = df['market_regime']
            else:
                # Create market regime based on moving averages
                sma_20 = df['close'].rolling(20).mean()
                sma_50 = df['close'].rolling(50).mean()
                sma_100 = df['close'].rolling(100).mean()
                
                y = np.where(
                    (sma_20 > sma_50) & (sma_50 > sma_100), 'uptrend',
                    np.where(
                        (sma_20 < sma_50) & (sma_50 < sma_100), 'downtrend',
                        'sideways'
                    )
                )
                y = pd.Series(y)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Remove rows with NaN targets
            valid_mask = ~pd.isna(y_encoded)
            X = features_df[valid_mask]
            y_encoded = y_encoded[valid_mask]
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Validate and clean features if data cleaner is available
            if self.data_cleaner:
                X, y_encoded = self.data_cleaner.validate_features(X, pd.Series(y_encoded))
                y_encoded = y_encoded.values  # Convert back to array
            
            # Use incremental learning if enabled and appropriate
            if self.use_incremental and not force_batch:
                logger.info("📈 Using incremental learning approach")
                
                # Prepare data for incremental training
                train_df = pd.concat([X, pd.Series(y_encoded, name='target', index=X.index)], axis=1)
                
                result = self.hybrid_manager.train_smart(
                    train_df, 'regime',
                    feature_cols=list(X.columns),
                    target_col='target',
                    force_batch=force_batch
                )
                
                if result['success']:
                    logger.info(f"✅ Incremental regime model trained - Samples: {result['total_samples_seen']:,}")
                    # Save label encoder
                    joblib.dump(label_encoder, self.model_paths['regime_label_encoder'])
                    return result
                else:
                    logger.warning("⚠️ Incremental training failed, falling back to batch")
                    # Fall through to batch training
            
            # BATCH TRAINING (original implementation)
            logger.info("📊 Using batch learning approach")
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
            
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=min(35, len(feature_names)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=180,
                max_depth=12,
                min_samples_split=25,
                min_samples_leaf=12,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Save model components
            joblib.dump(model, self.model_paths['regime_model'])
            joblib.dump(scaler, self.model_paths['regime_scaler'])
            joblib.dump(selector, self.model_paths['regime_selector'])
            joblib.dump(label_encoder, self.model_paths['regime_label_encoder'])
            
            self.models['regime'] = model
            self.scalers['regime'] = scaler
            self.feature_selectors['regime'] = selector
            
            result = {
                'success': True,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'features_used': selector.get_support().sum(),
                'training_samples': len(X_train),
                'regime_classes': list(label_encoder.classes_),
                'training_mode': 'batch'
            }
            
            logger.info(f"✅ Regime model trained (batch) - Test accuracy: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training regime model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_all_models(self, days_back: int = 90, force_refresh: bool = False, 
                        incremental: bool = True, force_batch: bool = False) -> Dict:
        """
        Train all ML models with smart incremental data loading and learning
        
        Args:
            days_back: Days of historical data to use (reduced default)
            force_refresh: Force fresh data download
            incremental: Use incremental loading for efficiency
            force_batch: Force batch training even if incremental is available
            
        Returns:
            Training results summary
        """
        logger.info("🚀 Starting enhanced ML model training...")
        
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'Scikit-learn not available'}
        
        # Show incremental learning status
        if self.use_incremental and not force_batch:
            logger.info("📈 Incremental/Cumulative Learning: ENABLED")
            logger.info("🔄 Models will build upon previous training sessions")
            
            # Show current cumulative stats
            for model_name in ['trend', 'signal', 'regime']:
                stats = self.incremental_trainer.get_cumulative_stats(model_name)
                if stats.get('training_sessions', 0) > 0:
                    logger.info(f"  {model_name}: {stats['total_samples_seen']:,} samples, "
                              f"{stats['training_sessions']} sessions")
        else:
            logger.info("📊 Batch Learning: Models will be retrained from scratch")
        
        # Fetch training data with incremental loading
        df = self.fetch_fresh_training_data(days_back, force_refresh, incremental)
        
        if df.empty:
            return {'success': False, 'error': 'No training data available'}
        
        logger.info(f"📊 Training with {len(df)} samples from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        results = {}
        
        # Train each model
        models_to_train = [
            ('trend', self.train_trend_prediction_model),
            ('signal', self.train_signal_success_model),
            ('regime', self.train_market_regime_model)
        ]
        
        for model_name, train_func in models_to_train:
            logger.info(f"\n🔄 Training {model_name} model...")
            result = train_func(df, force_batch=force_batch)
            results[model_name] = result
            
            if result['success']:
                mode = result.get('training_mode', 
                                result.get('is_incremental', False) and 'incremental' or 'batch')
                logger.info(f"✅ {model_name} model training completed ({mode})")
            else:
                logger.error(f"❌ {model_name} model training failed: {result.get('error', 'Unknown error')}")
        
        # Generate training report if using incremental learning
        if self.use_incremental:
            training_report = self.hybrid_manager.get_training_report()
            logger.info("\n📊 Cumulative Learning Report:")
            for model_name, model_stats in training_report['models'].items():
                logger.info(f"\n{model_name.upper()}:")
                logger.info(f"  Total samples accumulated: {model_stats['total_samples']:,}")
                logger.info(f"  Training sessions: {model_stats['training_sessions']}")
                logger.info(f"  Current accuracy: {model_stats.get('current_accuracy', 'N/A')}")
                if model_stats.get('improvement') is not None:
                    logger.info(f"  Improvement: {model_stats['improvement']:+.3f}")
                logger.info(f"  Model versions: {model_stats['versions']}")
        
        # Save training history (with JSON serialization fix)
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'data_samples': int(len(df)),
            'data_period': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'training_mode': 'incremental' if (self.use_incremental and not force_batch) else 'batch',
            'results': {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                           for kk, vv in v.items()} for k, v in results.items()},
            'models_trained': [k for k, v in results.items() if v['success']],
            'training_duration': 'completed'
        }
        
        # Save to file
        history_file = 'ml_training_history.json'
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
        except (json.JSONDecodeError, FileNotFoundError):
            # Start fresh if file is corrupted
            history = []
        
        history.append(training_record)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Summary
        successful_models = [k for k, v in results.items() if v['success']]
        
        summary = {
            'success': len(successful_models) > 0,
            'models_trained': successful_models,
            'total_models': len(models_to_train),
            'training_data_size': len(df),
            'training_mode': training_record['training_mode'],
            'results': results
        }
        
        logger.info(f"\n🎉 Training completed! {len(successful_models)}/{len(models_to_train)} models trained successfully")
        logger.info(f"📈 Training mode: {training_record['training_mode'].upper()}")
        
        return summary

def main():
    """Main training execution"""
    logger.info("🤖 CRYPTIX Enhanced ML Training System")
    logger.info("=" * 60)
    
    # Check if incremental learning is available
    if INCREMENTAL_AVAILABLE:
        logger.info("📈 Incremental/Cumulative Learning: AVAILABLE")
        logger.info("💡 Models will accumulate knowledge over training sessions")
    else:
        logger.info("📊 Batch Learning Only: AVAILABLE")
        logger.info("⚠️ Models will be retrained from scratch each time")
    
    logger.info("=" * 60)
    
    # Initialize trainer with incremental learning enabled by default
    trainer = EnhancedMLTrainer(use_incremental=True)
    
    # Show current cumulative stats if using incremental
    if trainer.use_incremental:
        logger.info("\n📊 Current Cumulative Learning Status:")
        for model_name in ['trend', 'signal', 'regime']:
            stats = trainer.incremental_trainer.get_cumulative_stats(model_name)
            sessions = stats.get('training_sessions', 0)
            samples = stats.get('total_samples_seen', 0)
            last_trained = stats.get('last_trained', 'Never')
            
            if sessions > 0:
                logger.info(f"\n{model_name.upper()} Model:")
                logger.info(f"  • Total samples accumulated: {samples:,}")
                logger.info(f"  • Training sessions: {sessions}")
                logger.info(f"  • Last trained: {last_trained[:10] if last_trained != 'Never' else 'Never'}")
                
                # Show learning curve
                curve = trainer.incremental_trainer.get_learning_curve(model_name)
                if len(curve) >= 2:
                    first_acc = curve[0].get('accuracy')
                    last_acc = curve[-1].get('accuracy')
                    # Only show improvement if both accuracies are available
                    if first_acc is not None and last_acc is not None:
                        improvement = last_acc - first_acc
                        logger.info(f"  • Improvement: {improvement:+.3f} (from {first_acc:.3f} to {last_acc:.3f})")
            else:
                logger.info(f"\n{model_name.upper()} Model: Not yet trained")
        
        logger.info("\n" + "=" * 60)
    
    # Train all models with incremental approach
    # Set force_batch=True to disable incremental learning for this session
    results = trainer.train_all_models(
        days_back=90, 
        force_refresh=False, 
        incremental=True,
        force_batch=False  # Set to True to force batch retraining
    )
    
    if results['success']:
        logger.info(f"\n✅ Training completed successfully!")
        logger.info(f"🎯 Models trained: {', '.join(results['models_trained'])}")
        logger.info(f"📊 Training data: {results['training_data_size']:,} samples")
        logger.info(f"📈 Training mode: {results['training_mode'].upper()}")
        
        # Show model performance
        logger.info("\n📈 Model Performance:")
        for model_name in results['models_trained']:
            model_result = results['results'][model_name]
            if 'test_accuracy' in model_result:
                logger.info(f"  {model_name}: {model_result['test_accuracy']:.3f} accuracy")
            elif 'accuracy' in model_result:
                logger.info(f"  {model_name}: {model_result['accuracy']:.3f} accuracy")
        
        # If incremental, show cumulative report
        if trainer.use_incremental and results['training_mode'] == 'incremental':
            logger.info("\n📊 Cumulative Learning Report:")
            report = trainer.hybrid_manager.get_training_report()
            for model_name, model_info in report['models'].items():
                logger.info(f"\n{model_name.upper()}:")
                logger.info(f"  • Total accumulated samples: {model_info['total_samples']:,}")
                logger.info(f"  • Total training sessions: {model_info['training_sessions']}")
                if model_info.get('improvement') is not None:
                    logger.info(f"  • Overall improvement: {model_info['improvement']:+.3f}")
                logger.info(f"  • Model versions saved: {model_info['versions']}")
    else:
        logger.error("❌ Training failed!")
        
    return results

if __name__ == "__main__":
    main()
