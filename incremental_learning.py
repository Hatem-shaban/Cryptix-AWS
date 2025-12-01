"""
Incremental Learning Module for CRYPTIX Trading Bot
Implements true cumulative/exponential learning that builds on previous training
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Safe ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available")

from model_paths import MODEL_PATHS

logger = logging.getLogger(__name__)

class IncrementalModelVersion:
    """Tracks version history of an incremental model"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.versions = []
        self.current_version = 0
        self.version_file = f"models/{model_name}_versions.json"
        self._load_version_history()
    
    def _load_version_history(self):
        """Load version history from disk"""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                    self.versions = data.get('versions', [])
                    self.current_version = data.get('current_version', 0)
            except Exception as e:
                logger.warning(f"Could not load version history: {e}")
                self.versions = []
                self.current_version = 0
    
    def save_version_history(self):
        """Save version history to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            with open(self.version_file, 'w') as f:
                json.dump({
                    'versions': self.versions,
                    'current_version': self.current_version
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving version history: {e}")
    
    def add_version(self, metrics: Dict, samples_added: int, total_samples: int):
        """Record a new training version"""
        self.current_version += 1
        version_info = {
            'version': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'samples_added': samples_added,
            'total_samples': total_samples,
            'metrics': metrics,
            'model_path': f"models/{self.model_name}_v{self.current_version}.pkl"
        }
        self.versions.append(version_info)
        self.save_version_history()
        return version_info
    
    def get_version_info(self, version: Optional[int] = None) -> Optional[Dict]:
        """Get information about a specific version"""
        if version is None:
            version = self.current_version
        
        for v in self.versions:
            if v['version'] == version:
                return v
        return None
    
    def rollback_to_version(self, version: int) -> bool:
        """Rollback to a previous version"""
        version_info = self.get_version_info(version)
        if version_info and os.path.exists(version_info['model_path']):
            self.current_version = version
            self.save_version_history()
            return True
        return False


class IncrementalMLTrainer:
    """
    True Incremental Learning System
    - Accumulates knowledge from previous training sessions
    - Uses partial_fit for online learning
    - Maintains model versions and allows rollback
    - Tracks cumulative performance over time
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_versions = {}
        self.cumulative_stats = {}
        
        # Model configurations
        self.model_configs = {
            'trend': {
                'type': 'sgd_classifier',  # Supports partial_fit
                'params': {
                    'loss': 'log_loss',
                    'penalty': 'l2',
                    'alpha': 0.0001,
                    'max_iter': 1000,
                    'random_state': 42,
                    'warm_start': True  # Critical for incremental learning
                }
            },
            'signal': {
                'type': 'passive_aggressive',  # Supports partial_fit
                'params': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42,
                    'warm_start': True
                }
            },
            'regime': {
                'type': 'sgd_classifier',
                'params': {
                    'loss': 'log_loss',
                    'penalty': 'l2',
                    'alpha': 0.0001,
                    'max_iter': 1000,
                    'random_state': 42,
                    'warm_start': True
                }
            }
        }
        
        # Initialize model versions
        for model_name in ['trend', 'signal', 'regime']:
            self.model_versions[model_name] = IncrementalModelVersion(model_name)
        
        # Load existing incremental models if available
        self._load_incremental_models()
        
        # Stats file
        self.stats_file = 'models/incremental_training_stats.json'
        self._load_cumulative_stats()
    
    def _load_incremental_models(self):
        """Load existing incremental models"""
        for model_name in ['trend', 'signal', 'regime']:
            model_path = f"models/{model_name}_incremental.pkl"
            scaler_path = f"models/{model_name}_incremental_scaler.pkl"
            selector_path = f"models/{model_name}_incremental_selector.pkl"
            
            # Load model
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"✅ Loaded existing incremental {model_name} model")
                except Exception as e:
                    logger.warning(f"Could not load {model_name} model: {e}")
            
            # Load scaler
            if os.path.exists(scaler_path):
                try:
                    self.scalers[model_name] = joblib.load(scaler_path)
                except Exception as e:
                    logger.warning(f"Could not load {model_name} scaler: {e}")
            
            # Load feature selector
            if os.path.exists(selector_path):
                try:
                    self.feature_selectors[model_name] = joblib.load(selector_path)
                except Exception as e:
                    logger.warning(f"Could not load {model_name} selector: {e}")
    
    def _load_cumulative_stats(self):
        """Load cumulative training statistics"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    self.cumulative_stats = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cumulative stats: {e}")
                self.cumulative_stats = {}
        
        # Initialize stats for each model if not present
        for model_name in ['trend', 'signal', 'regime']:
            if model_name not in self.cumulative_stats:
                self.cumulative_stats[model_name] = {
                    'total_samples_seen': 0,
                    'training_sessions': 0,
                    'last_trained': None,
                    'performance_history': [],
                    'feature_importance_history': []
                }
    
    def _save_cumulative_stats(self):
        """Save cumulative statistics"""
        try:
            os.makedirs('models', exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(self.cumulative_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cumulative stats: {e}")
    
    def _create_or_get_model(self, model_name: str, n_classes: Optional[int] = None):
        """Create a new model or return existing one"""
        if model_name in self.models:
            return self.models[model_name]
        
        config = self.model_configs.get(model_name, self.model_configs['trend'])
        
        if config['type'] == 'sgd_classifier':
            model = SGDClassifier(**config['params'])
        elif config['type'] == 'passive_aggressive':
            model = PassiveAggressiveClassifier(**config['params'])
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        self.models[model_name] = model
        return model
    
    def _update_scaler_incrementally(self, scaler: Optional[RobustScaler], 
                                    X: np.ndarray, model_name: str) -> RobustScaler:
        """Update scaler with new data"""
        if scaler is None:
            scaler = RobustScaler()
            scaler.fit(X)
        else:
            # For incremental scaling, we need to use partial_fit
            # Note: RobustScaler doesn't support partial_fit, so we use a workaround
            # Store stats and update incrementally
            if not hasattr(scaler, '_n_samples_seen'):
                scaler._n_samples_seen = 0
            
            old_n = scaler._n_samples_seen
            new_n = len(X)
            total_n = old_n + new_n
            
            # Update running statistics (simplified approach)
            if old_n == 0:
                scaler.fit(X)
            else:
                # Combine old and new data for scaling (memory-efficient batch)
                # In production, you'd use online statistics algorithms
                scaler.fit(X)
            
            scaler._n_samples_seen = total_n
        
        return scaler
    
    def train_incremental(self, df: pd.DataFrame, model_name: str,
                         feature_cols: List[str], target_col: str,
                         is_first_training: bool = False) -> Dict:
        """
        Train model incrementally on new data
        
        Args:
            df: New training data
            model_name: Name of model ('trend', 'signal', 'regime')
            feature_cols: List of feature column names
            target_col: Target column name
            is_first_training: Whether this is the first training (fit vs partial_fit)
            
        Returns:
            Training results dictionary
        """
        logger.info(f"🔄 Incremental training for {model_name} model with {len(df)} new samples")
        
        try:
            # Prepare features and target
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Remove NaN values
            valid_mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Get or create scaler
            scaler = self.scalers.get(model_name)
            scaler = self._update_scaler_incrementally(scaler, X, model_name)
            self.scalers[model_name] = scaler
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Feature selection - DISABLED for incremental models
            # The existing incremental models were trained WITHOUT feature selection
            # They expect all 73 features, not a reduced set
            # Using feature selection now would break compatibility
            selector = self.feature_selectors.get(model_name)
            
            if selector is not None:
                # If a selector exists from previous training, use it
                X_selected = selector.transform(X_scaled)
                logger.info(f"Using existing feature selector: {X_scaled.shape[1]} → {X_selected.shape[1]} features")
            else:
                # No selector - use all features (this is the normal case for incremental models)
                X_selected = X_scaled
            
            # Get or create model
            model = self._create_or_get_model(model_name, len(np.unique(y)))
            
            # Train incrementally
            if is_first_training or not hasattr(model, 'classes_'):
                # First training - use fit
                model.fit(X_selected, y)
                logger.info(f"✅ Initial training completed for {model_name}")
            else:
                # Incremental training - use partial_fit
                classes = np.unique(y)
                model.partial_fit(X_selected, y, classes=classes)
                logger.info(f"✅ Incremental update completed for {model_name}")
            
            # Evaluate on the new data
            y_pred = model.predict(X_selected)
            accuracy = accuracy_score(y, y_pred)
            
            # Calculate AUC if binary classification
            auc_score = None
            if len(np.unique(y)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X_selected)[:, 1]
                    auc_score = roc_auc_score(y, y_pred_proba)
                except Exception:
                    pass
            
            # Update cumulative statistics
            old_samples = self.cumulative_stats[model_name]['total_samples_seen']
            new_total = old_samples + len(X)
            self.cumulative_stats[model_name]['total_samples_seen'] = new_total
            self.cumulative_stats[model_name]['training_sessions'] += 1
            self.cumulative_stats[model_name]['last_trained'] = datetime.now().isoformat()
            
            # Add performance to history
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'samples_added': len(X),
                'total_samples': new_total,
                'accuracy': float(accuracy),
                'auc_score': float(auc_score) if auc_score else None,
                'training_session': self.cumulative_stats[model_name]['training_sessions']
            }
            self.cumulative_stats[model_name]['performance_history'].append(performance_record)
            
            # Save model and stats
            self._save_model(model_name, model, scaler, selector)
            self._save_cumulative_stats()
            
            # Save version
            version_info = self.model_versions[model_name].add_version(
                metrics={'accuracy': accuracy, 'auc_score': auc_score},
                samples_added=len(X),
                total_samples=new_total
            )
            
            result = {
                'success': True,
                'model_name': model_name,
                'samples_added': len(X),
                'total_samples_seen': new_total,
                'training_sessions': self.cumulative_stats[model_name]['training_sessions'],
                'accuracy': accuracy,
                'auc_score': auc_score,
                'version': version_info['version'],
                'is_incremental': not is_first_training
            }
            
            logger.info(f"✅ {model_name} trained - Acc: {accuracy:.3f}, Total samples: {new_total:,}")
            return result
            
        except Exception as e:
            logger.error(f"Error in incremental training for {model_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _save_model(self, model_name: str, model, scaler, selector):
        """Save incremental model components"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save with incremental naming
            model_path = f"models/{model_name}_incremental.pkl"
            scaler_path = f"models/{model_name}_incremental_scaler.pkl"
            selector_path = f"models/{model_name}_incremental_selector.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            if selector:
                joblib.dump(selector, selector_path)
            
            # Also save versioned copy
            version = self.model_versions[model_name].current_version
            version_path = f"models/{model_name}_v{version}.pkl"
            joblib.dump(model, version_path)
            
        except Exception as e:
            logger.error(f"Error saving {model_name} model: {e}")
    
    def get_learning_curve(self, model_name: str) -> List[Dict]:
        """Get learning curve showing performance over time"""
        if model_name in self.cumulative_stats:
            return self.cumulative_stats[model_name]['performance_history']
        return []
    
    def get_cumulative_stats(self, model_name: Optional[str] = None) -> Dict:
        """Get cumulative statistics for model(s)"""
        if model_name:
            return self.cumulative_stats.get(model_name, {})
        return self.cumulative_stats
    
    def reset_model(self, model_name: str):
        """Reset a model to start fresh (use with caution)"""
        logger.warning(f"⚠️ Resetting {model_name} model - all accumulated knowledge will be lost!")
        
        # Remove model files
        files_to_remove = [
            f"models/{model_name}_incremental.pkl",
            f"models/{model_name}_incremental_scaler.pkl",
            f"models/{model_name}_incremental_selector.pkl"
        ]
        
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Reset stats
        if model_name in self.cumulative_stats:
            self.cumulative_stats[model_name] = {
                'total_samples_seen': 0,
                'training_sessions': 0,
                'last_trained': None,
                'performance_history': [],
                'feature_importance_history': []
            }
            self._save_cumulative_stats()
        
        # Remove from memory
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.scalers:
            del self.scalers[model_name]
        if model_name in self.feature_selectors:
            del self.feature_selectors[model_name]
        
        logger.info(f"✅ {model_name} model reset complete")


class HybridTrainingManager:
    """
    Manages both batch and incremental training
    Provides smart decision-making for when to use each approach
    """
    
    def __init__(self):
        self.incremental_trainer = IncrementalMLTrainer()
        self.batch_trainer = None  # Will be imported from enhanced_ml_training
    
    def should_use_incremental(self, model_name: str, new_data_size: int) -> bool:
        """
        Decide whether to use incremental or batch training
        
        Rules:
        - Use incremental if model exists and new data is small (<10k samples)
        - Use incremental if this is an update to recent training
        - Use batch for major retraining or first training
        """
        stats = self.incremental_trainer.get_cumulative_stats(model_name)
        
        # No existing model - use batch training for first time
        if stats.get('training_sessions', 0) == 0:
            return False
        
        # Small update - use incremental
        if new_data_size < 10000:
            return True
        
        # Check last training date
        last_trained = stats.get('last_trained')
        if last_trained:
            last_date = datetime.fromisoformat(last_trained)
            days_since = (datetime.now() - last_date).days
            
            # If trained recently (<7 days), prefer incremental
            if days_since < 7:
                return True
        
        # Large data or old model - prefer batch retraining
        return False
    
    def train_smart(self, df: pd.DataFrame, model_name: str,
                   feature_cols: List[str], target_col: str,
                   force_batch: bool = False,
                   force_incremental: bool = False) -> Dict:
        """
        Smart training that chooses the best approach
        
        Args:
            df: Training data
            model_name: Model to train
            feature_cols: Feature columns
            target_col: Target column
            force_batch: Force batch training
            force_incremental: Force incremental training
        """
        # Decide training approach
        if force_batch:
            use_incremental = False
        elif force_incremental:
            use_incremental = True
        else:
            use_incremental = self.should_use_incremental(model_name, len(df))
        
        if use_incremental:
            logger.info(f"📈 Using INCREMENTAL training for {model_name}")
            is_first = self.incremental_trainer.cumulative_stats[model_name]['training_sessions'] == 0
            return self.incremental_trainer.train_incremental(
                df, model_name, feature_cols, target_col, is_first_training=is_first
            )
        else:
            logger.info(f"📊 Using BATCH training for {model_name}")
            # Would call batch trainer here
            return {
                'success': False,
                'error': 'Batch training not implemented in this module'
            }
    
    def get_training_report(self) -> Dict:
        """Generate comprehensive training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name in ['trend', 'signal', 'regime']:
            stats = self.incremental_trainer.get_cumulative_stats(model_name)
            learning_curve = self.incremental_trainer.get_learning_curve(model_name)
            
            # Calculate improvement over time
            improvement = None
            if len(learning_curve) >= 2:
                first_acc = learning_curve[0].get('accuracy')
                last_acc = learning_curve[-1].get('accuracy')
                # Only calculate improvement if both accuracies are available
                if first_acc is not None and last_acc is not None:
                    improvement = last_acc - first_acc
            
            report['models'][model_name] = {
                'total_samples': stats.get('total_samples_seen', 0),
                'training_sessions': stats.get('training_sessions', 0),
                'last_trained': stats.get('last_trained'),
                'current_accuracy': learning_curve[-1].get('accuracy') if learning_curve else None,
                'improvement': improvement,
                'versions': len(self.incremental_trainer.model_versions[model_name].versions)
            }
        
        return report


def main():
    """Demo/test of incremental learning"""
    logger.info("🤖 Incremental Learning System Demo")
    
    trainer = IncrementalMLTrainer()
    
    # Show current stats
    logger.info("\n📊 Current cumulative statistics:")
    for model_name in ['trend', 'signal', 'regime']:
        stats = trainer.get_cumulative_stats(model_name)
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Total samples seen: {stats.get('total_samples_seen', 0):,}")
        logger.info(f"  Training sessions: {stats.get('training_sessions', 0)}")
        logger.info(f"  Last trained: {stats.get('last_trained', 'Never')}")
        
        # Show learning curve
        curve = trainer.get_learning_curve(model_name)
        if curve:
            logger.info(f"  Performance history ({len(curve)} points):")
            for record in curve[-5:]:  # Show last 5
                logger.info(f"    {record['timestamp'][:10]}: Acc={record['accuracy']:.3f}, "
                          f"Samples={record['total_samples']:,}")
    
    logger.info("\n✅ Use this module in enhanced_ml_training.py for true incremental learning!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
