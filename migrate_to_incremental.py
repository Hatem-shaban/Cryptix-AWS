"""
Migration Utility for Converting to Incremental Learning
Converts existing batch-trained models to incremental learning format
"""

import os
import joblib
import json
from datetime import datetime
import logging
from pathlib import Path

from model_paths import MODEL_PATHS
from incremental_learning import IncrementalMLTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMigrator:
    """Migrates existing models to incremental learning format"""
    
    def __init__(self):
        self.incremental_trainer = IncrementalMLTrainer()
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
    
    def check_existing_models(self) -> dict:
        """Check which models exist in batch format"""
        model_status = {}
        
        for model_name in ['trend', 'signal', 'regime']:
            batch_path = MODEL_PATHS[f'{model_name}_model']
            incremental_path = f"models/{model_name}_incremental.pkl"
            
            model_status[model_name] = {
                'has_batch': os.path.exists(batch_path),
                'has_incremental': os.path.exists(incremental_path),
                'batch_path': batch_path,
                'incremental_path': incremental_path
            }
        
        return model_status
    
    def migrate_model(self, model_name: str, estimated_samples: int = 30000) -> bool:
        """
        Migrate a single model from batch to incremental format
        
        Args:
            model_name: Name of model to migrate
            estimated_samples: Estimated number of samples used in training
        """
        logger.info(f"🔄 Migrating {model_name} model to incremental format...")
        
        try:
            # Load batch model components
            batch_model_path = MODEL_PATHS[f'{model_name}_model']
            batch_scaler_path = MODEL_PATHS[f'{model_name}_scaler']
            batch_selector_path = MODEL_PATHS.get(f'{model_name}_selector')
            
            if not os.path.exists(batch_model_path):
                logger.error(f"❌ Batch model not found: {batch_model_path}")
                return False
            
            # Note: RandomForest and GradientBoosting don't support incremental learning
            # We need to initialize new SGD-based models instead
            logger.warning(f"⚠️ Cannot directly migrate {model_name} model")
            logger.info(f"   Batch models (RandomForest/GradientBoosting) don't support incremental learning")
            logger.info(f"   Recommendation: Train new incremental model with historical data")
            
            # Initialize incremental model stats to track migration
            stats = {
                'total_samples_seen': estimated_samples,
                'training_sessions': 1,
                'last_trained': datetime.now().isoformat(),
                'performance_history': [{
                    'timestamp': datetime.now().isoformat(),
                    'samples_added': estimated_samples,
                    'total_samples': estimated_samples,
                    'accuracy': None,
                    'auc_score': None,
                    'training_session': 1,
                    'note': 'Migrated from batch model (estimated)'
                }],
                'feature_importance_history': []
            }
            
            # Update cumulative stats
            self.incremental_trainer.cumulative_stats[model_name] = stats
            self.incremental_trainer._save_cumulative_stats()
            
            # Create version record
            version_info = self.incremental_trainer.model_versions[model_name].add_version(
                metrics={'accuracy': None, 'note': 'Baseline from batch model'},
                samples_added=estimated_samples,
                total_samples=estimated_samples
            )
            
            logger.info(f"✅ {model_name} migration tracking initialized")
            logger.info(f"   Estimated baseline: {estimated_samples:,} samples")
            logger.info(f"   Next training will use incremental learning")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating {model_name}: {e}")
            return False
    
    def migrate_all_models(self, estimated_samples: dict = None) -> dict:
        """
        Migrate all existing models
        
        Args:
            estimated_samples: Dict mapping model names to sample counts
        """
        if estimated_samples is None:
            # Use defaults based on typical training
            estimated_samples = {
                'trend': 30000,
                'signal': 30000,
                'regime': 30000
            }
        
        logger.info("🚀 Starting model migration to incremental learning...")
        
        # Check existing models
        status = self.check_existing_models()
        
        logger.info("\n📊 Current Model Status:")
        for model_name, info in status.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Batch model: {'✓' if info['has_batch'] else '✗'}")
            logger.info(f"  Incremental model: {'✓' if info['has_incremental'] else '✗'}")
        
        # Migrate each model
        results = {}
        for model_name in ['trend', 'signal', 'regime']:
            if status[model_name]['has_batch'] and not status[model_name]['has_incremental']:
                samples = estimated_samples.get(model_name, 30000)
                success = self.migrate_model(model_name, samples)
                results[model_name] = 'migrated' if success else 'failed'
            elif status[model_name]['has_incremental']:
                logger.info(f"✓ {model_name} already has incremental model")
                results[model_name] = 'already_incremental'
            else:
                logger.warning(f"⚠️ {model_name} has no batch model to migrate")
                results[model_name] = 'no_batch_model'
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 Migration Summary:")
        for model_name, result in results.items():
            logger.info(f"  {model_name}: {result}")
        
        logger.info("\n💡 Next Steps:")
        logger.info("  1. Run training with incremental mode enabled")
        logger.info("  2. New incremental models will be trained from scratch")
        logger.info("  3. Future trainings will accumulate knowledge")
        logger.info("="*60 + "\n")
        
        return results
    
    def create_training_comparison_report(self) -> dict:
        """Create a report comparing batch vs incremental learning"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'batch_vs_incremental': {
                'batch_learning': {
                    'pros': [
                        'Uses all data at once',
                        'Can use complex models (RandomForest, GradientBoosting)',
                        'Better for initial training',
                        'More stable predictions'
                    ],
                    'cons': [
                        'Replaces previous knowledge',
                        'Does not accumulate learning',
                        'Requires retraining from scratch',
                        'More computational overhead'
                    ]
                },
                'incremental_learning': {
                    'pros': [
                        'Accumulates knowledge over time',
                        'Updates existing models efficiently',
                        'Less computational overhead for updates',
                        'Tracks learning progress',
                        'Version control with rollback'
                    ],
                    'cons': [
                        'Limited to SGD-based models initially',
                        'May require more training sessions',
                        'Need careful feature management'
                    ]
                }
            },
            'recommendation': (
                'Use hybrid approach: '
                'Start with batch learning for foundation, '
                'then switch to incremental for regular updates'
            )
        }
        
        return report


def main():
    """Main migration process"""
    logger.info("🤖 Model Migration to Incremental Learning")
    logger.info("=" * 60)
    
    migrator = ModelMigrator()
    
    # Show comparison
    report = migrator.create_training_comparison_report()
    logger.info("\n📊 Batch vs Incremental Learning:\n")
    logger.info("BATCH LEARNING:")
    for pro in report['batch_vs_incremental']['batch_learning']['pros']:
        logger.info(f"  ✓ {pro}")
    for con in report['batch_vs_incremental']['batch_learning']['cons']:
        logger.info(f"  ✗ {con}")
    
    logger.info("\nINCREMENTAL LEARNING:")
    for pro in report['batch_vs_incremental']['incremental_learning']['pros']:
        logger.info(f"  ✓ {pro}")
    for con in report['batch_vs_incremental']['incremental_learning']['cons']:
        logger.info(f"  ✗ {con}")
    
    logger.info(f"\n💡 {report['recommendation']}\n")
    
    # Check if migration is needed
    status = migrator.check_existing_models()
    needs_migration = any(
        info['has_batch'] and not info['has_incremental'] 
        for info in status.values()
    )
    
    if needs_migration:
        response = input("\n❓ Do you want to migrate existing models? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            # Get estimated samples from training history if available
            estimated_samples = {}
            history_file = 'ml_training_history.json'
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                        if history:
                            latest = history[-1]
                            samples = latest.get('data_samples', 30000)
                            for model in ['trend', 'signal', 'regime']:
                                estimated_samples[model] = samples
                except Exception as e:
                    logger.warning(f"Could not read training history: {e}")
            
            # Perform migration
            results = migrator.migrate_all_models(estimated_samples)
            
            logger.info("\n✅ Migration process completed!")
        else:
            logger.info("\n⏭️ Migration skipped")
    else:
        logger.info("\n✅ All models are already in incremental format or no models to migrate")
    
    logger.info("\n📚 For documentation, see: INCREMENTAL_LEARNING.md")


if __name__ == "__main__":
    main()
