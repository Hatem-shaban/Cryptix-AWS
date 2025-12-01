# CRYPTIX-ML Models Directory

This directory contains all trained machine learning models, scalers, and feature selectors used by the CRYPTIX Trading Bot.

## Directory Structure

### Core ML Models
- `rf_price_trend_model.pkl` - Random Forest model for price trend prediction
- `rf_market_regime_model.pkl` - Market regime classification model
- `rf_pattern_recognition_model.pkl` - Pattern recognition model for signal success
- `rf_signal_success_model.pkl` - Signal success prediction model
- `rf_volatility_model.pkl` - Volatility prediction model (if created)

### Feature Scalers
- `rf_scaler.pkl` - Main feature scaler for trend models
- `rf_regime_scaler.pkl` - Scaler for market regime features
- `rf_pattern_scaler.pkl` - Scaler for pattern recognition features
- `rf_signal_scaler.pkl` - Scaler for signal success features
- `rf_volatility_scaler.pkl` - Scaler for volatility features (if created)

### Feature Selectors
- `rf_trend_selector.pkl` - Feature selector for trend prediction
- `rf_regime_selector.pkl` - Feature selector for regime classification
- `rf_pattern_selector.pkl` - Feature selector for pattern recognition
- `rf_signal_selector.pkl` - Feature selector for signal success
- `rf_volatility_selector.pkl` - Feature selector for volatility (if created)

### Additional Models
- `market_patterns.pkl` - Historical market pattern data
- `rf_regime_label_encoder.pkl` - Label encoder for market regime classes

## Model Management

All model paths are centrally managed through `model_paths.py` in the root directory. This ensures:

1. **Consistent Paths**: All modules use the same model file locations
2. **Easy Maintenance**: Model paths can be updated in one place
3. **Backward Compatibility**: Legacy code continues to work
4. **Migration Support**: Helper functions for moving models

## Usage

```python
from model_paths import MODEL_PATHS

# Get path to a specific model
trend_model_path = MODEL_PATHS['trend_model']

# Check if a model exists
from model_paths import model_exists
if model_exists('trend_model'):
    # Load the model
    pass

# Get list of missing models
from model_paths import get_missing_models
missing = get_missing_models()
```

## Training

Models are trained using the `EnhancedMLTrainer` class from `enhanced_ml_training.py`. Training automatically saves models to this directory using the centralized path configuration.

## Backup

It's recommended to backup this entire directory before:
- Major system updates
- Model retraining
- Code deployments

The models contain valuable learned patterns from historical market data and can take significant time to retrain.
