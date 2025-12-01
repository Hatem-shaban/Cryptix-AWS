"""
Centralized Model Paths Configuration
Manages all ML model file paths in a single location
"""

import os

# Base model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(filename: str) -> str:
    """Get full path for a model file"""
    return os.path.join(MODEL_DIR, filename)

# Model file definitions
MODEL_PATHS = {
    # Core ML Models
    'trend_model': get_model_path('rf_price_trend_model.pkl'),
    'regime_model': get_model_path('rf_market_regime_model.pkl'),
    'pattern_model': get_model_path('rf_pattern_recognition_model.pkl'),
    'signal_model': get_model_path('rf_signal_success_model.pkl'),
    'volatility_model': get_model_path('rf_volatility_model.pkl'),
    
    # Scalers
    'trend_scaler': get_model_path('rf_scaler.pkl'),
    'regime_scaler': get_model_path('rf_regime_scaler.pkl'),
    'pattern_scaler': get_model_path('rf_pattern_scaler.pkl'),
    'signal_scaler': get_model_path('rf_signal_scaler.pkl'),
    'volatility_scaler': get_model_path('rf_volatility_scaler.pkl'),
    
    # Feature Selectors
    'trend_selector': get_model_path('rf_trend_selector.pkl'),
    'regime_selector': get_model_path('rf_regime_selector.pkl'),
    'pattern_selector': get_model_path('rf_pattern_selector.pkl'),
    'signal_selector': get_model_path('rf_signal_selector.pkl'),
    'volatility_selector': get_model_path('rf_volatility_selector.pkl'),
    
    # Additional Models
    'market_patterns': get_model_path('market_patterns.pkl'),
    'regime_label_encoder': get_model_path('rf_regime_label_encoder.pkl'),
}

# Legacy compatibility - direct model file paths
TREND_MODEL = MODEL_PATHS['trend_model']
REGIME_MODEL = MODEL_PATHS['regime_model']
PATTERN_MODEL = MODEL_PATHS['pattern_model']
SIGNAL_MODEL = MODEL_PATHS['signal_model']
VOLATILITY_MODEL = MODEL_PATHS['volatility_model']

TREND_SCALER = MODEL_PATHS['trend_scaler']
REGIME_SCALER = MODEL_PATHS['regime_scaler']
PATTERN_SCALER = MODEL_PATHS['pattern_scaler']
SIGNAL_SCALER = MODEL_PATHS['signal_scaler']
VOLATILITY_SCALER = MODEL_PATHS['volatility_scaler']

MARKET_PATTERNS = MODEL_PATHS['market_patterns']
REGIME_LABEL_ENCODER = MODEL_PATHS['regime_label_encoder']

# Helper functions
def get_all_model_files():
    """Get list of all model files that should exist"""
    return list(MODEL_PATHS.values())

def model_exists(model_key: str) -> bool:
    """Check if a specific model file exists"""
    if model_key not in MODEL_PATHS:
        return False
    return os.path.exists(MODEL_PATHS[model_key])

def get_missing_models():
    """Get list of missing model files"""
    missing = []
    for key, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing.append(key)
    return missing

def cleanup_old_models():
    """Remove old model files from root directory (migration helper)"""
    root_dir = os.path.dirname(__file__)
    old_files = []
    
    for model_path in MODEL_PATHS.values():
        filename = os.path.basename(model_path)
        old_path = os.path.join(root_dir, filename)
        if os.path.exists(old_path) and old_path != model_path:
            old_files.append(old_path)
    
    return old_files
