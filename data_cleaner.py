"""
Data Cleaning and Validation Module for ML Training
Handles data preprocessing and validation for ML model training
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class DataCleaner:
    """Data cleaning and validation for ML training"""
    
    def __init__(self):
        self.outlier_threshold = 5  # Z-score threshold for outliers
        self.max_missing_ratio = 0.3  # Maximum allowed missing data ratio
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning for ML training
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"🧹 Starting data cleaning for {len(df)} records...")
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # 1. Handle infinite values
        logger.info("📊 Handling infinite values...")
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Replace infinity with NaN
            cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
        
        # 2. Handle extreme outliers (beyond reasonable trading values)
        logger.info("🎯 Removing extreme outliers...")
        
        for col in numeric_cols:
            if col in cleaned_df.columns:
                # Remove values beyond reasonable bounds
                q1 = cleaned_df[col].quantile(0.01)
                q99 = cleaned_df[col].quantile(0.99)
                iqr = cleaned_df[col].quantile(0.75) - cleaned_df[col].quantile(0.25)
                
                # Set extreme bounds
                lower_bound = q1 - 10 * iqr
                upper_bound = q99 + 10 * iqr
                
                # Cap extreme values
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 3. Handle missing values
        logger.info("📋 Handling missing values...")
        
        # Fill missing values with appropriate strategies
        for col in numeric_cols:
            if col in cleaned_df.columns:
                missing_ratio = cleaned_df[col].isna().sum() / len(cleaned_df)
                
                if missing_ratio > self.max_missing_ratio:
                    logger.warning(f"⚠️ Column {col} has {missing_ratio:.1%} missing values")
                
                # Fill with median for most columns
                if 'ratio' in col.lower() or 'returns' in col.lower():
                    # For ratios and returns, use 0 or 1 as appropriate
                    fill_value = 1 if 'ratio' in col.lower() else 0
                    cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                elif 'rsi' in col.lower():
                    cleaned_df[col] = cleaned_df[col].fillna(50)  # Neutral RSI
                elif 'volume' in col.lower():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                else:
                    # Use forward fill then backward fill, then median
                    cleaned_df[col] = cleaned_df[col].ffill().bfill()
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # 4. Validate data ranges
        logger.info("🔍 Validating data ranges...")
        
        # Ensure RSI is between 0 and 100
        if 'rsi' in cleaned_df.columns:
            cleaned_df['rsi'] = cleaned_df['rsi'].clip(0, 100)
        
        # Ensure percentages and ratios are reasonable
        percentage_cols = [col for col in cleaned_df.columns if any(x in col.lower() for x in ['ratio', 'position', 'distance'])]
        for col in percentage_cols:
            if col in cleaned_df.columns:
                # Cap ratios at reasonable bounds
                cleaned_df[col] = cleaned_df[col].clip(-10, 10)
        
        # 5. Remove rows with remaining critical missing values
        critical_cols = ['close', 'volume', 'timestamp']
        existing_critical_cols = [col for col in critical_cols if col in cleaned_df.columns]
        
        if existing_critical_cols:
            before_len = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=existing_critical_cols)
            after_len = len(cleaned_df)
            
            if before_len != after_len:
                logger.info(f"📉 Removed {before_len - after_len} rows with missing critical values")
        
        # 6. Final validation
        logger.info("✅ Final validation...")
        
        # Check for remaining infinite values
        inf_check = np.isinf(cleaned_df.select_dtypes(include=[np.number])).any().any()
        if inf_check:
            logger.warning("⚠️ Infinite values still present after cleaning")
            # Force replace any remaining infinities
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], cleaned_df[col].median())
        
        # Check for extremely large values
        for col in numeric_cols:
            if col in cleaned_df.columns:
                max_val = cleaned_df[col].max()
                if max_val > 1e10:  # Arbitrarily large threshold
                    logger.warning(f"⚠️ Very large values in {col}: max = {max_val}")
                    cleaned_df[col] = cleaned_df[col].clip(upper=1e6)  # Cap at 1 million
        
        logger.info(f"✅ Data cleaning completed: {len(cleaned_df)} records remaining")
        return cleaned_df
    
    def validate_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Validate and prepare features for ML training
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Tuple of (cleaned_X, cleaned_y)
        """
        logger.info(f"🔧 Validating features for ML training...")
        
        # Ensure no infinite or extremely large values in features
        X_clean = X.copy()
        
        # Replace any remaining problematic values
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Replace infinite values
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with median
            X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            
            # Cap extreme values
            std_val = X_clean[col].std()
            mean_val = X_clean[col].mean()
            
            if std_val > 0:
                # Cap at 5 standard deviations
                lower_cap = mean_val - 5 * std_val
                upper_cap = mean_val + 5 * std_val
                X_clean[col] = X_clean[col].clip(lower_cap, upper_cap)
        
        # Ensure target is valid
        y_clean = y.copy()
        if pd.api.types.is_numeric_dtype(y_clean):
            y_clean = y_clean.fillna(0)  # Fill missing targets with 0
        
        # Remove rows where either X or y is problematic
        valid_mask = ~(X_clean.isna().all(axis=1) | pd.isna(y_clean))
        
        X_final = X_clean[valid_mask]
        y_final = y_clean[valid_mask]
        
        logger.info(f"✅ Feature validation completed: {len(X_final)} samples ready for training")
        
        return X_final, y_final
