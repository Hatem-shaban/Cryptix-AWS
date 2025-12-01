# ML Training Best Practices - CRYPTIX-ML

## ✅ DO THIS

### Production-Ready Models (Session 74 - CURRENT STATE)
Your models are **already optimized** with 20 years of historical data:
- **SIGNAL: 84.96% accuracy** - Excellent for production trading
- **REGIME: 71.68% accuracy** - Strong market regime detection  
- **TREND: 50.23% accuracy** - Baseline (expected for trend prediction)
- **Total samples: 3.5M+** records from real Binance data

### When to Retrain
Only retrain models in these scenarios:
1. **Monthly major retraining** - Schedule a full batch training once per month
2. **Major market regime change** - Black swan events, crashes, regulatory changes
3. **Persistent prediction failures** - If models consistently fail for 1+ week
4. **New trading pairs** - Adding symbols not in original 20-year dataset

### Safe Training Workflow
```bash
# 1. ALWAYS backup current models first
Copy-Item models\*_model.pkl models\backup\  

# 2. Use incremental learning ONLY for large datasets (100K+ samples)
python enhanced_ml_training.py  # Only if you have massive new data

# 3. Verify results before deploying
# Check that accuracy IMPROVED, not decreased
```

## ❌ DON'T DO THIS

### Never Use Regular Training on Well-Trained Models
**What happened on Oct 29, 2025:**
- Session 74: SIGNAL 84.96%, REGIME 71.68% (EXCELLENT)
- Ran `enhanced_ml_training.py` with only 37K recent samples (90 days)
- Session 75: SIGNAL 72.0% (-13%), REGIME 42.7% (-29%) ⚠️ **DISASTER**

### Why It Failed
1. **Data imbalance**: 37K new samples vs 3.5M existing samples (1% of total)
2. **Recent bias**: 90-day data represents current market, not 20-year average
3. **Regime conflict**: Recent market conditions differ from historical patterns
4. **Dilution effect**: Small recent data "pollutes" well-established patterns

### The Math
```
Before: 3,563,734 samples (20 years) → 84.96% accuracy
Added:     37,772 samples (90 days)  → Different patterns
Result: Model confused by conflicting signals → 72.0% accuracy

REGIME model drop (71.7% → 42.7%):
- 42.7% is WORSE than random guessing (33.3% for 3 classes)
- Indicates severe pattern conflict
```

## 🎯 Best Practices Summary

### For Daily/Weekly Operations
- **DO NOT retrain models**
- Use existing Session 74 models as-is
- Monitor performance via trading results
- Only intervene if consistent failures occur

### For Monthly Maintenance
1. Download large historical dataset (1M+ samples, multiple years)
2. Backup current models: `Copy-Item models\*_model.pkl models\backup_YYYY_MM_DD\`
3. Train with incremental learning: `python enhanced_ml_training.py`
4. Verify accuracy IMPROVED across all models
5. If worse, restore backup: `Copy-Item models\backup_YYYY_MM_DD\*.pkl models\`

### For Emergency Recovery
```powershell
# Restore from specific version (Session 74 example)
Copy-Item models\trend_v74.pkl models\trend_model.pkl -Force
Copy-Item models\signal_v74.pkl models\signal_model.pkl -Force
Copy-Item models\regime_v74.pkl models\regime_model.pkl -Force

# Then manually edit incremental_training_stats.json
# Set training_sessions back to 74 and remove last entry
```

## 📊 Model Version History
- **Session 1-4**: Initial incremental setup (151K samples)
- **Session 5-42**: First 10-year training (1.98M samples)
- **Session 43-74**: Second 10-year training (3.5M total, 20 years) ✅ **OPTIMAL**
- **Session 75**: Bad regular training (37K recent data) ❌ **REVERTED**

## 🚨 Red Flags - When NOT to Deploy
- Accuracy decrease of >5% on any model
- REGIME model below 60% (indicates random performance)
- SIGNAL model below 75% (too unreliable for trading)
- Training on <100K samples when model already has millions

## 📝 Current Status (Restored Session 74)
```
✅ TREND:  3,563,734 samples | Session 74 | 50.23% accuracy
✅ SIGNAL: 3,563,734 samples | Session 74 | 84.96% accuracy  
✅ REGIME: 3,591,094 samples | Session 74 | 71.68% accuracy
```

**These models are production-ready. Do not retrain unless absolutely necessary.**
