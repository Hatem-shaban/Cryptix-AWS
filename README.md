# CRYPTIX-ML Trading Bot

An intelligent cryptocurrency trading bot with machine learning capabilities, technical analysis, automated risk management, and **automated ML model training**.

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd CRYPTIX-ML
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

3. **Test Configuration**
   ```bash
   python test_bot.py
   ```

4. **Run the Bot**
   ```bash
   python web_bot.py
   ```

## ✨ New Feature: Incremental ML Learning

The bot now includes **true incremental/cumulative ML learning** that accumulates knowledge over time:

- **📈 Knowledge Accumulation**: Models build upon previous training instead of replacing it
- **� Efficient Updates**: Fast incremental training (~45 seconds vs 8 minutes for batch)
- **📊 Performance Tracking**: Monitor accuracy improvements over time with learning curves
- **🎯 Version Control**: Rollback to previous versions if needed
- **🤖 Smart Mode Selection**: Automatically chooses batch or incremental based on context
- **☁️ Render Optimized**: Lightweight design perfect for cloud deployment

### New Capabilities:
```python
# Train with incremental learning (default)
python enhanced_ml_training.py

# View cumulative statistics
python incremental_learning.py

# Migrate existing models
python migrate_to_incremental.py

# Force batch retraining (monthly refresh)
trainer.train_all_models(force_batch=True)
```

**Key Benefits:**
- Models improve continuously with each training session
- Total samples accumulate (e.g., 30k → 32k → 34k → ...)
- Training history fully tracked with versioning
- 80-90% faster updates compared to batch retraining

📚 **Full Documentation**: See [INCREMENTAL_LEARNING.md](INCREMENTAL_LEARNING.md) for complete guide

## 📋 Prerequisites

- Python 3.8+
- Binance account with API access
- Telegram bot (optional, for notifications)

## 🔧 Configuration

For detailed configuration instructions, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).

### Quick Configuration Steps:

1. **Get Binance API Keys:**
   - Go to Binance → Account → API Management
   - Create new API key with "Spot & Margin Trading" permission
   - Add keys to your `.env` file

2. **Set Up Telegram (Optional):**
   - Message @BotFather to create a bot
   - Get your chat ID from @userinfobot
   - Add credentials to `.env` file

3. **Choose Trading Mode:**
   - Set `USE_TESTNET=true` for safe testing
   - Set `USE_TESTNET=false` for live trading

## 🎯 Features

- **Advanced Technical Analysis**: RSI, MACD, EMA, Stochastic, ADX, VWAP
- **Adaptive Trading Strategy**: Smart strategy that adapts to market conditions
- **Risk Management**: Position sizing, stop-loss, daily limits
- **Machine Learning**: Price trend prediction models
- **Telegram Notifications**: Real-time trade alerts and performance updates
- **Web Interface**: Monitor and control the bot via web dashboard
- **Intelligent Timing**: Market regime detection and optimal entry timing

## 📊 Trading Strategy

### Adaptive Strategy
- Composite scoring system
- Volatility adjustment
- Trend following capabilities
- Market regime detection
- Dynamic threshold adjustment

## 🛡️ Risk Management

- **Position Sizing**: Based on account balance and volatility
- **Daily Limits**: Maximum daily loss and consecutive losses
- **Portfolio Exposure**: Maximum percentage of portfolio at risk
- **Stop Loss**: Automated loss protection
- **Drawdown Protection**: Automatic trading halt on excessive losses

## 📁 Project Structure

```
CRYPTIX-ML/
├── config.py              # Main configuration
├── model_paths.py         # Centralized ML model path management
├── web_bot.py             # Main trading bot
├── ml_predictor.py        # Enhanced ML prediction engine
├── market_intelligence.py # Market analysis and regime detection
├── enhanced_ml_training.py # Advanced ML training with real market data
├── enhanced_historical_data.py # Comprehensive Binance data fetcher
├── data_cleaner.py        # Data validation and cleaning
├── telegram_notify.py     # Telegram notifications
├── .env.example          # Environment template
├── requirements.txt      # Python dependencies
├── models/               # Trained ML models, scalers, and selectors
│   ├── README.md         # Model directory documentation
│   ├── *.pkl             # Trained ML models and components
│   └── ...
├── logs/                 # Trading logs and history
└── docs/                 # Documentation
```

## 🧪 Testing

Always test your configuration before live trading:

```bash
# Test basic configuration
python test_bot.py

# Test with paper trading (testnet)
# Set USE_TESTNET=true in .env
python web_bot.py
```

## 📈 Monitoring

The bot provides multiple ways to monitor performance:

1. **Web Dashboard**: Access at `http://localhost:5000` when running
2. **Telegram Notifications**: Real-time updates on trades and performance
3. **Log Files**: Detailed logs in the `logs/` directory
4. **Performance Metrics**: Win rate, profit factor, and more

## ⚠️ Important Warnings

- **Start with Testnet**: Always begin with `USE_TESTNET=true`
- **Never Share API Keys**: Keep your `.env` file secure
- **Monitor Closely**: Especially during initial runs
- **Paper Trade First**: Test strategies thoroughly before live trading
- **Risk Management**: Never risk more than you can afford to lose

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Use at your own risk.

## 🆘 Support

- Check [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for configuration help
- Run `python test_bot.py` to validate your setup
- Review logs in the `logs/` directory for debugging
- Ensure all requirements are installed: `pip install -r requirements.txt`

---

**Disclaimer**: Cryptocurrency trading involves substantial risk. This bot is provided for educational purposes only. Always do your own research and never invest more than you can afford to lose.