-- Add Signals Table to CRYPTIX-ML Supabase Database
-- Run this in Supabase SQL Editor to add signal logging capability

-- 1. Signals Table (Trading signal history)
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    signal VARCHAR(10) NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    rsi DECIMAL(10, 2) DEFAULT 0,
    macd DECIMAL(20, 8) DEFAULT 0,
    macd_trend VARCHAR(20) DEFAULT '',
    sentiment VARCHAR(20) DEFAULT '',
    sma5 DECIMAL(20, 8) DEFAULT 0,
    sma20 DECIMAL(20, 8) DEFAULT 0,
    reason TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_signal ON signals(signal);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;

-- 4. Create Policy (Allow service_role access)
CREATE POLICY "Service role can do everything on signals" ON signals
    FOR ALL USING (auth.role() = 'service_role');

-- 5. Grant access to authenticated users (optional, for reading)
CREATE POLICY "Authenticated users can read signals" ON signals
    FOR SELECT USING (auth.role() = 'authenticated');

-- Success message
SELECT 'Signals table created successfully!' as message;
