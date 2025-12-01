-- CRYPTIX-ML Supabase Database Schema
-- Run this in Supabase SQL Editor

-- 1. Trades Table (Individual trade records)
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) DEFAULT 'bot' CHECK (source IN ('bot', 'binance_history', 'manual', 'positions_json_import')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Positions Table (Current aggregated positions)
CREATE TABLE IF NOT EXISTS positions (
    symbol VARCHAR(20) PRIMARY KEY,
    quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,
    avg_buy_price DECIMAL(20, 8) NOT NULL DEFAULT 0,
    total_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Configuration Table (Bot settings)
CREATE TABLE IF NOT EXISTS configuration (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Migration Status (Track data imports)
CREATE TABLE IF NOT EXISTS migration_status (
    migration_name VARCHAR(100) PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'pending',
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

-- 5. Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp DESC);

-- 6. Enable Row Level Security (RLS)
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration ENABLE ROW LEVEL SECURITY;
ALTER TABLE migration_status ENABLE ROW LEVEL SECURITY;

-- 7. Create Policies (Allow service_role access)
CREATE POLICY "Service role can do everything on trades" ON trades
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything on positions" ON positions
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything on configuration" ON configuration
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything on migration_status" ON migration_status
    FOR ALL USING (auth.role() = 'service_role');

-- 8. Create Functions for Position Calculation
CREATE OR REPLACE FUNCTION calculate_position(p_symbol VARCHAR(20))
RETURNS TABLE(
    symbol VARCHAR(20),
    quantity DECIMAL(20, 8),
    avg_buy_price DECIMAL(20, 8),
    total_cost DECIMAL(20, 8)
) AS $$
DECLARE
    total_bought DECIMAL(20, 8) := 0;
    total_sold DECIMAL(20, 8) := 0;
    weighted_cost DECIMAL(20, 8) := 0;
    final_quantity DECIMAL(20, 8);
    final_avg_price DECIMAL(20, 8);
    final_cost DECIMAL(20, 8);
BEGIN
    -- Calculate totals from trades
    SELECT 
        COALESCE(SUM(CASE WHEN action = 'BUY' THEN quantity ELSE 0 END), 0),
        COALESCE(SUM(CASE WHEN action = 'SELL' THEN quantity ELSE 0 END), 0),
        COALESCE(SUM(CASE WHEN action = 'BUY' THEN quantity * price ELSE 0 END), 0)
    INTO total_bought, total_sold, weighted_cost
    FROM trades 
    WHERE trades.symbol = p_symbol;
    
    final_quantity := total_bought - total_sold;
    
    IF final_quantity > 0 AND total_bought > 0 THEN
        final_avg_price := weighted_cost / total_bought;
        final_cost := final_quantity * final_avg_price;
    ELSE
        final_avg_price := 0;
        final_cost := 0;
    END IF;
    
    RETURN QUERY SELECT 
        p_symbol,
        final_quantity,
        final_avg_price,
        final_cost;
END;
$$ LANGUAGE plpgsql;

-- 9. Create Trigger to Auto-Update Positions
CREATE OR REPLACE FUNCTION update_position_on_trade()
RETURNS TRIGGER AS $$
DECLARE
    pos_record RECORD;
BEGIN
    -- Calculate new position
    SELECT * INTO pos_record FROM calculate_position(NEW.symbol);
    
    -- Upsert position
    INSERT INTO positions (symbol, quantity, avg_buy_price, total_cost, updated_at)
    VALUES (pos_record.symbol, pos_record.quantity, pos_record.avg_buy_price, pos_record.total_cost, NOW())
    ON CONFLICT (symbol) 
    DO UPDATE SET 
        quantity = EXCLUDED.quantity,
        avg_buy_price = EXCLUDED.avg_buy_price,
        total_cost = EXCLUDED.total_cost,
        updated_at = EXCLUDED.updated_at;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger
DROP TRIGGER IF EXISTS trigger_update_position ON trades;
CREATE TRIGGER trigger_update_position
    AFTER INSERT ON trades
    FOR EACH ROW
    EXECUTE FUNCTION update_position_on_trade();

-- 10. Insert Initial Configuration
INSERT INTO configuration (key, value) VALUES 
    ('last_sync', NOW()::TEXT),
    ('system_status', 'active')
ON CONFLICT (key) DO NOTHING;

-- 11. Insert Migration Status
INSERT INTO migration_status (migration_name, status) VALUES 
    ('binance_historical_import', 'pending'),
    ('positions_json_import', 'pending')
ON CONFLICT (migration_name) DO NOTHING;