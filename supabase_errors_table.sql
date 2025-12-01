-- Add Errors Table to CRYPTIX-ML Supabase Database
-- Run this in Supabase SQL Editor to add error logging capability

-- 1. Errors Table (Error and exception history)
CREATE TABLE IF NOT EXISTS errors (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    error_type VARCHAR(50) NOT NULL,
    error_message TEXT NOT NULL,
    function_name VARCHAR(100) DEFAULT '',
    severity VARCHAR(20) DEFAULT 'ERROR' CHECK (severity IN ('ERROR', 'WARNING', 'CRITICAL', 'INFO')),
    bot_status BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_errors_timestamp ON errors(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_errors_severity ON errors(severity);
CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(error_type);
CREATE INDEX IF NOT EXISTS idx_errors_function ON errors(function_name);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE errors ENABLE ROW LEVEL SECURITY;

-- 4. Create Policy (Allow service_role access)
CREATE POLICY "Service role can do everything on errors" ON errors
    FOR ALL USING (auth.role() = 'service_role');

-- 5. Grant access to authenticated users (optional, for reading)
CREATE POLICY "Authenticated users can read errors" ON errors
    FOR SELECT USING (auth.role() = 'authenticated');

-- Success message
SELECT 'Errors table created successfully!' as message;
