-- Enhanced Database Schema with Fixed VARCHAR Lengths
CREATE TABLE IF NOT EXISTS tle_data (
    id SERIAL PRIMARY KEY,
    satellite_name VARCHAR(255) NOT NULL,  -- Increased from 100
    line1 VARCHAR(69) NOT NULL,            -- Exact TLE line length
    line2 VARCHAR(69) NOT NULL,            -- Exact TLE line length
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Prevent duplicate TLE entries
    CONSTRAINT unique_tle UNIQUE (satellite_name, line1, line2)
);

-- Optional: Satellite metadata table
CREATE TABLE IF NOT EXISTS satellites (
    id SERIAL PRIMARY KEY,
    satellite_name VARCHAR(255) NOT NULL UNIQUE,
    norad_id VARCHAR(10),
    object_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_satellite_name ON tle_data(satellite_name);
CREATE INDEX IF NOT EXISTS idx_timestamp ON tle_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sat_timestamp ON tle_data(satellite_name, timestamp DESC);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;

-- View for statistics
CREATE OR REPLACE VIEW satellite_stats AS
SELECT 
    satellite_name,
    COUNT(*) as total_records,
    MIN(timestamp) as first_seen,
    MAX(timestamp) as last_seen
FROM tle_data
GROUP BY satellite_name
ORDER BY total_records DESC;

GRANT SELECT ON satellite_stats TO admin;