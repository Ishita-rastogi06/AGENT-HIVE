-- ══════════════════════════════════════════════════════════════════════════════
-- AgentHive — PostgreSQL Database Initialization & Seed Script
-- Auto-loaded by docker-entrypoint-initdb.d or native init script.
-- ══════════════════════════════════════════════════════════════════════════════

-- Create users table for authentication & session identity
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create audit_logs table (referenced by Phase 3/Phase 5 demo scenario)
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Seed initial system event
INSERT INTO audit_logs (event_type, description)
VALUES ('system_init', 'Database initialized successfully with seed schema');

-- Create customers table for data specialist analysis
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    account_balance NUMERIC(10, 2) DEFAULT 0.00,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Seed sample customer rows
INSERT INTO customers (name, email, status, account_balance) VALUES
    ('Alice Smith', 'alice@example.com', 'active', 1250.50),
    ('Bob Jones', 'bob@example.com', 'active', 450.00),
    ('Charlie Brown', 'charlie@example.com', 'pending', 0.00),
    ('Diana Prince', 'diana@example.com', 'active', 8900.25)
ON CONFLICT (email) DO NOTHING;

-- Create orders table for relational join queries
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    amount NUMERIC(10, 2) NOT NULL,
    status VARCHAR(50) DEFAULT 'completed',
    order_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Seed sample order rows
INSERT INTO orders (customer_id, amount, status) VALUES
    (1, 299.99, 'completed'),
    (1, 150.00, 'completed'),
    (2, 450.00, 'processing'),
    (4, 1200.50, 'completed')
ON CONFLICT DO NOTHING;
