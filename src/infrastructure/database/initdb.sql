-- SigmaForge - DuckDB Schema + Seed Data
-- Single source of truth for database initialization.
-- All CREATE TABLE are IF NOT EXISTS and INSERT are OR IGNORE for idempotency.

-- =========================================================================
-- SCHEMA
-- =========================================================================

-- config
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- system_prompts
CREATE TABLE IF NOT EXISTS system_prompts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    content TEXT NOT NULL,
    is_active BOOLEAN DEFAULT FALSE
);

-- models
CREATE TABLE IF NOT EXISTS models (
    repo_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    local_path TEXT,
    file_size BIGINT DEFAULT 0,
    status TEXT DEFAULT 'ready',
    dimension INTEGER,
    index_path TEXT,
    files TEXT,
    updated_at TEXT
);

-- doc_registry (unified document registry for all sources: GitHub, local, sigmaref)
CREATE TABLE IF NOT EXISTS doc_registry (
    url_hash TEXT PRIMARY KEY,
    org TEXT,
    repo TEXT,
    content_type TEXT,
    file_name TEXT,
    content_sha256 TEXT,
    file_size BIGINT,
    original_url TEXT NOT NULL,
    normalized_url TEXT,
    rule_id TEXT DEFAULT '00000000-0000-0000-0000-000000000000',
    title TEXT,
    timestamp TEXT,
    last_seen TEXT,
    embed_status TEXT DEFAULT 'discovery'
);

-- git_metadata
CREATE TABLE IF NOT EXISTS git_metadata (
    repo_key TEXT PRIMARY KEY,
    org TEXT NOT NULL,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    branch TEXT NOT NULL
);

-- git_selected_dirs
CREATE TABLE IF NOT EXISTS git_selected_dirs (
    repo_key TEXT NOT NULL,
    dir_path TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT '',
    updated TEXT,
    PRIMARY KEY (repo_key, dir_path)
);

-- sigma_spec (dedicated spec document table — multi-repo, org/repo aware)
CREATE TABLE IF NOT EXISTS sigma_spec (
    url_hash TEXT PRIMARY KEY,
    org TEXT,
    repo TEXT,
    content_type TEXT,
    file_name TEXT NOT NULL,
    content_sha256 TEXT,
    file_size BIGINT,
    original_url TEXT NOT NULL,
    normalized_url TEXT,
    title TEXT,
    timestamp TEXT,
    last_seen TEXT,
    embed_status TEXT DEFAULT 'discovery'
);

CREATE INDEX IF NOT EXISTS idx_sigma_spec_embed ON sigma_spec(embed_status);
CREATE INDEX IF NOT EXISTS idx_sigma_spec_org_repo ON sigma_spec(org, repo);

-- rule_references (junction table: rule_id ↔ reference url_hash — M:N)
CREATE TABLE IF NOT EXISTS rule_references (
    rule_id  TEXT NOT NULL,
    url_hash TEXT NOT NULL,
    ref_url  TEXT NOT NULL,
    created  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', current_timestamp)),
    PRIMARY KEY (rule_id, url_hash)
);

CREATE INDEX IF NOT EXISTS idx_rule_ref_url_hash ON rule_references(url_hash);

-- doc_error (failed URLs — 30x/40x errors to skip on retry)
CREATE TABLE IF NOT EXISTS doc_error (
    url_hash TEXT PRIMARY KEY,
    original_url TEXT NOT NULL,
    normalized_url TEXT NOT NULL,
    error_code INTEGER,
    error_message TEXT,
    org TEXT,
    repo TEXT,
    timestamp TEXT
);

-- worker_state
CREATE TABLE IF NOT EXISTS worker_state (
    worker_type TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'idle',
    current_task_id TEXT,
    progress_percent DOUBLE DEFAULT 0.0,
    current_file TEXT,
    last_heartbeat TEXT
);

-- release_cache (GitHub release tag cache — refreshed on demand)
CREATE TABLE IF NOT EXISTS release_cache (
    service TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

-- installed_versions (local binary versions — scanned on demand)
CREATE TABLE IF NOT EXISTS installed_versions (
    service TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    scanned_at TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_prompts_name ON system_prompts(name);
CREATE INDEX IF NOT EXISTS idx_doc_registry_embed ON doc_registry(embed_status);
CREATE INDEX IF NOT EXISTS idx_doc_registry_org_repo ON doc_registry(org, repo);

-- =========================================================================
-- SEED DATA
-- =========================================================================

-- Default app config
INSERT OR IGNORE INTO config (key, value) VALUES
    ('app_version', '"0.1.0"'),
    ('theme', '"dark"'),
    ('schema_version', '20260704');
