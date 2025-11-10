# ...existing code...
import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "config.db"

def create_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT NOT NULL,
        obs TEXT,
        creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_modification TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS tools (
        name TEXT PRIMARY KEY,
        url TEXT,
        description TEXT,
        creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_modification TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS collection (
        name TEXT PRIMARY KEY,
        description TEXT,
        creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_modification TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS config (
        name TEXT PRIMARY KEY,
        value TEXT DEFAULT '',
        creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_modification TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

def seed_config(conn: sqlite3.Connection):
    cur = conn.cursor()
    entries = [
        ("openai_api_key", "placeholder"),
        ("openai_baseurl", "http://10.246.47.184:10000/v1"),
        ("model", "qwen3:14b"),
        ("tools_config", ""),
        ("agent_name", "teste"),
        ("retrieval_function", "sentence_window_retrieval")
    ]
    cur.executemany("INSERT OR IGNORE INTO config (name, value) VALUES (?, ?);", entries)
    conn.commit()

def main():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    create_tables(conn)
    seed_config(conn)
    conn.close()
    print(f"SQLite DB created at: {DB_FILE}")

main()
