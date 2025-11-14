import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(".env")
db_path = os.getenv("SQLITE_PATH")

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

    CREATE TABLE IF NOT EXISTS collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        index_method TEXT NOT NULL,
        index_params TEXT NOT NULL,
        pdf_name TEXT
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
        ("openai_api_key",),
        ("openai_baseurl",),
        ("model",),
        ("agent_name",),
        ("retrieval_function",)
    ]
    cur.executemany("INSERT OR IGNORE INTO config (name) VALUES (?);", entries)
    conn.commit()

def main():
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    seed_config(conn)
    conn.close()
    print(f"SQLite DB created at: {db_path}")

main()
