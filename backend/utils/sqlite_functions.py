import os
import sqlite3
import json
from dotenv import load_dotenv

load_dotenv(".env")
db_path = os.getenv("SQLITE_PATH")

def connect():
    """Conecta ao banco de dados SQLite e retorna a conexão."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row # Permite acessar colunas por nome
    return conn

def create_tables(conn: sqlite3.Connection):
    """Cria as tabelas 'collections' e 'documents' se não existirem."""
    cursor = conn.cursor()
    
    # Tabela de Coleções
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        index_method TEXT NOT NULL,
        index_params TEXT NOT NULL 
    );
    """)
    
    # Tabela de Documentos (PDFs)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id INTEGER NOT NULL,
        pdf_name TEXT NOT NULL,
        FOREIGN KEY (collection_id) REFERENCES collections (id) ON DELETE CASCADE
    );
    """)
    conn.commit()

# --- Funções de Coleção ---

def add_collection(conn: sqlite3.Connection, name: str, index_method: str, index_params: dict):
    """Adiciona uma nova coleção ao banco de dados."""
    params_json = json.dumps(index_params)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO collections (name, index_method, index_params) VALUES (?, ?, ?)",
            (name, index_method, params_json)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError(f"Collection com o nome '{name}' já existe.")

def delete_collection(conn: sqlite3.Connection, name: str):
    """Remove uma coleção e seus documentos associados do SQLite."""
    cursor = conn.cursor()
    # O "ON DELETE CASCADE" na criação da tabela cuida de deletar os documentos.
    cursor.execute("DELETE FROM collections WHERE name = ?", (name,))
    conn.commit()

def get_all_collections_names(conn: sqlite3.Connection) -> list:
    """Retorna uma lista com os nomes de todas as coleções."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM collections ORDER BY name")
    rows = cursor.fetchall()
    return [row['name'] for row in rows]

def get_collection_details(conn: sqlite3.Connection, name: str) -> dict:
    """Busca os detalhes de uma coleção específica pelo nome."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM collections WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        details = dict(row)
        details['index_params'] = json.loads(details['index_params']) # Converte JSON de volta para dict
        return details
    return None

# --- Funções de Documento ---

def add_pdf_to_collection(conn: sqlite3.Connection, collection_name: str, pdf_name: str):
    """Associa um nome de PDF a uma coleção."""
    cursor = conn.cursor()
    # Primeiro, pega o ID da coleção
    cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
    collection_row = cursor.fetchone()
    if not collection_row:
        raise ValueError(f"Coleção '{collection_name}' não encontrada.")
    collection_id = collection_row['id']
    
    # Insere o documento
    cursor.execute(
        "INSERT INTO documents (collection_id, pdf_name) VALUES (?, ?)",
        (collection_id, pdf_name)
    )
    conn.commit()

def get_pdfs_for_collection(conn: sqlite3.Connection, collection_name: str) -> list:
    """Retorna uma lista de nomes de PDFs para uma dada coleção."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.pdf_name 
        FROM documents d
        JOIN collections c ON d.collection_id = c.id
        WHERE c.name = ?
    """, (collection_name,))
    rows = cursor.fetchall()
    return [row['pdf_name'] for row in rows]

def check_pdf_exists(conn: sqlite3.Connection, collection_name: str, pdf_name: str) -> bool:
    """Verifica se um PDF já existe em uma coleção."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 1 
        FROM documents d
        JOIN collections c ON d.collection_id = c.id
        WHERE c.name = ? AND d.pdf_name = ?
    """, (collection_name, pdf_name))
    return cursor.fetchone() is not None