import os
import sqlite3
import json
from dotenv import load_dotenv

load_dotenv(".env")
db_path = os.getenv("SQLITE_PATH")

'''def create_table_if_not_exists(db_path: str):
    """
    Cria a tabela 'collections' se ela ainda não existir.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        index_method TEXT NOT NULL,
        index_params TEXT NOT NULL,
        pdf_name TEXT
    );
    """)

    conn.commit()
    conn.close()
'''

def create_collection(db_path: str, name: str, index_method: str, index_params: dict):
    """
    Cria uma nova collection, sem pdf_name ainda.
    Se já existir (mesmo nome), não recria.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    INSERT OR IGNORE INTO collections (name, index_method, index_params)
    VALUES (?, ?, ?)
    """, (name, index_method, json.dumps(index_params)))

    conn.commit()
    conn.close()


def add_pdf_to_collection(db_path: str, collection_name: str, pdf_name: str):
    """
    Adiciona um PDF à lista existente da collection.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Busca a lista atual de PDFs
    cur.execute("SELECT pdf_name FROM collections WHERE name = ?", (collection_name,))
    row = cur.fetchone()

    # Converte JSON → lista
    pdf_list = json.loads(row[0]) if row[0] else []

    # Adiciona o novo PDF se ainda não existir
    if pdf_name not in pdf_list:
        pdf_list.append(pdf_name)

    # Salva de volta no banco
    cur.execute("""
    UPDATE collections
    SET pdf_name = ?
    WHERE name = ?
    """, (json.dumps(pdf_list), collection_name))

    conn.commit()
    conn.close()



def delete_collection(db_path: str, collection_name: str):
    """
    Remove a linha inteira da tabela 'collections' com o nome especificado.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DELETE FROM collections WHERE name = ?", (collection_name,))
    conn.commit()
    conn.close()



def list_collections(db_path: str):
    """
    Retorna uma lista apenas com os nomes das collections existentes.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM collections")
    rows = cur.fetchall()
    conn.close()

    # retorna lista simples de strings
    return [row[0] for row in rows]


def get_collection_params(db_path: str, collection_name: str):
    """
    Retorna os parâmetros e PDFs de uma collection específica.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    SELECT index_method, index_params, pdf_name
    FROM collections
    WHERE name = ?
    """, (collection_name,))

    row = cur.fetchone()
    conn.close()

    if not row:
        print(f"⚠️ Collection '{collection_name}' não encontrada.")
        return None

    index_method, index_params, pdf_name = row
    return {
        "index_method": index_method,
        "index_params": json.loads(index_params),
        "pdfs": json.loads(pdf_name) if pdf_name else []
    }
