import os
import sqlite3
import json
import datetime


db_path = "config.db"


# Functions to manage config table

def get_config_sqlite(variable_name: str) -> str:
  """Gets a config variable from the name column"""
  try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT value FROM config WHERE name = ?",(variable_name,))
    row = cur.fetchone()
    if row is None:
      raise ValueError(f"No variable with the name {variable_name}")
    return row[0]
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()


def get_tools_sqlite():
  """Getsthe tools config """
  try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM tools")
    row = cur.fetchone()
    if row is None:
      return None
    return row[0]
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()

def get_rag_tool_sqlite() -> str:
  """Returns the name of the RAG function"""
  try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT value FROM config WHERE name = ?", ('retrieval_function',))
    row = cur.fetchone()
    if row is None:
      raise ValueError("No prompts configured")
    return row[0]
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()

def update_config_sqlite(name:str, new_value: str):
    """
    Change the 'value' column of a especific name.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    UPDATE config
    SET value = ?
    WHERE name = ?
    """,(new_value,name))

    conn.commit()
    conn.close()

# Functions to manage tools table

def list_tools_sqlite():
    """Return a list off dictionaries with all the tools """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM tools")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {e}")
    finally:
        conn.close()

def add_tool_sqlite(name:str,url,description)->None:
  """Adds a tool to a database"""  
  try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""INSERT into tools (name, url, description, creation_date, last_modification)
                   VALUES (?, ?, ?, ?, ?) """,(name,
                                               url,
                                               description,
                                               datetime.datetime.now(),
                                               datetime.datetime.now()))
    conn.commit()
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()

def update_tool_sqlite(original_name:str, new_url:str, new_description:str):
    """Change a existing tool"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            UPDATE tools 
            SET url = ?, description = ?, last_modification = ?
            WHERE name = ?
        """, (new_url, new_description, datetime.datetime.now(), original_name))
        conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {e}")
    finally:
        conn.close()

def remove_tool_sqlite(name:str)->None:
  """removes a tool from the database"""
  try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""DELETE FROM tools where name = ?""",(name))
    conn.commit()
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()


# Functions to manage prompts table
 
def get_prompt_sqlite(prompt_id:int = 0 )->str:
  """gets a prompt, if no args are passed it takes the first"""
  try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM prompts where id = ?",(prompt_id,))
    row = cur.fetchone()
    if row is None:
      return None
    return row[0]
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()
    
def list_prompts_sqlite():
    """
    Retorna uma lista com TODOS os prompts para preencher o selectbox.
    Retorna uma lista de dicionários (ex: [{'id': 1, 'obs': 'Persona Financeira', ...}])
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row # Importante para acessar por nome da coluna
        cur = conn.cursor()
        cur.execute("SELECT * FROM prompts ORDER BY id ASC")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {e}")
    finally:
        conn.close()

def add_prompt_sqlite(obs:str, prompt_text:str):
    """Cria um novo prompt no banco."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prompts (obs, prompt, creation_date, last_modification)
            VALUES (?, ?, ?, ?)
        """, (obs, prompt_text, datetime.datetime.now(), datetime.datetime.now()))
        conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {e}")
    finally:
        conn.close()

def update_prompt_sqlite(prompt_id:int, new_obs:str, new_prompt_text:str):
    """Atualiza o texto ou o nome (obs) de um prompt existente pelo ID."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            UPDATE prompts 
            SET obs = ?, prompt = ?, last_modification = ?
            WHERE id = ?
        """, (new_obs, new_prompt_text, datetime.datetime.now(), prompt_id))
        conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {e}")
    finally:
        conn.close()

def delete_prompt_sqlite(prompt_id:int):
    """Remove um prompt do banco pelo ID."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
        conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {e}")
    finally:
        conn.close()
#Functions to manage collections table

def create_collection_sqlite(name: str, index_method: str, index_params: dict):
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


def add_pdf_to_collection_sqlite(collection_name: str, pdf_name: str):
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



def delete_collection_sqlite(collection_name: str):
    """
    Remove a linha inteira da tabela 'collections' com o nome especificado.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DELETE FROM collections WHERE name = ?", (collection_name,))
    conn.commit()
    conn.close()



def list_collections_sqlite():
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


def get_collection_params_sqlite(collection_name: str):
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
        print(f"Collection '{collection_name}' não encontrada.")
        return None

    index_method, index_params, pdf_name = row
    return {
        "index_method": index_method,
        "index_params": json.loads(index_params),
        "pdfs": json.loads(pdf_name) if pdf_name else []
    }
