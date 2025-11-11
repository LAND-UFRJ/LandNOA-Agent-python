import sqlite3
import datetime

def get_config(variable_name: str) -> str:
  """Gets a config varable from the database"""
  try:
    conn = sqlite3.connect("config.db")
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

#TODO check nisso
def get_tools():
# espera uma lista de dicionários com "name" e "url"
  """Getsthe tools config """
  try:
    conn = sqlite3.connect("config.db")
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

def add_tool_sqlite(name,url,description)->None:
  """Adds a tool to a database"""  
  try:
    conn = sqlite3.connect("config.db")
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

def remove_tool_sql(name:str)->None:
  """removes a tool from the database"""
  try:
    conn = sqlite3.connect("config.db")
    cur = conn.cursor()
    cur.execute("""DELETE FROM tools where name = ?""",(name))
    conn.commit()
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()

def get_prompt(prompt_id:int = 0 )->str:
  """gets a prompt, if no args are passed it takes the first"""
  try:
    conn = sqlite3.connect("config.db")
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

def get_rag_tool() -> str:
  """Returns the name of the RAG function"""
  try:
    conn = sqlite3.connect("config.db")
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
