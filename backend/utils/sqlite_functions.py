import sqlite3

def get_config(variable_name: str) -> str:
  """Gets a config varable from the database"""
  try:
    conn = sqlite3.connect("config.db")
    cur = conn.cursor()
    cur.execute(f"SELECT value FROM config WHERE name = {variable_name}")
    row = cur.fetchone()
    if row is None:
      raise ValueError(f"No variable with the name {variable_name}")
    return row[0]
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()

def get_tools():
# espera uma lista de dicionários com "name" e "url"
  """Getsthe tools config """
  try:
    conn = sqlite3.connect("config.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM tools")
    row = cur.fetchone()
    if row is None:
      raise ValueError("No tools configured")
    return row[0]
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()

def get_prompt(id:int = 0 )->str:
  """gets a prompt, if no args are passed it takes the first"""
  try:
    conn = sqlite3.connect("config.db")
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM prompts where id = {id}")
    row = cur.fetchone()
    if row is None:
      raise ValueError("No prompts configured")
    return row[0]
  except sqlite3.Error as e:
    raise RuntimeError(f"Database error: {e}")
  finally:
    conn.close()
