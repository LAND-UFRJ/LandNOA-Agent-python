import os
from pathlib import Path
from utils import agent_menager
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from chromadb import HttpClient
from utils.indexing import Splitter

load_dotenv()
#http:///.well-known/agent.json
#####################################################################################
#Onde parei: Endpoints para poder manipular (atualmente so penso em adicionar) coleções
#Onde parei: metricas (prometheus) de avaliação de do sistema (tokens, tempo ligado,requests recebidas)
#Possíveis amelhoramentos: Monitorar conversas (MongoDB), Validação do RAG, Fazer interface (que de para manipular RAG e parametros como um todo)
#Administrção de Sessão
#####################################################################################

AGENT_PATH = Path("agente/root_agent.py")

AGENT_SECRET_TOKEN = os.getenv("AGENT_SECRET_TOKEN")
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL")
FLASK_RUN_PORT = int(os.getenv("FLASK_RUN_PORT", '8010'))
chroma_url = os.getenv('CHROMADB_URL')
chroma_port = int(os.getenv('CHROMADB_PORT'))

app = Flask(__name__)

splitter = Splitter()

chroma_client = HttpClient(host=chroma_url,port=chroma_port)

def require_auth(f):
  """
  Decorator to enforce authentication on Flask route handlers.
  This decorator checks for the presence of an 'Authorization' header
  in the incoming request.
  If the header is missing or does not match the expected bearer
  token, it returns a 403 Forbidden
  response with an error message. Otherwise,
  it allows the wrapped function to execute.
  Args:
    f (function): The Flask route handler to be decorated.
  Returns:
    function: The decorated function that enforces authentication.
  """
  @wraps(f)
  def decorated_function(*args, **kwargs):
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
      return jsonify({"error": "Acesso não autorizado."}), 403
    return f(*args, **kwargs)
  return decorated_function

@app.route('/api/v1/health', methods=['GET'])
def health_check():
  """Endpoint para o Docker verificar se o AI Guide Agent está online."""
  return jsonify({"status": "ok"}), 200

@app.route('/api/v1/collections/list',methods=['GET'])
@require_auth
def list_collections():
  """List all available Chroma collections.

  Returns:
      dict: A dictionary with the result message and status code.
  """
  try:
    collections = chroma_client.list_collections()
    return {'result':f'The chroma Collections avaliable are {collections}'},200
  except Exception as e:
    return jsonify({"error":f"Error while listing chroma collections: {e}"}), 500


@app.route('/api/v1/collections/create',methods=['POST'])
@require_auth
def create_collections():
  """Create a new Chroma collection with the given name.

  Expects JSON payload with 'name' key.

  Returns:
      dict: A dictionary with the result message and status code.
  """
  name = request.get_json().get('name')
  try:
    if name in chroma_client.list_collections():
      return {'result':f'There is already a chroma collection with the name {name}'},200
    else:
      chroma_client.create_collection(name=name)
      return {'result':f'Created chroma Collections {name}'},200
  except Exception as e:
    return jsonify({"error": f"Error while creating chroma collections: {e}"}), 500

@app.route('/api/v1/collections/delete',methods=['POST'])
@require_auth
def delete_collections():
  """Delete a Chroma collection with the given name.

  Expects JSON payload with 'name' key.

  Returns:
      dict: A dictionary with the result message and status code.
  """
  name = request.get_json().get('name')
  try:
    if name in chroma_client.list_collections():
      chroma_client.delete_collection(name=name)
      return {'result':f'Deleted chroma collection with the name {name}'},200
    else:
      return {'result':f'There is no {name} Chroma Collection'},200
  except Exception as e:
    return jsonify({"error": f"Error while deleting chroma collections: {e}"}), 500


@app.route("/api/v1/mcp/add", methods=["POST"])
def add_tool():
  """Adds a tool to the Agent configuration JSON"""
  tool = request.get_json()
  try:
    agent_menager.add_tool(tool["name"], tool["url"])
    # Touch agent.py to trigger ADK hot reload
    AGENT_PATH.touch()
    return {"status": f"Tool {tool['name']} added successfully"}, 200
  except Exception as e:
    return jsonify({"error": str(e)}), 500

@app.route("/api/v1/mcp/remove", methods=["POST"])
def remove_tool():
  """Removes a tool from the Agent configuration JSON"""
  tool = request.get_json()
  try:
    if agent_menager.remove_tool(tool["name"]):
      AGENT_PATH.touch()
      return {"status": f"Tool {tool['name']} removed successfully"}, 200
    else:
      return {"status": f"Tool {tool['name']} not found"}, 404
  except Exception as e:
    return jsonify({"error": str(e)}), 500

@app.route("/api/v1/mcp/list", methods=["GET"])
def list_tools():
  """List the tools from the agent configuration file"""
  config = agent_menager.load_config()
  return {"tools": config.get("tools", [])}, 200

if __name__ == '__main__':
  app.run(port=FLASK_RUN_PORT, host='0.0.0.0')
