import os
import time
import threading
import atexit
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from functools import wraps
from chromadb import HttpClient
from llm import LlmConversation
from mcp import Client
from indexing import Splitter

load_dotenv()


#####################################################################################
#Onde parei: Funções de splitting só aceitam pdf, tem que poder aceitar texto puro tb.
#Onde parei: Endpoints para poder manipular (atualmente so penso em adicionar) coleções
#Onde parei: metricas (prometheus) de avaliação de do sistema (tokens, tempo ligado,requests recebidas)
#Possíveis amelhoramentos: Monitorar conversas (MongoDB), Validação do RAG, Fazer interface (que de para manipular RAG e parametros como um todo)
#####################################################################################
AGENT_ID = os.getenv("AI_GUIDE_AGENT_ID")
AGENT_SECRET_TOKEN = os.getenv("AI_GUIDE_AGENT_SECRET_TOKEN")
AGENT_BASE_URL = os.getenv("AI_GUIDE_AGENT_BASE_URL")
REGISTRY_BASE_URL = os.getenv("REGISTRY_BASE_URL")
FLASK_RUN_PORT = int(os.getenv("AI_GUIDE_FLASK_RUN_PORT", '8010'))
chorma_url = os.getenv('CHROMADB_URL')
chorma_port = os.getenv('CHROMADB_PORT')

app = Flask(__name__)

#todo:Inserir ferramentas MCP aqui tb
TOOLS_SCHEMA = [
    {
        "name": "responder_como_guia_de_ia",
        "description": """Use esta ferramenta para responder perguntas sobre o
          uso de IA de maneiras éticas e adequadas à uma 
          educação saudável e responsável.""",
        "parameters": {"pergunta": "string"}
    }
]

llm = LlmConversation()

mcp_client = Client()

splitter = Splitter()

chroma_client = HttpClient(host=chorma_url,port=chorma_port)

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

@app.route('/api/v1/send', methods=['POST'])
@require_auth
def execute_task():
  """Execute a task by invoking the LLM with the
    provided query and returning the response."""
  a2a_message = request.get_json()
  query = a2a_message.get("payload", {}).get("query")
  #user_uuid = a2a_message.get("payload", {}).get("uuid")
  if not query:
    return jsonify({"error": "A 'query' não foi recebida."}), 400
  try:
    response = llm.invoke(query)
    return jsonify({"result": response})
  except Exception as e:
    return jsonify({"error": f"Erro interno do agente Guia de IA: {e}"}), 500

@app.route('/api/v1/collections/list',methods=['GET'])
@require_auth
def list_collections():
  """List all available Chroma collections.

  Returns:
      dict: A dictionary with the result message and status code.
  """
  try:
    collections = chroma_client.list_collections()
    return {'result':f'The Chorma Collections avaliable are {collections}'},200
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
      return {'result':f'Created Chorma Collections {name}'},200
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

@require_auth
@app.route('/collections/<collection_name>/add', methods=['POST'])
def add_to_collection(collection_name):
  """Add a PDF or text content to the specified Chroma collection.
  Expects JSON payload with 'type' ('pdf' or 'text'), and 'content' (text string) or file upload for PDF.
  Args:
      collection_name (str): The name of the Chroma collection.
  Returns:
      dict: A dictionary with the status message and code.
  """
  try:
    # Check if collection exists
    collections = chroma_client.list_collections()
    if collection_name not in [c.name for c in collections]:
      return jsonify({"error": f"Collection '{collection_name}' does not exist."}), 404
    data = request.get_json()
    content_type = data.get('type')
    if content_type == 'text':
      text = data.get('content')
      if not text:
        return jsonify({"error": "No content provided for text type."}), 400
      # Split text into chunks (simple split for now)
      chunks = [text[i:i+750] for i in range(0, len(text), 750)]
    elif content_type == 'pdf':
      # For PDF, expect file upload
      if 'file' not in request.files:
        return jsonify({"error": "No file provided for PDF type."}), 400
      file = request.files['file']
      if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
      # Save file temporarily
      import tempfile
      with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
      # Process PDF
      from indexing import equal_chunks
      chunks = equal_chunks(temp_path)
      os.unlink(temp_path)  # Clean up
    else:
        return jsonify({"error": "Invalid type. Use 'pdf' or 'text'."}), 400
    # Embed chunks and add to collection
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    collection = chroma_client.get_collection(collection_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"{collection_name}_{i}" for i in range(len(chunks))]
    )
    return {"status": f"Added {len(chunks)} chunks to collection '{collection_name}'."}, 200
  except Exception as e:
      return jsonify({"error": f"Error adding to collection: {e}"}), 500
      

@app.route('/api/v1/mcp/add',methods=['POST'])
@require_auth
def add_tool():
  """Add a new tool to the MCP client."""
  try:
    tool = request.get_json()
    name,url = tool.get("name"), tool.get("url")
    mcp_client.add_tool(name=name,url=url)
    return{"status": f"tool {name} with url {url} added sucessfully"}, 200
  except Exception as e:
    return jsonify({"error": f"Error when adding tool {tool} with url {url}: {e}"}), 500

@app.route('/api/v1/mcp/remove',methods=['POST'])
@require_auth
def delete_tool():
  """Remove a tool from the MCP client."""
  tool = request.get_json()
  name = tool.get("name")
  try:
    if mcp_client.remove_tool(name = name):
      return {"status": f"tool {name} was removed sucessfully"}, 200
    else:
      return {"status": f"there is no tool {name} "}, 200
  except Exception as e:
    return jsonify({"error": f"Error when removing tool {tool} : {e}"}), 500

@app.route('/api/v1/mcp/list',methods=['GET'])
@require_auth
async def list_tools():
  """List avaliable tools"""
  try:
    cache_tools = mcp_client.array
    tools = await mcp_client.get_tools()
    return {"status": f"""The cached tools are {cache_tools}
                      and the binded tools are {tools} """}, 200
  except Exception as e:
    return jsonify({"error": f"Error : {e}"}), 500

@app.route('/api/v1/mcp/bind',methods=['POST'])
@require_auth
async def bind_tools():
  """Bind the available tools from MCP client to the LLM.

  Returns:
      dict: A dictionary with the status message and code.
  """
  try:
    tools = await mcp_client.get_tools()
    llm.bind_tools(tools=tools)
    return {"status": f"tools {tools} were sucessfully binded"}, 200
  except Exception as e:
    return jsonify({"error": f"Error while bindding tools {tools}: {e}"}), 500


def register_with_registry():
  """Register the agent with the registry by sending agent details."""
  payload = {"agent_id": AGENT_ID, "base_url": AGENT_BASE_URL, "tools": TOOLS_SCHEMA, "secret_token": AGENT_SECRET_TOKEN}
  try:
    requests.post(f"{REGISTRY_BASE_URL}/register", json=payload,timeout=5).raise_for_status()
    print(f"AI_GUIDE_AGENT ({AGENT_ID}): Registro/Heartbeat enviado com sucesso para o Registry!")
  except requests.exceptions.RequestException as e:
    print(f"AI_GUIDE_AGENT ({AGENT_ID}): Falha no registro/heartbeat. Erro: {e}")

def deregister_from_registry():
  """Deregister the agent from the registry."""
  try:
    requests.post(f"{REGISTRY_BASE_URL}/deregister", json={"agent_id": AGENT_ID}, timeout=2)
  except requests.exceptions.RequestException as e:
    print(f"AI_GUIDE_AGENT ({AGENT_ID}): Falha ao desregistrar. Erro: {e}")

def heartbeat_task():
  """
  Função será executada em segundo plano para enviar o re-registro periódico.
  """
  while True:
    time.sleep(40)
    register_with_registry()


if __name__ == '__main__':
  atexit.register(deregister_from_registry)
  time.sleep(2)
  register_with_registry()
  heartbeat_thread = threading.Thread(target=heartbeat_task, daemon=True)
  heartbeat_thread.start()
  app.run(port=FLASK_RUN_PORT, host='0.0.0.0')
