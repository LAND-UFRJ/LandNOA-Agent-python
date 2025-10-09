import os
import secrets
import requests
import time
from dotenv import load_dotenv
from chromadb import HttpClient
import atexit
from flask import Flask, request, jsonify
import threading
from llm import LLMConversation
from mcp import Client

load_dotenv()

AGENT_ID = os.getenv("AI_GUIDE_AGENT_ID")
AGENT_SECRET_TOKEN = os.getenv("AI_GUIDE_AGENT_SECRET_TOKEN")
AGENT_BASE_URL = os.getenv("AI_GUIDE_AGENT_BASE_URL")
REGISTRY_BASE_URL = os.getenv("REGISTRY_BASE_URL")
FLASK_RUN_PORT = int(os.getenv("AI_GUIDE_FLASK_RUN_PORT", 8010))

app = Flask(__name__)

#todo:Inserir ferramentas MCp aqui tb
TOOLS_SCHEMA = [
    {
        "name": "responder_como_guia_de_ia",
        "description": "Use esta ferramenta para responder perguntas sobre o uso de IA de maneiras éticas e adequadas à uma educação saudável e responsável.",
        "parameters": {"pergunta": "string"}
    }
]

llm = LLMConversation()

mcp_client = Client()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para o Docker verificar se o AI Guide Agent está online."""
    return jsonify({"status": "ok"}), 200

@app.route('/execute', methods=['POST'])
def execute_task():
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
        return jsonify({"error": "Acesso não autorizado."}), 403
    
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

#Todo    
@app.route('/collections/list',methods=['GET'])
def list_collections():
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
        return jsonify({"error": "Acesso não autorizado."}), 403

#Todo
@app.route('/collections/create',methods=['POST'])
def create_collections():
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
        return jsonify({"error": "Acesso não autorizado."}), 403

#Todo
@app.route('/collections/delete',methods=['POST'])
def delete_collections():
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
        return jsonify({"error": "Acesso não autorizado."}), 403

#Todo
@app.route('/mcp/add',methods=['POST'])
def add_tool():
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
        return jsonify({"error": "Acesso não autorizado."}), 403
    try:
      tool = request.get_json()
      name,url = tool.get("name"), tool.get("url")
      mcp_client.add_tool(name=name,url=url)
      return{"status": f"tool {name} with url {url} added sucessfully"}, 200
    except Exception as e:
        return jsonify({"error": f"Error when adding tool {tool} with url {url}: {e}"}), 500

#Todo
@app.route('/mcp/remove',methods=['POST'])
def delete_tool():
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
        return jsonify({"error": "Acesso não autorizado."}), 403
    tool = request.get_json()
    name = tool.get("name")
    try:  
      if mcp_client.remove_tool(name = name):
        return {"status": f"tool {name} was removed sucessfully"}, 200
      else:      
        return {"status": f"there is no tool {name} "}, 200
    except Exception as e:
        return jsonify({"error": f"Error when removing tool {tool} : {e}"}), 500


#Todo
@app.route('/mcp/bind',methods=['POST'])
def bind_tools():
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
        return jsonify({"error": "Acesso não autorizado."}), 403

def register_with_registry():
    payload = {"agent_id": AGENT_ID, "base_url": AGENT_BASE_URL, "tools": TOOLS_SCHEMA, "secret_token": AGENT_SECRET_TOKEN}
    try:
        requests.post(f"{REGISTRY_BASE_URL}/register", json=payload).raise_for_status()
        print(f"AI_GUIDE_AGENT ({AGENT_ID}): Registro/Heartbeat enviado com sucesso para o Registry!")
    except requests.exceptions.RequestException as e:
        print(f"AI_GUIDE_AGENT ({AGENT_ID}): Falha no registro/heartbeat. Erro: {e}")

def deregister_from_registry():
    try:
        requests.post(f"{REGISTRY_BASE_URL}/deregister", json={"agent_id": AGENT_ID}, timeout=2)
    except requests.exceptions.RequestException as e:
        print(f"AI_GUIDE_AGENT ({AGENT_ID}): Falha ao desregistrar. Erro: {e}")

def heartbeat_task():
    """
    Função que será executada em segundo plano para enviar o re-registro periódico.
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
