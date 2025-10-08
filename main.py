import os
import secrets
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from chromadb import HttpClient
from pydantic import BaseModel
from typing import List
import uvicorn
from .llm import LLMConversation

load_dotenv()

# Load environment variables (use python-dotenv if needed)
AGENT_SECRET_TOKEN = os.getenv("AGENT_SECRET_TOKEN", secrets.token_hex(16))
API_PORT = int(os.getenv("API_PORT", 8000))
CHROMA_URI = os.getenv("CHROMA_URI", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

@app.post("/query", dependencies=[Depends(authenticate)])

app = FastAPI()
security = HTTPBearer()

# Initialize ChromaDB (adjust as per your ChromaDBRetriever class)
chroma = HttpClient(host=CHROMA_URI, port=CHROMA_PORT)

llm = LLMConversation()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para o Docker verificar se o AI Guide Agent está online."""
    return jsonify({"status": "ok"}), 200

