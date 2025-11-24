import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)
from pathlib import Path
import uvicorn
from utils.agent_menager import build_agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from backend.utils.sqlite_functions import get_config_sqlite 
from a2a.types import AgentCard

load_dotenv()

HOST_IP = os.getenv("HOST_IP")
PORT = int(os.getenv("AGENT_PORT"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

root_agent = build_agent()
agent_name = get_config_sqlite("agent_name")
agent_card = AgentCard(name=agent_name,
    url=f"http://{HOST_IP}:{PORT}/",
    description= "Test agent from file",
    version="1.0.0",
    capabilities= {},
    skills=[],
    defaultInputModes= ["text/plain"],
    defaultOutputModes= ["text/plain"],
    supportsAuthenticatedExtendedCard= False,
)
a2a_app = to_a2a(root_agent,agent_card=agent_card)
a2a_app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
  expose_headers=["*"],  # Exposes all headers
)

if __name__ == '__main__':
  uvicorn.run(a2a_app, host='0.0.0.0', port=PORT)
