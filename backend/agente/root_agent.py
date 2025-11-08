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

load_dotenv()

IP = os.getenv("IP")
PORT = int(os.getenv("AGENT_PORT"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

root_agent = build_agent()

a2a_app = to_a2a(root_agent, port=PORT,host=IP)

a2a_app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
  expose_headers=["*"],  # Exposes all headers
)

if __name__ == '__main__':
  uvicorn.run(a2a_app, host=IP, port=PORT)
