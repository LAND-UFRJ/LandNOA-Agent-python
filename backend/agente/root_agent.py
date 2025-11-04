# agent/agent.py
import uvicorn
import sys
from pathlib import Path
from utils.agent_menager import build_agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

root_agent = build_agent()

a2a_app = to_a2a(root_agent, port=8001,host="10.246.3.42")

a2a_app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
  expose_headers=["*"],  # Exposes all headers
)

if __name__ == '__main__':
  uvicorn.run(a2a_app, host="10.246.3.42", port=8001)
