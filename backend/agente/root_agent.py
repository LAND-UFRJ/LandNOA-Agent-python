import sys
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

import uvicorn
from utils.agent_menager import build_agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from backend.utils.sqlite_functions import get_config_sqlite 
from a2a.types import AgentCard
import logging
import backend.utils.sqlite_functions as sf
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams
from backend.utils import retrieval

load_dotenv()
# login configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


# Reload logic

def _resolve_rag_tool():
  """Resolve the RAG tool object named in the DB/config.
  The value returned by get_rag_tool() should match an attribute exported
  by the retrieval module or an attribute of retrieval.Retriever.
  If it's a class, instantiate it; if it's an instance method, bind it to an instance.
  """
  name = sf.get_rag_tool_sqlite()
  if not name:
    raise ImportError("RAG tool name is empty. Check the 'retrieval_function' entry in the config DB (or SQLITE_PATH).")

  # module-level function or class
  if hasattr(retrieval, name):
    tool_obj = getattr(retrieval, name)
    # If it's a class, instantiate it
    if isinstance(tool_obj, type):
      return tool_obj()
    # module-level function — return as-is
    return tool_obj

  # attribute on Retriever (instance method or attribute) — instantiate Retriever and get the bound attribute
  if hasattr(retrieval, "Retriever") and hasattr(retrieval.Retriever, name):
    retriever_instance = retrieval.Retriever()
    tool_obj = getattr(retriever_instance, name)
    return tool_obj
  
@a2a_app.post("/api/reload_agent")
async def api_reload_agent():
    global root_agent
    logger.info("🔄 [Backend] Recebi pedido de reload...")

    try:
        # 1. Busca configurações frescas no SQLite
        new_model_name = sf.get_config_sqlite("model")
        new_openai_url = sf.get_config_sqlite("openai_baseurl")
        new_openai_key = sf.get_config_sqlite("openai_api_key")
        new_tools_config = sf.get_tools_sqlite()

        # 2. Atualiza o Cérebro (Model)
        new_model = LiteLlm(
            model=f'openai/{new_model_name}',
            api_base=new_openai_url,
            api_key=new_openai_key
        )
        root_agent.model = new_model
        
        # 3. Atualiza as Ferramentas (Tools)
        new_tools_list = []
        if new_tools_config is not None:
            new_tools_list.extend([
                McpToolset(connection_params=SseConnectionParams(url=tool["url"]))
                for tool in new_tools_config
            ])
        
        rag_tool = _resolve_rag_tool()
        if rag_tool:
            new_tools_list.append(rag_tool)
            
        root_agent.tools = new_tools_list
        
        logger.info(f"✅ [Backend] Agente atualizado para modelo: {new_model_name}")
        return {"status": "success"}

    except Exception as e:
        logger.error(f"❌ [Backend] Erro ao atualizar: {e}")
        return {"status": "error", "message": str(e)}




if __name__ == '__main__':
  uvicorn.run(a2a_app, host='0.0.0.0', port=PORT)
