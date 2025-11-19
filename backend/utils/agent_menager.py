from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams
from . import sqlite_functions as sf
from . import retrieval

OPENAI_URL = sf.get_config_sqlite("openai_baseurl")
OPENAI_KEY = sf.get_config_sqlite("openai_api_key")

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

  raise ImportError(f"RAG tool '{name}' not found in retrieval module")

def build_agent() -> LlmAgent:
  """Build the LlmAgent based on the current config.json."""
  model_name = sf.get_config_sqlite("model")
  agent_name = sf.get_config_sqlite("agent_name")
  cache_tools = sf.get_tools_sqlite()
  tools = []

  if cache_tools is not None:
    tools.extend([
      McpToolset(connection_params=SseConnectionParams(url=tool["url"]))
      for tool in cache_tools
    ])
  tools.append(_resolve_rag_tool())
  
  if sf.get_prompt_sqlite() is None:
    return LlmAgent(
    model=LiteLlm(model=f'openai/{model_name}',
                  api_base=OPENAI_URL,
                  api_key=OPENAI_KEY),
    name=agent_name,
    tools=tools,
  )
  else:
    return LlmAgent(
      model=LiteLlm(model=f'openai/{model_name}',
                  api_base=OPENAI_URL,
                  api_key=OPENAI_KEY),
      name=agent_name,
      tools=tools,
      instruction=sf.get_prompt_sqlite()
    )

def add_tool(name: str, url: str,description:str):
  """Add a tool and update config.json."""
  tools = sf.get_tools_sqlite()
  if any(t["name"] == name for t in tools["tools"]):
    raise ValueError(f"Tool {name} already exists.")
  sf.add_tool_sqlite(name,url,description)

def remove_tool(name: str) -> bool:
  """Remove a tool by name."""
  sf.remove_tool_sqlite(name)
