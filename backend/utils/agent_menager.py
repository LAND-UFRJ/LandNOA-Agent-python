from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams
from . import sqlite_functions as sf
from . import retrieval

def _resolve_rag_tool():
  """Resolve the RAG tool object named in the DB/config.
  The value returned by get_rag_tool() should match an attribute exported
  by the retrieval module or an attribute of retrieval.Retriever.
  If it's callable/class, instantiate it.
  """
  name = sf.get_rag_tool()
  if hasattr(retrieval, name):
    tool_obj = getattr(retrieval, name)
  elif hasattr(retrieval, "Retriever") and hasattr(retrieval.Retriever, name):
    tool_obj = getattr(retrieval.Retriever, name)
  else:
    raise ImportError(f"RAG tool '{name}' not found in retrieval module")
  return tool_obj() if callable(tool_obj) else tool_obj

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
  
  if sf.get_prompt() is None:
    return LlmAgent(
    model=LiteLlm(model=f'openai/{model_name}'),
    name=agent_name,
    tools=tools,
  )
  else:
    return LlmAgent(
      model=LiteLlm(model=f'openai/{model_name}'),
      name=agent_name,
      tools=tools,
      instruction=sf.get_prompt()
    )

def add_tool(name: str, url: str,description:str):
  """Add a tool and update config.json."""
  tools = sf.get_tools()
  if any(t["name"] == name for t in tools["tools"]):
    raise ValueError(f"Tool {name} already exists.")
  sf.add_tool_sqlite(name,url,description)

def remove_tool(name: str) -> bool:
  """Remove a tool by name."""
  sf.remove_tool_sql(name)
