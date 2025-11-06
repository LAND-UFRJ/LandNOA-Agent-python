from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams
from .sqlite_functions import get_config, get_tools, get_prompt

CONFIG_PATH = Path(__file__).resolve().parent.parent/"agente/config.json"

def build_agent() -> LlmAgent:
  """Build the LlmAgent based on the current config.json."""
  model_name = get_config("model")
  agent_name = get_config("agent_name")
  tools = [
    McpToolset(connection_params=SseConnectionParams(url=tool["url"]))
    for tool in get_tools()
  ]
  return LlmAgent(
    model=LiteLlm(model=f'openai/{model_name}'),
    name=agent_name,
    tools=tools,
    instruction=get_prompt()
  )

def add_tool(name: str, url: str):
  """Add a tool and update config.json."""
  tools = get_tools()
  if any(t["name"] == name for t in tools["tools"]):
    raise ValueError(f"Tool {name} already exists.")
  tools["tools"].append({"name": name, "url": url})


def remove_tool(name: str) -> bool:
  """Remove a tool by name."""
  tools= get_tools()
  before = len(tools["tools"])
  tools["tools"] = [t for t in tools["tools"] if t["name"] != name]
  if len(tools["tools"]) == before:
    return False
  return True
