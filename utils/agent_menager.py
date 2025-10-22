import json
from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams

CONFIG_PATH = Path(__file__).resolve().parent.parent/"agente/config.json"

def load_config():
  """Loads the configuration of the agent"""
  if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
  with open(CONFIG_PATH, "r",encoding="UTF-8") as f:
    return json.load(f)

def save_config(config: dict):
  """Saves the configuration of the agent"""
  with open(CONFIG_PATH, "w",encoding="UTF-8") as f:
    json.dump(config, f, indent=2)

def build_agent() -> LlmAgent:
  """Build the LlmAgent based on the current config.json."""
  config = load_config()
  model_name = config["model"]
  agent_name = config["agent_name"]
  tools = [
    McpToolset(connection_params=SseConnectionParams(url=tool["url"]))
    for tool in config.get("tools", [])
  ]
  return LlmAgent(
    model=LiteLlm(model=f'openai/{model_name}'),
    name=agent_name,
    tools=tools
  )

def add_tool(name: str, url: str):
  """Add a tool and update config.json."""
  config = load_config()
  if any(t["name"] == name for t in config["tools"]):
    raise ValueError(f"Tool {name} already exists.")
  config["tools"].append({"name": name, "url": url})
  save_config(config)

def remove_tool(name: str) -> bool:
  """Remove a tool by name."""
  config = load_config()
  before = len(config["tools"])
  config["tools"] = [t for t in config["tools"] if t["name"] != name]
  if len(config["tools"]) == before:
    return False
  save_config(config)
  return True
