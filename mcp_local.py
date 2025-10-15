from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

class Client:
  """A client for managing MCP (Model Context Protocol) tools.""" 
  def __init__(self) -> None:
    """Initialize the Client with an empty array and no bound client."""
    self.array: list[dict[str, str]] = []
    self.tools: list[McpToolset] = None
  def add_tool(self, name: str, url: str) -> None:
    """Add a tool to the internal registry.
    If the name already exists, append '_v2' to the name.
    Args:
        name (str): The name of the tool.
        url (str): The URL for the tool.
    """
    names = [tool['name'] for tool in self.array]
    if name not in names:
      self.array.append({"name": name, "url": url})
    else:
      self.array.append({"name": f"{name}_v2", "url": url})
  def remove_tool(self, name: str) -> bool:
    """Remove a tool by name from the internal registry.
    Returns True if the tool existed and was removed; False otherwise.
    Args:
        name (str): The name of the tool to remove.
    Returns:
        bool: True if removed, False if not found.
    """
    for i, tool in enumerate(self.array):
      if tool['name'] == name:
        del self.array[i]
        return True
    return False
  def get_tools(self) -> None:
    """Bind the necessary tools to the client instance.
    This method initializes the MultiServerMCPClient using
    the object's array attribute and assigns it to self.client.
    """
    urls = [self.array[i]['url'] for i in range(len(self.array))]
    self.tools = [McpToolset(connection_params=
                             SseConnectionParams(url=url)) for url in urls]
