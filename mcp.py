from langchain_mcp_adapters.client import MultiServerMCPClient

class Client:
  """A client for managing MCP (Model Context Protocol) tools.""" 
  def __init__(self) -> None:
    """Initialize the Client with an empty array and no bound client."""
    self.array: dict[str, dict[str, str]] = {}
    self.client = None
  def add_tool(self, name: str, url: str) -> None:
    """Add a tool to the internal registry.
    If the name already exists, append '_v2' to the name.
    Args:
        name (str): The name of the tool.
        url (str): The URL for the tool.
    """
    if name not in self.array:
      self.array[name] = {"transport": "sse", "url": url}
    else:
      self.array[f"{name}_v2"] = {"transport": "sse", "url": url}
  def remove_tool(self, name: str) -> bool:
    """Remove a tool by name from the internal registry.
    Returns True if the tool existed and was removed; False otherwise.
    Args:
        name (str): The name of the tool to remove.
    Returns:
        bool: True if removed, False if not found.
    """
    if name in self.array:
      del self.array[name]
      return True
    return False
  def bind_tools(self) -> None:
    """Bind the necessary tools to the client instance.
    This method initializes the MultiServerMCPClient using
    the object's array attribute and assigns it to self.client.
    """
    self.client = MultiServerMCPClient(self.array)  
  async def get_tools(self):
    """Asynchronously retrieve the tools from the bound client.
    Returns:
        The list of tools from the client.
    """
    tools = await self.client.get_tools()
    return tools