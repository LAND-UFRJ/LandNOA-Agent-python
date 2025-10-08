from langchain_mcp_adapters.client import MultiServerMCPClient

class Client:
    def __init__(self) -> None:
        self.array: dict[str, dict[str, str]] = {}
        self.client = None

    def addTool(self, name: str, url: str) -> None:
        if name not in self.array:
            self.array[name] = {"transport": "sse", "url": url}
        else:
            self.array[f"{name}_v2"] = {"transport": "sse", "url": url}
  
    def removeTool(self, name: str) -> bool:
        """Remove a tool by name from the internal registry.

        Returns True if the tool existed and was removed; False otherwise.
        """
        if name in self.array:
            del self.array[name]
    
    def bindTools(self):
      self.client = MultiServerMCPClient(self.array)
    
    async def getTools(self):
      tools = await self.client.get_tools()
      return tools