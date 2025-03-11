# Model Context Protocol (MCP) Integration Examples

This directory contains examples of integrating the OpenAI Agents Python framework with Model Context Protocol (MCP) servers.

## What is MCP?

Model Context Protocol (MCP) is an open standard that enables developers to build secure connections between LLM applications and external data sources or tools. MCP servers can provide:

- **Resources**: Data and context information
- **Tools**: Functions to perform actions
- **Prompts**: Templates for interactions

## Examples

This directory includes the following examples:

- **say_server.py**: An MCP server that provides text-to-speech capabilities (macOS only)
- **fetch_server.py**: An MCP server that fetches content from URLs
- **mcp_tools_example.py**: Demonstrates using MCP servers as tools with an OpenAI model

## Setup

1. Install the required dependencies:

```bash
pip install "openai-agents[mcp]" python-dotenv httpx
```

2. Get an API key from [Anthropic](https://console.anthropic.com/) for the MCP client and an API key from [OpenAI](https://platform.openai.com/) for the model.

3. Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Running the Examples

1. Start the MCP servers in separate terminals:

```bash
# Terminal 1: Start the Say server
python examples/mcp/say_server.py

# Terminal 2: Start the Fetch server
python examples/mcp/fetch_server.py
```

2. Run the example:

```bash
# Basic example with say and fetch servers
ANTHROPIC_API_KEY=your_anthropic_api_key OPENAI_API_KEY=your_openai_api_key python examples/mcp/mcp_tools_example.py
```

## How It Works

The integration uses MCP servers as tools that can be used with any model:

1. Create a `ServerConfig` for each MCP server you want to connect to
2. Create a `RunConfig` with the `with_mcp_tools()` factory method
3. Run an agent with the configuration to access MCP server tools

Example:

```python
from agents import Agent, RunConfig, Runner, ServerConfig

# Configure MCP servers
say_server = ServerConfig(
    name="say-service",
    command="python",
    args=["say_server.py"]
)

fetch_server = ServerConfig(
    name="fetch-service",
    command="python",
    args=["fetch_server.py"]
)

# Create RunConfig with MCP tools
config = RunConfig.with_mcp_tools(
    servers=[say_server, fetch_server],
    model="gpt-4o",  # Can use any model, not just Claude
    anthropic_api_key="your_anthropic_api_key"  # Still needed for MCP client
)

# Run agent with MCP tools
result = await Runner.run(
    starting_agent=agent,
    input="Fetch the content from https://example.com",
    run_config=config
)
```

## Creating Your Own MCP Server

You can create your own MCP servers following the pattern in the example servers:

```python
from mcp.server.fastmcp import FastMCP

# Create a server
mcp = FastMCP("My Server")

# Add a tool
@mcp.tool()
def my_tool(param1: str, param2: int) -> str:
    """Tool description"""
    # Implement tool logic
    return "Result"

# Add a resource
@mcp.resource("my-resource://{id}")
def get_resource(id: str) -> str:
    """Resource description"""
    # Implement resource logic
    return f"Resource {id} data"

# Run the server
if __name__ == "__main__":
    mcp.run()
```

## Additional Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) - A tool for testing MCP servers