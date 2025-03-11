#!/usr/bin/env python3

import asyncio
import os
from dotenv import load_dotenv

from agents import Agent, RunConfig, Runner, ServerConfig

# Load environment variables from .env file
load_dotenv()

async def main():
    """
    Demonstrates how to use MCP as a tool provider with OpenAI Agents.
    
    This example connects to two MCP servers as tools rather than as a model provider:
    1. Say Server: Provides text-to-speech functionality (macOS only)
    2. Fetch Server: Provides URL fetching capabilities
    
    To run this example:
    1. First, start both servers in separate terminals:
       python examples/mcp/say_server.py
       python examples/mcp/fetch_server.py
       
    2. Then run this example with the required API keys:
       ANTHROPIC_API_KEY=your_key OPENAI_API_KEY=your_key python examples/mcp/mcp_tools_example.py
    """
    
    # Check for required API keys
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is required for MCP client")
        return
        
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is required for OpenAI model")
        return
    
    # Configure the MCP servers
    say_server = ServerConfig(
        name="say-service",
        command="python",
        args=["examples/mcp/say_server.py"]
    )
    
    fetch_server = ServerConfig(
        name="fetch-service",
        command="python",
        args=["examples/mcp/fetch_server.py"]
    )
    
    # Create an agent that will use OpenAI model but MCP tools
    assistant = Agent.from_prompt(
        name="MCP Tools Assistant",
        prompt="""You are a helpful assistant with access to special tools from MCP servers.
        
        1. Text-to-speech tools: You can speak text aloud using different voices
        2. Fetch tools: You can fetch content from URLs
        
        Use these tools appropriately when a user asks for them.
        For speaking text, only use the say tool when explicitly requested.
        For fetching URLs, make sure the URL is valid and safe before fetching.
        """
    )
    
    # Configure the agent to use OpenAI model with MCP tools
    config = RunConfig.with_mcp_tools(
        servers=[say_server, fetch_server],
        include_all_servers=True,
        model="gpt-4o",  # Using OpenAI model
        anthropic_api_key=anthropic_api_key,  # Still needed for MCP client
    )
    
    # Example queries to demonstrate capabilities
    queries = [
        "Can you fetch the content from https://example.com?",
        "List all available voices for text-to-speech.",
        "Say 'Hello, this is a test using the tool flow instead of model flow' using the voice Alex.",
        "Fetch the headers from https://github.com and explain what they mean.",
    ]
    
    for i, query in enumerate(queries):
        print(f"\n[Query {i+1}] {query}")
        
        result = await Runner.run(
            starting_agent=assistant,
            input=query,
            run_config=config,
        )
        
        print("\nAssistant's Response:")
        print(result.output)
        print("-" * 50)
    
if __name__ == "__main__":
    asyncio.run(main())