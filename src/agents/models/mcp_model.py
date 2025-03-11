from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional, Tuple, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.http import http_client
from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
)

from ..agent_output import AgentOutputSchema
from ..exceptions import AgentsException
from ..handoffs import Handoff
from ..items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..logger import logger
from ..model_settings import ModelSettings
from ..tool import Tool, FunctionTool
from ..tracing import generation_span
from ..usage import Usage
from .fake_id import FAKE_RESPONSES_ID
from .interface import Model, ModelProvider, ModelTracing
from ..function_schema import FunctionSchema, JSONSchemaValue


class ServerConfig:
    """Configuration for connecting to an MCP server."""

    def __init__(
        self,
        name: str,
        *,
        # For HTTP server
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        
        # For stdio server
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.command = command
        self.args = args or []
        self.env = env or {}
        
        # Validate configuration
        if base_url is None and command is None:
            raise ValueError(f"Server {name} must have either base_url or command specified")
        
        if base_url is not None and command is not None:
            raise ValueError(f"Server {name} cannot have both base_url and command specified")


class MCPTool(FunctionTool):
    """A tool that executes functions from an MCP server.
    
    This class wraps MCP server tools as OpenAI Agents tools, allowing seamless
    integration of MCP capabilities into the agents framework.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        client_session: ClientSession,
        mcp_schema: Dict[str, Any],
    ):
        self.name = name
        self.description = description
        self._client_session = client_session
        self._mcp_schema = mcp_schema
        
        # Create function schema from MCP tool schema
        parameters = self._mcp_schema.get("parameters", {})
        schema = FunctionSchema(
            name=name,
            description=description,
            parameters=JSONSchemaValue(parameters),
        )
        
        super().__init__(self._call_mcp_tool, schema)
    
    async def _call_mcp_tool(self, **kwargs):
        """Call the MCP tool with the given parameters."""
        try:
            result = await self._client_session.call_tool(
                self.name,
                arguments=kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.name}: {e}")
            raise AgentsException(f"MCP tool error ({self.name}): {str(e)}")


async def create_mcp_tools(client_session: ClientSession) -> List[Tool]:
    """Create OpenAI Agents tools from MCP server tools.
    
    Args:
        client_session: MCP client session to use for creating tools
        
    Returns:
        A list of OpenAI Agents tools wrapping MCP server tools
    """
    tools = []
    
    # Get available tools from the MCP server
    mcp_tools = await client_session.list_tools()
    
    # Convert each MCP tool to an MCPTool
    for mcp_tool in mcp_tools:
        tool = MCPTool(
            name=mcp_tool["name"],
            description=mcp_tool.get("description", ""),
            client_session=client_session,
            mcp_schema=mcp_tool,
        )
        tools.append(tool)
    
    return tools


class MCPToolProvider:
    """Provider for MCP (Model Context Protocol) tools."""

    def __init__(
        self,
        servers: Optional[List[ServerConfig]] = None,
        default_server: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        import os
        self._anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._anthropic_api_key:
            raise ValueError("Anthropic API key must be provided for MCP integration")
            
        self._servers = servers or []
        self._default_server = default_server
        
        # Validate server configurations
        if not self._servers:
            raise ValueError("At least one MCP server must be configured")
            
        if not self._default_server:
            self._default_server = self._servers[0].name
            
        # Session cache
        self._sessions: Dict[str, ClientSession] = {}
    
    def _get_or_create_session(self, server_name: str) -> ClientSession:
        """Get or create a client session for the specified server."""
        # Check if we already have a session
        if server_name in self._sessions:
            return self._sessions[server_name]
            
        # Find server config
        server_config = next((s for s in self._servers if s.name == server_name), None)
        if not server_config:
            raise ValueError(f"No server configuration found for {server_name}")
            
        # Create a new session
        loop = asyncio.new_event_loop()
        try:
            if server_config.base_url:
                # HTTP server
                read_stream, write_stream = loop.run_until_complete(
                    http_client(
                        base_url=server_config.base_url,
                        api_key=server_config.api_key
                    )
                )
            else:
                # Stdio server
                server_params = StdioServerParameters(
                    command=server_config.command,
                    args=server_config.args,
                    env=server_config.env
                )
                read_stream, write_stream = loop.run_until_complete(
                    stdio_client(server_params)
                )
                
            # Create and initialize client session
            session = ClientSession(read_stream, write_stream)
            loop.run_until_complete(session.initialize())
            
            # Cache the session
            self._sessions[server_name] = session
            
            return session
        finally:
            loop.close()
    
    def get_tools_from_server(self, server_name: Optional[str] = None) -> List[Tool]:
        """Get tools from the specified MCP server.
        
        Args:
            server_name: Name of the server to get tools from. If None, uses the default server.
            
        Returns:
            A list of OpenAI Agents tools wrapping MCP server tools
        """
        if server_name is None:
            server_name = self._default_server
            
        # Get or create session
        session = self._get_or_create_session(server_name)
        
        # Create tools from session
        loop = asyncio.new_event_loop()
        try:
            tools = loop.run_until_complete(create_mcp_tools(session))
            return tools
        finally:
            loop.close()
    
    def get_all_tools(self) -> List[Tool]:
        """Get tools from all configured MCP servers.
        
        Returns:
            A list of OpenAI Agents tools wrapping MCP server tools from all servers
        """
        all_tools = []
        
        # Get tools from each server
        for server in self._servers:
            tools = self.get_tools_from_server(server.name)
            all_tools.extend(tools)
            
        return all_tools