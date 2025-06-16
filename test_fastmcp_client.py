#!/usr/bin/env python3
"""
Test MCP server using the official FastMCP Client library
"""
import asyncio
import os
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

# Server configuration
SERVER_URL = "http://localhost:8085"
TOKEN = os.getenv("TOKEN", "your_token_here")

async def test_with_fastmcp_client():
    """Test the MCP server using FastMCP Client"""
    print("🚀 Testing MCP Server with FastMCP Client")
    print(f"Server: {SERVER_URL}")
    print("=" * 50)
    
    # Create client with proper authentication
    # For Streamable HTTP, we need to pass the full URL with /mcp path
    mcp_url = f"{SERVER_URL}/mcp"
    
    # Create transport with authentication headers
    from fastmcp.client import StreamableHttpTransport
    
    try:
        # Create transport with auth headers
        transport = StreamableHttpTransport(
            url=mcp_url,
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        client = Client(transport)
        
        async with client:
            print("✅ Connected to MCP server")
            
            # Test 1: List available tools
            print("\n🛠️  Listing available tools...")
            try:
                tools = await client.list_tools()
                print(f"Available tools: {[tool.name for tool in tools]}")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
            except Exception as e:
                print(f"❌ Error listing tools: {e}")
            
            # Test 2: Call validate tool
            print("\n🔍 Testing validate tool...")
            try:
                result = await client.call_tool("validate")
                print(f"✅ Validate result: {result}")
            except Exception as e:
                print(f"❌ Error calling validate: {e}")
            
            # Test 3: Call resume tool
            print("\n📄 Testing resume tool...")
            try:
                result = await client.call_tool("resume")
                print("✅ Resume tool executed successfully")
                if result and len(result) > 0:
                    preview = result[0].text[:200] if hasattr(result[0], 'text') else str(result[0])[:200]
                    print(f"Resume preview: {preview}...")
            except Exception as e:
                print(f"❌ Error calling resume: {e}")
            
            # Test 4: Call fetch tool
            print("\n🌐 Testing fetch tool...")
            try:
                result = await client.call_tool("fetch", {
                    "url": "https://httpbin.org/json",
                    "max_length": 1000
                })
                print(f"✅ Fetch result: {result}")
            except Exception as e:
                print(f"❌ Error calling fetch: {e}")
                
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
    
    print("\n✨ Testing complete!")

if __name__ == "__main__":
    asyncio.run(test_with_fastmcp_client()) 