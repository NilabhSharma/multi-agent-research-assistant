import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    server_url = "http://127.0.0.1:8000/mcp"

    # Connect to the running MCP server over HTTP
    async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # The MCP "handshake" - establishes the session
            await session.initialize()

            # Ask the server: what tools do you offer?
            tools_response = await session.list_tools()
            print("Tools available on this MCP server:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description.strip().splitlines()[0]}")

            # Now actually call the web_search tool
            print("\nCalling web_search tool...\n")
            result = await session.call_tool(
                "web_search",
                arguments={"query": "What is the Model Context Protocol?", "max_results": 2}
            )

            print("Result from MCP server:")
            for content_block in result.content:
                print(content_block.text)


if __name__ == "__main__":
    asyncio.run(main())