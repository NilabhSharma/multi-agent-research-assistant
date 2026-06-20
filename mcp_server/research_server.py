import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found. Check your .env file.")

tavily_client = TavilyClient(api_key=tavily_api_key)

# Create the MCP server instance. "research-tools" is just a name identifying
# this server - it'll show up when clients connect.
mcp = FastMCP("research-tools")


# The @mcp.tool() decorator is what turns a normal Python function into a
# standardized MCP tool. MCP automatically reads the function's type hints
# and docstring to describe the tool to any connecting client - this is the
# "standardized, type-safe interface" your project description mentions.
@mcp.tool()
def web_search(query: str) -> str:
    """
    Search the web for current information on a given query.

    Args:
        query: The search query string.

    Returns:
        A formatted string containing titles, URLs, and content snippets
        from the top search results.
    """
    response = tavily_client.search(query=query, max_results=3)

    formatted_results = []
    for result in response["results"]:
        formatted_results.append(
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content: {result['content'][:500]}"
        )

    return "\n\n---\n\n".join(formatted_results)


if __name__ == "__main__":
    # Run the server using "streamable-http" transport, meaning it listens
    # on a real network port, just like a small web server.
    mcp.run(transport="streamable-http")