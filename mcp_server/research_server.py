import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found. Check your .env file.")

tavily_client = TavilyClient(api_key=tavily_api_key)

mcp = FastMCP("research-tools")
@mcp.tool()
def web_search(query: str) -> str:
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
    mcp.run(transport="streamable-http")