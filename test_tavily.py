import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load variables from .env into the environment
load_dotenv()

# Read the key
api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError("TAVILY_API_KEY not found. Check your .env file.")

# Create a client
client = TavilyClient(api_key=api_key)

# Run a simple search
response = client.search(query="What is LangGraph used for?", max_results=3)

print("Search results from Tavily:\n")
for i, result in enumerate(response["results"], start=1):
    print(f"{i}. {result['title']}")
    print(f"   URL: {result['url']}")
    print(f"   Snippet: {result['content'][:150]}...")
    print()