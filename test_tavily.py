import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError("TAVILY_API_KEY not found. Check your .env file.")

client = TavilyClient(api_key=api_key)
response = client.search(query="What is LangGraph used for?", max_results=3)
print("Search results from Tavily:\n")
for i, result in enumerate(response["results"], start=1):
    print(f"{i}. {result['title']}")
    print(f"   URL: {result['url']}")
    print(f"   Snippet: {result['content'][:150]}...")
    print()