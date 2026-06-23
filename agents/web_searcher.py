import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

class MiniState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

async def build_mini_app():
    client = MultiServerMCPClient({
        "research-tools": {
            "url": MCP_SERVER_URL,
            "transport": "streamable_http",
        }
    })

    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MiniState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state: MiniState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    mini_graph = StateGraph(MiniState)
    mini_graph.add_node("agent", call_model)
    mini_graph.add_node("tools", tool_node)
    mini_graph.set_entry_point("agent")
    mini_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    mini_graph.add_edge("tools", "agent")

    return mini_graph.compile()
def contains_leaked_tool_syntax(text):
    suspicious_markers = ["<function=", "</function>", "function=web_search"]
    return any(marker in text for marker in suspicious_markers)


def web_searcher_node(state):
    plan = state["plan"]
    search_results = []

    print(f"\n[WEB SEARCHER] Researching {len(plan)} sub-questions (via MCP)...")

    async def run_all():
        mini_app = await build_mini_app()

        for i, sub_question in enumerate(plan, start=1):
            print(f"  ({i}/{len(plan)}) {sub_question}")
            max_attempts = 3
            finding = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = await mini_app.ainvoke({
                        "messages": [
                            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                            {"role": "user", "content": sub_question}
                        ]
                    })
                    candidate = result["messages"][-1].content

                    if contains_leaked_tool_syntax(candidate):
                        raise ValueError(
                            "Response contained leaked/malformed tool-call syntax"
                        )

                    finding = candidate
                    break  # success, no need to retry
                except Exception as e:
                    print(f"    [retry {attempt}/{max_attempts}] tool call failed: {e}")
                    if attempt == max_attempts:
                        finding = (
                            "Could not retrieve information for this sub-question "
                            "after multiple attempts due to a tool-calling error."
                        )

            search_results.append(f"Q: {sub_question}\nA: {finding}")

    asyncio.run(run_all())

    print("[WEB SEARCHER] Done.\n")

    return {
        "search_results": search_results,
        "next": "summarizer"
    }

if __name__ == "__main__":
    test_state = {
        "plan": [
            "What specific software engineering tasks are most likely to be automated by AI agents?",
            "What new skills are emerging as a result of AI agents in software engineering?"
        ]
    }
    result = web_searcher_node(test_state)
    print("=" * 60)
    print("SEARCH RESULTS:")
    print("=" * 60)
    for r in result["search_results"]:
        print(r)
        print("-" * 40)