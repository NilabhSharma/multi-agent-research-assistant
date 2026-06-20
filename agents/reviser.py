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

REVISER_SYSTEM_PROMPT = """You are a research report editor. You will be
given an existing markdown research report and a follow-up instruction
asking you to revise or expand part of it.

Rules you MUST follow:
- If the instruction requires facts, figures, or claims not already present
  in the report, use the search tool to find real, current information.
  NEVER invent citations, statistics, or sources.
- Preserve the report's existing markdown structure (headers, sources
  section) unless told to change it.
- Keep all existing source citations intact.
- Add any new source URLs you used to the Sources section.
- Output the FULL revised report (not just the changed section), in
  markdown format."""


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


def reviser_node(state):
    report = state["final_report"]
    instruction = state["follow_up_instruction"]

    user_message = (
        f"Existing report:\n{report}\n\n"
        f"Follow-up instruction: {instruction}"
    )

    print(f"\n[REVISER] Applying follow-up instruction: \"{instruction}\"")

    async def run():
        mini_app = await build_mini_app()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = await mini_app.ainvoke({
                    "messages": [
                        {"role": "system", "content": REVISER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ]
                })
                return result["messages"][-1].content
            except Exception as e:
                print(f"    [retry {attempt}/{max_attempts}] failed: {e}")
                if attempt == max_attempts:
                    print("    Giving up, keeping original report unchanged.")
                    return report

    revised_report = asyncio.run(run())

    print(f"  New length: {len(revised_report)} characters")

    return {
        "final_report": revised_report,
        "next": "END"
    }


# Quick standalone test
if __name__ == "__main__":
    test_state = {
        "final_report": (
            "# AI in Software Engineering\n\n"
            "## Overview\nAI is changing how software is built.\n\n"
            "## Automation\nCode generation and testing are increasingly automated.\n\n"
            "## Sources\n* https://example.com/a"
        ),
        "follow_up_instruction": "Expand the Automation section with more current, real examples."
    }
    result = reviser_node(test_state)
    print("=" * 60)
    print("REVISED REPORT:")
    print("=" * 60)
    print(result["final_report"])