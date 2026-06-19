import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

WRITER_PROMPT = """You are a professional research report writer. Using the
summary below, write a polished, well-structured research report in MARKDOWN
format about the given topic.

Requirements:
- Start with a single # Title (clear, descriptive)
- Include a brief 2-3 sentence ## Overview section
- Organize the body into logical ## sections with headers based on themes
  in the summary
- Preserve ALL source citations from the summary, formatted as markdown
  links where possible, e.g. [source](URL)
- End with a ## Sources section listing all unique URLs referenced, as a
  bulleted list
- Use clear, professional prose. Avoid filler phrases like "in conclusion"
  or "in today's world"

Topic: {topic}

Summary to base the report on:
{summary}

Write the full markdown report now:"""


def writer_node(state):
    topic = state["topic"]
    summary = state["summary"]

    prompt = WRITER_PROMPT.format(topic=topic, summary=summary)
    response = llm.invoke(prompt)

    print("\n[WRITER] Final report generated.")
    print(f"  Length: {len(response.content)} characters")

    return {
        "final_report": response.content,
        "next": "END"
    }


# Quick standalone test
if __name__ == "__main__":
    test_state = {
        "topic": "The impact of AI agents on software engineering jobs",
        "summary": (
            "The integration of AI agents in software engineering is transforming "
            "the profession. Tasks like code generation, debugging, and testing "
            "are becoming automated (Source: https://example.com/a). New skills "
            "like system design, AI oversight, and prompt engineering are growing "
            "in importance (Source: https://example.com/b)."
        )
    }
    result = writer_node(test_state)
    print("=" * 60)
    print("FINAL REPORT:")
    print("=" * 60)
    print(result["final_report"])