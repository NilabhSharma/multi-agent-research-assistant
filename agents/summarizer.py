import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

SUMMARIZER_PROMPT = """You are a research summarizer. Below are raw findings
gathered for several sub-questions about a research topic. Each finding may
include source URLs.

Your job: condense these findings into a single, coherent summary that:
- Removes redundancy between findings
- Preserves all important facts and figures
- Keeps track of which source URL supports which claim, using inline format like (Source: URL)
- Is organized by theme, not necessarily by the original sub-question order
- Is written in clear prose paragraphs, not just bullet lists

Topic: {topic}

Raw findings:
{findings}

Write the condensed summary now:"""


def summarizer_node(state):
    topic = state["topic"]
    findings = "\n\n".join(state["search_results"])

    prompt = SUMMARIZER_PROMPT.format(topic=topic, findings=findings)
    response = llm.invoke(prompt)

    print("\n[SUMMARIZER] Condensed summary generated.")
    print(f"  Length: {len(response.content)} characters")

    return {
        "summary": response.content,
        "next": "writer"
    }


# Quick standalone test
if __name__ == "__main__":
    test_state = {
        "topic": "The impact of AI agents on software engineering jobs",
        "search_results": [
            "Q: What tasks are most likely to be automated?\nA: Code generation, debugging, and testing are increasingly automated. (Source: https://example.com/a)",
            "Q: What new skills are emerging?\nA: System design, AI oversight, and prompt engineering are growing in importance. (Source: https://example.com/b)"
        ]
    }
    result = summarizer_node(test_state)
    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(result["summary"])