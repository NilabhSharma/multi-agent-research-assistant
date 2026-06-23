import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

PLANNER_PROMPT = """You are a research planner. Given a research topic, break it down into 3-4 focused, specific sub-questions that together would give a comprehensive understanding of the topic.

Respond with ONLY the sub-questions, one per line, no numbering, no extra commentary.

Topic: {topic}"""

def planner_node(state):
    topic = state["topic"]

    prompt = PLANNER_PROMPT.format(topic=topic)
    response = llm.invoke(prompt)

    sub_questions = [
        line.strip() for line in response.content.split("\n") if line.strip()
    ]

    print(f"\n[PLANNER] Generated {len(sub_questions)} sub-questions:")
    for q in sub_questions:
        print(f"  - {q}")

    return {
        "plan": sub_questions,
        "next": "web_searcher"
    }

if __name__ == "__main__":
    test_state = {"topic": "The impact of AI agents on software engineering jobs"}
    result = planner_node(test_state)
    print("\nReturned state update:")
    print(result)