from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    
    topic: str 
    messages: Annotated[list, add_messages]

    plan: List[str]
    search_results: List[str]
    summary: str
    final_report: str

    next: str
    follow_up_instruction: str