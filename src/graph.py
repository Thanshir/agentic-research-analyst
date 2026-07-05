"""
The core agent graph.

Architecture (this is the part to walk through in interviews):

    START -> planner -> [has tool calls?] -> tool_executor -> planner (loop)
                       -> [no tool calls]  -> reflector -> [sufficient?] -> finalize -> END
                                                          -> [insufficient & iterations left] -> planner (loop)

This project builds a custom LangGraph StateGraph rather than the newer
`langchain.agents.create_agent` (LangChain 1.0's high-level agent builder),
because the self-reflection loop requires branching logic that goes beyond
create_agent's default ReAct pattern. This is the correct choice per
LangChain's own guidance: use create_agent for standard tool-calling agents,
and drop down to a custom StateGraph when you need fine-grained control over
routing — exactly the tradeoff worth explaining in an interview.

This is NOT a simple single-pass RAG chain. It's a loop with a decision
point that can send the agent back to do more research if its own
self-critique decides the answer is weak. That reflection loop is the
single biggest thing that separates "I called an LLM" from "I built an
agent system."
"""
import operator
from typing import Annotated, List, TypedDict, Optional

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.config import get_llm
from src.tools import TOOLS
from src.schemas import ResearchAnswer, ReflectionResult

MAX_ITERATIONS = 3
MAX_TOOL_CALLS = 6  # hard cap on planner<->tool_executor loops per run


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    research_question: str
    iterations: int
    final_answer: Optional[ResearchAnswer]


PLANNER_SYSTEM_PROMPT = """You are a rigorous research analyst agent.
You have access to tools: search_web (for facts/current info) and calculator
(for arithmetic). Use them whenever you're not 100% certain of a fact.
Do not guess at numbers, prices, or current events from memory alone.
Once you have enough information, respond with your findings in plain text
(do not call any more tools) so the reflection step can review your work."""


import time
from groq import BadRequestError


def planner_node(state: AgentState):
    messages = state["messages"]
    if len(messages) == 0 or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + messages

    last_error = None
    for attempt, temp in enumerate([0.0, 0.3, 0.5]):
        try:
            llm = get_llm(temperature=temp).bind_tools(TOOLS)
            response = llm.invoke(messages)
            return {"messages": [response]}
        except BadRequestError as e:
            last_error = e
            if attempt < 2:
                time.sleep(1)
                continue
    return {"messages": [AIMessage(content=(
        f"[Tool-call generation failed after retries: {last_error}. "
        "Proceeding without further tool use for this turn.]"
    ))]}


def route_after_planner(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_call_turns = sum(
            1 for m in state["messages"] if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        if tool_call_turns > MAX_TOOL_CALLS:
            return "reflector"
        return "tool_executor"
    return "reflector"


tool_executor_node = ToolNode(TOOLS)


REFLECTOR_SYSTEM_PROMPT = """You are a strict quality reviewer for a research
analyst agent. Given the original question and the analyst's draft findings,
decide if the answer is complete, accurate-sounding, and well-supported.
Be skeptical: vague answers, missing numbers, or unverified claims should be
marked insufficient. If insufficient, provide ONE precise follow-up search
query that would fill the gap."""


def reflector_node(state: AgentState):
    llm = get_llm().with_structured_output(ReflectionResult)
    conversation_summary = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"] if getattr(m, "content", None)
    )
    prompt = [
        SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Original question: {state['research_question']}\n\n"
            f"Conversation so far:\n{conversation_summary}"
        ),
    ]
    result: ReflectionResult = llm.invoke(prompt)

    if not result.is_sufficient and state["iterations"] < MAX_ITERATIONS:
        follow_up = result.follow_up_query or "Please refine and expand your previous answer."
        return {
            "messages": [HumanMessage(content=f"Reviewer feedback: {result.critique}\n"
                                               f"Please address this: {follow_up}")],
            "iterations": state["iterations"] + 1,
        }
    # Sufficient, or out of iterations -> proceed to finalize
    return {"iterations": state["iterations"] + 1}


def route_after_reflector(state: AgentState, config=None):
    # If reflector just added a new HumanMessage asking for more work, loop back
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage) and state["iterations"] <= MAX_ITERATIONS:
        return "planner"
    return "finalize"


FINALIZE_SYSTEM_PROMPT = """Given the full research conversation, produce a
final structured answer. Extract concrete sources mentioned during web
searches (title + url) if any were used."""


def finalize_node(state: AgentState):
    llm = get_llm().with_structured_output(ResearchAnswer)
    conversation_summary = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"] if getattr(m, "content", None)
    )
    prompt = [
        SystemMessage(content=FINALIZE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Original question: {state['research_question']}\n\n"
            f"Full conversation:\n{conversation_summary}"
        ),
    ]
    result: ResearchAnswer = llm.invoke(prompt)
    return {"final_answer": result}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner", route_after_planner, {"tool_executor": "tool_executor", "reflector": "reflector"}
    )
    graph.add_edge("tool_executor", "planner")
    graph.add_conditional_edges(
        "reflector", route_after_reflector, {"planner": "planner", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(question: str) -> ResearchAnswer:
    app = build_graph()
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "research_question": question,
        "iterations": 0,
        "final_answer": None,
    }
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})
    return final_state["final_answer"]


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    q = " ".join(sys.argv[1:]) or "Compare AWS EC2 vs GCP Compute Engine pricing for a t3.medium-equivalent instance running 24/7 for a month."
    answer = run_agent(q)
    print("\n=== FINAL ANSWER ===")
    print(answer.model_dump_json(indent=2))
