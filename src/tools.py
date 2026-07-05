"""
Tools available to the agent.

Keeping tools in their own module (vs. inline in the graph) mirrors how
real agent codebases are organized, and makes it trivial to add/remove
tools without touching the graph logic.
"""
import os
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

_search_tool = None


def get_search_tool():
    """Lazy-init so importing this module doesn't require the API key immediately."""
    global _search_tool
    if _search_tool is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not set. Get a free key at https://tavily.com"
            )
        _search_tool = TavilySearch(max_results=4, tavily_api_key=api_key)
    return _search_tool


@tool
def search_web(query: str) -> str:
    """Search the internet for current, factual information on a topic.
    Use this whenever you need up-to-date facts, prices, comparisons, or
    anything you're not fully certain about from memory.
    Args:
        query: the exact search query string to look up, e.g. "AWS EC2 pricing 2026"
    """
    tool_instance = get_search_tool()
    results = tool_instance.invoke({"query": query})
    # Tavily returns a dict with a 'results' list
    items = results.get("results", []) if isinstance(results, dict) else results
    formatted = []
    for item in items:
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")[:500]
        formatted.append(f"Title: {title}\nURL: {url}\nContent: {content}\n")
    return "\n---\n".join(formatted) if formatted else "No results found."


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '1200 * 0.15' or '(500-200)/2'.
    Only use this for arithmetic — not for anything requiring web lookup."""
    # A restricted eval — only digits, operators, parentheses, decimal points
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


TOOLS = [search_web, calculator]
