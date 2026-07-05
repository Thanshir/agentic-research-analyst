# Agentic Research Analyst

A self-reflecting research agent built with **LangGraph 1.0**, **LangChain 1.0**, **Groq (Llama 3.3 70B)**, and **Tavily search**. Unlike a basic RAG chatbot, this agent plans its own tool usage, critiques its own draft answers, and loops back to do more research when it judges its own output insufficient — up to a bounded number of iterations.

> Built on the current stable major versions (LangChain/LangGraph 1.0, released Oct 2025) — not the older 0.x API that most tutorials still teach. Worth mentioning in interviews: it signals you're working with the current framework, not copying stale tutorial code.

## Why this project exists

Most beginner LangChain projects are single-pass RAG chatbots: retrieve → stuff into prompt → answer. This project instead demonstrates:

- **Agent orchestration with LangGraph 1.0** — a stateful graph with conditional routing, not a linear chain. (LangChain 1.0 also ships a high-level `create_agent` builder for standard ReAct agents — this project uses a custom `StateGraph` instead because the reflection loop needs branching logic `create_agent` doesn't support out of the box. Knowing when to reach for the low-level API vs. the high-level one is itself worth explaining in interviews.)
- **Tool use** — the agent decides when to call `web_search` or `calculator`, not a hardcoded pipeline
- **Self-reflection loop** — a dedicated reviewer step that scores the draft answer and can send the agent back to do more research (bounded at 3 iterations to avoid infinite loops)
- **Structured, validated output** — every response is a Pydantic model (`ResearchAnswer`), not raw text, so it's safe to consume downstream
- **Automated evaluation** — a 10-case test suite scored with a keyword-match + LLM-as-judge hybrid, producing a measurable pass rate
- **Observability** — LangSmith tracing enabled, so every run's full trace (tool calls, retries, latency, token usage) is inspectable
- **Provider-agnostic LLM layer** — swap between Groq (cloud, reliable tool-calling) and Ollama (local, free, offline) via one env var
- **Served as a real API** (FastAPI) with a demo UI (Streamlit), containerized with Docker

## Architecture

```
START -> planner -> [tool calls?] -> tool_executor -> planner (loop)
                   -> [no tool calls] -> reflector -> [insufficient?] -> planner (loop, max 3x)
                                                     -> [sufficient] -> finalize -> END
```

- **planner**: LLM decides what to do — call a tool, or produce a draft answer
- **tool_executor**: runs whichever tool(s) the planner requested
- **reflector**: a separate LLM call critiques the draft against the original question; if weak, it generates a precise follow-up query and loops back
- **finalize**: converts the final conversation into a validated `ResearchAnswer` object

## Setup

```bash
git clone <your-repo-url>
cd agentic-research-analyst
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `GROQ_API_KEY` — free at https://console.groq.com
- `TAVILY_API_KEY` — free tier at https://tavily.com
- (optional) `LANGCHAIN_API_KEY` — free at https://smith.langchain.com, enables tracing

## Running it

**CLI (single question):**
```bash
python -m src.graph "Compare AWS EC2 vs GCP Compute Engine pricing for a small workload"
```

**API server:**
```bash
uvicorn app:app --reload --port 8000
# then: curl -X POST localhost:8000/research -H "Content-Type: application/json" -d '{"question": "..."}'
```

**Streamlit UI:**
```bash
streamlit run ui.py
```

**Run the eval suite:**
```bash
python -m eval.evaluate
# writes eval/results.json with per-case scores and an overall average
```

**Docker:**
```bash
docker build -t research-agent .
docker run -p 8000:8000 --env-file .env research-agent
```

## Project structure

```
agentic-research-analyst/
├── src/
│   ├── config.py      # LLM provider abstraction (Groq / Ollama)
│   ├── schemas.py      # Pydantic output models
│   ├── tools.py         # web_search, calculator
│   └── graph.py         # LangGraph agent definition
├── eval/
│   ├── test_cases.json  # 10 evaluation questions across categories
│   └── evaluate.py      # scoring harness
├── app.py                # FastAPI service
├── ui.py                 # Streamlit demo UI
├── Dockerfile
└── requirements.txt
```

## Known limitations (worth stating honestly in interviews)

- Reflection loop is bounded at 3 iterations to control cost/latency — a production system might use adaptive stopping based on confidence trends
- Eval suite uses keyword-matching for factual questions and LLM-as-judge for open-ended ones; a larger production eval would use human-labeled golden answers
- No persistent memory across sessions yet — each question is independent
