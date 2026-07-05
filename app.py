"""
FastAPI service exposing the agent as an HTTP API.

Running this + pointing the Streamlit UI at it (or calling it with curl/Postman)
demonstrates you can ship an agent as a real service, not just a script.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.graph import run_agent
from src.schemas import ResearchAnswer

load_dotenv()

app = FastAPI(
    title="Agentic Research Analyst",
    description="A self-reflecting research agent built with LangGraph.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchAnswer)
def research(request: QuestionRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return run_agent(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn app:app --reload --port 8000
