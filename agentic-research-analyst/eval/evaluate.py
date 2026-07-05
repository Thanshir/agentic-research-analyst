"""
Automated evaluation harness.

Why this matters (resume talking point):
Almost no beginner AI projects measure whether their system actually works.
This script runs a fixed test set through the agent and scores it two ways:
  1. Keyword/substring checks (fast, deterministic, for factual questions)
  2. LLM-as-judge (for open-ended questions where exact match doesn't apply)

Run with: python -m eval.evaluate
Produces eval/results.json with per-case scores + an overall pass rate you
can literally screenshot and put in your README or resume.
"""
import json
import time
from pathlib import Path
from groq import RateLimitError

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph import run_agent
from src.config import get_llm

load_dotenv()

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

JUDGE_PROMPT = """You are grading an AI research agent's answer.
Question: {question}
Agent's answer: {answer}

Score the answer from 0 to 10 on: relevance, factual plausibility, and
completeness. Respond with ONLY a single integer 0-10, nothing else."""


def keyword_score(answer_text: str, expected_contains: list) -> float:
    if not expected_contains:
        return None
    answer_lower = answer_text.lower()
    return 1.0 if any(kw.lower() in answer_lower for kw in expected_contains) else 0.0


def llm_judge_score(question: str, answer_text: str) -> float:
    llm = get_llm()
    prompt = [
        SystemMessage(content="You are a strict, concise grading assistant."),
        HumanMessage(content=JUDGE_PROMPT.format(question=question, answer=answer_text)),
    ]
    response = llm.invoke(prompt)
    try:
        score = int("".join(c for c in response.content if c.isdigit())[:2] or "0")
        return min(score, 10) / 10
    except Exception:
        return 0.0


def run_evaluation():
    test_cases = json.loads(TEST_CASES_PATH.read_text())
    results = []
    total_score = 0.0

    for case in test_cases:
        print(f"Running: {case['id']}...")
        start = time.time()
        try:
            answer = run_agent(case["question"])
            answer_text = answer.answer
            error = None
        except Exception as e:
            answer_text = ""
            error = str(e)
        elapsed = round(time.time() - start, 2)

        if error:
            score = 0.0
        else:
            kw_score = keyword_score(answer_text, case.get("expected_contains", []))
            score = kw_score if kw_score is not None else llm_judge_score(case["question"], answer_text)

        total_score += score
        results.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": answer_text,
            "score": round(score, 2),
            "latency_seconds": elapsed,
            "error": error,
        })
        print(f"  -> score: {score:.2f}, latency: {elapsed}s")

    summary = {
        "total_cases": len(test_cases),
        "average_score": round(total_score / len(test_cases), 3),
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Average score: {summary['average_score']*100:.1f}% across {len(test_cases)} cases ===")
    print(f"Full results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
