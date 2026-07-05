"""
LLM provider abstraction.

Why this exists (resume talking point):
Hard-coding a single LLM provider is brittle and expensive to change later.
This module lets the whole app swap between a free cloud provider (Groq, fast
+ reliable tool-calling) and a fully local provider (Ollama, zero cost, works
offline) with a single environment variable. This is the kind of provider
abstraction real production LLM systems need.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()


def get_llm(temperature: float = 0.0):
    """
    Returns a chat model based on LLM_PROVIDER env var.

    Groq is the default because small local models (7B-8B) are unreliable
    at structured tool-calling, which this agent depends on heavily.
    Ollama is kept as a fallback for fully offline / zero-cost demos.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        return ChatOllama(model=model, temperature=temperature)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'groq' or 'ollama'.")
