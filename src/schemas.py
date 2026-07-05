"""
Structured output schemas.

Why this matters (resume talking point):
Most beginner LangChain projects just return raw strings from the LLM.
Production systems need *validated, structured* output so downstream code
(APIs, databases, UIs) can rely on a guaranteed shape. Pydantic + LangChain's
`with_structured_output` enforces this at the model layer.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str = Field(description="Title or short description of the source")
    url: Optional[str] = Field(default=None, description="URL if available")


class ResearchAnswer(BaseModel):
    """Final structured answer produced by the agent."""

    answer: str = Field(description="The direct, complete answer to the research question")
    confidence: float = Field(
        description="Model's self-assessed confidence in this answer, from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Brief explanation of how the answer was derived")
    sources: List[Source] = Field(
        default_factory=list, description="Sources used to support the answer"
    )
    needs_more_research: bool = Field(
        description="True if the model believes the answer is incomplete or low quality"
    )


class ReflectionResult(BaseModel):
    """Output of the self-critique / reflection step."""

    is_sufficient: bool = Field(
        description="True if the current answer sufficiently addresses the question"
    )
    critique: str = Field(description="What is missing or wrong, if anything")
    follow_up_query: Optional[str] = Field(
        default=None,
        description="If more research is needed, the exact follow-up search query to run",
    )
