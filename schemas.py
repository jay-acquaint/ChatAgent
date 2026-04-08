# Schema
from pydantic import BaseModel, Field
from typing import Optional


class RAGOutput(BaseModel):
    answer: str = Field(description="Clear explanation")
    summary: list[str] = Field(
        description="Exactly 3 bullet points",
        min_items=1,
        max_items=3
    )
    example: str = Field(description="Real-world example")
    sources: list[str] = Field(
        description="List of supporting context snippets",
        min_items=1
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )


class QueryDecomposerOutput(BaseModel):
    queries: list[str] = Field(
        description="List of decompressed queries",
        min_items=1,
        max_items=3
    )


class VerificationOutput(BaseModel):
    is_valid: bool = Field(description="Whether answer is grounded in context")
    score: float = Field(description="Confidence score between 0 and 1")
    feedback: str = Field(description="Why the answer is valid/invalid")


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    summary: list[str]
    confidence: float
    sources: list