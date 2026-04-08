from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

from llm import get_ollama_llm


# Schema (SAFE parsing)
class QueryRefinerOutput(BaseModel):
    queries: List[str] = Field(
        description="List of improved queries",
        min_items=1,
        max_items=3
    )


# Main Function
def refine_query(
    original_query: str,
    retries: int,
    past_queries: Optional[List[str]] = None,
    failures: Optional[List[str]] = None
) -> List[str]:

    past_queries = past_queries or []
    failures = failures or []

    parser = PydanticOutputParser(pydantic_object=QueryRefinerOutput)

    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant improving search queries.

    Original query:
    {query}

    Previous queries (DO NOT repeat):
    {past_queries}

    Failures:
    {failures}

    Retry attempt: {retries}

    TASK:
    Generate EXACTLY 3 NEW and DIFFERENT queries:
    - Avoid repeating previous queries
    - Fix issues from failures
    - Make them clearer and more specific
    - Focus on the core concept

    {format_instructions}
    """)

    formatted = prompt.format_prompt(
        query=original_query,
        retries=retries,
        past_queries=past_queries,
        failures=failures,
        format_instructions=parser.get_format_instructions()
    )

    llm = get_ollama_llm(model="mistral", temperature=0.3)

    response = llm.invoke(formatted)

    try:
        parsed = parser.parse(response.content)

        # Extra safety: remove duplicates
        unique_queries = list(dict.fromkeys(parsed.queries))

        # Remove already used queries
        filtered = [q for q in unique_queries if q not in past_queries]

        return filtered if filtered else [original_query]

    except Exception as e:
        print(f"[QueryRefiner Error] {e}")

        # Safe fallback
        return [original_query]