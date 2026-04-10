from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from decomposer import query_decomposer
from rag import get_retriever
from reranker import rerank
from research_agent import generate_answer
from verifier import verify_answer
from query_refiner import refine_query
from tools import web_search_tool
from reflection import reflect_answer
from confidence import compute_confidence


# STATE (shared memory)
class GraphState(TypedDict):
    query: str
    queries: List[str]
    chat_history: List[str]
    docs: List
    reranked_docs: List
    result: Optional[dict]
    verification: Optional[dict]
    retries: int
    past_queries: List[str]
    failed_attempts: List[str]
    reflection: str


# NODES
def decompose_node(state: GraphState):
    queries = query_decomposer(state["query"])
    print("\n[Decompose] ->", queries)
    return {
        "queries": queries,
        "past_queries": queries[:]
    }


def retrieve_node(state: GraphState):
    retriever = get_retriever()
    all_docs = []

    for q in state["queries"]:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # Deduplicate
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    print(f"\n[Retrieve] -> {len(unique_docs)} docs")
    return {"docs": unique_docs}


def rerank_node(state: GraphState):
    reranked = rerank(state["query"], state["docs"], top_k=5)

    # FILTER BAD DOCS
    filtered = [(doc, score) for doc, score in reranked if score > 0]

    print("\n[Rerank] Top filtered docs:")
    for doc, score in filtered:
        print(f"Score: {score:.4f}")

    return {"reranked_docs": filtered}


def generate_node(state: GraphState):
    # Handle no relevant docs
    if not state["reranked_docs"]:
        print("\n[Generate] No relevant docs → returning fallback")

        return {
            "result": {
                "answer": "I don't know based on the provided context.",
                "summary": [
                    "No relevant data found",
                    "Cannot answer from context",
                    "Try another query"
                ],
                "example": "N/A",
                "sources": [],
                "confidence": 0.0
            }
        }

    top_docs = [doc for doc, _ in state["reranked_docs"]]

    result = generate_answer(state["query"], top_docs)

    print("\n[Generate] Done")
    return {"result": result}


def verify_node(state: GraphState):
    # Skip verification if no docs
    if not state["reranked_docs"]:
        print("\n[Verify] Skipped (no docs)")
        return {"verification": None}

    top_docs = [doc for doc, _ in state["reranked_docs"]]

    result = state["result"]

    # Handle dict vs object
    answer = (
        result.get("answer")
        if isinstance(result, dict)
        else result.answer
    )

    verification = verify_answer(
        state["query"],
        answer,
        top_docs
    )

    print(f"\n[Verify] Valid: {verification.is_valid}, Score: {verification.score}")
    print(f"[Verify] Feedback: {verification.feedback}")

    return {"verification": verification}


def evaluate_node(state: GraphState):
    result = state.get("result")
    reranked_docs = state.get("reranked_docs", [])
    verification = state.get("verification")

    fused_confidence = compute_confidence(
        reranked_docs,
        verification,
        result
    )

    retries = state.get("retries", 0)

    print(f"\n[Evaluate] Fused Confidence: {fused_confidence}, Retries: {retries}")

    # Inject fused confidence into result
    if isinstance(result, dict):
        result["confidence"] = fused_confidence
    else:
        result.confidence = fused_confidence

    return {
        "result": result,
        "retries": retries
    }


def retry_node(state: GraphState):
    retries = state.get("retries", 0) + 1

    print(f"\n[Retry] Attempt #{retries}")

    past_queries = state.get("past_queries", [])
    failed_attempts = state.get("failed_attempts", [])
    reflection = state.get("reflection", "")

    if reflection:
        failed_attempts.append(f"Reflection: {reflection[:100]}")

    # PASS MEMORY HERE (FIX)
    new_queries = refine_query(
        original_query=state["query"],
        retries=retries,
        past_queries=past_queries,
        failures=failed_attempts
    )

    new_queries = [q for q in new_queries if q not in past_queries]

    print(f"[Retry] Filtered New Queries: {new_queries}")
    print(f"[Memory] Reflection used: {reflection[:100]}")

    # Update memory
    updated_past = past_queries + new_queries

    return {
        "retries": retries,
        "queries": new_queries if new_queries else state["queries"],
        "past_queries": updated_past,
        "failed_attempts": failed_attempts
    }


def tool_node(state: GraphState):
    print("\n[Tool] Using Web Search Tool 🌍")

    query = state["query"]

    tool_result = web_search_tool(query)

    print(f"[Tool Output]: {tool_result[:200]}...")

    # Convert to pseudo-doc format
    class ToolDoc:
        def __init__(self, content):
            self.page_content = content

    docs = [ToolDoc(tool_result)]

    return {
        "docs": docs,
        "reranked_docs": [(docs[0], 1.0)]
    }


def reflect_node(state: GraphState):
    result = state["result"]

    answer = (
        result.get("answer")
        if isinstance(result, dict)
        else result.answer
    )

    improved = reflect_answer(state["query"], answer)

    print("\n[Reflect] Improved Answer Generated")

    reflection_feedback = improved

    # Update result
    if isinstance(result, dict):
        result["answer"] = improved
    else:
        result.answer = improved

    return {
        "result": result,
        "reflection": reflection_feedback
    }


def context_node(state: GraphState):
    history = state.get("chat_history", [])

    if not history:
        return {"query": state["query"]}

    # Build conversational context
    formatted_history = "\n".join(
        [f"User: {msg}" for msg in history]
    )

    full_query = f"""
    Conversation so far:
    {formatted_history}

    Current question:
    {state["query"]}
    """

    print("\n[Context] Injected chat history")

    return {"query": full_query}


# ROUTER (CORE LOGIC)
def route_after_evaluate(state: GraphState):
    result = state["result"]

    # Confidence
    confidence = (
        result.get("confidence", 0.0)
        if isinstance(result, dict)
        else getattr(result, "confidence", 0.0)
    )

    # Verifier
    verification = state.get("verification")
    is_valid = (
        verification.is_valid
        if verification is not None
        else False
    )

    retries = state.get("retries", 0)

    print(f"\n[Routing] Confidence: {confidence}, Valid: {is_valid}, Retries: {retries}")

    if confidence > 0.8:
        print("[Decision] High confidence → Finish ✅")
        return "end"

    # Decision logic
    if retries < 2:
        if not is_valid or confidence < 0.6:
            print("[Decision] Verifier failed → Retry 🔁")
            return "retry"

    # After retries exhausted → USE TOOL
    if not is_valid or confidence < 0.6:
        print("[Decision] Use Tool 🌍")
        return "tool"

    print("[Decision] → Finish ✅")
    return "end"


# BUILD GRAPH
def build_graph():
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("context", context_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("verify", verify_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("retry", retry_node)
    builder.add_node("tool", tool_node)

    # Entry
    builder.set_entry_point("context")

    # Flow
    builder.add_edge("context", "decompose")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "reflect")
    builder.add_edge("reflect", "verify")
    builder.add_edge("verify", "evaluate")

    # Conditional routing
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "retry": "retry",
            "tool": "tool",
            "end": END
        }
    )

    builder.add_edge("tool", "generate")
    builder.add_edge("retry", "retrieve") # Retry goes back to retrieval

    return builder.compile()