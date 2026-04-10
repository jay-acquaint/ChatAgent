import uuid
import hashlib
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from graph import build_graph
from schemas import QueryRequest, QueryResponse
from fastapi.responses import StreamingResponse
import asyncio
from memory_redis import get_history, save_message
from rag import load_retriever
from memory_redis import redis_client

# Initialize FastAPI
app = FastAPI(title="RAG API")


@app.on_event("startup")
async def startup_event():
    load_retriever()  # warm the index once
    print("[Startup] FAISS index loaded ✅")


# Build graph once at startup
graph = build_graph()


# cache handler
def get_cache_key(query: str) -> str:
    return f"response_cache:{hashlib.md5(query.strip().lower().encode()).hexdigest()}"


# API Endpoint
@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Get or generate the session_id
    session_id = request.session_id or str(uuid.uuid4())
    cache_key = get_cache_key(request.query)

    print(f"\n[API] Query: {request.query}")
    print(f"[API] Session: {request.session_id}")

    cached = await asyncio.get_event_loop().run_in_executor(
        None, redis_client.get, cache_key
    )
    if cached:
        print("[Cache HIT] Returning cached response")
        response = json.loads(cached)
        response["session_id"] = session_id
        save_message(session_id, f"User: {request.query}")
        save_message(session_id, f"AI: {response["answer"]}")
        return response

    # Load memory
    history = await asyncio.get_event_loop().run_in_executor(
        None, get_history, session_id
    )

    # Build state
    state = {
        "query": request.query,
        "chat_history": history,
        "retries": 0,
        "past_queries": [],
        "failed_attempts": []
    }
    print("state--------->>>>>>>>", state)
    # Run graph
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, graph.invoke, state)
    final = result.get("result")

    # Extract answer
    if isinstance(final, dict):
        answer = final["answer"]
        response = final
    else:
        answer = final.answer
        response = {
            "answer": final.answer,
            "summary": final.summary,
            "confidence": final.confidence,
            "sources": final.sources,
        }

    # Save conversation
    save_message(session_id, f"User: {request.query}")
    save_message(session_id, f"AI: {answer}")

    response["session_id"] = session_id

    # Store in cache (TTL: 1 hour)
    redis_client.setex(cache_key, 3600, json.dumps({
        "answer": answer,
        "summary": response.get("summary", []),
        "confidence": response.get("confidence", 0.0),
        "sources": response.get("sources", []),
    }))

    return response


async def stream_response(graph, state):
    result = graph.invoke(state)

    final = result.get("result")

    answer = (
        final.get("answer")
        if isinstance(final, dict)
        else final.answer
    )

    # Simulate token streaming
    for word in answer.split():
        yield word + " "
        await asyncio.sleep(0.05)


@app.post("/ask-stream")
async def ask_stream(request: QueryRequest):
    import uuid

    session_id = request.session_id or str(uuid.uuid4())

    history = get_history(session_id)

    state = {
        "query": request.query,
        "chat_history": history,
        "retries": 0,
        "past_queries": [],
        "failed_attempts": []
    }

    async def generator():
        result = graph.invoke(state)
        final = result.get("result")

        answer = (
            final.get("answer")
            if isinstance(final, dict)
            else final.answer
        )

        # Save memory
        save_message(session_id, f"User: {request.query}")
        save_message(session_id, f"AI: {answer}")

        # Stream
        for char in answer:
            yield char
            await asyncio.sleep(0.01)

    return StreamingResponse(generator(), media_type="text/plain", headers={
        "X-Session-ID": session_id
    })


# Health Check
@app.get("/")
def root():
    return {"message": "Agentic AI API is running 🚀"}