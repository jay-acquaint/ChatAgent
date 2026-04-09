import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from graph import build_graph
from schemas import QueryRequest, QueryResponse
from fastapi.responses import StreamingResponse
import asyncio
from memory_redis import get_history, save_message

# Initialize FastAPI
app = FastAPI(title="Agentic AI API")

# Build graph once at startup
graph = build_graph()

# API Endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    # Get or generate the session_id
    session_id = request.session_id or str(uuid.uuid4())

    print(f"\n[API] Query: {request.query}")
    print(f"[API] Session: {request.session_id}")

    # Load memory
    history = get_history(session_id)

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
    result = graph.invoke(state)
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