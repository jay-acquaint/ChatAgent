from graph import build_graph


# Query
query = "What is RAG?"
# query = "What is quantum physics?"


# Execute graph
graph = build_graph()
result = graph.invoke({
    "query": query,
    "retries": 0,
    "past_queries": [],
    "failed_attempts": []
})


# FINAL OUTPUT
print("\n=== FINAL ANSWER ===")
print(result["result"])