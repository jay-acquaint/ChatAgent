from sentence_transformers import CrossEncoder

# Load once (IMPORTANT)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, docs: list, top_k: int = 3):
    """
    Rerank documents based on relevance to query
    """
 
    # Prepare pairs: (query, doc)
    pairs = [(query, doc.page_content) for doc in docs]

    # Get scores
    scores = model.predict(pairs)

    # Attach scores
    scored_docs = list(zip(docs, scores))

    # Sort by score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Take top_k
    top_docs = scored_docs[:top_k]

    return top_docs