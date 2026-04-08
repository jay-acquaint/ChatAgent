from typing import List, Tuple


def compute_confidence(reranked_docs: List[Tuple], verification, result):
    """
    Combine multiple signals into one confidence score
    """

    # 1. RERANK SCORE
    if not reranked_docs:
        rerank_score = 0.0
    else:
        scores = [float(score) for _, score in reranked_docs]
        avg_score = sum(scores) / len(scores)

        # normalize (simple scaling)
        rerank_score = max(0.0, min(1.0, avg_score / 10))


    # 2. VERIFIER SCORE
    if verification is None:
        verifier_score = 0.0
    else:
        verifier_score = verification.score if verification.is_valid else 0.0


    # 3. LLM CONFIDENCE
    llm_conf = (
        result.get("confidence", 0.0)
        if isinstance(result, dict)
        else getattr(result, "confidence", 0.0)
    )


    # FINAL FUSION
    final_confidence = (
        0.4 * rerank_score +
        0.4 * verifier_score +
        0.2 * llm_conf
    )

    return round(final_confidence, 3)