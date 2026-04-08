from langchain_core.prompts import ChatPromptTemplate
from llm import get_ollama_llm



def reflect_answer(query: str, answer: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
    You are an AI reviewer.

    Your job:
    Improve the given answer.

    Rules:
    - Keep it factually correct
    - Make it clearer and more complete
    - Do NOT add new information not related to the query
    - Keep it concise but better

    QUESTION:
    {query}

    CURRENT ANSWER:
    {answer}

    Return ONLY the improved answer.
    """)

    formatted = prompt.format_prompt(
        query=query,
        answer=answer
    )

    llm = get_ollama_llm(model="mistral", temperature=0)
    response = llm.invoke(formatted)

    return response.content.strip()