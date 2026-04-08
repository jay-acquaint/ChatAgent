from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from llm import get_ollama_llm 



def web_search_tool(query: str) -> str:
    """
    Simulated web search using LLM knowledge
    (Replace later with real API)
    """

    prompt = ChatPromptTemplate.from_template("""
    You are a web search engine.

    Provide a factual and concise answer to the query.

    Query:
    {query}
    """)

    formatted = prompt.format_prompt(query=query)

    llm = get_ollama_llm(model="mistral", temperature=0.7)
    response = llm.invoke(formatted)

    return response.content