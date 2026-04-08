from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import llm
from schemas import RAGOutput
from llm import get_ollama_llm



def generate_answer(query: str, docs: list):
    """
    Research Agent:
    - Uses reranked docs
    - Produces grounded, structured answer
    """
    parser = PydanticOutputParser(pydantic_object=RAGOutput)

    # Build context with IDs (important for sources)
    context_blocks = []
    for i, doc in enumerate(docs):
        context_blocks.append(f"[{i}] {doc.page_content}")

    context = "\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_template("""
    You are an advanced AI Research Assistant.

    STRICT RULES:
    - UsSE ONLY THE PROVIDED CONTEXT
    - Do NOT use prior knowledge
    - IF ANSWER IS MISSING THAN SAY ONLY THIS - "I don't know"
    - Be precise and factual

    TASK:
    1. Understand the question
    2. Extract relevant info from context
    3. Generate a clear answer
    4. Provide EXACTLY 3 bullet points summary
    5. Give a real-world example
    6. Cite sources using [index] numbers

    CONTEXT:
    {context}

    {format_instructions}

    Question: {question}
    """)

    formatted = prompt.format_prompt(
        context=context,
        question=query,
        format_instructions=parser.get_format_instructions()
    )

    llm = get_ollama_llm(model="mistral", temperature=0.7)
    response = llm.invoke(formatted)

    result = parser.parse(response.content)
    result.sources = [
        doc.page_content[:200]
        for doc in docs[:3]
    ]

    return result