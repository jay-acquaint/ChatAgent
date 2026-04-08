from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import VerificationOutput
from llm import get_ollama_llm


def verify_answer(query: str, answer: str, docs: list):
    parser = PydanticOutputParser(pydantic_object=VerificationOutput)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template("""
    You are a strict AI verifier.

    Your job is to check if the answer is FULLY supported by the provided context.

    RULES:
    - If answer contains information NOT present in context → INVALID
    - If answer is partially supported → INVALID
    - If context is irrelevant → INVALID
    - Be strict (no guessing)

    CONTEXT:
    {context}

    ANSWER:
    {answer}

    {format_instructions}

    Question: {question}
    """)

    formatted = prompt.format_prompt(
        context=context,
        answer=answer,
        question=query,
        format_instructions=parser.get_format_instructions()
    )

    llm = get_ollama_llm(model="mistral", temperature=0.7)
    response = llm.invoke(formatted)

    result = parser.parse(response.content)

    return result