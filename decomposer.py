from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import QueryDecomposerOutput
from llm import get_ollama_llm



def query_decomposer(query: str) -> list[str]:

    # Ollama LLM
    llm = get_ollama_llm(model="mistral", temperature=0.7)

    # Initialize the parser for structured output
    parser = PydanticOutputParser(pydantic_object=QueryDecomposerOutput)

    multi_query_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant.

        Generate exactly 3 different versions of the user question
        to improve document retrieval.

        Return ONLY a JSON object in this format:

        {format_instructions}

        Question: {question}
        """)

    mq_formatted = multi_query_prompt.format_prompt(
        question=query,
        format_instructions=parser.get_format_instructions()
    )
    mq_response = llm.invoke(mq_formatted)

    parsed = parser.parse(mq_response.content)

    return parsed.queries
