"""
In this we define different LLMs we want to use across the app. This way, if we want to switch to a different model, we can do it in one place.
"""
from langchain_ollama import ChatOllama


# Ollsma + mistral model
def get_ollama_llm(model="mistral", temperature=0.7):
    return ChatOllama(
        model=model,
        temperature=temperature,
        # base_url="http://ollama:11434"  # Use the service name defined in docker-compose   
    )