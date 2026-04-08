from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_retriever():
    # Load document
    loader = TextLoader("data/docs.txt")
    documents = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Return retriever
    return vectorstore.as_retriever(search_kwargs={"k": 5})