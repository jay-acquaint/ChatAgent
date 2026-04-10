from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

_retriever = None  # module-level singleton

def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    loader = TextLoader("data/docs.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,      # larger chunks = fewer LLM tokens needed
        chunk_overlap=40
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so it survives restarts
    vectorstore.save_local("faiss_index")

    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever


def load_retriever():
    """Load persisted index at startup instead of rebuilding."""
    global _retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception:
        _retriever = get_retriever()
    return _retriever