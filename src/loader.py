from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import chromadb
import os

PDF_PATH = "./source-data/my-text.pdf"
PAGE_LIMIT = 126
CHROMA_PATH = "./chroma_db"
COLLECTION_OPENAI = "my_pdf_collection"
COLLECTION_OLLAMA = "my_ollama"


def _load_pdf(pdf_path: str):
    if not pdf_path:
        raise ValueError("pdf_path must not be empty")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()[:PAGE_LIMIT]
    print(f"Loaded {len(pages)} pages")
    return pages


def _reset_collection(collection_name: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        pass  # collection didn't exist


def _verify_search(vectorstore: Chroma):
    print("Verifying with similarity search...")
    results = vectorstore.similarity_search("What is the first step?", k=3)
    for doc in results:
        print(doc.page_content)
        print(doc.metadata)


def load_data_ollama(pdf_path: str):
    """Load PDF into ChromaDB using local Ollama embeddings (nomic-embed-text)."""
    pages = _load_pdf(pdf_path)

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,   # ~512 tokens
        chunk_overlap=400, # ~20% overlap
    )
    chunks = splitter.split_documents(pages)

    _reset_collection(COLLECTION_OLLAMA)

    print("Creating embeddings using Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_OLLAMA,
    )
    print(f"Loaded {len(chunks)} chunks into ChromaDB")
    _verify_search(vectorstore)


def load_data_openai(pdf_path: str):
    """Load PDF into ChromaDB using OpenAI embeddings (requires OPENAI_API_KEY)."""
    pages = _load_pdf(pdf_path)

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=256,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)

    _reset_collection(COLLECTION_OPENAI)

    print("Creating embeddings using OpenAI...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_OPENAI,
    )
    print(f"Loaded {len(chunks)} chunks into ChromaDB")
    _verify_search(vectorstore)


if __name__ == "__main__":
    print("Starting data loading process...")
    load_data_ollama(PDF_PATH)
    print("Done.")
