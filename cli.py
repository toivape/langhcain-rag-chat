"""
PDF Q&A CLI using ChromaDB + OllamaEmbeddings + ChatAnthropic.

Usage:
  python cli.py index --pdf ./source-data/my-text.pdf
  python cli.py chat
"""

import os
import sys
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

import chromadb
from tqdm import tqdm
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION = "lang_chain_qa"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant that answers questions based solely on the provided document context.\n\n"
        "{context}\n\n"
        "If the answer is not in the context, say you don't have enough information in the document."
    )),
    ("human", "{input}"),
])


# ── helpers ───────────────────────────────────────────────────────────────────

def make_embeddings(ollama_url: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)


def load_pdf(pdf_path: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    print(f"Loading PDF: {pdf_path}")
    pages = PyPDFLoader(pdf_path).load()
    print(f"  {len(pages)} pages loaded")
    return pages


def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=400)
    chunks = splitter.split_documents(pages)
    print(f"  {len(chunks)} chunks created")
    return chunks


# ── index mode ────────────────────────────────────────────────────────────────

def _reset_collection(chroma_dir: str, collection: str) -> None:
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(collection)
        print(f"  Deleted existing collection '{collection}'")
    except Exception:
        pass


def build_index(pdf_path: str, chroma_dir: str, collection: str, ollama_url: str) -> None:
    try:
        embeddings = make_embeddings(ollama_url)
    except Exception as e:
        sys.exit(f"Failed to initialise Ollama embeddings: {e}\nIs Ollama running at {ollama_url}?")

    try:
        pages = load_pdf(pdf_path)
    except FileNotFoundError as e:
        sys.exit(str(e))

    chunks = chunk_documents(pages)
    _reset_collection(chroma_dir, collection)
    print("Embedding chunks and storing in ChromaDB...")
    batch_size = 50
    try:
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        vectorstore = None
        for batch in tqdm(batches, desc="Embedding", unit="batch"):
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=chroma_dir,
                    collection_name=collection,
                )
            else:
                vectorstore.add_documents(batch)
    except Exception as e:
        sys.exit(f"Embedding failed: {e}\nEnsure Ollama is running and nomic-embed-text is available.")

    print(f"ChromaDB collection '{collection}' saved to: {chroma_dir}/")


# ── chat mode ─────────────────────────────────────────────────────────────────

def load_index(chroma_dir: str, collection: str, embeddings: OllamaEmbeddings) -> Chroma:
    return Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
        collection_name=collection,
    )


def make_chain(vectorstore: Chroma):
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.2, max_tokens=2000)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, PROMPT))


def interactive_loop(chain) -> None:
    console.print(Panel("[bold green]PDF Q&A ready[/bold green] — type [italic]quit[/italic] or [italic]exit[/italic] to stop."))
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            question = session.prompt("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]Bye.[/bold]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[bold]Bye.[/bold]")
            break

        try:
            result = chain.invoke({"input": question})
        except Exception as e:
            msg = str(e)
            if "ANTHROPIC_API_KEY" in msg or "authentication" in msg.lower():
                console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY is not set or is invalid.")
            else:
                console.print(f"[bold red]Error:[/bold red] {e}")
            continue

        console.print(Markdown(result["answer"]))

        pages = sorted({
            doc.metadata["page"] + 1
            for doc in result.get("context", [])
            if "page" in doc.metadata
        })
        if pages:
            page_list = ", ".join(str(p) for p in pages)
            console.print(f"[dim](Sources: pages {page_list})[/dim]")


def run_chat(chroma_dir: str, collection: str, ollama_url: str) -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    try:
        print("Initialising embeddings...")
        embeddings = make_embeddings(ollama_url)
    except Exception as e:
        sys.exit(f"Failed to initialise Ollama embeddings: {e}")

    try:
        print(f"Loading vectorstore from ChromaDB collection '{collection}'...")
        vectorstore = load_index(chroma_dir, collection, embeddings)
    except Exception as e:
        sys.exit(f"Failed to load ChromaDB collection: {e}")

    if vectorstore.get(limit=1)["ids"] == []:
        sys.exit(
            f"Collection '{collection}' is empty.\n"
            f"Run:  python cli.py index --pdf <your.pdf>"
        )

    print(f"Loaded ChromaDB collection '{collection}' from '{chroma_dir}'")
    interactive_loop(make_chain(vectorstore))


# ── CLI ───────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, help="PDF Q&A via ChromaDB + OllamaEmbeddings + ChatAnthropic")


@app.command()
def index(
    pdf: Path = typer.Option(..., help="Path to the PDF file"),
    ollama_url: str = typer.Option(DEFAULT_OLLAMA_URL, help="Ollama base URL"),
    chroma_dir: str = typer.Option(DEFAULT_CHROMA_DIR, help="ChromaDB persist directory"),
    collection: str = typer.Option(DEFAULT_COLLECTION, help="ChromaDB collection name"),
):
    """Load a PDF into ChromaDB."""
    build_index(str(pdf), chroma_dir, collection, ollama_url)


@app.command()
def chat(
    ollama_url: str = typer.Option(DEFAULT_OLLAMA_URL, help="Ollama base URL"),
    chroma_dir: str = typer.Option(DEFAULT_CHROMA_DIR, help="ChromaDB persist directory"),
    collection: str = typer.Option(DEFAULT_COLLECTION, help="ChromaDB collection name"),
):
    """Interactive Q&A against an existing ChromaDB collection."""
    run_chat(chroma_dir, collection, ollama_url)


if __name__ == "__main__":
    app()
