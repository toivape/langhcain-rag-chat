# PDF Question-Answering RAG System

A chatbot that answers questions from PDF documents using LangChain, ChromaDB, Ollama, and Anthropic Claude.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with the `nomic-embed-text` model
- Anthropic API key

### Install

```bash
uv sync
```

### API key

```bash
export ANTHROPIC_API_KEY='your-key-here'
```

## Usage

### Option A — CLI (`cli.py`)

#### 1. Index a PDF

```bash
uv run python cli.py index --pdf source-data/my-text.pdf
```

This embeds the PDF and stores it in ChromaDB (`./chroma_db`, collection `lang_chain_qa`).

#### 2. Start interactive Q&A

```bash
uv run python cli.py chat
```

Type your question at the prompt. Enter `quit` or `exit` to stop.

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--chroma-dir` | `./chroma_db` | ChromaDB persist directory |
| `--collection` | `lang_chain_qa` | ChromaDB collection name |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL |

---

### Option B — Gradio web UI (`src/`)

#### 1. Load a PDF

```bash
uv run python -c "from src.loader import load_data_ollama; load_data_ollama('path/to/your.pdf')"
```

#### 2. Start the chat

```bash
uv run python src/chat.py
```

The interface will be available at http://localhost:7860.
