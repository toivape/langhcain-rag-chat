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

### 1. Load a PDF

```bash
uv run python -c "from src.loader import load_data_ollama; load_data_ollama('path/to/your.pdf')"
```

### 2. Start the chat

```bash
uv run python src/chat.py
```

The interface will be available at http://localhost:7860.
