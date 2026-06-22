# PDF Question-Answering RAG System

A chatbot that answers questions from PDF documents using LangChain, ChromaDB, and Ollama.

## Setup

### Prerequisites

**Common**

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

Pull the embedding model used by both options:

```bash
ollama pull nomic-embed-text
```

Start Ollama if it isn't already running as a background service:

```bash
ollama serve
```

**Option A only** (fully local)

Pull the LLM:

```bash
ollama pull qwen2.5:14b
```

**Option B only**

- Anthropic API key

### Install

```bash
uv sync
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

#### Voice input (optional)

Ask questions by speaking instead of typing (English, push-to-talk):

```bash
uv run python cli.py chat --voice
```

Press Enter to start recording, speak your question, then press Enter to stop.
The question is transcribed locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(the `base.en` model downloads automatically on first use) and answered as text.
Press Ctrl-C to stop. Requires a working microphone.

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--voice`, `-v` | off | Ask questions by speaking (push-to-talk, English) |
| `--chroma-dir` | `./chroma_db` | ChromaDB persist directory |
| `--collection` | `lang_chain_qa` | ChromaDB collection name |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL |

---

### Option B — Gradio web UI (`src/`)

#### Prerequisites

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY='your-key-here'
```

#### 1. Load a PDF

```bash
uv run python -c "from src.loader import load_data_ollama; load_data_ollama('path/to/your.pdf')"
```

#### 2. Start the chat

```bash
uv run python src/chat.py
```

The interface will be available at http://localhost:7860.
