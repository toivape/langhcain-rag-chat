"""
Gradio chat interface for PDF question-answering using RAG.
"""

import re
import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_vectorstore() -> Chroma:
    """Load the existing ChromaDB vectorstore."""
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="my_ollama",
    )


def format_docs(docs) -> str:
    """Format retrieved documents into a single string with metadata."""
    formatted = []
    for doc in docs:
        page_info = (
            f"[Page {doc.metadata.get('page', 'unknown') + 1}]"
            if "page" in doc.metadata
            else ""
        )
        formatted.append(f"{page_info}\n{doc.page_content}")
    return "\n\n".join(formatted)


def _format_source_citation(source_docs) -> str:
    """Return a source citation footer, or empty string if no page metadata."""
    pages = {doc.metadata["page"] + 1 for doc in source_docs if "page" in doc.metadata}
    if not pages:
        return ""
    return f"\n\n*Source: Pages {', '.join(str(p) for p in sorted(pages))}*"


def create_rag_chain(vectorstore: Chroma):
    """Create a RAG chain using LangChain Expression Language (LCEL)."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 30,
            "lambda_mult": 0.5,  # balance between relevance (0) and diversity (1)
        },
    )

    template = """You are a helpful AI assistant that answers questions based on the provided context from a PDF document.

Context from the document (multiple excerpts):
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- The context includes multiple excerpts from different pages - synthesize information across all relevant excerpts
- If the answer cannot be found in any of the context excerpts, say "I don't have enough information in the document to answer that question."
- Be concise but thorough in your answers
- When you find the answer, cite the specific page number(s) where you found it
- Use a friendly, professional tone

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def extract_page_number(query: str) -> int | None:
    """
    Extract page number from a user query.

    Examples:
        "summarize page 21" -> 21
        "what's on page 5?" -> 5
        "tell me about pages 10-12" -> 10
    """
    patterns = [
        r"\bpage\s+(\d+)",
        r"\bpages\s+(\d+)",
        r"\bp\.\s*(\d+)",
        r"\bpg\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return int(match.group(1))
    return None


def is_comprehensive_query(query: str) -> bool:
    """Detect if the query asks for comprehensive information (e.g., "all 12 steps")."""
    patterns = [
        r"\ball\s+\d+",
        r"\ball\s+(?:the\s+)?steps",
        r"\bevery\s+step",
        r"\beach\s+step",
        r"\bcomplete\s+list",
        r"\bfull\s+list",
        r"\bentire\s+",
        r"\bwhole\s+",
        r"\ball\s+chapters?",
    ]
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in patterns)


def chat_function(message: str, history: list) -> str:
    """Process user message and return AI response."""
    try:
        page_number = extract_page_number(message)

        if page_number is not None:
            print(f"Page-specific query detected: page {page_number}")
            page_retriever = _vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10, "filter": {"page": page_number - 1}},
            )
            source_docs = page_retriever.invoke(message)
            if not source_docs:
                return f"⚠️ No content found for page {page_number}. The document may not have that many pages, or that page might be empty."

            template = """You are a helpful AI assistant that answers questions based on the provided context from a PDF document.

Context from page {page_num}:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information provided from page {page_num}
- Be concise but thorough in your answers
- Use a friendly, professional tone
- If the user asked for a summary, provide a comprehensive summary of the page content

Answer:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | _llm | StrOutputParser()
            answer = chain.invoke({"context": format_docs(source_docs), "question": message, "page_num": page_number})
            return answer + f"\n\n*Source: Page {page_number}*"

        if is_comprehensive_query(message):
            print("Comprehensive query detected - retrieving more chunks")
            comprehensive_retriever = _vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 30},
            )
            source_docs = comprehensive_retriever.invoke(message)
            if not source_docs:
                return "⚠️ No relevant content found in the document."

            template = """You are a helpful AI assistant that answers questions based on the provided context from a PDF document.

Context from the document (comprehensive excerpts from multiple pages):
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- The context includes many excerpts - synthesize ALL relevant information comprehensively
- For questions asking about "all" items, list every one you can find in the context
- If some items are missing from the context, clearly state which ones you found and which are missing
- Organize your answer clearly (use numbered lists or bullet points when appropriate)
- Cite page numbers for each item when available
- Use a friendly, professional tone

Answer:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | _llm_comprehensive | StrOutputParser()
            answer = chain.invoke({"context": format_docs(source_docs), "question": message})
            return answer + _format_source_citation(source_docs)

        # Normal RAG query
        answer = rag_chain.invoke(message)
        source_docs = retriever.invoke(message)
        return answer + _format_source_citation(source_docs)

    except Exception as e:
        error_msg = str(e)
        if "ANTHROPIC_API_KEY" in error_msg or "anthropic" in error_msg.lower():
            return "⚠️ Anthropic API key not found. Please set your ANTHROPIC_API_KEY environment variable."
        elif "Collection my_ollama does not exist" in error_msg:
            return "⚠️ ChromaDB collection not found. Please run the loader first to load your PDF into the database."
        elif "Connection refused" in error_msg or "11434" in error_msg:
            return "⚠️ Cannot connect to Ollama. Please make sure Ollama is running (try: ollama serve) and the nomic-embed-text model is available."
        else:
            return f"⚠️ Error: {error_msg}\n\nMake sure ChromaDB is properly loaded and Ollama is running with the nomic-embed-text model."


def create_gradio_interface() -> gr.ChatInterface:
    """Create and configure the Gradio chat interface."""
    return gr.ChatInterface(
        fn=chat_function,
        title="📚 PDF Question-Answering Chatbot",
        description="""
        Ask questions about the PDF document loaded in ChromaDB.
        The AI will search through the document and provide answers based on the content.

        **Setup:** Make sure you've loaded your PDF using `src/loader.py` first!
        """,
        examples=[
            "What is the main topic of this document?",
            "Can you summarize the first chapter?",
            "What are the key findings?",
            "What is discussed on page 10?",
        ],
    )


def main():
    """Launch the Gradio chat interface."""
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)


# Initialize shared resources at module load
print("Loading vectorstore and initializing RAG chain...")
_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.2, max_tokens=2000)
_llm_comprehensive = ChatAnthropic(model="claude-haiku-4-5", temperature=0.2, max_tokens=3000)
_vectorstore = load_vectorstore()
rag_chain, retriever = create_rag_chain(_vectorstore)
print("RAG chain ready!")


if __name__ == "__main__":
    main()
