import faiss
import pickle
import requests
import numpy as np
import os
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Directory where FAISS index and metadata are stored
BASE_DIR = "faiss_index"


# ---------------------------------------------------------
# Lazy-loaded global resources
# (loaded once and reused across requests)
# ---------------------------------------------------------

_embedding_model = None   # SentenceTransformer model
_index = None             # FAISS vector index
_chunks = None            # Text chunks from Wikipedia
_metadata = None          # Metadata (title, URL) per chunk


# ---------------------------------------------------------
# Resource Loader (IMPORTANT)
# ---------------------------------------------------------

def load_resources():
    """
    Lazily load all heavy ML resources.
    This function ensures that:
    - Embedding model is loaded only once
    - FAISS index is loaded only once
    - Wikipedia chunks & metadata are loaded only once

    This avoids Django startup crashes and improves performance.
    """
    global _embedding_model, _index, _chunks, _metadata

    # Load embedding model if not already loaded
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load FAISS index and stored chunks if not already loaded
    if _index is None or _chunks is None or _metadata is None:
        index_path = os.path.join(BASE_DIR, "index.faiss")
        chunks_path = os.path.join(BASE_DIR, "chunks.pkl")

        # Safety check: ingestion must be done first
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise RuntimeError(
                "FAISS index not found. Please run ingest_wikipedia.py first."
            )

        # Load FAISS index
        _index = faiss.read_index(index_path)

        # Load chunks and metadata (title + Wikipedia URL)
        with open(chunks_path, "rb") as f:
            _chunks, _metadata = pickle.load(f)


# ---------------------------------------------------------
# Retrieval Logic (Vector Search)
# ---------------------------------------------------------

def retrieve_context(query, k=5):
    """
    Retrieve top-k most relevant Wikipedia chunks for a given query.

    Args:
        query (str): User question
        k (int): Number of chunks to retrieve

    Returns:
        contexts (list[str]): Relevant Wikipedia text chunks
        sources (dict): {title: wikipedia_url} for citations
    """
    load_resources()

    # Convert query to embedding
    query_embedding = _embedding_model.encode([query])

    # Search FAISS index for nearest neighbors
    _, indices = _index.search(np.array(query_embedding), k)

    contexts = []
    sources = {}

    # Collect retrieved chunks and citation info
    for idx in indices[0]:
        contexts.append(_chunks[idx])

        meta = _metadata[idx]  # {"title": ..., "url": ...}
        sources[meta["title"]] = meta["url"]

    return contexts, sources


# ---------------------------------------------------------
# LLaMA (Ollama) API Call
# ---------------------------------------------------------

def call_llama(prompt):
    """
    Send prompt to local LLaMA model via Ollama API.

    Args:
        prompt (str): Final prompt to be sent to LLaMA

    Returns:
        str: Generated answer from the model
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "temperature": 0.2,
            "stream": False
        },
        timeout=120
    )

    return response.json()["response"]


# ---------------------------------------------------------
# RAG Pipeline with Short-Term Memory
# ---------------------------------------------------------

def ask_question(question, chat_history=None):
    """
    Full Retrieval-Augmented Generation (RAG) pipeline
    with short-term conversational memory.

    Memory policy:
    - Only last 2 Q&A pairs are remembered
    - Memory is used ONLY to resolve references ("it", "that", etc.)
    - Wikipedia remains the factual source of truth

    Args:
        question (str): User's current question
        chat_history (list): Previous Q&A stored in session

    Returns:
        answer (str): Final generated answer
        sources (dict): Wikipedia citations
    """
    # Retrieve relevant Wikipedia context
    contexts, sources = retrieve_context(question)

    # Build short-term memory block (last 2 chats only)
    memory_block = ""

    if chat_history:
        recent_chats = chat_history[-2:]  # Limit memory size

        memory_lines = []
        for item in recent_chats:
            memory_lines.append(f"User: {item['question']}")
            memory_lines.append(f"Assistant: {item['answer']}")

        memory_block = "\n".join(memory_lines)

    # Final prompt sent to LLaMA
    prompt = f"""
You are an AI assistant.
Use the Wikipedia context to answer the question.
You may use the previous conversation ONLY to understand references like "it", "that", etc.
Do NOT hallucinate.

Previous conversation:
{memory_block}

Wikipedia context:
{chr(10).join(contexts)}

Current question:
{question}

Answer:
"""

    # Generate final answer
    answer = call_llama(prompt)

    return answer, sources
