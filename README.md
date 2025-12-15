# 🧠 AI / ML Wikipedia RAG System (LLaMA + Django)

A local, domain-specific **Retrieval-Augmented Generation (RAG)** system that answers
questions about **AI, Machine Learning, GenAI, LLMs, Deep Learning, and Neural Networks**
using **Wikipedia as the knowledge source**.

---

## 🚀 Features

- Semantic search over AI-related Wikipedia pages
- Retrieval-Augmented Generation (RAG)
- Local LLaMA 3.1 inference via Ollama
- Wikipedia citations with clickable links
- Chat history with session-based memory
- Short-term memory (last 2 chats only)
- Clear chat option
- Loading spinner for better UX
- Fully offline (no external APIs)

---

## 🧰 Tech Stack

- Backend: Django
- LLM: LLaMA 3.1 (Ollama)
- Vector DB: FAISS
- Embeddings: Sentence-Transformers (MiniLM)
- Data Source: Wikipedia
- Frontend: HTML, CSS, JavaScript

---

## ⚙️ Setup & Run Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shreesi/wiki-ai-rag.git
cd wiki-ai-rag
