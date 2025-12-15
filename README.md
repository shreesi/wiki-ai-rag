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
```
### 2️⃣ Install Dependencies
``` bash
pip install -r requirements.txt
```
This installs Django, FAISS, Sentence-Transformers, Wikipedia API, and other required libraries.

### 3️⃣ Install Ollama (LLaMA Runtime)
Download Ollama from the official website:
👉 https://ollama.com/download

Verify installation:
```bash
ollama --version
```
Start the Ollama server (keep this terminal running):
```bash
ollama serve
```

### 4️⃣ Download LLaMA Model
```push
ollama pull llama3.1:8b
```

### 5️⃣ Ingest Wikipedia Data (One-Time Setup)
```bash
python ingest_wikipedia.py
```
This step:

Downloads AI-related Wikipedia pages

Splits text into chunks

Generates embeddings

Stores them in FAISS

⚠️ This step is required only once.

### 6️⃣ Run Django Server
```bash
python manage.py migrate
python manage.py runserver
```
UI Reference:
<img width="2928" height="1476" alt="image" src="https://github.com/user-attachments/assets/79ce7cbb-e260-4ee6-bbab-4de4064721c1" />

