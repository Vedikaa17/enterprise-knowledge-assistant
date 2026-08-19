# 📘 Enterprise Knowledge Assistant

A RAG-based (Retrieval-Augmented Generation) document search assistant that answers natural language questions using company documents — with accurate source citations.

🔗 **Live Demo:** https://vedikaa17.github.io/enterprise-knowledge-assistant/
🔗 **Backend API:** https://enterprise-knowledge-assistant-tts6.onrender.com

## Overview

Instead of manually searching through policy documents, users can ask questions in plain English and get instant, accurate answers — grounded in the actual document content, with the source file cited for every answer.

## Features

- **Semantic Search** — understands the meaning behind a question, not just keywords
- **Cited Answers** — every response tells you exactly which document it came from
- **Hallucination Prevention** — if the answer isn't in the documents, the system says so instead of guessing
- **Clean UI** — glassmorphism-styled frontend with a landing page and interactive Q&A interface

## Tech Stack

- **Backend:** FastAPI (Python)
- **Vector Database:** ChromaDB (local, persistent)
- **LLM & Embeddings:** Google Gemini API (`gemini-embedding-2` for embeddings, `gemini-3.6-flash` for generation)
- **Frontend:** HTML, CSS, JavaScript (no framework)
- **Deployment:** Render (backend), GitHub Pages (frontend)

## How It Works

1. Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`
2. Each chunk is converted into a vector embedding using Gemini's embedding model
3. Embeddings are stored in ChromaDB for fast similarity search
4. When a user asks a question, it's embedded the same way and matched against stored chunks
5. The most relevant chunks are passed to Gemini as context, along with the question
6. Gemini generates an answer using only the provided context, citing its source

## Project Structure

enterprise-knowledge-assistant/
├── documents/ # Source documents (.txt files)
├── chunk_documents.py # Document loading + chunking
├── embed_and_store.py # Embedding generation + Chroma storage
├── retrieve.py # Semantic retrieval logic
├── generate_answer.py # RAG answer generation
├── main.py # FastAPI application
├── index.html # Landing page
├── app.html # Q&A interface
└── requirements.txt # Python dependencies


## Running Locally

```bash
# Clone the repo
git clone https://github.com/Vedikaa17/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant

# Set up virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key to a .env file
# GEMINI_API_KEY=your_key_here

# Run the server
uvicorn main:app --reload
```

Then open `app.html` in your browser.

## Author

**Vedika Welukar**
B.Tech Artificial Intelligence & Data Science