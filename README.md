---
title: RAG Knowledge Base
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - rag
  - knowledge-base
  - customer-support
  - groq
  - langchain
  - chromadb
  - streamlit
  - chatbot
  - llm
  - python
---

# 💬 RAG Knowledge Base — Customer Support Chatbot

> Pre-indexed AI knowledge base for customer support — ask questions about your product without uploading files. Drop your PDFs in `/docs`, index once, answer forever. ⚡

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Groq](https://img.shields.io/badge/Groq-llama3.3-black)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vectorstore-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

- 📁 **Pre-indexed knowledge base** — no file upload needed from the user
- 🔍 **Semantic search** over your documents via ChromaDB
- ⚡ **Ultra-fast responses** powered by Groq (llama-3.3-70b)
- 💬 **Chat interface** with conversation history and source display
- 🛠️ **Admin ingestion script** — run once to index all your PDFs
- 🐳 **Docker ready** for easy deployment

---

## 🏗️ Architecture

```
Admin (you)                    End User
     │                             │
     ▼                             ▼
Drop PDFs in /docs          Asks questions
     │                             │
     ▼                             ▼
python src/ingest.py    →   streamlit run src/app.py
     │                             │
     ▼                             ▼
ChromaDB (indexed)      ←   RAG Pipeline (Groq + LangChain)
```

---

## 📦 Tech Stack

| Component | Tool | Role |
|-----------|------|------|
| LLM | Groq (llama-3.3-70b-versatile) | Answer generation |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Text vectorization |
| Vector Store | ChromaDB | Semantic search |
| Orchestration | LangChain | RAG pipeline |
| UI | Streamlit | Chat interface |

---

## 🚀 Getting Started (Local)

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/salmazenn/rag-knowledge-base.git
cd rag-knowledge-base
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv-kb
source .venv-kb/bin/activate   # Mac/Linux
# .venv-kb\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install langchain-community \
            langchain-groq \
            langchain-huggingface \
            langchain-chroma \
            langchain-text-splitters \
            chromadb \
            pypdf \
            streamlit \
            sentence-transformers \
            python-dotenv \
            numpy
```

### 4. Set up environment variables

Create a `.env` file at the root of the project:

```bash
cp .env.example .env
```

Then edit `.env` with your values:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional — defaults shown
GROQ_MODEL=llama-3.3-70b-versatile
CHROMA_PERSIST_DIR=./data/chroma
DOCS_DIR=./docs
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
TOP_K=8
```

### 5. Add your documents

Drop your PDF files into the `/docs` folder:

```bash
cp your-faq.pdf docs/
cp your-product-guide.pdf docs/
# Add as many PDFs as needed
```

### 6. Run the ingestion script

```bash
python src/ingest.py
```

You should see:

```
🚀 Starting ingestion...

📁 Source folder: ./docs
💾 Vector store: ./data/chroma

📄 Loading: your-faq.pdf
   ✅ 12 pages loaded

✂️  163 chunks created

🔍 Creating embeddings (all-MiniLM-L6-v2)...
💾 163 chunks saved in: ./data/chroma

🎉 Ingestion complete! 163 chunks indexed and ready.
👉 Now run: streamlit run src/app.py
```

### 7. Launch the app

```bash
streamlit run src/app.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## ⚙️ Configuration Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `DOCS_DIR` | `./docs` | Folder containing your PDFs |
| `CHUNK_SIZE` | `1000` | Size of each text chunk (tokens) |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K` | `8` | Number of chunks retrieved per query |

### Tuning tips

- **Increase `CHUNK_SIZE`** (e.g. 1500) for documents with long paragraphs
- **Increase `TOP_K`** (e.g. 10-12) for broad questions requiring more context
- **Decrease `TOP_K`** (e.g. 4) for precise factual questions — faster and more accurate
- **Re-run `ingest.py`** every time you add or modify documents in `/docs`

---

## 📁 Project Structure

```
rag-knowledge-base/
├── src/
│   ├── ingest.py       # Admin script — index PDFs into ChromaDB
│   ├── rag.py          # RAG engine — retrieval + generation
│   └── app.py          # Streamlit chat interface
├── docs/               # Drop your PDFs here
├── data/
│   └── chroma/         # ChromaDB vector store (auto-generated)
├── .env                # Your API keys (never commit this!)
├── .env.example        # Template for environment variables
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔄 Updating the Knowledge Base

When you add new documents or update existing ones:

```bash
# 1. Add/replace PDFs in /docs
cp new-document.pdf docs/

# 2. Re-run ingestion (old index is automatically replaced)
python src/ingest.py

# 3. Restart the app
streamlit run src/app.py
```

---

## ☁️ Deploy on HuggingFace Spaces

### 1. Create a Space
- Go to **huggingface.co/new-space**
- SDK: **Docker** → Blank
- Visibility: **Public**

### 2. Add secrets
In your Space settings → **Variables and secrets**:
- `GROQ_API_KEY` → your Groq key

### 3. Push to HuggingFace

```bash
git remote add hf https://huggingface.co/spaces/salmazen/rag-knowledge-base
git push hf main
```

> ⚠️ Important: the ChromaDB index must be pre-built and committed to the repo before deploying. Run `python src/ingest.py` locally and commit the `data/chroma/` folder.

---

## 🔭 Roadmap

- [ ] Multi-language support
- [ ] Admin dashboard to manage documents
- [ ] Automatic RAGAS evaluation
- [ ] Agents layer on top of the knowledge base
- [ ] Authentication for the chat interface

---

## 📄 License

MIT — free to use, modify and distribute.
