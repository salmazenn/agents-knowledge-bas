---
title: Agents Knowledge Base
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - agent
  - rag
  - knowledge-base
  - customer-support
  - groq
  - chromadb
  - streamlit
  - llm
  - python
---

# 🤖 Agents Knowledge Base — Customer Support Agent

> AI-powered customer support agent with two tools: search a pre-indexed knowledge base and escalate to human support when needed. Built with Groq + ChromaDB + Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Groq](https://img.shields.io/badge/Groq-llama3.3-black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vectorstore-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

- 🔍 **Knowledge base search** — semantic search over pre-indexed PDFs
- 🚨 **Human escalation** — automatic escalation with ticket creation
- 💬 **Chat interface** — conversation history with Streamlit
- 📋 **Escalation dashboard** — view pending tickets in the sidebar
- ⚡ **Ultra-fast** — powered by Groq inference

---

## 🧠 How It Works

\`\`\`
Customer question
       │
       ▼
Escalation keywords detected?
   YES → Create ticket → Human agent
   NO  → Search knowledge base
              │
        Found answer?
          YES → Answer customer
          NO  → Create ticket → Human agent
\`\`\`

**Escalation triggers :**
- Customer explicitly requests a human agent
- Keywords detected: urgent, complaint, frustrated...
- No relevant answer found in the knowledge base

---

## 📦 Tech Stack

| Component | Tool | Role |
|-----------|------|------|
| LLM | Groq (llama-3.3-70b-versatile) | Answer generation |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Text vectorization |
| Vector Store | ChromaDB | Semantic search |
| UI | Streamlit | Chat interface |

---

## 🚀 Getting Started (Local)

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com) (free)

### 1. Clone the repository

\`\`\`bash
git clone https://github.com/salmazenn/agents-knowledge-bas.git
cd agents-knowledge-bas
\`\`\`

### 2. Create a virtual environment

\`\`\`bash
python3 -m venv .venv-agents
source .venv-agents/bin/activate
\`\`\`

### 3. Install dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Set up environment variables

\`\`\`bash
cp env.example .env
\`\`\`

Edit \`.env\`:

\`\`\`env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
GROQ_MODEL=llama-3.3-70b-versatile
CHROMA_PERSIST_DIR=./data/chroma
DOCS_DIR=./docs
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
TOP_K=3
ESCALATION_LOG=./data/escalations.json
\`\`\`

### 5. Add your documents

\`\`\`bash
cp your-faq.pdf docs/
\`\`\`

### 6. Index your documents

\`\`\`bash
python src/ingest.py
\`\`\`

### 7. Launch the app

\`\`\`bash
streamlit run src/app.py
\`\`\`

Open **http://localhost:8501** 🎉

---

## ⚙️ Configuration Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| \`GROQ_API_KEY\` | — | **Required.** Your Groq API key |
| \`GROQ_MODEL\` | \`llama-3.3-70b-versatile\` | Groq model |
| \`CHROMA_PERSIST_DIR\` | \`./data/chroma\` | ChromaDB storage path |
| \`DOCS_DIR\` | \`./docs\` | Folder with your PDFs |
| \`CHUNK_SIZE\` | \`1000\` | Chunk size in tokens |
| \`CHUNK_OVERLAP\` | \`100\` | Overlap between chunks |
| \`TOP_K\` | \`3\` | Number of chunks retrieved |
| \`ESCALATION_LOG\` | \`./data/escalations.json\` | Escalation tickets log |

---

## 📁 Project Structure

\`\`\`
agents-knowledge-base/
├── src/
│   ├── ingest.py       # Index PDFs into ChromaDB
│   ├── agent.py        # Agent logic (search + escalation)
│   └── app.py          # Streamlit chat interface
├── docs/               # Drop your PDFs here
├── data/
│   ├── chroma/         # ChromaDB (auto-generated)
│   └── escalations.json # Escalation tickets
├── .env                # API keys (never commit!)
├── env.example         # Environment template
├── requirements.txt
├── Dockerfile
└── README.md
\`\`\`

---

## 🔭 Roadmap

- [ ] Email notifications for escalation tickets
- [ ] Admin dashboard to manage tickets
- [ ] Multi-language support
- [ ] Integration with CRM (Salesforce, HubSpot)
- [ ] Analytics dashboard

---

## 📄 License

MIT — free to use, modify and distribute.
