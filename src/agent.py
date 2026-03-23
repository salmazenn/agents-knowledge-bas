"""
Agents Knowledge Base — Agent de support client
Approche : chaîne custom avec détection d'escalade
"""

import os
import json
import chromadb
from datetime import datetime
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
TOP_K              = int(os.getenv("TOP_K", "3"))
ESCALATION_LOG     = os.getenv("ESCALATION_LOG", "./data/escalations.json")

# ── Vector store ──────────────────────────────────────────────────────────────

def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return Chroma(
        client=client,
        collection_name="knowledge_base",
        embedding_function=embeddings
    )

# ── Escalation ────────────────────────────────────────────────────────────────

def create_escalation_ticket(reason: str) -> str:
    ticket = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "status": "pending"
    }
    log_path = Path(ESCALATION_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tickets = []
    if log_path.exists():
        with open(log_path) as f:
            tickets = json.load(f)
    tickets.append(ticket)
    with open(log_path, "w") as f:
        json.dump(tickets, f, indent=2)
    return f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}"

# ── Agent ─────────────────────────────────────────────────────────────────────

ESCALATION_KEYWORDS = [
    "human", "agent", "person", "representative", "supervisor",
    "humain", "personne", "représentant", "superviseur", "parler à",
    "urgent", "angry", "frustrated", "frustrated", "complaint", "plainte"
]

def should_escalate(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in ESCALATION_KEYWORDS)

def search_kb(vectorstore: Chroma, query: str) -> str:
    docs = vectorstore.similarity_search(query, k=TOP_K)
    if not docs:
        return ""
    return "\n\n".join([doc.page_content for doc in docs])

def build_agent(vectorstore: Chroma):
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return {"vectorstore": vectorstore, "llm": llm}

def ask_agent(agent: dict, question: str, history: list = None) -> dict:
    vectorstore = agent["vectorstore"]
    llm = agent["llm"]

    # Détection d'escalade directe
    if should_escalate(question):
        ref = create_escalation_ticket(f"Customer requested: {question}")
        return {
            "question": question,
            "answer": (
                f"Je comprends votre urgence. 🚨\n\n"
                f"J'ai créé un ticket d'escalade pour vous mettre en contact avec un agent humain.\n\n"
                f"**Référence :** `{ref}`\n"
                f"**Délai de réponse :** Dans les 4 heures\n\n"
                f"En attendant, vous pouvez aussi nous contacter directement :\n"
                f"- 📧 support@techpro-solutions.com\n"
                f"- 📞 1-800-TECHPRO\n\n"
                f"— Alex, Support TechPro"
            ),
            "escalated": True
        }

    # Recherche dans la KB
    context = search_kb(vectorstore, question)

    if not context:
        ref = create_escalation_ticket(f"No KB answer found for: {question}")
        return {
            "question": question,
            "answer": (
                f"Je n'ai pas trouvé de réponse dans notre base de connaissances. 😕\n\n"
                f"J'ai créé un ticket pour qu'un agent humain vous réponde.\n\n"
                f"**Référence :** `{ref}`\n\n"
                f"— Alex, Support TechPro"
            ),
            "escalated": True
        }

    # Génération de réponse
    prompt = f"""You are Alex, a customer support agent for TechPro Solutions.
Answer the customer question using ONLY the context below.
Be concise, professional and helpful. Answer in the same language as the question.
Sign with "— Alex, TechPro Support"

Context:
{context}

Customer question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return {
        "question": question,
        "answer": response.content,
        "escalated": False
    }