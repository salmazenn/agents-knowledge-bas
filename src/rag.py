"""
RAG Knowledge Base — Moteur RAG
"""

import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
TOP_K              = int(os.getenv("TOP_K", "8"))

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """
Tu es un assistant de support client expert et bienveillant.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement et suggère de contacter le support.
Sois concis, précis et professionnel.

Contexte :
{context}

Question : {question}

Réponse :"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ── Core ──────────────────────────────────────────────────────────────────────

def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return Chroma(
        client=client,
        collection_name="knowledge_base",
        embedding_function=embeddings
    )

def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

def ask(chain: RetrievalQA, question: str) -> dict:
    result = chain.invoke({"query": question})
    sources = list({
        doc.metadata.get("source", "Source inconnue")
        for doc in result.get("source_documents", [])
    })
    return {
        "question": question,
        "answer": result["result"].strip(),
        "sources": sources
    }