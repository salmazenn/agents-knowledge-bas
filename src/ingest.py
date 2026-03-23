"""
RAG Knowledge Base — Script d'ingestion
À lancer une seule fois pour indexer les documents du dossier /docs
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

DOCS_DIR          = os.getenv("DOCS_DIR", "./docs")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Functions ─────────────────────────────────────────────────────────────────

def load_docs(docs_dir: str) -> list:
    """Charge tous les PDFs du dossier /docs."""
    docs = []
    path = Path(docs_dir)
    pdf_files = list(path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"Aucun PDF trouvé dans : {docs_dir}")

    for pdf in pdf_files:
        print(f"📄 Chargement : {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        # Ajoute le nom du fichier comme source
        for page in pages:
            page.metadata["source"] = pdf.name
        docs.extend(pages)
        print(f"   ✅ {len(pages)} pages chargées")

    return docs

def split_docs(documents: list) -> list:
    """Découpe les documents en chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"\n✂️  {len(chunks)} chunks créés au total")
    return chunks

def build_vectorstore(chunks: list):
    """Crée et persiste le vectorstore ChromaDB."""
    print(f"\n🔍 Création des embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Supprime la collection existante si elle existe
    try:
        client.delete_collection("knowledge_base")
        print("🗑️  Ancienne collection supprimée")
    except:
        pass

    collection = client.get_or_create_collection("knowledge_base")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]
    embeddings_list = embeddings.embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids
    )

    print(f"💾 {collection.count()} chunks sauvegardés dans : {CHROMA_PERSIST_DIR}")
    return collection.count()

def main():
    print("🚀 Démarrage de l'ingestion...\n")
    print(f"📁 Dossier source : {DOCS_DIR}")
    print(f"💾 Vectorstore : {CHROMA_PERSIST_DIR}\n")

    documents = load_docs(DOCS_DIR)
    print(f"\n📚 Total : {len(documents)} pages chargées")

    chunks = split_docs(documents)
    count = build_vectorstore(chunks)

    print(f"\n🎉 Ingestion terminée ! {count} chunks indexés et prêts.")
    print("👉 Lance maintenant : streamlit run src/app.py")

if __name__ == "__main__":
    main()