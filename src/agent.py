"""
Agents Knowledge Base — Agent de support client
Outils : recherche KB + escalade vers humain
"""

import os
import chromadb
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
TOP_K              = int(os.getenv("TOP_K", "6"))
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

# ── Tools ─────────────────────────────────────────────────────────────────────

def make_tools(vectorstore: Chroma):

    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Search the knowledge base to answer customer questions about
        products, pricing, account management, technical issues,
        billing, and company policies.
        Use this tool first for any customer question.
        """
        docs = vectorstore.similarity_search(query, k=TOP_K)
        if not docs:
            return "No relevant information found in the knowledge base."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            results.append(f"[Source: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    @tool
    def escalate_to_human(
        reason: str,
        customer_message: str,
        urgency: str = "normal"
    ) -> str:
        """
        Escalate the conversation to a human support agent when:
        - The knowledge base doesn't contain the answer
        - The issue is complex or sensitive
        - The customer is frustrated or angry
        - The question involves account security or billing disputes
        - The customer explicitly requests a human agent

        Args:
            reason: Why you are escalating (be specific)
            customer_message: The customer's original message
            urgency: 'low', 'normal', or 'high'
        """
        import json
        from pathlib import Path

        ticket = {
            "timestamp": datetime.now().isoformat(),
            "urgency": urgency,
            "reason": reason,
            "customer_message": customer_message,
            "status": "pending"
        }

        # Sauvegarde le ticket
        log_path = Path(ESCALATION_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        tickets = []
        if log_path.exists():
            with open(log_path) as f:
                tickets = json.load(f)
        tickets.append(ticket)
        with open(log_path, "w") as f:
            json.dump(tickets, f, indent=2)

        urgency_labels = {
            "low": "dans les 24 heures",
            "normal": "dans les 4 heures",
            "high": "dans les 30 minutes"
        }
        delay = urgency_labels.get(urgency, "dans les 4 heures")

        return (
            f"✅ Ticket d'escalade créé avec succès.\n"
            f"Un agent humain vous contactera {delay}.\n"
            f"Référence : ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}\n"
            f"Urgence : {urgency.upper()}"
        )

    return [search_knowledge_base, escalate_to_human]


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es Alex, un agent de support client professionnel et bienveillant pour TechPro Solutions.

Tu as accès à deux outils :
1. **search_knowledge_base** : pour chercher des réponses dans notre documentation
2. **escalate_to_human** : pour escalader vers un agent humain quand nécessaire

## Processus de décision :

1. Pour toute question client, commence TOUJOURS par chercher dans la knowledge base
2. Si tu trouves une réponse claire et complète → réponds directement
3. Si la réponse est partielle → réponds avec ce que tu as et propose d'escalader
4. Si tu ne trouves RIEN de pertinent → escalade vers un humain
5. Escalade TOUJOURS si :
   - La question concerne un litige de facturation
   - Le client exprime de la frustration ou colère
   - La question implique la sécurité du compte
   - Le client demande explicitement un humain

## Style de communication :
- Poli, professionnel et empathique
- Réponses concises et structurées
- Toujours en français sauf si le client écrit en anglais
- Signe toujours tes messages avec "— Alex, Support TechPro"

Date et heure actuelle : {current_datetime}
"""

def build_agent(vectorstore: Chroma) -> AgentExecutor:
    """Construit l'agent LangChain."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    tools = make_tools(vectorstore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )

def ask_agent(agent: AgentExecutor, question: str, history: list = None) -> dict:
    """Pose une question à l'agent."""
    if history is None:
        history = []

    result = agent.invoke({
        "input": question,
        "chat_history": history,
        "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    return {
        "question": question,
        "answer": result["output"],
    }