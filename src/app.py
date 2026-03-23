"""
Agents Knowledge Base — Interface Streamlit
Agent de support client avec outils
"""

import streamlit as st
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent))
from agent import get_vectorstore, build_agent, ask_agent
from langchain_core.messages import HumanMessage, AIMessage

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Support Client — TechPro",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Support Client TechPro")
st.caption("Bonjour ! Je suis Alex, votre assistant de support. Comment puis-je vous aider ?")

# ── Session state ─────────────────────────────────────────────────────────────

if "agent" not in st.session_state:
    with st.spinner("Initialisation de l'agent..."):
        try:
            vectorstore = get_vectorstore()
            st.session_state.agent = build_agent(vectorstore)
            st.session_state.ready = True
        except Exception as e:
            st.session_state.ready = False
            st.session_state.error = str(e)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Status ────────────────────────────────────────────────────────────────────

if not st.session_state.get("ready", False):
    st.error(f"⚠️ Erreur : {st.session_state.get('error', 'Impossible de charger l agent.')}")
    st.info("💡 Admin : lancez `python src/ingest.py` pour indexer les documents.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🤖 Alex — Support TechPro")
    st.markdown("""
    **Ce que je peux faire :**
    - 🔍 Répondre à vos questions
    - 📋 Consulter notre documentation
    - 🚨 Escalader vers un humain si besoin

    **Besoin urgent ?**

    📧 support@techpro-solutions.com
    📞 1-800-TECHPRO
    """)

    st.divider()

    # Tickets d'escalade
    escalation_log = Path("./data/escalations.json")
    if escalation_log.exists():
        with open(escalation_log) as f:
            tickets = json.load(f)
        pending = [t for t in tickets if t["status"] == "pending"]
        if pending:
            st.warning(f"🚨 {len(pending)} ticket(s) en attente")
            with st.expander("Voir les tickets"):
                for t in pending[-3:]:
                    st.markdown(f"**{t['timestamp'][:16]}** — {t['urgency'].upper()}")
                    st.markdown(f"_{t['reason']}_")
                    st.divider()

    st.divider()

    if st.button("🗑️ Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("**Agent :** Alex ⚡")
    st.markdown("**Modèle :** Groq llama-3.3-70b")

# ── Chat ──────────────────────────────────────────────────────────────────────

# Message de bienvenue
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "Bonjour ! 👋 Je suis **Alex**, votre assistant de support TechPro Solutions.\n\n"
            "Je peux répondre à vos questions sur nos produits, abonnements, et problèmes techniques. "
            "Si je ne trouve pas la réponse, je vous mettrai en contact avec notre équipe humaine.\n\n"
            "Comment puis-je vous aider aujourd'hui ?"
        )

# Historique
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Alex réfléchit..."):
            result = ask_agent(
                st.session_state.agent,
                prompt,
                st.session_state.chat_history
            )
        st.markdown(result["answer"])

    # Mise à jour de l'historique
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=result["answer"])
    ])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })

    st.rerun()