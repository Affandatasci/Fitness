"""
app.py — Streamlit chat UI for the Forge Physique Coaching RAG demo.

Run with:
    streamlit run app.py

Heavy resources (embedding model, reranker, Qdrant client, Groq
client, the compiled LangGraph graph, and the semantic cache) are
loaded ONCE per app process via @st.cache_resource — not once per
user message, and not once per browser tab. Chat history is kept in
st.session_state instead, since that IS per-browser-tab/session.
Mixing these two up is the single most common Streamlit mistake in
a RAG app: put a heavy model load in session_state and it reloads on
every message; put chat history in cache_resource and every visitor
sees every other visitor's conversation.
"""

import streamlit as st
from groq import Groq

import config
from retriever import HybridRetriever
from agent import build_graph, answer_question
from cache import SemanticCache


st.set_page_config(page_title="Forge Physique Coaching — Assistant", page_icon="\U0001F4AA")


@st.cache_resource(show_spinner="Loading models and connecting to Qdrant (first load only)...")
def load_resources():
    retriever = HybridRetriever()
    groq_client = Groq(api_key=config.GROQ_API_KEY)
    graph = build_graph(retriever, groq_client)

    def embed_fn(text: str):
        # Reuses the retriever's already-loaded embedding model instead
        # of loading a second copy just for the cache.
        return retriever.embedding_model.encode(text, normalize_embeddings=True)

    # NOTE: because this whole function is @st.cache_resource, this one
    # SemanticCache instance is shared across every visitor to the
    # deployed app, not created fresh per browser tab. See cache.py's
    # docstring for why that's the correct choice here.
    cache = SemanticCache(embed_fn=embed_fn)

    return retriever, groq_client, graph, cache


retriever, groq_client, graph, cache = load_resources()

st.title("\U0001F4AA Forge Physique Coaching")
st.caption("Ask anything about programs, pricing, or how coaching works.")

with st.sidebar:
    st.header("Settings")
    deep_reasoning = st.toggle(
        "Deep reasoning mode",
        value=config.DEEP_REASONING_MODE_DEFAULT,
        help=(
            "Off (default): single retrieve-then-answer pass, roughly "
            "2-5 second responses. On: adds a self-correction loop "
            "(grade the retrieved chunks, rewrite the question and "
            "retrieve again if they're insufficient) — slower, but "
            "shows the deeper reasoning path to a technical prospect."
        ),
    )
    st.divider()
    st.caption(f"Cached answers this session: {len(cache)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: {', '.join(message['sources'])}")

user_question = st.chat_input("Ask a question about Forge Physique Coaching...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(
                question=user_question,
                graph=graph,
                cache=cache,
                deep_reasoning=deep_reasoning,
            )
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption(f"Sources: {', '.join(result['sources'])}")
        elif result["from_cache"]:
            st.caption("(answered from cache)")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
