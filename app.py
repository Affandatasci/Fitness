import streamlit as st
from openai import OpenAI
import config
from retriever import HybridRetriever
from cache import SemanticCache
from agent import build_graph, answer_question

st.set_page_config(
    page_title="Your Online Fitness Coach",
    page_icon="💪",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 0.6rem !important; }
.stApp { background-color: #000000; }
section[data-testid="stSidebar"] { display: none; }

body, p, div, span, li, td, th, label        { color: #FFFFFF !important; }
.stMarkdown p, .stMarkdown li                { color: #FFFFFF !important; font-size: 1.2rem !important; line-height: 1.7; }
h1                                           { color: #FFFFFF !important; font-size: 3.4rem !important; }
h2, h3                                       { color: #CCCCCC !important; }
strong, b                                    { color: #FFFFFF !important; }
.stCaption, .stCaption p                     { color: #AAAAAA !important; font-size: 1.99rem !important; }
.ask-line { font-size: 2.10rem !important; color: #CCCCCC !important; margin-bottom: 14px; }

.chat-box {
    background-color: #111111;
    border: 2px solid #333333;
    border-radius: 14px;
    padding: 20px 24px;
    min-height: 210px;
    margin-top: -20px;
    margin-bottom: 16px;
}

.msg-user {
    background-color: #1a1a2e;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 14px;
    border-left: 3px solid #4CAF50;
}
.msg-bot {
    background-color: #1c1c1c;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 14px;
    border-left: 3px solid #888888;
}
.chat-box p, .chat-box li, .chat-box ul, .chat-box ol,
.chat-box div, .chat-box span {
    color: #EEEEEE !important;
    font-size: 1.5rem !important;
    line-height: 1.7;
}
.chat-box strong, .chat-box b { color: #FFFFFF !important; }
.msg-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 6px;
}
.msg-user .msg-label { color: #4CAF50 !important; }
.msg-bot  .msg-label { color: #888888 !important; }
.chat-placeholder { color: #555555 !important; text-align: center; padding: 60px 0; font-size: 1rem !important; }

[data-testid="stChatInput"] textarea {
    font-size: 1.2rem   !important;
    min-height: 72px    !important;
    padding: 18px 24px  !important;
    background-color: #111111 !important;
    color: #FFFFFF      !important;
    border: 2px solid #444444 !important;
    border-radius: 12px !important;
    width: 100%         !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #666666 !important; }
[data-testid="stChatInput"] button { background-color: #333333 !important; border-radius: 8px !important; }
hr { border-color: rgba(255,255,255,0.15) !important; }
.stSpinner > div { color: #AAAAAA !important; }
</style>
""", unsafe_allow_html=True)

# ── Load resources once ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and connecting to Qdrant (first load only)...")
def load_resources():
    retriever = HybridRetriever()
    groq_client = OpenAI(
        api_key=config.GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    graph = build_graph(retriever, groq_client)
    embed_fn = lambda text: retriever.embedding_model.encode(
        text, normalize_embeddings=True
    ).tolist()
    cache = SemanticCache(embed_fn)
    return graph, cache

graph, cache = load_resources()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Answer pending question BEFORE rendering ─────────────────────────────────
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        result = answer_question(
            st.session_state.messages[-1]["content"],
            graph, cache
        )
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "from_cache": result["from_cache"]
    })

# ── Centered layout ───────────────────────────────────────────────────────────
_, main, _ = st.columns([0.5, 7, 0.5])

with main:
    st.markdown('<h1 style="font-size:3.4rem;color:#FFFFFF;font-weight:800;margin-bottom:6  44px;">💪 Your Online Fitness Coach</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.99rem;color:#AAAAAA;margin-bottom:16px;">Ask me about workouts, nutrition, meal plans, recovery, and supplements.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:2.10rem;color:#CCCCCC;margin-bottom:35px;">Ask me anything about fitness — I\'ll give you answers based on expert coaching content.</p>', unsafe_allow_html=True)

    # ── Chat box ──────────────────────────────────────────────────────────────
    chat_html = '<div class="chat-box">'
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += (
                    f'<div class="msg-user">'
                    f'<span class="msg-label">You</span>'
                    f'<p>{msg["content"]}</p></div>'
                )
            else:
                sources_html = ""
                if msg.get("sources"):
                    sources_html = f'<p style="font-size:0.8rem;color:#888;margin-top:8px">📄 {", ".join(msg["sources"])}</p>'
                cache_html = ""
                if msg.get("from_cache"):
                    cache_html = '<p style="font-size:0.75rem;color:#666;margin-top:4px">⚡ Answered from cache</p>'
                chat_html += (
                    f'<div class="msg-bot">'
                    f'<span class="msg-label">Coach</span>'
                    f'<p>{msg["content"]}</p>'
                    f'{sources_html}{cache_html}</div>'
                )
    else:
        chat_html += '<p class="chat-placeholder">Ask a question to get started...</p>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input = st.chat_input("e.g. How many calories should I eat to lose fat?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()

    st.markdown("---")
    st.caption("💪 Forge Physique Coaching · hello@forgephysiquecoaching.com")