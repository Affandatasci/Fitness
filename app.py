import streamlit as st
import time
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
.stApp { background-color: #000000; }
section[data-testid="stSidebar"] { display: none; }

body, p, div, span, li, td, th, label        { color: #FFFFFF !important; }
.stMarkdown p, .stMarkdown li                { color: #FFFFFF !important; font-size: 1.2rem !important; line-height: 1.7; }
h2, h3                                       { color: #CCCCCC !important; }
strong, b                                    { color: #FFFFFF !important; }
.stCaption, .stCaption p                     { color: #AAAAAA !important; font-size: 1.2rem !important; text-align: center !important; }
.subtitle { font-size: 1.2rem !important; color: #AAAAAA !important; text-align: center !important; margin-bottom: 8px; }
.subtitle2 { font-size: 1.2rem !important; color: #CCCCCC !important; text-align: center !important; margin-bottom: 32px; }

.fx-title { color: #FFFFFF !important; font-size: 3rem; font-weight: 700; text-align: center; margin: 0 0 20px 0; }

.new-copy p, .new-copy li { font-size: 19px !important; line-height: 1.7; margin-bottom: 10px; }
.new-copy h3 { font-size: 23px !important; margin: 0 0 14px 0; }
.new-copy strong { color: #4CAF50 !important; }
.new-copy ol { padding-left: 1.4em; }

.stButton > button {
    background-color: #111111 !important;
    color: #FFFFFF !important;
    border: 2px solid #444444 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    font-size: 18px !important;
}
.stButton > button:hover {
    background-color: #222222 !important;
    border-color: #4CAF50 !important;
    color: #FFFFFF !important;
}
.stButton > button p, .stButton > button div, .stButton > button span {
    color: #FFFFFF !important;
    font-size: 18px !important;
}

.chat-box {
    background-color: #111111;
    border: 2px solid #333333;
    border-top: 4px solid #4CAF50;
    border-radius: 14px;
    padding: 20px 24px;
    min-height: 210px;
    margin-bottom: 16px;
    width: 100% !important;
    box-sizing: border-box;
}

.msg-user {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 14px;
    border-left: 3px solid #4CAF50;
}
.msg-bot {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 14px;
    border-left: 3px solid #888888;
}
.chat-box p, .chat-box li, .chat-box ul, .chat-box ol,
.chat-box div, .chat-box span {
    color: #111111 !important;
    font-size: 1.1rem !important;
    line-height: 1.7;
}
.chat-box strong, .chat-box b { color: #000000 !important; }
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
footer { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Left/right panels: problem / solution / sample questions, and how it works + price ──
left, main, right = st.columns([1.2, 2.6, 1.2])

SAMPLE_QUESTIONS = [
    "What's included in the Momentum tier?",
    "How many calories should  eat to lose fat?",
]

PANEL_CSS = """
<style>
.st-key-fx_left, .st-key-fx_right {
    background: #111111;
    border: 1px solid #333333;
    border-top: 4px solid #4CAF50;
    border-radius: 14px;
    padding: 20px 20px 22px 20px;
}
.st-key-fx_left [data-testid="stMarkdownContainer"],
.st-key-fx_right [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }

.st-key-fx_left p, .st-key-fx_right li, .st-key-fx_right .fx-invest {
    font-size: 19px !important;
    line-height: 1.5 !important;
    color: #FFFFFF !important;
}
.st-key-fx_left p { margin: 0 0 14px 0 !important; }
.st-key-fx_left b, .st-key-fx_right b { color: #4CAF50 !important; font-weight: 800 !important; }
.st-key-fx_left p.fx-try { font-size: 20px !important; margin: 6px 0 0 0 !important; }
.st-key-fx_right .fx-title {
    font-size: 23px !important;
    font-weight: 800 !important;
    line-height: 1.3 !important;
    color: #4CAF50 !important;
    margin-bottom: 14px !important;
}
.st-key-fx_right ol { margin: 0 0 18px 0; padding-left: 1.3em; }
.st-key-fx_right li { margin-bottom: 10px; }
.st-key-fx_right .fx-invest {
    background: #1A1A1A;
    border-left: 4px solid #4CAF50;
    border-radius: 8px;
    padding: 12px 14px;
}

.st-key-fx_left .stElementContainer, .st-key-fx_left .stButton { width: 100% !important; }
.st-key-fx_left .stButton > button {
    width: 100% !important;
    height: auto;
    min-height: 0;
    background: #000000 !important;
    color: #FFFFFF !important;
    border: 1.5px solid #4CAF50 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}
.st-key-fx_left .stButton > button * { justify-content: flex-start !important; text-align: left !important; }
.st-key-fx_left .stButton > button p {
    font-size: 18px !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
    margin: 0 !important;
    color: #FFFFFF !important;
}
.st-key-fx_left .stButton > button:hover {
    background: #4CAF50 !important;
    color: #000000 !important;
    border-color: #4CAF50 !important;
}
.st-key-fx_left .stButton > button:hover p,
.st-key-fx_left .stButton > button:focus p,
.st-key-fx_left .stButton > button:active p {
    color: #000000 !important;
}
</style>
"""

with left:
    with st.container(key="fx_left"):
        st.markdown(
            PANEL_CSS
            + '<div class="new-copy">'
            "<p><strong>The Problem:</strong> Online coaches get 30-80 messages a week asking about programs, pricing, and nutrition.</p>"
            "<p><strong>The Solution:</strong> This AI agent answers client questions instantly from real coaching content, 24/7.</p>"
            "<p><strong>Result:</strong> Cuts repetitive support time by 60-80%. Try it below with sample data.</p>"
            '<p class="fx-try"><strong>Try asking:</strong></p></div>',
            unsafe_allow_html=True,
        )
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if st.button(q, key=f"sample_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

with right:
    with st.container(key="fx_right"):
        st.markdown(
            '<div class="new-copy">'
            '<div class="fx-title">What This Would Look Like For Your Business</div>'
            "<ol>"
            "<li>You send me your program details, pricing, and FAQs</li>"
            "<li>I build and train the AI on YOUR specific content (3-4 days)</li>"
            "<li>I deploy it to your website, WhatsApp, or email</li>"
            "<li>Your clients get instant, accurate answers 24/7</li>"
            "</ol>"
            '<div class="fx-invest"><strong>Investment:</strong> Starting at <strong>$99</strong> for a basic setup. Includes 30 days of free support.</div>'
            "</div>",
            unsafe_allow_html=True,
        )

# ── Load resources once ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and connecting to Qdrant (first load only)...")
def load_resources():
    retriever = HybridRetriever()
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    graph = build_graph(retriever, anthropic_client)
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
        start_time = time.time()
        result = answer_question(
            st.session_state.messages[-1]["content"],
            graph, cache
        )
        elapsed_time = time.time() - start_time
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "from_cache": result["from_cache"],
        "elapsed": elapsed_time
    })
with main:
    # ── Title ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="fx-title">💪 Your Online Fitness Coach</div>', unsafe_allow_html=True)
    # st.markdown('<p class="subtitle">Ask me about workouts, nutrition, meal plans, recovery, and supplements.</p>', unsafe_allow_html=True)
    # st.markdown('<p class="subtitle2">Ask me anything about fitness — I\'ll give you answers based on expert coaching content.</p>', unsafe_allow_html=True)

    # st.info("🔹 This is a live demo using sample coaching content. Your real program details would replace this.")
    # ── Chat box ──────────────────────────────────────────────────────────────
    if st.session_state.messages:
        chat_html = '<div class="chat-box">'
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
                timing_html = ""
                if msg.get("elapsed") is not None:
                    timing_html = f'<p style="font-size:0.75rem;color:#666;margin-top:4px">⚡ Answered in {msg["elapsed"]:.1f}s</p>'
                chat_html += (
                    f'<div class="msg-bot">'
                    f'<span class="msg-label">Coach</span>'
                    f'<p>{msg["content"]}</p>'
                    f'{sources_html}{cache_html}{timing_html}</div>'
                )
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input = st.chat_input("e.g. How many calories should I eat to lose fat?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()

    st.markdown("---")
    # st.caption("Note: this demo currently answers English-language FAQ-style questions. Complex multi-step requests are escalated to a human.")
    st.markdown('<p style="color:#AAAAAA; font-size:1.8rem; text-align:center;">💪 Forge Physique Coaching · <a href="mailto:hello@forgephysiquecoaching.com" style="color:#4CAF50 !important; text-decoration:none;">hello@forgephysiquecoaching.com</a></p>', unsafe_allow_html=True)