import streamlit as st
import time
import config
from retriever import HybridRetriever
from cache import SemanticCache
from agent import build_graph, answer_question

st.set_page_config(
    page_title="Your Online Fitness Coach",
    page_icon="💪",
    layout="centered"
)

st.markdown("""
<style>
.block-container { padding-top: 3rem !important; max-width: 1300px !important; }
.stApp { background-color: #000000; }
section[data-testid="stSidebar"] { display: none; }

body, p, div, span, li, td, th, label        { color: #FFFFFF !important; }
.stMarkdown p, .stMarkdown li                { color: #FFFFFF !important; font-size: 1.2rem !important; line-height: 1.7; }
h1                                           { color: #FFFFFF !important; font-size: 3.4rem !important; text-align: center !important; }
h2, h3                                       { color: #CCCCCC !important; }
strong, b                                    { color: #FFFFFF !important; }
.stCaption, .stCaption p                     { color: #AAAAAA !important; font-size: 1.2rem !important; text-align: center !important; }
.subtitle { font-size: 1.2rem !important; color: #AAAAAA !important; text-align: center !important; margin-bottom: 8px; }
.subtitle2 { font-size: 1.2rem !important; color: #CCCCCC !important; text-align: center !important; margin-bottom: 32px; }

.new-copy p, .new-copy li { font-size: calc(1.2rem * 1.3) !important; line-height: 1.7; margin-bottom: 10px; }
.new-copy h3 { font-size: calc(1.5rem * 1.3) !important; margin: 20px 0 10px; }
.new-copy strong { color: #4CAF50 !important; }
.new-copy ol { padding-left: 1.4em; }

.stButton > button {
    background-color: #111111 !important;
    color: #FFFFFF !important;
    border: 2px solid #444444 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    width: 100% !important;
    font-size: 1.5rem !important;
}
.stButton > button:hover {
    background-color: #222222 !important;
    border-color: #4CAF50 !important;
    color: #FFFFFF !important;
}
.stButton > button p, .stButton > button div, .stButton > button span {
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
}

.chat-box {
    background-color: #111111;
    border: 2px solid #333333;
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
[data-testid="stBottom"] { background-color: #000000 !important; border-top: none !important; }
[data-testid="stBottom"] > div { background-color: #000000 !important; max-width: 1300px !important; width: 100% !important; margin: 0 auto !important; }
[data-testid="stBottomBlockContainer"] { max-width: 1300px !important; width: 100% !important; padding-left: 0 !important; padding-right: 0 !important; }
.stChatFloatingInputContainer { background-color: #000000 !important; border-top: none !important; max-width: 1300px !important; width: 100% !important; margin: 0 auto !important; }
.stChatFloatingInputContainer > div { background-color: #000000 !important; max-width: 1300px !important; width: 100% !important; }
[data-testid="stChatInput"] { max-width: 1300px !important; width: 100% !important; }
[data-testid="stChatInputContainer"] { max-width: 1300px !important; width: 100% !important; }
section[data-testid="stBottom"] { background: #000000 !important; }
</style>
""", unsafe_allow_html=True)

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

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<h1>💪 Your Online Fitness Coach</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">Ask me about workouts, nutrition, meal plans, recovery, and supplements.</p>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle2">Ask me anything about fitness — I\'ll give you answers based on expert coaching content.</p>', unsafe_allow_html=True)

# ── Problem / Solution / Result ─────────────────────────────────────────────
st.markdown("""
<div class="new-copy">
<p><strong>The Problem:</strong> Online coaches get 30-80 messages a week asking about programs, pricing, and nutrition.</p>
<p><strong>The Solution:</strong> This AI agent answers client questions instantly from real coaching content, 24/7.</p>
<p><strong>Result:</strong> Cuts repetitive support time by 60-80%. Try it below with sample data.</p>
</div>
""", unsafe_allow_html=True)



# st.info("🔹 This is a live demo using sample coaching content. Your real program details would replace this.")
# ── Chat box ──────────────────────────────────────────────────────────────────
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




# ── Sample questions ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<p class="new-copy" style="font-weight:700;">Try asking:</p>', unsafe_allow_html=True)
    sample_questions = [
        "What's included in the 12-week transformation program?",
        "How many calories should I eat to lose fat?",
        
    ]
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        if cols[i % 2].button(q, key=f"sample_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("e.g. How many calories should I eat to lose fat?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

st.markdown("---")
st.markdown("""
<div class="new-copy">
<h3>What This Would Look Like For Your Business</h3>
<ol>
<li>You send me your program details, pricing, and FAQs</li>
<li>I build and train the AI on YOUR specific content (3-4 days)</li>
<li>I deploy it to your website, WhatsApp, or email</li>
<li>Your clients get instant, accurate answers 24/7</li>
</ol>
# <p><strong>Investment:</strong> Starting at $99 for a basic setup. Includes 30 days of free support.</p>
</div>
""", unsafe_allow_html=True)
# st.caption("Note: this demo currently answers English-language FAQ-style questions. Complex multi-step requests are escalated to a human.")
st.markdown("---")
st.markdown('<p style="color:#AAAAAA; font-size:1.8rem; text-align:center;">💪 Forge Physique Coaching · <a href="mailto:hello@forgephysiquecoaching.com" style="color:#4CAF50 !important; text-decoration:none;">hello@forgephysiquecoaching.com</a></p>', unsafe_allow_html=True)