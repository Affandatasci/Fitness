"""
app.py — Gradio chat UI for the Forge Physique Coaching RAG demo.

Replaces the Streamlit version for HuggingFace Spaces deployment
(SDK: Gradio, Hardware: CPU Basic — FREE, 16 GB RAM).

All backend files (agent.py, retriever.py, cache.py, config.py)
are 100% unchanged — this is a pure UI swap.

Deploy steps:
  1. Create HF Space → SDK: Gradio → Hardware: CPU Basic (FREE)
  2. Push this repo (all .py files + data/ folder)
  3. Add secrets in Space settings:
       GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
"""

import gradio as gr
from groq import Groq

import config
from retriever import HybridRetriever
from agent import build_graph, answer_question
from cache import SemanticCache
import spaces   # add at top with other imports

@spaces.GPU     # add this decorator
def respond(message: str, history: list, deep_reasoning: bool):
    if not message.strip():
        return history, ""
    result = answer_question(
        ...
    )
    ...

# ---------------------------------------------------------------------------
# Load heavy resources ONCE at startup — shared across ALL visitors.
# Equivalent of Streamlit's @st.cache_resource. Do NOT move this inside
# the chat function or it will reload on every message (same mistake as
# putting it in st.session_state in the Streamlit version).
# ---------------------------------------------------------------------------

print("Loading models and connecting to Qdrant (first load only)...")

retriever   = HybridRetriever()
groq_client = Groq(api_key=config.GROQ_API_KEY)
graph       = build_graph(retriever, groq_client)

# Reuse the retriever's already-loaded embedding model for the cache —
# no second copy in memory.
def _embed_fn(text: str):
    return retriever.embedding_model.encode(text, normalize_embeddings=True)

cache = SemanticCache(embed_fn=_embed_fn)

print("Ready.")


# ---------------------------------------------------------------------------
# Chat function — called by Gradio on every user message
# ---------------------------------------------------------------------------

def respond(message: str, history: list, deep_reasoning: bool):
    """
    Parameters
    ----------
    message       : the new user message
    history       : list of [user_msg, assistant_msg] pairs (Gradio format)
    deep_reasoning: from the checkbox widget

    Returns
    -------
    (updated history, empty string)  — empty string clears the input box
    """
    if not message.strip():
        return history, ""

    result = answer_question(
        question=message,
        graph=graph,
        cache=cache,
        deep_reasoning=deep_reasoning,
    )

    reply = result["answer"]

    # Append a small metadata footer to the assistant reply
    if result["sources"]:
        sources_str = ", ".join(result["sources"])
        reply += f"\n\n📄 *Sources: {sources_str}*"
    elif result["from_cache"]:
        reply += "\n\n⚡ *Answered from semantic cache*"

    history = history + [[message, reply]]
    return history, ""


def _cache_count_label() -> str:
    return f"**Cached answers this session:** {len(cache)}"


# ---------------------------------------------------------------------------
# Gradio UI — gr.Blocks for full layout control
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Forge Physique Coaching Assistant",
) as demo:

    gr.Markdown(
        """
        # 💪 Forge Physique Coaching — Assistant
        Ask anything about programs, pricing, nutrition, or how coaching works.
        Powered by **Groq** (LLM) + **Qdrant** (hybrid vector search).
        """
    )

    with gr.Row():

        # ── Left column: chat ──────────────────────────────────────────────
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
            height=480,
            show_label=False,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about Forge Physique Coaching...",
                    show_label=False,
                    scale=8,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")
            clear_btn = gr.Button("🗑️  Clear chat", size="sm", variant="secondary")

        # ── Right column: settings ─────────────────────────────────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### ⚙️ Settings")
            deep_mode = gr.Checkbox(
                label="Deep Reasoning Mode",
                value=config.DEEP_REASONING_MODE_DEFAULT,
                info=(
                    "Off (default): single retrieve → answer pass, ~2-5 s. "
                    "On: adds a grade + rewrite loop for harder questions — "
                    "slower, but shows the full agentic reasoning path."
                ),
            )
            gr.Markdown("---")
            cache_label = gr.Markdown(_cache_count_label())

    gr.Markdown(
        "<div style='text-align:center; color:#aaa; font-size:0.82em; margin-top:4px;'>"
        "Portfolio demo · Responses grounded in the Forge Physique knowledge base only"
        "</div>"
    )

    # ── Event wiring ───────────────────────────────────────────────────────
    # Both the Send button and pressing Enter trigger the same respond fn.
    # After respond(), we also refresh the cache counter in the sidebar.

    send_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot, deep_mode],
        outputs=[chatbot, msg_box],
    ).then(fn=_cache_count_label, outputs=cache_label)

    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot, deep_mode],
        outputs=[chatbot, msg_box],
    ).then(fn=_cache_count_label, outputs=cache_label)

    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, msg_box],
    )


demo.launch(theme=gr.themes.Soft())
