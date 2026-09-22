"""
agent.py — LangGraph orchestration for the Forge Physique RAG demo.

Default path (DEEP_REASONING_MODE_DEFAULT = False, set in config.py):
    retrieve -> generate                                   (1 LLM call, ~2-5s)

Deep reasoning path (toggle on in the UI):
    retrieve -> grade -> [insufficient? rewrite -> retrieve -> grade]* -> generate
    (up to 3 LLM calls, capped at MAX_GRADE_ATTEMPTS retries so it can
    never loop forever)

This split exists because running the full self-correction loop on
every query is what made the old BiomedicalRAG-Agent project take
multiple minutes to respond — see config.py's comment on
DEEP_REASONING_MODE_DEFAULT for the full explanation. Keeping it off
by default fixes both the latency complaint and the Groq free-tier
rate-limit complaint (1 call per query instead of 4-6) at the same time.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

import config


MAX_GRADE_ATTEMPTS = 2

SYSTEM_PROMPT_TEMPLATE = """You are the AI assistant for Forge Physique Coaching, an online physique coaching business run by Marcus Cole.

Answer the client's question using ONLY the information in the CONTEXT below. This is a strict rule:
- If the answer is fully contained in the context, answer clearly and directly.
- If the context only partially answers the question, answer what you can and say plainly what you don't have information on.
- If the context does not contain the answer at all, say so plainly and suggest the client email hello@forgephysiquecoaching.com — never invent a price, policy, or fact that isn't present in the context.

Do not mention "the context" or "the documents" to the client — speak naturally as Forge Physique Coaching's assistant.

CONTEXT:
{context}
"""


class GraphState(TypedDict):
    question: str            # the retrieval query — may be rewritten during the deep-reasoning loop
    original_question: str   # what the client actually typed — always answered in these terms
    chunks: list              # [{"text": ..., "source": ...}, ...] from the last retrieval
    sources: list              # deduplicated source filenames, surfaced to the UI
    answer: str
    deep_reasoning: bool
    is_sufficient: bool
    attempts: int


def build_graph(retriever, groq_client):
    """
    Wires the retriever and Groq client into the graph's node functions
    as closures, then compiles and returns the graph. Call this once
    (e.g. behind @st.cache_resource in app.py) — it's cheap to build,
    but there's no reason to rebuild it on every question.
    """

    def retrieve_node(state: GraphState) -> dict:
        chunks = retriever.retrieve(state["question"])
        sources = sorted({c["source"] for c in chunks})
        return {"chunks": chunks, "sources": sources}

    def grade_node(state: GraphState) -> dict:
        context = "\n\n".join(c["text"] for c in state["chunks"])
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Does the retrieved context contain enough information to fully "
            f"answer the question? Respond with exactly one word: YES or NO."
        )
        response = groq_client.chat.completions.create(
            model=config.LIGHT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return {
            "is_sufficient": verdict.startswith("YES"),
            "attempts": state["attempts"] + 1,
        }

    def rewrite_node(state: GraphState) -> dict:
        prompt = (
            f"This question did not retrieve enough relevant information from "
            f"a knowledge base: \"{state['question']}\"\n\n"
            f"Rewrite it to be more specific and more likely to match relevant "
            f"document content. Return ONLY the rewritten question, nothing else."
        )
        response = groq_client.chat.completions.create(
            model=config.LIGHT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        rewritten = response.choices[0].message.content.strip()
        return {"question": rewritten}

    def generate_node(state: GraphState) -> dict:
        context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in state["chunks"])
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        response = groq_client.chat.completions.create(
            model=config.MAIN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["original_question"]},
            ],
            temperature=0.3,
        )
        return {"answer": response.choices[0].message.content}

    def route_after_retrieve(state: GraphState) -> str:
        return "grade" if state["deep_reasoning"] else "generate"

    def route_after_grade(state: GraphState) -> str:
        if state["is_sufficient"] or state["attempts"] >= MAX_GRADE_ATTEMPTS:
            return "generate"
        return "rewrite"

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve)
    graph.add_conditional_edges("grade", route_after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def answer_question(question: str, graph, cache, deep_reasoning: bool = None) -> dict:
    """
    Top-level entry point used by app.py. Checks the semantic cache
    first (fast path, no LLM call on a hit); otherwise runs the
    LangGraph pipeline and stores the result in the cache for next time.

    Returns {"answer": str, "sources": list, "from_cache": bool}.
    """
    if deep_reasoning is None:
        deep_reasoning = config.DEEP_REASONING_MODE_DEFAULT

    cached_answer = cache.get(question)
    if cached_answer is not None:
        return {"answer": cached_answer, "sources": [], "from_cache": True}

    initial_state = {
        "question": question,
        "original_question": question,
        "chunks": [],
        "sources": [],
        "answer": "",
        "deep_reasoning": deep_reasoning,
        "is_sufficient": False,
        "attempts": 0,
    }
    final_state = graph.invoke(initial_state)

    cache.add(question, final_state["answer"])

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "from_cache": False,
    }
