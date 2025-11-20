# utils/assistant.py

from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from models.embeddings import EmbeddingClient
from utils.rag import retrieve_relevant_chunks
from utils.search import web_search


# Keywords that usually need fresh, external info
OUTAGE_KEYWORDS = [
    "outage",
    "down",
    "status",
    "today",
    "current",
    "latest",
    "incident",
    "issue",
    "maintenance",
]


def should_use_web_search(user_query: str, rag_results: List[Dict]) -> bool:
    """
    Decide whether we should call web search.

    Heuristics:
    - If query mentions 'outage', 'status', 'latest', etc. -> True
    - If no RAG results or very low similarity -> True
    - Otherwise -> False
    """
    q = (user_query or "").lower()

    # 1. Keyword-based trigger
    if any(kw in q for kw in OUTAGE_KEYWORDS):
        return True

    # 2. If no RAG results at all
    if not rag_results:
        return True

    # 3. If average similarity is too low -> docs not helpful
    avg_score = sum(r["score"] for r in rag_results) / len(rag_results)
    if avg_score < 0.25:
        return True

    return False


def build_context_block(rag_results: List[Dict], web_results: List[Dict]) -> str:
    """
    Convert RAG + web results into a single text block for the LLM.
    """
    parts: List[str] = []

    if rag_results:
        rag_lines = ["Internal documentation:"]
        for i, r in enumerate(rag_results, 1):
            rag_lines.append(f"[Doc {i} | {r['source']}] {r['text']}")
        parts.append("\n".join(rag_lines))

    if web_results:
        web_lines = ["Web search results:"]
        for i, w in enumerate(web_results, 1):
            web_lines.append(
                f"[Web {i} | {w['title']}] {w['snippet']} (URL: {w['url']})"
            )
        parts.append("\n".join(web_lines))

    return "\n\n".join(parts).strip()


def build_system_prompt(mode: str) -> str:
    """
    Build system prompt based on response mode: 'concise' or 'detailed'.
    """
    base = (
        "You are SmartAssist, an AI customer support agent for a SaaS product.\n"
        "You must always be accurate and honest. If the context is insufficient, say so clearly.\n"
        "Internal documentation is the main source of truth. Use web results only for fresh/external information.\n\n"
    )

    mode = (mode or "").lower()
    if mode == "concise":
        base += (
            "Respond in 2–4 short sentences. Be crisp, direct, and easy to skim.\n"
            "If steps are needed, use at most 3 short bullet points.\n"
        )
    else:  # detailed
        base += (
            "Provide a detailed, step-by-step answer.\n"
            "Use numbered or bulleted lists where helpful.\n"
            "When relevant, mention which source you used, e.g., 'According to Doc 1' or 'Based on Web 2'.\n"
        )

    return base


def answer_query(
    user_query: str,
    mode: str,
    chat_model,
    embed_client: EmbeddingClient,
    vectorstore: Dict,
    top_k: int = 5,
) -> Dict:
    """
    End-to-end pipeline:

    1. Get RAG results from vectorstore.
    2. Decide if web search is needed.
    3. Build combined context block.
    4. Call chat_model with system + user messages.
    5. Return answer + metadata (sources).
    """
    # 1. Retrieve from internal docs
    rag_results = retrieve_relevant_chunks(
        query=user_query,
        embed_client=embed_client,
        vectorstore=vectorstore,
        top_k=top_k,
    )

    # 2. Decide web search usage
    use_web = should_use_web_search(user_query, rag_results)

    # 3. Web search if needed
    web_results: List[Dict] = web_search(user_query, k=3) if use_web else []

    # 4. Build context + system prompt
    context_block = build_context_block(rag_results, web_results)
    system_prompt = build_system_prompt(mode)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"User question:\n{user_query}\n\n"
                f"Here is the available context from internal docs and web (if any):\n{context_block}\n\n"
                "Using ONLY this information, answer the question. "
                "If the context does not contain enough information, say that explicitly."
            )
        ),
    ]

    # 5. Call the LLM
    try:
        response = chat_model.invoke(messages)
        answer_text = response.content
    except Exception as e:
        answer_text = f"Error getting response from model: {str(e)}"

    return {
        "answer": answer_text,
        "rag_results": rag_results,
        "web_results": web_results,
        "used_web": use_web,
    }
