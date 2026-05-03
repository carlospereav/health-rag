"""
RAG answer step: retrieve top-k chunks with search.py, then answer via Groq
using only the retrieved context and explicit source indices [1]…[k].
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Import after load_dotenv so Qdrant env is available for search.py side effects.
import search  # noqa: E402

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a careful assistant for a retrieval-augmented health knowledge base.

Rules:
- Answer using ONLY the numbered context passages below. If the context does not contain enough information, say so clearly instead of guessing.
- When you use information from a passage, cite it with the matching bracket label: [1], [2], etc., corresponding to passage order.
- Do not cite or invent sources, titles, or facts that are not supported by the context.
- Write in the same language as the user's question.
- Be clear and concise."""


def _require_groq_key() -> str:
    if not GROQ_API_KEY or not GROQ_API_KEY.strip():
        sys.exit("GROQ_API_KEY is not set or is empty. Check your .env file.")
    return GROQ_API_KEY.strip()


def format_context(hits: list[dict]) -> str:
    """Build a numbered context block from search hits (1-based indices)."""
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        author = meta.get("author", "N/A")
        title = meta.get("source_title", "N/A")
        stype = meta.get("source_type", "N/A")
        paradigm = meta.get("paradigm", "N/A")
        header = (
            f"[{i}] {author} — {title} ({stype}, paradigm: {paradigm})"
        )
        parts.append(f"{header}\n{hit['text'].strip()}")
    return "\n\n".join(parts)


def build_messages(question: str, context_block: str) -> list[dict]:
    user_content = (
        "Context passages (use only these; cite with [1], [2], … as needed):\n\n"
        f"{context_block}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer(
    question: str,
    *,
    top_k: int = search.DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    model: str | None = None,
) -> str:
    """Retrieve chunks, then return the assistant reply from Groq."""
    hits = search.search(question, top_k=top_k)
    if not hits:
        return ""

    context_block = format_context(hits)
    resolved_model = (model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip()
    client = Groq(api_key=_require_groq_key())
    messages = build_messages(question, context_block)

    try:
        completion = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
        )
    except Exception as exc:
        sys.exit(f"Groq API error: {exc}")

    choice = completion.choices[0].message
    return (choice.content or "").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Answer a question using retrieved context and Groq.",
    )
    p.add_argument("question", help="User question")
    p.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=search.DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default {search.DEFAULT_TOP_K})",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE})",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"Groq model id (default: env GROQ_MODEL or {DEFAULT_GROQ_MODEL})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text = answer(
        args.question,
        top_k=args.top_k,
        temperature=args.temperature,
        model=args.model,
    )
    if not text:
        print("No results found. Is the collection populated?")
    else:
        print(text)
