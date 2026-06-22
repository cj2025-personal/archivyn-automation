"""
Semantic splitting helper.

This module is intentionally decoupled so it can be wired in when needed.
"""
from typing import List

try:
    from semantic_text_splitter import TextSplitter
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError(
        "semantic_splitter requires: semantic-text-splitter, sentence-transformers, transformers"
    ) from e


def semantic_split(
    text: str,
    max_tokens: int = 200,
    overlap_tokens: int = 30,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    tokenizer_name: str = "gpt2",
) -> List[str]:
    """
    Split text into semantically coherent chunks with overlap.
    """
    if not text or not text.strip():
        return []

    model = SentenceTransformer(embedding_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    splitter = TextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap=overlap_tokens,
        model=model,
    )
    return [c.strip() for c in splitter.chunks(text) if c and c.strip()]
