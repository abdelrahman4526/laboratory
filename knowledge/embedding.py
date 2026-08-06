"""
embedding.py
Generates embeddings for approved knowledge using Gemini's gemini-embedding-001 model.

Embedding input = Name + Description + Keywords + Aliases (combined into search_text).
"""

import logging
import threading

import numpy as np

from .schemas import ApprovedKnowledge
from .utils import get_gemini_client

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768  



def build_embedding_text(name: str, approved: ApprovedKnowledge) -> str:
    """Combines Name + Description + Keywords + Aliases into search_text string."""
    if approved.search_text:
        return approved.search_text

    an = approved.alias_names
    aliases_str = ", ".join(an.aliases)
    extra_names = [n for n in (an.alias, an.equivalent_name) if n and n.strip()]
    if extra_names:
        aliases_str = ", ".join([aliases_str, *extra_names]) if aliases_str else ", ".join(extra_names)

    keywords_str = ", ".join(approved.keywords) if isinstance(approved.keywords, list) else str(approved.keywords or "")
    return f"{name}\n{approved.description}\nالمرادفات والأسماء البديلة: {aliases_str}\nالكلمات المفتاحية: {keywords_str}".strip()


def generate_embedding(name: str, approved: ApprovedKnowledge) -> list[float]:
    
    client_or_genai = get_gemini_client()
    text_input = build_embedding_text(name, approved)

    if hasattr(client_or_genai, "models"):
        res = client_or_genai.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text_input,
            config={"output_dimensionality": EMBEDDING_DIM}
        )
        embedding = res.embeddings[0].values
    else:
        result = client_or_genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text_input,
            task_type="retrieval_document",
            title=name,
            output_dimensionality=EMBEDDING_DIM,
        )
        embedding = result["embedding"]

    logger.info("Generated embedding of length %s for '%s'", len(embedding), name)

    vec = np.array(embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()