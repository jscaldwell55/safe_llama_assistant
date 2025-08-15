import logging
import re
from typing import Any, Iterable, List, Tuple, Union

try:
    from config import MAX_CONTEXT_LENGTH
except Exception:
    MAX_CONTEXT_LENGTH = 4000  # sensible fallback

logger = logging.getLogger(__name__)

_Textish = Union[str, dict, Tuple[Any, ...]]

def _extract_text(item: _Textish) -> str:
    """
    Robustly extract text from various chunk shapes:
    - str
    - dicts with 'text' or 'chunk' or 'content'
    - tuples where the first element looks like text
    """
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("text", "chunk", "content"):
            if k in item and isinstance(item[k], str):
                return item[k]
    if isinstance(item, tuple) and item and isinstance(item[0], str):
        return item[0]
    return str(item)

def _clean_piece(text: str) -> str:
    """
    Light cleanup: collapse internal whitespace, strip fence artefacts,
    keep punctuation/line breaks reasonable.
    """
    if not text:
        return ""
    # Remove accidental prompt/role labels if present inside chunks
    text = re.sub(r"^\s*(User|Assistant)\s*:\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    # Collapse long runs of whitespace but keep single newlines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _dedupe_keep_order(pieces: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in pieces:
        key = p.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out

def _truncate_chars(s: str, limit: int) -> str:
    if limit and len(s) > limit:
        s = s[:limit].rstrip()
        # avoid cutting mid-word if possible
        cut = re.search(r".*\b", s[::-1])
        if cut and cut.end() > 0:
            s = s[: len(s) - cut.end()]
    return s

def format_retrieved_context(chunks: Iterable[_Textish], max_chars: int = MAX_CONTEXT_LENGTH) -> str:
    """
    Assemble a compact, deduplicated context block for the LLM.
    Returns plain text (no 'Context:' header) so the prompt wrapper can add its own label.
    """
    if not chunks:
        logger.info("Formatted 0 chunks into 0 chars")
        return ""

    # 1) Extract + clean each piece
    pieces = [_clean_piece(_extract_text(c)) for c in chunks]
    pieces = [p for p in pieces if p]  # drop empties
    if not pieces:
        logger.info("Formatted 0 chunks into 0 chars")
        return ""

    # 2) Deduplicate by content
    pieces = _dedupe_keep_order(pieces)

    # 3) Join with a lightweight separator to keep sections distinct
    block = "\n---\n".join(pieces)

    # 4) Trim obvious trailing garbage markers inside the block
    block = re.sub(r"(?:#+\s*)+$", "", block).rstrip()

    # 5) Enforce char budget
    block = _truncate_chars(block, max_chars)

    logger.info(f"Formatted {len(pieces)} chunks into {len(block)} chars")
    return block

# Backward-compatible alias so code like `from context_formatter import context_formatter`
# continues to work without touching rag.py
def context_formatter(chunks: Iterable[_Textish], max_chars: int = MAX_CONTEXT_LENGTH) -> str:
    return format_retrieved_context(chunks, max_chars=max_chars)

def format_enhanced_context(
    chunks: Iterable[_Textish],
    query: str | None = None,
    max_chars: int = MAX_CONTEXT_LENGTH
) -> str:
    """
    Kept for callers that pass (chunks, query). We currently ignore `query` during formatting,
    since retrieval already handles relevance, but the signature remains stable.
    """
    return format_retrieved_context(chunks, max_chars=max_chars)

__all__ = ["format_retrieved_context", "format_enhanced_context", "context_formatter"]
