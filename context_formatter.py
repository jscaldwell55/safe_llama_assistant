# context_formatter.py
import logging
import re
from typing import Iterable, List, Union, Dict

logger = logging.getLogger(__name__)

# ---------- helpers to keep context clean and sentence-complete ----------

_SENT_END = re.compile(r'[.!?)]\s*$')

def _clip_to_sentence_boundaries(text: str) -> str:
    """Trim a paragraph so it ends on a sentence boundary."""
    if not text:
        return text
    text = text.strip()

    # Already ends cleanly?
    if _SENT_END.search(text):
        return text

    # Otherwise, cut back to the last terminator if any
    m = list(re.finditer(r'[.!?)]', text))
    if m:
        return text[:m[-1].end()].rstrip()

    # Fallback: drop a very short dangling last line
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and len(lines[-1].split()) <= 5:
        return "\n".join(lines[:-1]).strip()
    return text


def _dedupe_exact_lines(text: str) -> str:
    """Remove exact duplicate lines, preserving first occurrence."""
    seen = set()
    out_lines: List[str] = []
    for ln in [l for l in text.splitlines() if l.strip()]:
        key = ln.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out_lines.append(ln)
    return "\n".join(out_lines)


def finalize_context_block(raw: str) -> str:
    """
    Apply sentence-boundary clipping per paragraph and exact-line dedupe.
    Keep it conservative—do not paraphrase or remove medically-relevant content.
    """
    paras = [p.strip() for p in re.split(r'\n{2,}', raw or '') if p.strip()]
    clipped = [_clip_to_sentence_boundaries(p) for p in paras]
    final_text = "\n\n".join(clipped).strip()
    final_text = _dedupe_exact_lines(final_text)
    return final_text


# ---------- primary formatting API ----------

def _extract_text(chunk: Union[str, Dict[str, str]]) -> str:
    """Accepts either raw strings or dicts with 'text'/'content' keys."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for k in ("text", "content", "chunk", "body"):
            if k in chunk and isinstance(chunk[k], str):
                return chunk[k]
    return ""


def format_context(chunks: Iterable[Union[str, Dict[str, str]]],
                   max_chars: int = 4000) -> str:
    """
    Build a compact, readable context block from retrieved chunks.
    - Joins paragraphs with a blank line
    - Clips to sentence boundaries
    - Light exact-line dedupe
    - Enforces a soft max length (character-based)
    """
    texts: List[str] = []
    running_len = 0

    for ch in chunks or []:
        t = _extract_text(ch).strip()
        if not t:
            continue
        # keep each chunk as its own paragraph to preserve local structure
        candidate = t if t.endswith(("\n", "\r")) else f"{t}"
        if running_len + len(candidate) > max_chars and running_len > 0:
            break
        texts.append(candidate)
        running_len += len(candidate)

    raw_block = "\n\n".join(texts).strip()
    formatted = finalize_context_block(raw_block)

    logger.info(f"Formatted {len(texts)} chunks into {len(formatted)} chars")
    return formatted


__all__ = [
    "format_context",
    "finalize_context_block",
]
