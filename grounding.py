# grounding.py
from typing import List, Dict, Tuple
import re
import numpy as np

# --- Sentence splitting & boilerplate filters ---

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

_SKIP_SENTENCE_PATTERNS = [
    r'^according to (the )?(medication guide|document|label)\b',
    r'^this is not a complete list\b',
    r'^(for (more|full) information|see (the )?(medication guide|label))\b',
    r'^(please|always) (talk|speak|consult) (with|to) your healthcare provider\b',
    r'^(note|disclaimer):\s*',
]
_SKIP_SENTENCE_RE = re.compile("|".join(_SKIP_SENTENCE_PATTERNS), re.IGNORECASE)

def split_sentences(text: str) -> List[str]:
    """Split into sentences and drop trivial/boilerplate ones that won't ground."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]
    out: List[str] = []
    for s in sents:
        if len(s) < 12:
            continue
        if _SKIP_SENTENCE_RE.search(s):
            continue
        out.append(s)
    return out

# --- Context chunk utilities ---

_HEADER_RE = re.compile(r'^Source:\s.*?\n', flags=re.IGNORECASE)

def strip_header(chunk: str) -> str:
    """Remove 'Source: ..., Chunk N' header lines from formatted context chunks."""
    return _HEADER_RE.sub("", chunk or "").strip()

def window_chunks(chunks: List[str], window_chars: int = 800) -> List[Tuple[int, str]]:
    """Create sliding windows over each chunk to improve alignment granularity."""
    windows: List[Tuple[int, str]] = []
    for i, ch in enumerate(chunks):
        ch = (ch or "").strip()
        if not ch:
            continue
        if len(ch) <= window_chars:
            windows.append((i, ch))
            continue
        step = max(100, window_chars // 2)
        for start in range(0, len(ch), step):
            seg = ch[start:start + window_chars].strip()
            if len(seg) > 50:
                windows.append((i, seg))
    return windows

# --- Similarity & grounding ---

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def per_sentence_grounding(
    sentences: List[str],
    chunks: List[str],
    embed_fn,                      # callable: List[str] -> np.ndarray [N, d]
    threshold: float = 0.55,
    window_chars: int = 800,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Returns:
      best_scores: {sent_idx -> best_cosine}
      best_chunk_index: {sent_idx -> original_chunk_id that matched best}
    """
    # Strip headers from chunks to avoid polluting embeddings
    base_chunks = [strip_header(c) for c in (chunks or []) if c and c.strip()]
    # Build windows over chunks to better match sentence granularity
    chunk_windows = window_chunks(base_chunks, window_chars=window_chars)  # [(orig_idx, text)]
    if not chunk_windows or not sentences:
        return {}, {}

    # Embed all at once (efficiency)
    sent_vecs = embed_fn(sentences).astype('float32')
    win_texts = [w[1] for w in chunk_windows]
    win_vecs = embed_fn(win_texts).astype('float32')

    # Normalize for cosine via dot
    sent_vecs /= np.linalg.norm(sent_vecs, axis=1, keepdims=True) + 1e-8
    win_vecs  /= np.linalg.norm(win_vecs,  axis=1, keepdims=True) + 1e-8

    best_scores: Dict[int, float] = {}
    best_chunk_index: Dict[int, int] = {}

    # Compute dense sims in blocks to avoid memory spikes
    block = 256
    W = win_vecs.T  # [d, W]
    for i in range(0, len(sentences), block):
        s_block = sent_vecs[i:i+block]               # [B, d]
        sims = s_block @ W                           # [B, W]
        top_idx = np.argmax(sims, axis=1)            # [B]
        top_val = sims[np.arange(sims.shape[0]), top_idx]
        for j, (idx, val) in enumerate(zip(top_idx, top_val)):
            sent_id = i + j
            best_scores[sent_id] = float(val)
            best_chunk_index[sent_id] = chunk_windows[idx][0]  # original chunk id

    return best_scores, best_chunk_index


