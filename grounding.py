# grounding.py
from typing import List, Dict, Tuple
import re
import numpy as np

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]
    # prune trivial fragments
    return [s for s in sents if len(s) >= 12]

def window_chunks(chunks: List[str], window_chars: int = 800) -> List[Tuple[int, str]]:
    """Create sliding windows over each chunk to improve alignment granularity."""
    windows: List[Tuple[int, str]] = []
    for i, ch in enumerate(chunks):
        ch = ch.strip()
        if len(ch) <= window_chars:
            windows.append((i, ch))
            continue
        step = window_chars // 2
        for start in range(0, len(ch), step):
            seg = ch[start:start+window_chars].strip()
            if len(seg) > 50:
                windows.append((i, seg))
    return windows

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def per_sentence_grounding(
    sentences: List[str],
    chunks: List[str],
    embed_fn,                      # callable: List[str] -> np.ndarray [N, d]
    threshold: float = 0.55
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Returns:
      best_scores: {sent_idx -> best_cosine}
      best_chunk_index: {sent_idx -> original_chunk_id that matched best}
    """
    # Build windows over chunks to better match sentence granularity
    chunk_windows = window_chunks(chunks, window_chars=800)  # [(orig_idx, text)]
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
    for i in range(0, len(sentences), block):
        s_block = sent_vecs[i:i+block]               # [B, d]
        sims = s_block @ win_vecs.T                  # [B, W]
        top_idx = np.argmax(sims, axis=1)            # [B]
        top_val = sims[np.arange(sims.shape[0]), top_idx]
        for j, (idx, val) in enumerate(zip(top_idx, top_val)):
            sent_id = i + j
            best_scores[sent_id] = float(val)
            best_chunk_index[sent_id] = chunk_windows[idx][0]  # original chunk id

    return best_scores, best_chunk_index

def attach_citations(sentences: List[str], best_chunk_index: Dict[int, int]) -> str:
    """Append [S{chunk_id}] to each sentence; join back into a paragraph."""
    out = []
    for i, s in enumerate(sentences):
        tag = ""
        if i in best_chunk_index:
            tag = f" [S{best_chunk_index[i]}]"
        # ensure sentence ends with punctuation before tag
        s = s.rstrip()
        if s and s[-1] not in ".!?":
            s += "."
        out.append(s + tag)
    return " ".join(out)
