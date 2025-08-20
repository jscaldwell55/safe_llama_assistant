# safe_sanitizer.py
import re
from typing import List, Dict, Any

_INJECTION_PATTERNS = [
    r'^\s*(system|assistant|developer)\s*:\s*',          # role-like headers
    r'(?i)\b(ignore (all|the) (previous|above) (rules|instructions))\b',
    r'(?i)\b(as an ai|you must|you will|do not follow|override)\b',
    r'(?i)\b(deactivate|disable) (safety|guard|policy)\b',
    r'(?i)\b(#\s*prompt|###\s*instructions|begin\s*system\s*prompt)\b',
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS))

def sanitize_chunk_text(text: str, max_len: int = 4000) -> str:
    """Remove role headers, jailbreak-y directives, and noisy boilerplate."""
    if not text:
        return ""
    # Strip role-like headers and known instruction markers line-by-line
    cleaned_lines = []
    for line in text.splitlines():
        if _INJECTION_RE.search(line):
            continue
        # Trim super long lines and normalize whitespace
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()

    # Guardrail: drop chunks that are mostly instructions
    bad_density = len(_INJECTION_RE.findall(text)) >= 2
    if bad_density or len(cleaned) < 50:
        return ""  # signal to drop this chunk

    # Keep a sane upper bound to avoid context bloat
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rsplit(" ", 1)[0].strip()
    return cleaned

def sanitize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return same shape as input, with possibly dropped/cleaned items."""
    sanitized = []
    for r in results:
        t = sanitize_chunk_text(r.get("text", ""))
        if not t:
            continue
        out = dict(r)
        out["text"] = t
        sanitized.append(out)
    return sanitized
