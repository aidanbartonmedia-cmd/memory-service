"""Approximate token counting for the /recall budget.

The contract allows approximation ("don't blow past it by 2x"). We use the
standard ~4 chars/token heuristic for English prose, which under-counts dense
code and over-counts whitespace, but stays comfortably inside 2x for the
chat-style text this service emits.
"""

from __future__ import annotations


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
