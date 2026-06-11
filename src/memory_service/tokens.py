"""Approximate token counting for the /recall budget.

The contract allows approximation ("don't blow past it by 2x"). ~4 chars per
token is right for English prose but undercounts CJK and other dense scripts
by up to ~4x (they tokenize near 1 token/char), which could breach the 2x
ceiling for non-Latin profiles — so non-ASCII characters are charged a full
token each. Conservative (over-counts accented Latin text), which errs on
the safe side of the budget contract.
"""

from __future__ import annotations


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_count = len(text) - non_ascii
    return max(1, ascii_count // 4 + non_ascii)
