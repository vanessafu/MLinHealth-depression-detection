"""Prompt template (Section 4)."""
from __future__ import annotations

NO_EVIDENCE = "The interview does not discuss this item directly."


def _truncate(text: str, max_words: int) -> str:
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words]) + " ..."


def build_prompt(k: int, item_text: str, pairs: list[dict], ev: int, max_answer_words: int) -> str:
    lines = [
        "You are rating one item of the PHQ-8 for the past two weeks.",
        f"Item {k}: {item_text}. 0 = not at all, 1 = several days, "
        "2 = more than half the days, 3 = nearly every day.",
        "Evidence from the interview (question / answer, in time order):",
    ]
    for n, p in enumerate(pairs, 1):
        lines.append(f"[{n}] Q: {p['question']} A: {_truncate(p['answer'], max_answer_words)}")
    if not ev:
        lines.append(NO_EVIDENCE)
    lines += ["Answer with a single digit 0-3.", "Score:"]
    return "\n".join(lines)
