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


BIN_LEVELS = ("0 = minimal (total 0-4), 1 = mild (5-9), 2 = moderate (10-14), "
              "3 = moderately severe or severe (15-24)")


def build_bin_prompt(evidence_lines: list[str]) -> str:
    """整体严重程度区间（4 档），证据为若干行文本（Q-A 对或转写行）。"""
    return "\n".join([
        "You are estimating the overall PHQ-8 depression severity of the participant for the past two weeks.",
        f"Severity levels: {BIN_LEVELS}.",
        "Evidence from the interview (in time order):", *evidence_lines,
        "Answer with a single digit 0-3.", "Level:"])


def pair_lines(pairs: list[dict], max_answer_words: int) -> list[str]:
    return [f"[{n}] Q: {p['question']} A: {_truncate(p['answer'], max_answer_words)}" for n, p in enumerate(pairs, 1)]
