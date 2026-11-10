"""Steps 6–7: build Q-A pairs and trim opening / closing."""
from __future__ import annotations

from ..config import Cfg


def build_qa_pairs(events: list[dict], cfg: Cfg) -> tuple[list[dict], list[tuple[float, float]]]:
    """Returns (pairs, ellie_intervals).

    A main question opens a new pair; follow-ups are appended to the question;
    backchannels are ignored; participant speech until the next main question is the answer.
    """
    qc = cfg.preprocessing.qa
    ellie_intervals = [(e["start"], e["end"]) for e in events if e["role"] == "ellie"]

    start = 0
    if qc.trim_opening:
        start = next((i for i, e in enumerate(events) if e["role"] == "ellie" and e["type"] == "main"),
                     len(events))
    end = len(events)
    if qc.trim_closing:
        end = next((i for i, e in enumerate(events) if i >= start and e["role"] == "ellie"
                    and e["type"] == "closing"), len(events))

    pairs: list[dict] = []
    cur: dict | None = None
    for e in events[start:end]:
        if e["role"] == "ellie":
            if e["type"] == "main":
                cur = {"question": e["text"], "items": list(e["items"]), "rapport": e["rapport"],
                       "q_start": e["start"], "answer_words": [], "answer_utts": []}
                pairs.append(cur)
            elif e["type"] == "followup" and cur is not None:
                cur["question"] += " " + e["text"]
            # backchannel / opening / closing inside the interview: ignored
        elif cur is not None:
            cur["answer_words"].extend(e["words"])
            cur["answer_utts"].append(e["text"])

    out = []
    for p in pairs:
        if p["rapport"] and not qc.keep_rapport:
            continue
        if not p["answer_words"]:
            continue
        out.append({
            "pair_id": len(out),
            "question": p["question"],
            "answer": " ".join(p["answer_utts"]).strip(),
            "t_start": float(p["answer_words"][0]["start"]),
            "t_end": float(p["answer_words"][-1]["end"]),
            "items": p["items"],
            "rapport": p["rapport"],
            "words": p["answer_words"],
        })
    return out, ellie_intervals
