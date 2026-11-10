"""Steps 4–5: identify Ellie's speaker label and classify her sentences via a question bank.

Question bank CSV columns:
  text      Ellie's sentence (any casing/punctuation)
  type      main / followup / backchannel / opening / closing
  rapport   1 if it's a rapport question (e.g. "where are you from originally"), else 0
  items     PHQ item numbers this question maps to, ';'-separated (e.g. "3" or "2;6"), may be empty
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from ..config import Cfg, resolve

_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")


def normalize(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


@dataclass
class BankEntry:
    text: str
    type: str
    rapport: bool
    items: list[int]


class QuestionBank:
    def __init__(self, cfg: Cfg):
        ec = cfg.preprocessing.ellie
        df = pd.read_csv(resolve(ec.question_bank), dtype=str).fillna("")
        self.entries = [
            BankEntry(text=normalize(r["text"]), type=r["type"].strip(),
                      rapport=str(r["rapport"]).strip() in ("1", "1.0", "true", "True"),
                      items=[int(x) for x in str(r["items"]).replace(".0", "").split(";") if x.strip()])
            for _, r in df.iterrows()
        ]
        self.texts = [e.text for e in self.entries]
        self.threshold = ec.match_threshold * 100

    def match(self, sentence: str) -> tuple[BankEntry | None, float]:
        from rapidfuzz import fuzz, process

        q = normalize(sentence)
        if not q:
            return None, 0.0
        hit = process.extractOne(q, self.texts, scorer=fuzz.ratio)
        if hit is None or hit[1] < self.threshold:
            return None, float(hit[1]) if hit else 0.0
        return self.entries[hit[2]], float(hit[1])


def identify_ellie(utterances: list[dict], bank: QuestionBank) -> tuple[str | None, dict]:
    """Speaker whose sentences match the bank most often."""
    counts: Counter = Counter()
    for u in utterances:
        for s in split_sentences(u["text"]):
            if bank.match(s)[0] is not None:
                counts[u["speaker"]] += 1
    if not counts:
        return None, {}
    return counts.most_common(1)[0][0], dict(counts)


def classify_ellie_sentences(utterances: list[dict], ellie: str, bank: QuestionBank, cfg: Cfg) -> list[dict]:
    """Break Ellie's utterances into typed sentences; participant utterances are passed through.

    Sentence-level timestamps are approximated by the parent utterance's span.
    """
    unmatched = cfg.preprocessing.ellie.unmatched_as
    events = []
    for u in utterances:
        if u["speaker"] != ellie:
            events.append({"role": "participant", **u})
            continue
        for s in split_sentences(u["text"]):
            entry, score = bank.match(s)
            events.append({
                "role": "ellie", "text": s, "start": u["start"], "end": u["end"],
                "type": entry.type if entry else unmatched,
                "rapport": bool(entry and entry.rapport),
                "items": entry.items if entry else [],
                "match_score": score, "matched": entry is not None,
            })
    return events
