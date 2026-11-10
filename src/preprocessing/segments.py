"""Step 8: cut participant audio segments from Q-A answers (length rules in config)."""
from __future__ import annotations

from ..config import Cfg


def _overlaps(s: float, e: float, intervals: list[tuple[float, float]]) -> bool:
    return any(s < b and e > a for a, b in intervals)


def _split_at_ellie(words: list[dict], intervals: list[tuple[float, float]]) -> list[list[dict]]:
    """Break the answer wherever Ellie speaks between two consecutive words (e.g. a follow-up)."""
    runs, cur = [], []
    for w in words:
        if cur and _overlaps(cur[-1]["end"], w["start"], intervals):
            runs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        runs.append(cur)
    return runs


def _split_long(words: list[dict], max_sec: float, pause: float) -> list[list[dict]]:
    """Greedy split at word boundaries; prefer the last pause >= `pause` before exceeding max_sec."""
    chunks, cur = [], []
    for w in words:
        if cur and w["end"] - cur[0]["start"] > max_sec:
            cut = None
            for i in range(len(cur) - 1, 0, -1):
                if cur[i]["start"] - cur[i - 1]["end"] >= pause:
                    cut = i
                    break
            if cut is None:
                chunks.append(cur)
                cur = []
            else:
                chunks.append(cur[:cut])
                cur = cur[cut:]
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def make_segments(pairs: list[dict], ellie_intervals: list[tuple[float, float]],
                  total_dur: float, cfg: Cfg) -> list[dict]:
    sc = cfg.preprocessing.segments
    segs = []
    for p in pairs:
        # 剔除与 Ellie 重叠的词
        words = [w for w in p["words"] if not _overlaps(w["start"], w["end"], ellie_intervals)]
        if not words or words[-1]["end"] - words[0]["start"] < sc.min_sec:
            continue
        chunks = [c for run in _split_at_ellie(words, ellie_intervals)
                  for c in _split_long(run, sc.max_sec, sc.split_pause_sec)]
        for chunk in chunks:
            s, e = chunk[0]["start"], chunk[-1]["end"]
            if e - s < sc.min_sec:
                continue
            # 两端加余量，但不进入 Ellie 的语音
            s2 = max(0.0, s - sc.margin_sec)
            e2 = min(total_dur, e + sc.margin_sec)
            for a, b in ellie_intervals:
                if s2 < b <= s:
                    s2 = b
                if e <= a < e2:
                    e2 = a
            segs.append({"seg_id": len(segs), "pair_id": p["pair_id"], "start": round(s2, 3),
                         "end": round(e2, 3), "pos": round(((s2 + e2) / 2) / total_dur, 5)})
    return segs
