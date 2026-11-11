"""对比哪种 ASR 能保留语气词（um / uh / hmm …）：large-v3、large-v3 + 逐字提示词、CrisperWhisper。
只转写 loud 轨（VAD 0.12 区间），对齐后统计语气词数，并打印例句。
  python scripts/analysis/filler_test.py --sids 308 309
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import whisperx  # noqa: E402

from src.audio import load_wav  # noqa: E402
from src.preprocessing.vad import mask_non_speech  # noqa: E402

FILLER = re.compile(r"\b(u+m+|u+h+m*|h+m+|m+h*m+|e+r+m*|a+h+|e+h+)\b", re.I)
PROMPT = "Umm, let me think like, hmm... Okay, uh, here's what I'm, like, thinking. Mm-hmm, I mean, uh, yeah."
VARIANTS = {
    "large-v3": ("large-v3", None),
    "large-v3+prompt": ("large-v3", PROMPT),
    "crisper": ("nyrahealth/faster_CrisperWhisper", None),
}

ap = argparse.ArgumentParser()
ap.add_argument("--sids", nargs="*", type=int, default=[308, 309])
a = ap.parse_args()
align_model, meta = whisperx.load_align_model(language_code="en", device="cuda")
out = ROOT / "data/interim/filler_test"
out.mkdir(parents=True, exist_ok=True)
for name, (model, prompt) in VARIANTS.items():
    opts = {"initial_prompt": prompt, "suppress_tokens": []} if prompt or "crisper" in name else None
    asr = whisperx.load_model(model, "cuda", compute_type="float16", language="en", asr_options=opts)
    for sid in a.sids:
        wav = load_wav(ROOT / f"data/raw/{sid}_AUDIO.wav")
        speech = json.load(open(ROOT / f"data/interim/vad/{sid}.json"))["speech_by_threshold"]["0.12"]
        x = mask_non_speech(wav, speech, 16000)
        r = asr.transcribe(x, batch_size=16, language="en")
        r = whisperx.align(r["segments"], align_model, meta, x, "cuda", return_char_alignments=False)
        segs = r["segments"]
        json.dump(segs, open(out / f"{sid}_{name}.json", "w"), ensure_ascii=False, indent=1)
        text = " ".join(s["text"] for s in segs)
        f = FILLER.findall(text)
        print(f"{sid} {name:16s} words={len(text.split()):5d} fillers={len(f):3d} {sorted(set(x.lower() for x in f))}",
              flush=True)
        ex = [s["text"].strip() for s in segs if FILLER.search(s["text"])][:6]
        for e in ex:
            print("      ", e[:110])
    del asr
