"""Smoke tests on synthetic data (no audio, no GPU models needed)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import N_ITEMS
from src.config import load_config
from src.evaluation.metrics import ccc
from src.evaluation.report import build_final, evaluate
from src.labels import ITEM_COLS, consistency_report, load_labels
from src.preprocessing import feature_path, interim_path
from src.preprocessing.ellie import QuestionBank, classify_ellie_sentences, identify_ellie
from src.preprocessing.qa_pairs import build_qa_pairs
from src.preprocessing.segments import make_segments
from src.splits import make_folds
from src.stage1.dataset import Stage1Sessions
from src.stage1.losses import class_weights, stage1_loss
from src.stage1.teacher import save_teacher, teacher_features
from src.stage2.dataset import Bags
from src.stage2.model import ResidualMIL
from src.stage2.train import run_cv


@pytest.fixture()
def cfg(tmp_path):
    c = load_config("config/default.yaml", [
        f"paths.labels_dir={tmp_path}/labels", f"paths.interim={tmp_path}/interim",
        f"paths.features={tmp_path}/features", f"paths.outputs={tmp_path}/out", "experiment.device=cpu",
        "stage2.train.max_epochs=3", "stage2.train.patience=2", "stage2.train.seeds=[0]",
    ])
    return c


def _write_labels(cfg, n_train=40, n_dev=10, seed=0):
    rng = np.random.default_rng(seed)
    d = (cfg.paths.labels_dir)
    import os
    os.makedirs(d, exist_ok=True)
    sids = np.arange(300, 300 + n_train + n_dev)
    sids[n_train // 2:n_train] += 300   # half the train set in the AI range
    items = rng.integers(0, 4, size=(len(sids), N_ITEMS))
    total = items.sum(1)
    det = pd.DataFrame(items, columns=cfg.labels.item_cols)
    det.insert(0, "Participant_ID", sids)
    det.to_csv(f"{d}/{cfg.labels.detailed_file}", index=False)
    for split, sl in [("train", slice(0, n_train)), ("dev", slice(n_train, None))]:
        pd.DataFrame({"Participant_ID": sids[sl], "Gender": rng.integers(0, 2, len(sids))[sl],
                      "PHQ_Binary": (total[sl] >= 10).astype(int), "PHQ_Score": total[sl]}
                     ).to_csv(f"{d}/{cfg.labels.split_files[split]}", index=False)
    return sids


def test_config_inheritance():
    c = load_config("config/eval_baseline.yaml")
    assert c.experiment.name == "baseline" and c.stage2.teacher.kind == "zeroshot"
    assert c.stage2.loss.w_tired == 0.5           # inherited
    a = load_config("config/ablations/A6_no_aux.yaml", ["stage2.loss.lambda_res=0.1"])
    assert a.stage2.loss.w_tired == 0.0 and a.stage2.loss.lambda_res == 0.1


def test_labels_and_folds(cfg):
    _write_labels(cfg)
    lab = load_labels(cfg)
    rep = consistency_report(lab)
    assert rep["sum_mismatch_sids"] == [] and rep["binary_mismatch_sids"] == []
    folds = make_folds(lab, cfg)
    assert sorted(folds.fold.unique()) == list(range(5)) and len(folds) == 40


def _fake_utts():
    w = lambda t, s, e, spk: {"word": t, "start": s, "end": e, "speaker": spk}  # noqa: E731
    utts = [
        ("A", 0, 3, "hi i'm ellie thanks for coming in today."),
        ("A", 3, 5, "where are you from originally?"),
        ("B", 5.5, 8, "i am from ohio"),
        ("A", 9, 12, "mhm. how easy is it for you to get a good night's sleep?"),
        ("B", 12.5, 40, " ".join(["word"] * 60)),
        ("A", 41, 43, "can you tell me more about that?"),
        ("B", 43.5, 46, "it is hard"),
        ("A", 47, 49, "thanks for sharing your thoughts with me."),
        ("B", 49.5, 50, "bye"),
    ]
    out = []
    for spk, s, e, text in utts:
        toks = text.split()
        step = (e - s) / len(toks)
        words = [w(t, s + i * step, s + (i + 1) * step - 0.01, spk) for i, t in enumerate(toks)]
        out.append({"speaker": spk, "start": s, "end": e, "text": text, "words": words})
    return out


def test_ellie_qa_segments(cfg):
    bank = QuestionBank(cfg)
    utts = _fake_utts()
    ellie, _ = identify_ellie(utts, bank)
    assert ellie == "A"
    events = classify_ellie_sentences(utts, ellie, bank, cfg)
    pairs, ellie_iv = build_qa_pairs(events, cfg)
    # rapport question dropped (keep_rapport=false); closing trimmed; follow-up merged
    assert len(pairs) == 1 and pairs[0]["items"] == [3]
    assert "tell me more" in pairs[0]["question"] and "it is hard" in pairs[0]["answer"]
    segs = make_segments(pairs, ellie_iv, 60.0, cfg)
    assert all(s["end"] - s["start"] <= 20.0 + 2 * 0.2 + 1e-6 for s in segs) and len(segs) >= 2
    for s in segs:   # never overlaps Ellie
        assert not any(s["start"] < b and s["end"] > a for a, b in ellie_iv)

    cfg.preprocessing.qa.keep_rapport = True
    pairs2, _ = build_qa_pairs(events, cfg)
    assert len(pairs2) == 2


def test_stage1_loss_and_prompt(cfg):
    y = np.random.default_rng(0).integers(0, 4, (20, N_ITEMS))
    w = class_weights(y)
    assert torch.allclose(torch.tensor([w[k, y[:, k]].mean().item() for k in range(N_ITEMS)]),
                          torch.ones(N_ITEMS), atol=1e-5)
    logits = torch.randn(2, N_ITEMS, 4, requires_grad=True)
    loss, _ = stage1_loss(logits, torch.tensor(y[:2]), torch.tensor(y[:2].sum(1), dtype=torch.float32),
                          w, 0.5, 0.5)
    loss.backward()
    assert logits.grad is not None

    # prompt building from cached Q-A pairs
    sid = 300
    pairs = [{"pair_id": j, "question": f"q{j}", "answer": f"a{j}", "t_start": j, "t_end": j + 1,
              "items": [], "words": []} for j in range(10)]
    json.dump({"pairs": pairs}, open(interim_path(cfg, "qa_pairs", sid, "json"), "w"))
    ret = {sid: {k: {"ranked": list(range(9, 1, -1)), "scores": [0.5] * 8, "max_r": 0.5, "ev": int(k != 5)}
                 for k in range(1, 9)}}
    ds = Stage1Sessions(cfg, [sid], ret)
    p = ds.get(0)["prompts"]
    assert len(p) == 8 and p[0].endswith("Score:") and "[6]" in p[0] and "[7]" not in p[0]
    assert "does not discuss" in p[4] and "does not discuss" not in p[0]


def test_teacher_features():
    probs = np.full((3, N_ITEMS, 4), 0.25)
    tf = teacher_features(probs)
    assert np.allclose(tf["y_text"], 12.0)
    assert np.allclose(tf["tau"][:, 0], np.log(3)) and np.allclose(tf["tau"][:, 1], 0.0)


def test_stage2_zero_init():
    m = ResidualMIL(n_layers=5, feat_dim=16)
    B, M = 3, 7
    yt = torch.tensor([3.0, 10.0, 22.0])
    tau, nu = torch.randn(B, 2), torch.randn(B)
    mask = torch.ones(B, M, dtype=torch.bool)
    mask[0, 4:] = False
    out = m(torch.randn(B, M, 5, 16), torch.rand(B, M), mask, yt, tau, nu)
    assert torch.allclose(out["y_tot"], yt) and torch.allclose(out["tired_logits"], tau)
    assert torch.allclose(out["attn"].sum(-1), torch.ones(B)) and (out["attn"][0, 4:] == 0).all()


def test_stage2_cv_and_eval(cfg):
    sids = _write_labels(cfg)
    lab = load_labels(cfg)
    folds = make_folds(lab, cfg)
    rng = np.random.default_rng(0)
    for s in sids:
        m = int(rng.integers(3, 12))
        torch.save({"feats": torch.randn(m, 5, 16).half(), "seg_ids": torch.arange(m),
                    "pos": torch.rand(m), "layers": "all"}, feature_path(cfg, int(s)))
    probs = rng.dirichlet(np.ones(4), size=(len(sids), N_ITEMS)).astype(np.float32)
    tdir = cfg.paths.outputs + "/main/stage1"
    import os
    os.makedirs(tdir, exist_ok=True)
    split = lab.set_index("sid").loc[sids, "split"].to_numpy()
    save_teacher(f"{tdir}/teacher_oof.npz", sids, split, probs)

    bags = Bags(cfg, [int(s) for s in sids], f"{tdir}/teacher_oof.npz", lab)
    dev_idx = [i for i, s in enumerate(bags.sids) if s in set(lab.sid[lab.split == "dev"])]
    res = run_cv(cfg, bags, folds, {"dev": dev_idx}, "cpu")
    assert np.isfinite(res["cv_ccc"]) and res["eval"]["dev"]["y_tot"].shape == (10,)

    from src.stage2.train import predictions_frame
    pred = pd.concat([predictions_frame(bags, res["train_idx"], res["oof"], "train"),
                      predictions_frame(bags, dev_idx, res["eval"]["dev"], "dev")])
    final, info = build_final(pred, lab, list(cfg.fusion.beta_grid))
    t = {"sids": np.asarray(sids), "split": split, "probs": probs}
    metrics = evaluate(final, lab, t, cfg)
    assert 0 <= info["beta"] <= 1 and "dev" in metrics and "items" in metrics["dev"]
    assert set(metrics["dev"]["items"]) == set(f"item{k}" for k in range(1, 9))
    assert ITEM_COLS[0] == "item1"


def test_ccc():
    y = np.arange(10.0)
    assert ccc(y, y) == pytest.approx(1.0) and ccc(y, y + 5) < 1.0


# ---------------------------------------------------------------- transcribe.method = compressed
def _seg(t0, t1, text):
    toks = text.split()
    step = (t1 - t0) / len(toks)
    return {"start": t0, "end": t1, "text": text,
            "words": [{"word": w, "start": t0 + i * step, "end": t0 + (i + 1) * step - 0.01, "score": 0.9}
                      for i, w in enumerate(toks)]}


def test_compress_lifts_quiet_keeps_loud():
    from src.preprocessing.compress import compress

    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    wav = np.concatenate([1e-4 * rng.standard_normal(sr), 0.3 * tone, 0.003 * tone,
                          1e-4 * rng.standard_normal(sr)]).astype(np.float32)
    out, _ = compress(wav, sr)
    rms = lambda x: float(np.sqrt((x ** 2).mean()))  # noqa: E731
    assert abs(rms(out[sr + 4000:2 * sr - 4000]) / rms(wav[sr + 4000:2 * sr - 4000]) - 1) < 0.1   # loud untouched
    assert rms(out[2 * sr + 4000:3 * sr - 4000]) > 10 * rms(wav[2 * sr + 4000:3 * sr - 4000])    # quiet lifted
    assert rms(out[:sr - 4000]) < 2 * rms(wav[:sr - 4000])                                       # noise gated


def test_asr_merge_repeats_drift_and_fill():
    from src.preprocessing.asr_merge import collapse_repeats, drift_guard, merge_passes

    loop = [_seg(10 + i, 10.8 + i, "i'm from new york") for i in range(5)]
    out, n = collapse_repeats(loop, [_seg(10, 11, "something else")])
    assert len(out) == 1 and n == 4
    real = [_seg(10 + i, 10.8 + i, "i'm trying") for i in range(4)]
    assert len(collapse_repeats(real, [_seg(10, 14, "i'm trying i'm trying i'm trying")])[0]) == 4

    raw = [_seg(20, 22, "think of me as a friend today"), _seg(30, 32, "your answers are totally confidential"),
           _seg(50, 52, "where are you from originally")]
    comp = [_seg(14, 16, "think of me as a friend today"), _seg(22, 24, "your answers are totally confidential"),
            _seg(50, 52, "where are you from originally"), _seg(60, 62, "only the quiet pass heard this")]
    kept, demoted = drift_guard(comp, [raw])
    assert [s["text"] for s in demoted][:2] == [comp[0]["text"], comp[1]["text"]]
    assert comp[3] in kept

    units, info = merge_passes({"comp": comp, "raw": raw, "novad": raw}, ["comp", "raw"], refs=["raw", "novad"])
    texts = [u["text"] for u in units]
    assert "only the quiet pass heard this" in texts                     # comp-only content kept
    assert sum("think of me as a friend" in t for t in texts) == 1       # drifted duplicate not doubled
    starts = {u["text"]: u["start"] for u in units}
    assert abs(starts["think of me as a friend today"] - 20) < 0.5       # raw timing wins in drifted region


def test_script_score_and_voice_classify():
    from src.preprocessing.asr_merge import to_units
    from src.preprocessing.speaker import text_scores
    from src.preprocessing.voiceprint import classify

    units = to_units([_seg(0, 2, "Hi, I'm Allie."), _seg(3, 5, "I moved here last year.")], "comp", 2.0)
    sc = text_scores(units, ["hi i'm ellie", "thanks for coming in today"])
    assert sc[0] > 0.8 and sc[1] < 0.5

    rng = np.random.default_rng(0)
    cE, cP = rng.standard_normal(16), rng.standard_normal(16)
    is_e = np.array([1] * 8 + [0] * 12, bool)
    emb = np.stack([(cE if e else cP) + 0.3 * rng.standard_normal(16) for e in is_e])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    txt = np.where(is_e, 0.6, 0.3)
    txt[:3] = 0.95                                                        # Ellie anchors
    nw = np.full(20, 6)
    vc = {"anchor_e": 0.85, "anchor_p": 0.5, "knn": 5, "voice_weight": 3.0, "text_weight": 4.0}
    pred, _, info = classify(emb, txt, nw, np.full(20, 2.0), np.zeros(20), vc)
    assert (pred == is_e).all() and info["n_anchor_E"] == 3


def test_switch_to_two_track_preset():
    from src.config import load_config

    d, t = load_config("config/default.yaml"), load_config("config/two_track.yaml")
    assert d.preprocessing.transcribe.method == "compressed" and t.preprocessing.transcribe.method == "two_track"
    assert d.paths.interim != t.paths.interim                                    # 缓存分开
    assert d.to_dict()["stage1"] == t.to_dict()["stage1"]                         # 其余不变
