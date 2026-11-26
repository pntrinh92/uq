"""Pilot: are certified OOD flags on LLM hidden states broken by harmless formatting changes?

For a fine-tuned Qwen2.5-0.5B classifier (nlp_ladder, one seed) this script
  1. re-encodes calA and the ID test sentences under label-preserving format transforms
     (trailing space, double spaces, newline, final period, capitalisation, upper case, prefix, quotes),
  2. extracts last-token states at the scaffold layers and the logits,
  3. computes conformal p-values against the *unperturbed* A2 half of calA (marginal split conformal),
     and reports accuracy and the ID false-alarm rate at alpha = 5% per transform, plus TPR on the
     cached OOD sets (AG News, 20 Newsgroups, random tokens),
  4. remedy: leave-one-transform-out nuisance projection. For each held-out transform, the top-k
     directions of the feature differences h(T(x)) - h(x) over A1 and all *other* transforms are
     projected out of every feature at that layer, the class-conditional tied Gaussian is refitted on the
     projected training features, and validity and power are measured again.

Usage: LADDER=<sst2|cola> NLP_SEED=0 $PY pilot.py extract   # GPU (or CPU), caches features
       LADDER=<sst2|cola> NLP_SEED=0 $PY pilot.py eval      # CPU, writes results json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "imagenet_ladder"))
sys.path.insert(0, os.path.join(HERE, "..", "nlp_ladder"))
from inl_common import CKPT, FEAT, LADDER, NLP_SEED, OOD_SETS, REGEN_LAYERS, val_split  # noqa: E402

OUT = f"/ephemeral/trinhpn_uq/nuisance_pilot/{LADDER}/seed{NLP_SEED}"
TRANSFORMS = ["orig", "space_toggle", "double_space", "newline", "period_toggle", "capitalize_toggle",
              "upper", "prefix", "quotes"]
LAYER_IDX = [l if l != "z" else 23 for l in REGEN_LAYERS]   # z = after final norm = index 23 of _cls
ALPHA = 0.05


def transform(text, ending, kind):
    """text: sentence without trailing whitespace; ending: what the training format appends (' ' or '')."""
    if kind == "orig":
        return text + ending
    if kind == "space_toggle":
        return text if ending else text + " "
    if kind == "double_space":
        return text.replace(" ", "  ") + ending
    if kind == "newline":
        return text + "\n"
    if kind == "period_toggle":
        return (text.rstrip(" .") if text.endswith(".") else text + ".") + ending
    if kind == "capitalize_toggle":
        t = text[:1].lower() + text[1:] if text[:1].isupper() else text[:1].upper() + text[1:]
        return t + ending
    if kind == "upper":
        return text.upper() + ending
    if kind == "prefix":
        return "Text: " + text + ending
    if kind == "quotes":
        return '"' + text + '"' + ending
    raise ValueError(kind)


def extract():
    import torch
    from transformers import AutoTokenizer
    import nlp_prepare as npp
    tok = AutoTokenizer.from_pretrained(npp.MODEL)
    data = npp.load_data()
    ids_val, y_val = data["sets"]["val"]
    sp = val_split()
    idx = np.r_[sp["calA"], sp["test"]]
    texts = [tok.decode(ids_val[i]) for i in idx]
    ending = " " if LADDER == "sst2" else ""
    if ending:
        assert all(t.endswith(" ") for t in texts)
    base = [t.rstrip() for t in texts]
    try:
        device = npp.pick_gpu(min_free_mb=3500)
    except RuntimeError as e:
        print("CPU fallback:", e, flush=True)
        device = torch.device("cpu")
    model = npp.load_model(device, f"{CKPT}/model").eval()
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/idx.npy", idx)
    for kind in TRANSFORMS:
        t0 = time.time()
        seqs = [ids[:npp.MAX_LEN] for ids in tok([transform(b, ending, kind) for b in base],
                                                  add_special_tokens=False)["input_ids"]]
        cls, z, lg = forward(model, seqs, device)
        np.save(f"{OUT}/{kind}_layers.npy", cls[:, LAYER_IDX].astype(np.float32))
        np.save(f"{OUT}/{kind}_logits.npy", lg)
        acc = np.mean(lg.argmax(1)[len(sp["calA"]):] == y_val[sp["test"]])
        print(f"[{LADDER} s{NLP_SEED}] {kind:18s} {time.time() - t0:5.0f}s test acc {acc:.4f}", flush=True)


def forward(model, seqs, device):
    import torch
    import nlp_prepare as npp
    n, L, d = len(seqs), model.config.num_hidden_layers, model.config.hidden_size
    cls, z, logits = np.zeros((n, L, d), np.float32), np.zeros((n, d), np.float32), np.zeros((n, 2), np.float32)
    with torch.no_grad():
        for b in npp.batches(seqs, max_tokens=4096):
            x, m = npp.collate([seqs[j] for j in b])
            last = m.sum(1) - 1
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(input_ids=x.to(device), attention_mask=m.to(device), output_hidden_states=True)
            hs = torch.stack([h[torch.arange(len(b)), last.to(device)] for h in out.hidden_states[1:]], 1).float()
            cls[b] = hs.cpu().numpy()
            logits[b] = out.logits.float().cpu().numpy()
    return cls, z, logits


# ----------------------------------------------------------------- evaluation
class TiedGauss:
    """Class-conditional Gaussian with a shared covariance; score = min over classes of Mahalanobis^2."""

    def __init__(self, x, y):
        self.mu = np.stack([x[y == c].mean(0) for c in np.unique(y)]).astype(np.float64)
        r = x - self.mu[np.searchsorted(np.unique(y), y)]
        cov = (r.T.astype(np.float64) @ r) / len(x)
        cov += 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov))
        self.L = np.linalg.cholesky(np.linalg.inv(cov))

    def __call__(self, x):
        out = np.full(len(x), np.inf)
        for m in self.mu:
            out = np.minimum(out, (((x - m) @ self.L) ** 2).sum(1))
        return out


def pvalue(s, cal):
    cal = np.sort(cal)
    return (1 + len(cal) - np.searchsorted(cal, s, side="left")) / (len(cal) + 1)


def scores(feats, train, y_train, A1, proj=None):
    """feats: dict name -> (n, n_layers, d). Returns dict score name -> dict name -> (n,)."""
    d2 = {}
    for j in range(len(LAYER_IDX)):
        P = proj[j] if proj is not None else None
        f = (lambda a: a - (a @ P) @ P.T) if P is not None else (lambda a: a)
        g = TiedGauss(f(train[:, j]), y_train)
        d2[j] = {k: g(f(v[:, j].astype(np.float64))) for k, v in feats.items()}
    med = [np.median(d2[j]["orig"][A1]) for j in d2]
    u = {k: np.stack([np.log(np.maximum(d2[j][k], 1e-12) / med[j]) for j in d2], 1) for k in feats}
    return {"maha_z": {k: d2[len(LAYER_IDX) - 1][k] for k in feats}, "maha_multi": {k: u[k].mean(1) for k in feats}}


def run_eval():
    sp = val_split()
    nA = len(sp["calA"])
    perm = np.random.default_rng(0).permutation(nA)   # same A1/A2 as conformal_ood.py
    A1, A2 = np.sort(perm[: nA // 2]), np.sort(perm[nA // 2:])
    test = np.arange(nA, nA + len(sp["test"]))
    y_all = np.load(f"{FEAT}/val_labels.npy")[np.load(f"{OUT}/idx.npy")]
    feats = {k: np.load(f"{OUT}/{k}_layers.npy") for k in TRANSFORMS}
    logits = {k: np.load(f"{OUT}/{k}_logits.npy").astype(np.float64) for k in TRANSFORMS}
    for o in OOD_SETS:
        feats[o] = np.load(f"{FEAT}/{o}_cls.npy", mmap_mode="r")[:, LAYER_IDX]
        logits[o] = np.load(f"{FEAT}/{o}_logits.npy")[:, 0].astype(np.float64)
    # sanity: re-extracted unperturbed features match the cached ones
    cached = np.load(f"{FEAT}/val_cls.npy", mmap_mode="r")[np.load(f"{OUT}/idx.npy")][:, LAYER_IDX[-1]]
    rel = np.linalg.norm(cached - feats["orig"][:, -1]) / np.linalg.norm(cached)
    print(f"re-extraction vs cache, z: relative difference {rel:.2e}", flush=True)
    train = np.load(f"{FEAT}/train_cls.npy", mmap_mode="r")
    y_train = np.load(f"{FEAT}/train_labels.npy").astype(np.int64)
    if len(train) > 30_000:   # SST-2: 67k phrases; a fixed random subset keeps the covariance fits fast
        sub = np.sort(np.random.default_rng(1).choice(len(train), 30_000, replace=False))
        train, y_train = train[sub], y_train[sub]
    train = np.asarray(train[:, LAYER_IDX], np.float64)

    def lse(l):
        return l.max(1) + np.log(np.exp(l - l.max(1, keepdims=True)).sum(1))

    def report(sc, tag):
        rows = {}
        for name, s in sc.items():
            cal = s["orig"][A2]
            r = {"far": {}, "tpr": {}}
            for k in TRANSFORMS:
                r["far"][k] = 100 * float((pvalue(s[k][test], cal) <= ALPHA).mean())
            for o in OOD_SETS:
                r["tpr"][o] = 100 * float((pvalue(s[o], cal) <= ALPHA).mean())
            rows[name] = r
            print(f"  [{tag}] {name:11s} FAR " + " ".join(f"{k}={v:.1f}" for k, v in r["far"].items()) +
                  " | TPR " + " ".join(f"{k}={v:.1f}" for k, v in r["tpr"].items()), flush=True)
        return rows

    out = {"acc": {k: 100 * float((logits[k][test].argmax(1) == y_all[test]).mean()) for k in TRANSFORMS},
           "flip_vs_orig": {k: 100 * float((logits[k][test].argmax(1) != logits["orig"][test].argmax(1)).mean())
                            for k in TRANSFORMS},
           "rel_shift_z": {k: float(np.median(np.linalg.norm(feats[k][test, -1] - feats["orig"][test, -1], axis=1) /
                                              np.linalg.norm(feats["orig"][test, -1], axis=1))) for k in TRANSFORMS}}
    print("acc", {k: round(v, 2) for k, v in out["acc"].items()})
    print("flip", {k: round(v, 2) for k, v in out["flip_vs_orig"].items()})
    base_sc = scores(feats, train, y_train, A1)
    base_sc["msp"] = {k: -np.abs(v[:, 1] - v[:, 0]) for k, v in logits.items()}
    base_sc["energy"] = {k: -lse(v) for k, v in logits.items()}
    out["baseline"] = report(base_sc, "baseline")

    # leave-one-transform-out nuisance projection, learned on A1 only
    nuis = [k for k in TRANSFORMS if k != "orig"]
    out["projection"] = {}
    for kdim in (4, 16):
        res = {}
        for held in nuis:
            proj = []
            for j in range(len(LAYER_IDX)):
                D = np.concatenate([feats[k][A1, j] - feats["orig"][A1, j] for k in nuis if k != held]).astype(np.float64)
                U = np.linalg.svd(D, full_matrices=False)[2][:kdim].T
                proj.append(U)
            sc = scores(feats, train, y_train, A1, proj)
            rows = report(sc, f"k={kdim} held={held}")
            res[held] = {n: {"far_held": r["far"][held], "far_orig": r["far"]["orig"], "tpr": r["tpr"]}
                         for n, r in rows.items()}
        out["projection"][str(kdim)] = res
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1)
    print("saved", f"{OUT}/results.json")


if __name__ == "__main__":
    {"extract": extract, "eval": run_eval}[sys.argv[1]]()
