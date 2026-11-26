"""Pilot 2: remedies for format-fragile conformal OOD flags on LLM hidden states.

Pipelines (the classifier and the detector see the same text; the Gaussian is fitted on that pipeline's
training features, conformal calibration uses the unperturbed A2 half of calA through the same pipeline):
  raw     the text as given. Detector readouts: last token (as now), last *content* token (last token whose
          string has a letter or digit), mean over tokens.
  canon   c(x) = core(x) + training ending, with core(x) = collapse whitespace, then strip trailing
          spaces, newlines and periods. Every input whose difference from x lies in the "ending group"
          (trailing space / newline / period, internal whitespace runs) has the same canonical form, so its
          score has exactly the distribution of the score of x: split-conformal validity carries over.
  orbit   s_orbit(x) = min over e in ENDINGS of s(core(x) + e), Gaussian of the raw last-token pipeline.
          The orbit {core(x) + e} is the same for every member of the ending group: exact invariance.

Transforms tested on ID (calA + test): in the ending group: space_toggle, double_space, newline,
period_toggle; outside it: capitalize_toggle, upper, prefix, quotes.

Usage: LADDER=<sst2|cola> NLP_SEED=<s> $PY pilot2.py extract | eval
"""
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "imagenet_ladder"))
sys.path.insert(0, os.path.join(HERE, "..", "nlp_ladder"))
sys.path.insert(0, HERE)
from inl_common import CKPT, FEAT, LADDER, NLP_SEED, OOD_SETS, REGEN_LAYERS, val_split  # noqa: E402
from pilot import ALPHA, TRANSFORMS, TiedGauss, pvalue, transform  # noqa: E402

OUT = f"/ephemeral/trinhpn_uq/nuisance_pilot2/{LADDER}/seed{NLP_SEED}"
LAYER_IDX = [l if l != "z" else 23 for l in REGEN_LAYERS]
ENDINGS = [p + w for p in ["", ".", " ."] for w in ["", " ", "\n"]]
IN_GROUP = ["space_toggle", "double_space", "newline", "period_toggle"]
OUT_GROUP = ["capitalize_toggle", "upper", "prefix", "quotes"]
TRAIN_END = " " if LADDER == "sst2" else ""
N_TRAIN = 30_000
R_SPLITS = 200


def core(text):
    return re.sub(r"\s+", " ", text).strip().rstrip(" .\n")


def canon(text):
    return core(text) + TRAIN_END


# ----------------------------------------------------------------- extraction
def extract():
    import torch
    from transformers import AutoTokenizer
    import nlp_prepare as npp
    tok = AutoTokenizer.from_pretrained(npp.MODEL)
    content = np.array([any(ch.isalnum() for ch in tok.decode([i])) for i in range(len(tok))])
    data = npp.load_data()["sets"]
    sp = val_split()
    idx = np.r_[sp["calA"], sp["test"]]
    ids_val = data["val"][0]
    base = [tok.decode(ids_val[i]).rstrip() for i in idx]
    ids_tr, y_tr = data["train"]
    sub = (np.sort(np.random.default_rng(1).choice(len(ids_tr), N_TRAIN, replace=False))
           if len(ids_tr) > N_TRAIN else np.arange(len(ids_tr)))
    while True:
        try:
            device = npp.pick_gpu(min_free_mb=3500)
            break
        except RuntimeError as e:
            print("waiting for a GPU:", e, flush=True)
            time.sleep(120)
    model = npp.load_model(device, f"{CKPT}/model").eval()
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/idx.npy", idx)
    np.save(f"{OUT}/train_labels.npy", y_tr[sub])
    enc = lambda texts: [i[:npp.MAX_LEN] for i in tok(texts, add_special_tokens=False)["input_ids"]]  # noqa: E731
    dec = lambda seqs: [tok.decode(s) for s in seqs]  # noqa: E731

    def run(name, seqs, readouts):
        t0 = time.time()
        r = forward(model, seqs, device, content, readouts)
        for k, v in r.items():
            np.save(f"{OUT}/{name}_{k}.npy", v)
        print(f"[{LADDER} s{NLP_SEED}] {name:28s} n={len(seqs):6d} {time.time() - t0:5.0f}s", flush=True)

    # raw pipeline: three readouts
    run("raw_train", [ids_tr[i] for i in sub], ("last", "lastc", "mean"))
    for o in OOD_SETS:
        run(f"raw_{o}", data[o][0], ("last", "lastc", "mean"))
    val_text = {k: [transform(b, TRAIN_END, k) for b in base] for k in TRANSFORMS}
    for k in TRANSFORMS:
        run(f"raw_{k}", enc(val_text[k]), ("last", "lastc", "mean"))
    # canonical pipeline
    run("canon_train", enc([canon(t) for t in dec([ids_tr[i] for i in sub])]), ("last",))
    ood_text = {o: dec(data[o][0]) for o in OOD_SETS}
    for o in OOD_SETS:
        run(f"canon_{o}", enc([canon(t) for t in ood_text[o]]), ("last",))
    for k in TRANSFORMS:
        run(f"canon_{k}", enc([canon(t) for t in val_text[k]]), ("last",))
    # orbit: every ending on the core
    for j, e in enumerate(ENDINGS):
        for o in OOD_SETS:
            run(f"orbit{j}_{o}", enc([core(t) + e for t in ood_text[o]]), ("last",))
        for k in TRANSFORMS:
            run(f"orbit{j}_{k}", enc([core(t) + e for t in val_text[k]]), ("last",))
    open(f"{OUT}/EXTRACT_DONE", "w").close()


def forward(model, seqs, device, content, readouts):
    import torch
    import nlp_prepare as npp
    n, d, nl = len(seqs), model.config.hidden_size, len(LAYER_IDX)
    out = {r: np.zeros((n, nl, d), np.float16) for r in readouts}
    out["logits"] = np.zeros((n, 2), np.float32)
    lidx = torch.tensor(LAYER_IDX)
    with torch.no_grad():
        for b in npp.batches(seqs, max_tokens=4096):
            x, m = npp.collate([seqs[j] for j in b])
            last = m.sum(1) - 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(input_ids=x.to(device), attention_mask=m.to(device), output_hidden_states=True)
            hs = torch.stack(o.hidden_states[1:], 1)[:, lidx].float()        # (B, nl, T, d)
            ar = torch.arange(len(b), device=device)
            if "last" in readouts:
                out["last"][b] = hs[ar, :, last.to(device)].cpu().numpy()
            if "lastc" in readouts:
                c = torch.from_numpy(content[x.numpy()]) & m.bool()
                pos = torch.where(c.any(1), (c * torch.arange(x.shape[1])).argmax(1), last)
                out["lastc"][b] = hs[ar, :, pos.to(device)].cpu().numpy()
            if "mean" in readouts:
                mm = m.to(device)[:, None, :, None].float()
                out["mean"][b] = ((hs * mm).sum(2) / mm.sum(2)).cpu().numpy()
            out["logits"][b] = o.logits.float().cpu().numpy()
    return out


# ----------------------------------------------------------------- evaluation
def run_eval():
    sp = val_split()
    nA = len(sp["calA"])
    perm = np.random.default_rng(0).permutation(nA)
    A1, A2 = np.sort(perm[: nA // 2]), np.sort(perm[nA // 2:])
    test = np.arange(nA, nA + len(sp["test"]))
    y = np.load(f"{FEAT}/val_labels.npy")[np.load(f"{OUT}/idx.npy")]
    y_tr = np.load(f"{OUT}/train_labels.npy").astype(np.int64)
    L = lambda name: np.load(f"{OUT}/{name}.npy")  # noqa: E731
    sets = TRANSFORMS + list(OOD_SETS)

    def lse(l):
        return l.max(1) + np.log(np.exp(l - l.max(1, keepdims=True)).sum(1))

    def gauss(train, feats):
        """feats: dict set -> (n, nl, d) float16. Returns per-layer d2 dict."""
        d2 = []
        for j in range(len(LAYER_IDX)):
            g = TiedGauss(train[:, j].astype(np.float64), y_tr)
            d2.append({k: g(v[:, j].astype(np.float64)) for k, v in feats.items()})
        return d2

    def combine(d2, med):
        return {"maha_z": {k: d2[-1][k] for k in d2[-1]},
                "maha_multi": {k: np.mean([np.log(np.maximum(d2[j][k], 1e-12) / med[j]) for j in range(len(d2))], 0)
                               for k in d2[-1]}}

    def logit_scores(lg):
        return {"msp": {k: -np.abs(v[:, 1] - v[:, 0]) for k, v in lg.items()},
                "energy": {k: -lse(v.astype(np.float64)) for k, v in lg.items()}}

    # Validity over calibration draws: the pipelines fit nothing on calA except per-layer medians (a constant
    # shift of maha_multi), so calA and test can be pooled and re-split R times into |A2| calibration points
    # (unperturbed) and the rest (transformed).
    pool = np.r_[np.arange(nA), test]
    rng = np.random.default_rng(123)
    splits = [rng.permutation(pool) for _ in range(R_SPLITS)]

    def metrics(sc, lg):
        res = {}
        for name, s in sc.items():
            cal = s["orig"][A2]
            far_rs = {k: [] for k in TRANSFORMS}
            tpr_rs = {o: [] for o in OOD_SETS}
            for p in splits:
                c, t = s["orig"][p[:len(A2)]], p[len(A2):]
                for k in TRANSFORMS:
                    far_rs[k].append((pvalue(s[k][t], c) <= ALPHA).mean())
                for o in OOD_SETS:
                    tpr_rs[o].append((pvalue(s[o], c) <= ALPHA).mean())
            res[name] = {"far": {k: 100 * float((pvalue(s[k][test], cal) <= ALPHA).mean()) for k in TRANSFORMS},
                         "tpr": {o: 100 * float((pvalue(s[o], cal) <= ALPHA).mean()) for o in OOD_SETS},
                         "far_rs": {k: 100 * float(np.mean(v)) for k, v in far_rs.items()},
                         "far_rs_q90": {k: 100 * float(np.quantile(v, 0.9)) for k, v in far_rs.items()},
                         "tpr_rs": {o: 100 * float(np.mean(v)) for o, v in tpr_rs.items()},
                         "auroc": {o: auroc(s["orig"][test], s[o]) for o in OOD_SETS}}
        acc = {k: 100 * float((lg[k][test].argmax(1) == y[test]).mean()) for k in TRANSFORMS}
        return {"scores": res, "acc": acc}

    out = {}
    # raw pipeline, three readouts
    lg = {k: L(f"raw_{k}_logits") for k in sets}
    for r in ("last", "lastc", "mean"):
        d2 = gauss(L(f"raw_train_{r}"), {k: L(f"raw_{k}_{r}") for k in sets})
        med = [np.median(d[ "orig"][A1]) for d in d2]
        sc = combine(d2, med)
        if r == "last":
            sc.update(logit_scores(lg))
            raw_last_d2, raw_med = d2, med
        out[f"raw_{r}"] = metrics(sc, lg)
        print_row(f"raw_{r}", out[f"raw_{r}"])
    # canonical pipeline
    lg = {k: L(f"canon_{k}_logits") for k in sets}
    d2 = gauss(L("canon_train_last"), {k: L(f"canon_{k}_last") for k in sets})
    sc = combine(d2, [np.median(d["orig"][A1]) for d in d2])
    sc.update(logit_scores(lg))
    out["canon_last"] = metrics(sc, lg)
    print_row("canon_last", out["canon_last"])
    # orbit-min, raw last-token Gaussian
    train = L("raw_train_last")
    gs = [TiedGauss(train[:, j].astype(np.float64), y_tr) for j in range(len(LAYER_IDX))]
    per = {"maha_z": [], "maha_multi": [], "msp": [], "energy": []}
    for j in range(len(ENDINGS)):
        feats = {k: L(f"orbit{j}_{k}_last") for k in sets}
        lgj = {k: L(f"orbit{j}_{k}_logits") for k in sets}
        d2 = [{k: gs[l](v[:, l].astype(np.float64)) for k, v in feats.items()} for l in range(len(LAYER_IDX))]
        sc = combine(d2, raw_med)
        sc.update(logit_scores(lgj))
        for n in per:
            per[n].append(sc[n])
    sc = {n: {k: np.min([v[k] for v in per[n]], 0) for k in sets} for n in per}
    out["orbit_last"] = metrics(sc, {k: L(f"raw_{k}_logits") for k in sets})
    print_row("orbit_last", out["orbit_last"])
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1)
    print("saved", f"{OUT}/results.json", flush=True)


def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    return 100 * float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def print_row(tag, r):
    print(f"== {tag}: acc " + " ".join(f"{k}={v:.1f}" for k, v in r["acc"].items()), flush=True)
    for n, s in r["scores"].items():
        print(f"  {n:11s}" +
              " | FARrs " + " ".join(f"{v:5.1f}" for v in s["far_rs"].values()) +
              " | TPRrs " + " ".join(f"{v:5.1f}" for v in s["tpr_rs"].values()) +
              " | AUROC " + " ".join(f"{v:5.1f}" for v in s["auroc"].values()), flush=True)


if __name__ == "__main__":
    {"extract": extract, "eval": run_eval}[sys.argv[1]]()
