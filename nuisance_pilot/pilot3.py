"""Pilot 3: ID-first calibration. Worst-case (max over a nuisance library) calibration scores.

Test inputs are scored once, as given (raw pipeline). Each calibration point x gets
    s_N(x) = max_{g in N} s(g . x),  N = {orig} + a library of format transforms.
For every g in N, s(g . x) <= s_N(x), so P(flag(g . x)) <= alpha in finite samples (marginal over the
calibration draw) without making the score invariant. Leave-one-transform-out: the held-out transform
is not in N; its false-alarm rate measures generalisation to an unseen nuisance.

Re-uses the pilot2 features (raw pipeline, readouts last / lastc). 200 calibration re-draws as in pilot2.
Usage: LADDER=<task> NLP_SEED=<s> $PY pilot3.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot2 import ALPHA, LAYER_IDX, OOD_SETS, OUT, R_SPLITS, TRANSFORMS, TiedGauss, pvalue, val_split  # noqa: E402

NUIS = [k for k in TRANSFORMS if k != "orig"]
FLIP_MAX = 5.0


def main():
    sp = val_split()
    nA, nT = len(sp["calA"]), len(sp["test"])
    n_cal = nA - nA // 2                                   # |A2|, as in pilot2
    pool = np.arange(nA + nT)
    rng = np.random.default_rng(123)
    splits = [rng.permutation(pool) for _ in range(R_SPLITS)]
    y_tr = np.load(f"{OUT}/train_labels.npy").astype(np.int64)
    sets = TRANSFORMS + list(OOD_SETS)
    # "Harmless" transforms: prediction flip rate <= FLIP_MAX on calA (never on test)
    lg = {k: np.load(f"{OUT}/raw_{k}_logits.npy").argmax(1) for k in TRANSFORMS}
    flips = {k: 100 * float((lg[k][:nA] != lg["orig"][:nA]).mean()) for k in NUIS}
    harmless = [k for k in NUIS if flips[k] <= FLIP_MAX]
    print("calA flip %:", {k: round(v, 1) for k, v in flips.items()}, "-> harmless:", harmless, flush=True)
    out = {"flips_calA": flips, "harmless": harmless}
    for r in ("last", "lastc"):
        train = np.load(f"{OUT}/raw_train_{r}.npy")
        feats = {k: np.load(f"{OUT}/raw_{k}_{r}.npy") for k in sets}
        d2 = []
        for j in range(len(LAYER_IDX)):
            g = TiedGauss(train[:, j].astype(np.float64), y_tr)
            d2.append({k: g(v[:, j].astype(np.float64)) for k, v in feats.items()})
        sc = {"maha_z": {k: d2[-1][k] for k in sets},
              "maha_multi": {k: np.mean([np.log(np.maximum(d2[j][k], 1e-12)) for j in range(len(d2))], 0) for k in sets}}
        for name, s in sc.items():
            res = {}
            libs = {"none": ["orig"], "all": ["orig", *NUIS], "harmless": ["orig", *harmless],
                    **{f"lofo_{h}": ["orig", *[k for k in NUIS if k != h]] for h in NUIS}}
            for lib, members in libs.items():
                for mode in ("inputmax", "qmax"):
                    if mode == "inputmax" and lib == "none":
                        continue
                    cal_score = np.max([s[k] for k in members], 0)   # worst case over the library, per input
                    far = {k: [] for k in TRANSFORMS}
                    tpr = {o: [] for o in OOD_SETS}
                    for p in splits:
                        ci, t = p[:n_cal], p[n_cal:]
                        if mode == "inputmax":
                            pv = lambda x: pvalue(x, cal_score[ci])  # noqa: E731
                        else:   # threshold = max over g of the conformal quantile of s(g . x): valid for each fixed g
                            pv = lambda x: np.max([pvalue(x, s[g][ci]) for g in members], 0)  # noqa: E731
                        for k in TRANSFORMS:
                            far[k].append((pv(s[k][t]) <= ALPHA).mean())
                        for o in OOD_SETS:
                            tpr[o].append((pv(s[o]) <= ALPHA).mean())
                    res[f"{lib}_{mode}"] = {"far": {k: 100 * float(np.mean(v)) for k, v in far.items()},
                                            "tpr": {o: 100 * float(np.mean(v)) for o, v in tpr.items()}}
            out[f"{r}_{name}"] = res
            for mode in ("inputmax", "qmax"):
                lo = {h: res[f"lofo_{h}_{mode}"]["far"][h] for h in NUIS}
                a = res[f"all_{mode}"]
                print(f"[{r} {name} {mode}] all: FAR " + " ".join(f"{k[:6]}={v:.1f}" for k, v in a["far"].items())
                      + " | TPR " + " ".join(f"{v:.1f}" for v in a["tpr"].values())
                      + "\n      held-out FAR " + " ".join(f"{k[:6]}={v:.1f}" for k, v in lo.items()), flush=True)
            h = res["harmless_qmax"]
            print(f"[{r} {name} harmless qmax] FAR " + " ".join(f"{k[:6]}={v:.1f}" for k, v in h["far"].items())
                  + " | TPR " + " ".join(f"{v:.1f}" for v in h["tpr"].values()), flush=True)
            print(f"[{r} {name}] none: TPR " + " ".join(f"{v:.1f}" for v in res["none_qmax"]["tpr"].values()), flush=True)
    json.dump(out, open(f"{OUT}/results_idfirst.json", "w"), indent=1)


if __name__ == "__main__":
    main()
