"""Evaluate every rung of the ladder on cached features.

Backbones: "fm" (ViT-B/16 itself) and "ode" (the compiled flow of rung 2).
On each backbone: deterministic, + temperature scaling, + Brownian same-marginal SDE
at z (rung 3, MALA targeting a 1000-class tied GMM), + regeneration scaffold (rung 4,
CLS after blocks 2, 4, 6, 8, 10 and z). Baselines on the FM: dropout injection,
3-checkpoint ensemble, multi-layer Mahalanobis OOD score.

Selection uses calB only (validation NLL); conformal thresholds use calA only;
all reported numbers are on the 40k test split, the shift set and the OOD sets.

Usage: python ladder_eval.py fm ode
"""
import itertools
import os
import sys
import time

import numpy as np
import timm
import torch

from inl_common import (CORRUPTIONS, FEAT, HAS_REAL, NUM_CLASSES, OOD_SETS, REGEN_LAYERS, TiedGMM, auroc,
                        classification_metrics, entropy, fit_temperature, get_npy, head_weights, save_json, val_split)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from common import langevin, make_preconditioner  # noqa: E402
from exp3_regeneration import conformal_quantile, jump_probs, population_predictives, predictive  # noqa: E402

torch.set_num_threads(int(os.environ.get("NUM_THREADS", "32")))
OOD = OOD_SETS
EVAL_SETS = ["val", "shift", *OOD]
ALPHAS, BETAS, A0S = [0.01, 0.05], [0.1, 0.3, 1.0, 3.0], [1e-3, 1e-2]
H_GRID, K_STEPS, CHAINS = [0.003, 0.01, 0.03, 0.1, 0.3], 5, 8


def softmax_np(lg):
    lg = lg.astype(np.float64)
    lg = lg - lg.max(-1, keepdims=True)
    p = np.exp(lg)
    return p / p.sum(-1, keepdims=True)


def mcc(pred, y):
    tp, tn = np.sum((pred == 1) & (y == 1)), np.sum((pred == 0) & (y == 0))
    fp, fn = np.sum((pred == 1) & (y == 0)), np.sum((pred == 0) & (y == 1))
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def load_backbone(B, name):
    pre = "" if B == "fm" else f"{B}_"
    cls = get_npy(f"{pre}{name}_cls")
    z = np.asarray(get_npy(f"{pre}{name}_z"), dtype=np.float32)
    lg = get_npy(f"{pre}{name}_logits")
    lg = np.asarray(lg[:, 0] if lg.ndim == 3 else lg, dtype=np.float32)
    layers = {l: (z if l == "z" else np.asarray(cls[:, l], dtype=np.float32)) for l in REGEN_LAYERS}
    return layers, lg


class Ctx:
    """Labels, splits, ReaL sets and the per-set probability bookkeeping shared by all rows."""

    def __init__(self):
        sp = val_split()
        self.calA, self.calB, self.test = np.sort(sp["calA"]), np.sort(sp["calB"]), np.sort(sp["test"])
        self.y_val = get_npy("val_labels", mmap=False).astype(np.int64)
        self.y_shift = get_npy("shift_labels", mmap=False).astype(np.int64)   # CIFAR-C stores uint8
        self.corr = get_npy("shift_corruption", mmap=False)
        sev = f"{FEAT}/shift_severity.npy"
        self.sev = np.load(sev) if os.path.exists(sev) else None
        # ReaL label sets (ImageNet); on CIFAR each image has its single label as its set.
        self.real = (np.load(f"{FEAT}/val_real.npy", allow_pickle=True) if HAS_REAL
                     else np.array([[int(y)] for y in self.y_val] + [[]], dtype=object)[:-1])

    def metrics(self, probs, ref_probs=None):
        """probs: dict set -> (N, 1000) on the full set (val is indexed here)."""
        pt = probs["val"][self.test]
        yt = self.y_val[self.test]
        out = {"test": classification_metrics(torch.tensor(pt), torch.tensor(yt))}
        pred = pt.argmax(1)
        has = np.array([len(r) > 0 for r in self.real[self.test]])
        out["test"]["real_acc"] = float(np.mean([p in r for p, r in zip(pred[has], self.real[self.test][has])]))
        if NUM_CLASSES == 2:   # CoLA is scored by MCC (DIRECTOR, Table 1)
            out["test"]["mcc"] = mcc(pred, yt)
        if ref_probs is not None:
            ref = ref_probs["val"][self.test].argmax(1)
            flip = pred != ref
            rs = self.real[self.test]
            ok_new = np.array([p in r for p, r in zip(pred, rs)])
            ok_old = np.array([p in r for p, r in zip(ref, rs)])
            out["flip"] = float(flip.mean())
            out["flip_harmful_real"] = float((flip & has & ok_old & ~ok_new).mean())
            out["flip_benign_real"] = float((flip & has & ok_old & ok_new).mean())
        out["shift"] = classification_metrics(torch.tensor(probs["shift"]), torch.tensor(self.y_shift))
        if NUM_CLASSES == 2:
            out["shift"]["mcc"] = mcc(probs["shift"].argmax(1), self.y_shift)
        out["shift_per_corruption"] = {
            c: classification_metrics(torch.tensor(probs["shift"][self.corr == c]), torch.tensor(self.y_shift[self.corr == c]))
            for c in CORRUPTIONS}
        if self.sev is not None:   # CIFAR-C: also by severity (DIRECTOR, Fig. 3)
            out["shift_per_severity"] = {
                int(v): classification_metrics(torch.tensor(probs["shift"][self.sev == v]),
                                               torch.tensor(self.y_shift[self.sev == v])) for v in np.unique(self.sev)}
        ent_test = entropy(torch.tensor(pt)).numpy()
        out["ood_auroc"] = {o: auroc(ent_test, entropy(torch.tensor(probs[o])).numpy()) for o in OOD}
        out["ood_auroc"]["mean"] = float(np.mean([out["ood_auroc"][o] for o in OOD]))
        return out

    def calb_nll(self, p):
        y = self.y_val[self.calB]
        return float(-np.log(np.clip(p[self.calB][np.arange(len(y)), y], 1e-12, None)).mean())

    def temperature(self, probs):
        """Fit T on calB on log-probabilities, apply to every set."""
        lp = {k: np.log(np.clip(v, 1e-12, None)) for k, v in probs.items()}
        T = fit_temperature(torch.tensor(lp["val"][self.calB]), torch.tensor(self.y_val[self.calB]))
        return {k: softmax_np(v / T) for k, v in lp.items()}, T


def gmm_distances(B, layers_by_set, y_train):
    d2 = {}
    for l in REGEN_LAYERS:
        t0 = time.time()
        g = TiedGMM(torch.tensor(layers_by_set["train"][l]), torch.tensor(y_train))
        d2[l] = {k: np.concatenate([g.min_mahalanobis(torch.tensor(v[l][i:i + 20_000])).numpy()
                                    for i in range(0, len(v[l]), 20_000)]) for k, v in layers_by_set.items()}
        print(f"  [{B}] distances layer {l}: {time.time() - t0:.0f}s", flush=True)
    return d2


def regeneration(ctx, d2, p_det, layers, label):
    """Rung 4 over the (alpha, beta, a0) grid, selected on calB NLL."""
    best = None
    for alpha, beta, a0 in itertools.product(ALPHAS, BETAS, A0S):
        q = {l: conformal_quantile(d2[l]["val"][ctx.calA], alpha) for l in layers}
        a = {k: {l: jump_probs(d2[l][k], q[l], beta, a0) for l in layers} for k in ["train", *EVAL_SETS]}
        m, usable = population_predictives(a["train"], p_det["train"], layers)
        pred = {k: predictive(a[k], p_det[k], m, usable, layers) for k in EVAL_SETS}
        nll = ctx.calb_nll(pred["val"][0])
        if best is None or nll < best[0]:
            best = (nll, dict(alpha=alpha, beta=beta, a0=a0), pred, q)
    nll, cfg, pred, q = best
    exceed = np.zeros(len(ctx.test), bool)
    for l in layers:
        exceed |= d2[l]["val"][ctx.test] > q[l]
    probs = {k: v[0] for k, v in pred.items()}
    jump = {k: float(v[1].mean() if k != "val" else v[1][ctx.test].mean()) for k, v in pred.items()}
    print(f"  [{label}] selected {cfg} calB NLL {nll:.4f}", flush=True)
    return probs, dict(config=cfg, calB_nll=nll, id_exceed=float(exceed.mean()), bound_L_alpha=cfg["alpha"] * len(layers),
                       mean_jump=jump)


def brownian(ctx, z_by_set, y_train, z_train, W, b, label):
    """Rung 3: Lie-Trotter, exact FM then K MALA steps at z targeting the GMM; S chains per input."""
    gmm = TiedGMM(torch.tensor(z_train), torch.tensor(y_train))
    L, Lp = make_preconditioner(gmm, "precond")
    gen = torch.Generator().manual_seed(0)

    def run(z, h):
        probs, acc = np.zeros((len(z), NUM_CLASSES)), []
        for i in range(0, len(z), 5000):
            zr = torch.tensor(z[i:i + 5000]).double().repeat_interleave(CHAINS, 0)
            zs, a = langevin(zr, gmm, L, Lp, h, K_STEPS, True, gen)
            acc.append(a)
            lg = (zs.float() @ W.T + b).view(-1, CHAINS, NUM_CLASSES)
            probs[i:i + 5000] = torch.softmax(lg, -1).mean(1).double().numpy()
        return probs, float(np.mean(acc))

    sel = {}
    for h in H_GRID:
        p, a = run(z_by_set["val"][ctx.calB], h)
        y = ctx.y_val[ctx.calB]
        sel[h] = (float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()), a)
        print(f"  [{label}] h={h} calB NLL {sel[h][0]:.4f} accept {a:.2f}", flush=True)
    h = min(sel, key=lambda k: sel[k][0])
    probs, accepts = {}, []
    for k in EVAL_SETS:
        probs[k], a = run(z_by_set[k], h)
        accepts.append(a)
    return probs, dict(h=h, K=K_STEPS, chains=CHAINS, calB=sel, accept=float(np.mean(accepts)))


def evaluate_backbone(B, ctx, W, b):
    print(f"== backbone {B}", flush=True)
    data = {k: load_backbone(B, k) for k in ["train", *EVAL_SETS]}
    layers_by_set = {k: v[0] for k, v in data.items()}
    p_det = {k: softmax_np(v[1]) for k, v in data.items()}
    y_train = get_npy("train_labels", mmap=False)
    rows, info = {}, {}
    rows[f"{B}"] = ctx.metrics({k: p_det[k] for k in EVAL_SETS})
    p_ts, T = ctx.temperature({k: p_det[k] for k in EVAL_SETS})
    rows[f"{B}+TS"] = ctx.metrics(p_ts, p_det)
    info[f"{B}+TS"] = {"T": T}

    d2 = gmm_distances(B, layers_by_set, y_train)
    for name, layers in (("regen", REGEN_LAYERS), ("regen_z", ["z"])):
        probs, inf = regeneration(ctx, d2, p_det, layers, f"{B}+{name}")
        rows[f"{B}+{name}"] = ctx.metrics(probs, p_det)
        info[f"{B}+{name}"] = inf
    med = {l: float(np.median(d2[l]["val"][ctx.calA])) for l in REGEN_LAYERS}
    ml = {k: sum(d2[l][k] / med[l] for l in REGEN_LAYERS) for k in EVAL_SETS}
    info[f"{B}+multilayer_mahalanobis"] = {o: auroc(ml["val"][ctx.test], ml[o]) for o in OOD}

    z_by_set = {k: layers_by_set[k]["z"] for k in EVAL_SETS}
    probs, inf = brownian(ctx, z_by_set, y_train, layers_by_set["train"]["z"], W, b, f"{B}+brownian_z")
    rows[f"{B}+brownian_z"] = ctx.metrics(probs, p_det)
    info[f"{B}+brownian_z"] = inf
    return rows, info, p_det


def baselines(ctx, p_det):
    rows = {}
    if get_npy("val_logits").shape[1] > 1:
        ens = {k: softmax_np(np.asarray(get_npy(f"{k}_logits"), dtype=np.float32)).mean(1) for k in EVAL_SETS}
        rows["ensemble_3ckpt"] = ctx.metrics(ens, p_det)
    if os.path.exists(f"{FEAT}/dropout_val_logits.npy"):
        drop = {k: softmax_np(np.load(f"{FEAT}/dropout_{k}_logits.npy").astype(np.float32)).mean(1) for k in EVAL_SETS}
        rows["dropout_injection"] = ctx.metrics(drop, p_det)
    return rows


if __name__ == "__main__":
    backbones = sys.argv[1:] or ["fm", "ode"]
    ctx = Ctx()
    W, b = head_weights()
    for B in backbones:
        rows, info, p_det = evaluate_backbone(B, ctx, W, b)
        if B == "fm":
            rows.update(baselines(ctx, {k: p_det[k] for k in EVAL_SETS}))
        if B.startswith("ode"):
            fm_pred = softmax_np(np.asarray(get_npy("val_logits")[:, 0], dtype=np.float32))[ctx.test].argmax(1)
            info[f"{B}_agreement_with_fm_test"] = float((p_det["val"][ctx.test].argmax(1) == fm_pred).mean())
        save_json({"rows": rows, "info": info}, f"ladder_{B}.json")
        for k, v in rows.items():
            print(f"{k:24s} acc={v['test']['acc']:.4f} real={v['test']['real_acc']:.4f} nll={v['test']['nll']:.4f} "
                  f"ece={v['test']['ece']:.4f} sh_nll={v['shift']['nll']:.3f} sh_ece={v['shift']['ece']:.3f} "
                  f"ood={v['ood_auroc']['mean']:.3f} flip={v.get('flip', 0):.4f}", flush=True)
