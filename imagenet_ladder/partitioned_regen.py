"""Partitioned regeneration: the refinement suggested by the ImageNet rung-4 result.

The plain scaffold regenerates from a p / E_p[a] and reverts to a population predictive
m with m_y ~ 1/C. Since q_y - p_y = J (m_y - p_y), it raises the true-class probability
only when m_y > p_y, which with C = 1000 classes almost never happens.

Refinement: take any soft partition w_c(x) >= 0 with sum_c w_c = 1, define
pi_c = E_p[w_c] and p_c = w_c p / pi_c, and regenerate from a w_c p / E_p[a w_c] with
probability a(x) w_c(x). The marginal is still exact:
    inflow(x') = sum_c E_p[a w_c] * a(x') w_c(x') p(x') / E_p[a w_c] = a(x') p(x') = outflow(x').
We use w(x) = softmax(log p(.|x) / tau): tau = 1 reverts towards the predicted class,
tau = inf gives back the population version. The predictive stays closed-form:
    q(y|x) = S_L p(y|x) + sum_l S_{l-1} a_l(x) [w_l(x) M_l](y),
with M_l[c] the predictive after regenerating at layer l inside partition c.

Usage: python partitioned_regen.py fm [ode8 ...]
"""
import itertools
import sys
import time

import numpy as np
import torch

from inl_common import auroc, classification_metrics, entropy, get_npy, save_json
from ladder_eval import (ALPHAS, A0S, BETAS, EVAL_SETS, OOD, REGEN_LAYERS, Ctx, conformal_quantile, gmm_distances,
                         jump_probs, load_backbone, softmax_np)

TAUS = [1.0, 3.0, 10.0, float("inf")]


def partition(p, tau):
    if np.isinf(tau):
        return np.full(p.shape, 1.0 / p.shape[1], dtype=np.float32)
    return softmax_np(np.log(np.clip(p, 1e-12, None)) / tau).astype(np.float32)


def fit_M(a_tr, w_tr, p_tr, layers):
    """Backward recursion over layers: M_l (C x C) from the training pool."""
    M = {}
    for li in reversed(range(len(layers))):
        l = layers[li]
        surv = np.ones(len(p_tr), np.float32)
        tail = np.zeros_like(p_tr)
        for k in layers[li + 1:]:
            tail += (surv * a_tr[k])[:, None] * (w_tr @ M[k])
            surv = surv * (1 - a_tr[k])
        tail += surv[:, None] * p_tr
        A = a_tr[l][:, None] * w_tr                      # weight of pool point j in partition c
        den = A.sum(0)
        pop = (a_tr[l][:, None] * tail).sum(0) / a_tr[l].sum()
        M[l] = np.where(den[:, None] > 0, (A.T @ tail) / np.maximum(den, 1e-30)[:, None], pop[None])
    return M


def predict(a, w, p, M, layers):
    surv = np.ones(len(p), np.float32)
    q = np.zeros_like(p)
    for l in layers:
        q += (surv * a[l])[:, None] * (w @ M[l])
        surv = surv * (1 - a[l])
    return q + surv[:, None] * p, 1 - surv


def run(B, ctx):
    data = {k: load_backbone(B, k) for k in ["train", *EVAL_SETS]}
    layers_by_set = {k: v[0] for k, v in data.items()}
    p = {k: softmax_np(v[1]).astype(np.float32) for k, v in data.items()}
    d2 = gmm_distances(B, layers_by_set, get_npy("train_labels", mmap=False))
    layers = REGEN_LAYERS
    grid, best = [], None
    for tau in TAUS:
        w = {k: partition(p[k], tau) for k in p}
        for alpha, beta, a0 in itertools.product(ALPHAS, BETAS, A0S):
            t0 = time.time()
            q = {l: conformal_quantile(d2[l]["val"][ctx.calA], alpha) for l in layers}
            a = {k: {l: jump_probs(d2[l][k], q[l], beta, a0).astype(np.float32) for l in layers} for k in p}
            M = fit_M(a["train"], w["train"], p["train"], layers)
            qv, _ = predict(a["val"], w["val"], p["val"], M, layers)
            nll = ctx.calb_nll(qv)
            cfg = dict(tau=tau, alpha=alpha, beta=beta, a0=a0, calB_nll=nll)
            grid.append(cfg)
            print(f"  [{B}] {cfg} {time.time() - t0:.0f}s", flush=True)
            if best is None or nll < best[0]["calB_nll"]:
                best = (cfg, a, w, M, q)
    cfg, a, w, M, q = best
    pred = {k: predict(a[k], w[k], p[k], M, layers) for k in EVAL_SETS}
    probs = {k: v[0].astype(np.float64) for k, v in pred.items()}
    out = ctx.metrics(probs, {k: p[k] for k in EVAL_SETS})
    jt = pred["val"][1][ctx.test]
    out["ood_auroc_jump"] = {o: auroc(jt, pred[o][1]) for o in OOD}
    exceed = np.zeros(len(ctx.test), bool)
    for l in layers:
        exceed |= d2[l]["val"][ctx.test] > q[l]
    info = dict(selected=cfg, id_exceed=float(exceed.mean()), bound_L_alpha=cfg["alpha"] * len(layers),
                mean_jump={k: float(v[1].mean() if k != "val" else jt.mean()) for k, v in pred.items()},
                best_per_tau={str(t): min((g for g in grid if g["tau"] == t), key=lambda g: g["calB_nll"]) for t in TAUS})
    return out, info


if __name__ == "__main__":
    torch.set_num_threads(32)
    ctx = Ctx()
    for B in sys.argv[1:] or ["fm"]:
        row, info = run(B, ctx)
        save_json({"row": row, "info": info}, f"partitioned_regen_{B}.json")
        print(f"{B}+partitioned_regen acc={row['test']['acc']:.4f} nll={row['test']['nll']:.4f} "
              f"ece={row['test']['ece']:.4f} sh_nll={row['shift']['nll']:.3f} sh_ece={row['shift']['ece']:.3f} "
              f"ood={row['ood_auroc']['mean']:.3f} flip={row['flip']:.4f}", flush=True)
        print(info["best_per_tau"], flush=True)
