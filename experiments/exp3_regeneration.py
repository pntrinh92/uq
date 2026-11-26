"""Experiment 3: the proposed regeneration scaffold on the pilot.

At each layer l (blocks 1-4, then z) the state either stays (prob 1 - a_l(x)) or
is regenerated from the training features at that layer, drawn with weight
proportional to a_l. This kernel leaves the empirical marginal p_l unchanged:

    int p(x) K(x, A) dx = p(A) - int_A a p + E_p[a] * int_A a p / E_p[a] = p(A).

No score model is needed. Because a regenerated state forgets the input, the
predictive has a closed form (first-jump decomposition, Markov property):

    q(y|x) = prod_l (1 - a_l) p_FM(y|x) + sum_l prod_{k<l} (1 - a_k) a_l m_l(y),

with m_l(y) the population predictive after regenerating at layer l, computed
once on training features. The jump probability is distance-aware and
conformally thresholded:

    a_l(x) = a0 + (1 - a0) * (1 - exp(-beta * (d_l(x)^2 / q_l - 1)_+)),

where d_l is the min class Mahalanobis distance (tied GMM on training features)
and q_l is the split-conformal (1 - alpha) quantile of d_l^2 on half of the
validation set. The floor a0 > 0 keeps the regeneration target a p / E_p[a]
well defined (training features rarely exceed q_l). Flip guarantee: below all
thresholds the total jump mass is J <= 1 - (1 - a0)^L, and argmax q = argmax p_FM
whenever the FM top-2 margin exceeds J / (1 - J); so
    P_ID(flip) <= P(any d_l^2 > q_l) + P(margin <= J / (1 - J)),
and the first term is at most L * alpha by exchangeability (split conformal).
beta, alpha, a0 and a temperature are picked on the other half (validation NLL).
"""
import itertools
import math

import numpy as np
import torch

from common import (TiedGMM, auroc, classification_metrics, entropy, fit_temperature, get_models,
                    save_json)

N_SPLITS = 5
ALPHAS = [0.01, 0.05, 0.1]
BETAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
A0S = [1e-3, 1e-2, 3e-2]
LAYER_SETS = {"all": ["block1", "block2", "block3", "block4", "z"], "z only": ["z"]}


@torch.no_grad()
def layer_feats(model, X):
    logits, z, states = model(X, return_all=True)
    feats = {f"block{i + 1}": s.flatten(1) for i, s in enumerate(states)}
    feats["z"] = z
    return feats, torch.softmax(logits, 1).double()


def conformal_quantile(scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


def jump_probs(d2, q, beta, a0):
    return a0 + (1 - a0) * (1 - np.exp(-beta * np.clip(d2 / q - 1, 0, None)))


def population_predictives(a_tr, p_tr, layers):
    """m_l for every layer, by backward recursion over training points.

    a_tr[l]: (n,) jump probabilities of training points; p_tr: (n, C) FM predictive.
    Layers whose training jump mass is zero cannot regenerate exactly; their a is forced to 0.
    """
    m, usable = {}, {}
    for li in reversed(range(len(layers))):
        l = layers[li]
        surv = np.ones(len(p_tr))
        # q_{>l}(y | j): continue from training point j after layer l (later jumps folded into m_k).
        mix = np.zeros_like(p_tr)
        for k in layers[li + 1:]:
            if usable[k]:
                mix += (surv * a_tr[k])[:, None] * m[k][None]
                surv = surv * (1 - a_tr[k])
        tail = surv[:, None] * p_tr + mix
        w = a_tr[l]
        usable[l] = w.sum() > 0
        m[l] = (w[:, None] * tail).sum(0) / w.sum() if usable[l] else np.full(p_tr.shape[1], np.nan)
    return m, usable


def predictive(a, p_fm, m, usable, layers):
    surv = np.ones(len(p_fm))
    q = np.zeros_like(p_fm)
    for l in layers:
        if not usable[l]:
            continue
        q += (surv * a[l])[:, None] * m[l][None]
        surv = surv * (1 - a[l])
    return q + surv[:, None] * p_fm, 1 - surv


def run_split(seed):
    data, models = get_models(seed)
    fm = models[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data["Xva"]))
    cal_a, cal_b = perm[: len(perm) // 2], perm[len(perm) // 2:]

    feats, probs = {}, {}
    sets = {"train": data["Xtr"], "val": data["Xva"], "test": data["Xte"], "shift": data["Xsh"], **data["ood"]}
    for k, X in sets.items():
        feats[k], probs[k] = layer_feats(fm, X)
    all_layers = LAYER_SETS["all"]
    gmms = {l: TiedGMM(feats["train"][l], data["ytr"]) for l in all_layers}
    d2 = {k: {l: gmms[l].min_mahalanobis(feats[k][l]).numpy() for l in all_layers} for k in sets}
    p_np = {k: v.numpy() for k, v in probs.items()}
    yb = data["yva"][cal_b]

    out = {"grid": []}
    # Baseline: multi-layer Mahalanobis OOD score (sum of d_l^2 normalised by its calibration median).
    med = {l: float(np.median(d2["val"][l][cal_a])) for l in all_layers}
    ml = {k: sum(d2[k][l] / med[l] for l in all_layers) for k in sets}
    out["multilayer_mahalanobis_auroc"] = {o: auroc(ml["test"], ml[o]) for o in data["ood"]}
    for lname, layers in LAYER_SETS.items():
        best = None
        for alpha, beta, a0 in itertools.product(ALPHAS, BETAS, A0S):
            q = {l: conformal_quantile(d2["val"][l][cal_a], alpha) for l in layers}
            a = {k: {l: jump_probs(d2[k][l], q[l], beta, a0) for l in layers} for k in sets}
            m, usable = population_predictives(a["train"], p_np["train"], layers)
            pred = {k: predictive(a[k], p_np[k], m, usable, layers) for k in sets}
            qv = torch.tensor(pred["val"][0][cal_b])
            r = dict(layers=lname, alpha=alpha, beta=beta, a0=a0, usable=[l for l in layers if usable[l]],
                     calb_nll=classification_metrics(qv, yb)["nll"],
                     calb_brier=classification_metrics(qv, yb)["brier"])
            r["test"] = classification_metrics(torch.tensor(pred["test"][0]), data["yte"])
            r["shift"] = classification_metrics(torch.tensor(pred["shift"][0]), data["yte"])
            ent = {k: entropy(torch.tensor(pred[k][0])).numpy() for k in sets}
            jump = {k: pred[k][1] for k in sets}
            r["ood_entropy"] = {o: auroc(ent["test"], ent[o]) for o in data["ood"]}
            r["ood_jump"] = {o: auroc(jump["test"], jump[o]) for o in data["ood"]}
            # Guarantee check on ID test: threshold exceedances and small-margin inputs.
            n_use = sum(usable[l] for l in layers)
            exceed = np.zeros(len(data["Xte"]), dtype=bool)
            for l in layers:
                if usable[l]:
                    exceed |= d2["test"][l] > q[l]
            J = 1 - (1 - a0) ** n_use
            top2 = np.sort(p_np["test"], 1)[:, -2:]
            small_margin = (top2[:, 1] - top2[:, 0]) <= J / (1 - J)
            r["id_exceed"] = float(exceed.mean())
            r["bound_L_alpha"] = alpha * n_use
            r["id_small_margin"] = float((small_margin & ~exceed).mean())
            r["flip_bound_empirical"] = float((exceed | small_margin).mean())
            r["flip"] = float((pred["test"][0].argmax(1) != p_np["test"].argmax(1)).mean())
            r["mean_jump"] = {k: float(jump[k].mean()) for k in ("test", "shift", *data["ood"])}
            out["grid"].append(r)
            if best is None or r["calb_nll"] < best[0]["calb_nll"]:
                best = (r, pred)
        r, pred = best
        # Temperature on cal_b, applied to log q.
        T = fit_temperature(torch.tensor(np.log(np.clip(pred["val"][0][cal_b], 1e-12, None))), yb)
        ts = lambda p: torch.softmax(torch.tensor(np.log(np.clip(p, 1e-12, None))) / T, 1)
        sel = dict(r, T=T,
                   test_ts=classification_metrics(ts(pred["test"][0]), data["yte"]),
                   shift_ts=classification_metrics(ts(pred["shift"][0]), data["yte"]))
        # Finite-class Hoeffding certificate on Brier (bounded in [0, 2]) over the |grid| candidates.
        n, G, delta = len(cal_b), len(ALPHAS) * len(BETAS) * len(A0S), 0.05
        sel["brier_certificate"] = sel["calb_brier"] + 2 * math.sqrt(math.log(2 * G / delta) / (2 * n))
        sel["jump_values"] = {k: pred[k][1].tolist() for k in ("test", "shift", *data["ood"])}
        out[f"selected_{lname}"] = sel
        print(f"split{seed} [{lname}] alpha={r['alpha']} beta={r['beta']} acc={r['test']['acc']:.3f} "
              f"nll={r['test']['nll']:.3f} ece={r['test']['ece']:.3f} sh_nll={r['shift']['nll']:.3f} "
              f"sh_ece={r['shift']['ece']:.3f} ood={np.mean(list(r['ood_entropy'].values())):.3f} "
              f"a0={r['a0']} exceed={r['id_exceed']:.3f}(Lalpha={r['bound_L_alpha']:.2f}) flip={r['flip']:.3f}<={r['flip_bound_empirical']:.3f} "
              f"T={T:.2f} cert={sel['brier_certificate']:.3f} test_brier={r['test']['brier']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    runs = [run_split(s) for s in range(N_SPLITS)]
    save_json(runs, "exp3_regeneration.json")
