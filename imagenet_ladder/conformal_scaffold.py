"""Learned distance, conformal threshold and input-level guarantees for the regeneration scaffold.

Path scaffold. One regeneration decision after the last layer, with a rate that may depend on the
whole feature path (CLS after the scaffold layers, z). Regenerating replaces the path by the path of
a pool (training) point j drawn with weight a(x_j) w_c(x_j) inside soft partition c, so the joint law
of all layer states, hence every layer marginal, is unchanged:
    inflow(x') = sum_c E_p[a w_c] a(x') w_c(x') p(x') / E_p[a w_c] = a(x') p(x') = outflow(x').
Predictive (closed form):  q(.|x) = (1 - a(x)) p(.|x) + a(x) w(x) M,
    M[c] = sum_j a_j w_c(x_j) p(.|x_j) / sum_j a_j w_c(x_j),   w(x) = softmax(log p(.|x) / tau)
(tau = inf: population target; tau = 1: revert towards the predicted class).

Score.  s_theta(x) = sum_l theta_l log(d_l^2(x) / med_l), theta on the simplex, d_l the min-class
Mahalanobis distance at layer l (tied covariance fitted on the pool).
Jump rate on the conformal p-value pi_G(x) of s within the group G(x) (FM confidence quintile):
    a(x) = a0 + (1 - a0) h(pi_G(x)),   h(u) = 1 - (1 - (1 - u/alpha)_+)^kappa.

Held-out protocol (the FM's training data is never used to calibrate): calA = A1 + A2, calB, test.
  1. Learn the distance on A1: theta by non-negative logistic regression of the FM's errors on the
     layer distances ("err"), the uniform weights ("unif"), or theta jointly with (b, beta, a0, tau)
     by the NLL of q ("nll", ablation). theta is chosen by error AUROC on calB.
  2. Calibrate on A2: per-group split-conformal thresholds / p-values of the learned score.
  3. Certify on calB with Learn-then-Test: for every (alpha, kappa, tau) test H: R_flip > eps_flip
     or R_jump > eps_jump (Hoeffding-Bentkus p-values, Bonferroni); every rejected candidate meets both
     bounds with probability >= 1 - delta. Pick (i) the lowest calB NLL, (ii) the most sensitive
     certified candidate (largest calB jump mass: Neyman-Pearson, power at a certified ID cost).
  4. Test, shift and OOD sets: report only.

Input-level guarantees (exchangeable ID inputs; G computable from x):
  (A) P(s(X) > b_hat_G | X in G) <= alpha                               (Mondrian split conformal)
  (B) E[a(X) | X in G] <= a0 + (1 - a0) alpha kappa / (kappa + 1)        (pi_G super-uniform, h decreasing),
      and TV(q(.|x), p(.|x)) <= a(x), so the same bound holds for the expected ID perturbation
  (C) P(flip | X in G) <= alpha + P(margin <= a0 / (1 - a0) | G)        (one combined score: alpha, not L alpha)
  (D) P(Y in C(X) | X in G) >= 1 - alpha                                (Mondrian label sets on q, calibrated
      on calC, a quarter of the test split, since q depends on A1, A2 and calB)
  (E) with prob >= 1 - delta over A2: P(s(X) > b_hat | A2) <= Beta_{1-delta}(n + 1 - l, l), l = ceil((n+1)(1-alpha))

Base predictive (SCAFFOLD_BASE): "fm" scaffolds the FM's softmax; "ts" first applies temperature scaling
fitted on A1, then scaffolds the tempered predictive (TS for calibration, scaffold for atypical inputs).
T depends on A1 only, so A2, calB and calC stay exchangeable with test given everything learned before.

Learn-then-Test family control (LTT_MODE): "bonferroni" tests every candidate at delta / (2 |Lambda|);
"fst" uses fixed-sequence testing (Angelopoulos et al., 2021, Sec. 4) along each tau chain ordered by the
budget bound, at delta / |TAUS| per chain, with p = max(p_flip, p_jump) (valid for the union null).
Fixed-sequence needs no correction inside a chain, which matters when calB is small (NLP: 450-2500).

Usage: python conformal_scaffold.py fm [ode8 ...]   (LADDER selects ImageNet, CIFAR or an NLP task)
"""
import itertools
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist, binom

import director_metrics
from inl_common import OOD_SETS, REGEN_LAYERS, auroc, fit_temperature, get_npy, save_json
from ladder_eval import EVAL_SETS, Ctx, conformal_quantile, gmm_distances, load_backbone, softmax_np

BASE = os.environ.get("SCAFFOLD_BASE", "fm")
LTT_MODE = os.environ.get("LTT_MODE", "bonferroni")
assert BASE in ("fm", "ts") and LTT_MODE in ("bonferroni", "fst")
SUFFIX = ("_ts" if BASE == "ts" else "") + ("_fst" if LTT_MODE == "fst" else "")

N_GROUPS = 5                                  # FM confidence (top-2 log-margin) quintiles, cut points from A1
ALPHA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2]
KAPPAS = [0.5, 1.0, 2.0, 4.0]
TAUS = [1.0, float("inf")]
A0 = 1e-3
EPS_FLIP, EPS_JUMP, DELTA = 0.01, 0.05, 0.1
ALPHA_DEMO = 0.05                             # level of the marginal-vs-Mondrian comparison
POOL_LEARN = 20_000
SET_ALPHAS = [0.05, 0.1]


# ----------------------------------------------------------------- scaffold pieces
def partition(p, tau):
    if not np.isfinite(tau):
        return np.full(p.shape, 1.0 / p.shape[1], np.float32)
    return softmax_np(np.log(np.clip(p, 1e-12, None)) / tau).astype(np.float32)


def hinge_rate(s, b, beta, a0):
    return a0 + (1 - a0) * (1 - np.exp(-beta * np.clip(s - b, 0, None)))


def pvalue_rate(pv, alpha, kappa, a0=A0):
    return a0 + (1 - a0) * (1 - (1 - np.clip(1 - pv / alpha, 0, 1)) ** kappa)


def fit_M(a_pool, w_pool, p_pool):
    A = a_pool[:, None] * w_pool
    den = A.sum(0)
    pop = (a_pool[:, None] * p_pool).sum(0) / a_pool.sum()
    return np.where(den[:, None] > 0, (A.T @ p_pool) / np.maximum(den, 1e-30)[:, None], pop[None]).astype(np.float32)


def predict(a, w, p, M):
    return (1 - a)[:, None] * p + a[:, None] * (w @ M)


# ----------------------------------------------------------------- stage 1: learn the distance on A1
def learn_nll(u_a1, p_a1, y_a1, u_pool, p_pool, steps=300, k_smooth=10.0):
    """Ablation: (theta, beta, a0, tau) and the threshold b jointly, by the A1 NLL of q (smooth hinge)."""
    u1, p1, y1 = (torch.tensor(u_a1, dtype=torch.float64), torch.tensor(p_a1, dtype=torch.float64),
                  torch.tensor(y_a1))
    up, pp = torch.tensor(u_pool, dtype=torch.float64), torch.tensor(p_pool, dtype=torch.float64)
    t = torch.zeros(u1.shape[1], dtype=torch.float64, requires_grad=True)
    log_beta = torch.zeros((), dtype=torch.float64, requires_grad=True)
    logit_a0 = torch.tensor(math.log(0.01 / 0.99), dtype=torch.float64, requires_grad=True)
    log_tau = torch.zeros((), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(np.quantile(u_a1.mean(1), 0.95)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([t, log_beta, logit_a0, log_tau, b], lr=0.03)
    lp1, lpp = p1.clamp_min(1e-12).log(), pp.clamp_min(1e-12).log()

    def rate(u, theta, beta, a0):
        return a0 + (1 - a0) * (1 - torch.exp(-beta * F.softplus(k_smooth * (u @ theta - b)) / k_smooth))

    hist = []
    for it in range(steps + 1):
        theta, beta, a0, tau = torch.softmax(t, 0), log_beta.exp(), torch.sigmoid(logit_a0), log_tau.exp()
        a1, ap = rate(u1, theta, beta, a0), rate(up, theta, beta, a0)
        w1, wp = torch.softmax(lp1 / tau, 1), torch.softmax(lpp / tau, 1)
        A = ap[:, None] * wp
        M = (A.T @ pp) / A.sum(0).clamp_min(1e-30)[:, None]
        q = (1 - a1)[:, None] * p1 + a1[:, None] * (w1 @ M)
        loss = -q[torch.arange(len(y1)), y1].clamp_min(1e-12).log().mean()
        if it < steps:
            opt.zero_grad()
            loss.backward()
            opt.step()
        if it % 100 == 0:
            hist.append(dict(step=it, nll=loss.item(), beta=beta.item(), a0=a0.item(), tau=tau.item(), b=b.item()))
            print("   learn_nll", hist[-1], flush=True)
    with torch.no_grad():
        return dict(theta=torch.softmax(t, 0).numpy().tolist(), beta=float(log_beta.exp()),
                    a0=float(torch.sigmoid(logit_a0)), tau=float(log_tau.exp()), b=float(b), history=hist)


def learn_err(u_a1, err_a1, l2=1e-3):
    """theta >= 0 by logistic regression of the FM error indicator on the standardized layer scores."""
    mu, sd = u_a1.mean(0), u_a1.std(0) + 1e-8
    X = (u_a1 - mu) / sd
    y = err_a1.astype(np.float64)

    def f(v):
        w, c = v[:-1], v[-1]
        z = X @ w + c
        loss = np.mean(np.logaddexp(0, z) - y * z) + l2 * w @ w
        g = 1 / (1 + np.exp(-z)) - y
        return loss, np.append(X.T @ g / len(y) + 2 * l2 * w, g.mean())

    d = X.shape[1]
    res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=[(0, None)] * d + [(None, None)])
    w = res.x[:-1] / sd                       # back to the unstandardized log-distance scale
    return (w / w.sum()).tolist() if w.sum() > 0 else [1.0 / d] * d


# ----------------------------------------------------------------- stage 2: conformal calibration
def confidence(p):
    """FM top-2 log-margin log(p1 / p2): orders inputs like max-prob but does not saturate at p1 = 1."""
    top2 = np.sort(p, 1)[:, -2:].astype(np.float64)
    return np.log(np.maximum(top2[:, 1], 1e-300)) - np.log(np.maximum(top2[:, 0], 1e-300))


def groups_of(p, edges):
    return np.searchsorted(edges, confidence(p)).astype(np.int64)


def mondrian_threshold(s_cal, g_cal, g, alpha):
    qg = np.array([conformal_quantile(s_cal[g_cal == k], alpha) for k in range(N_GROUPS)])
    return qg[g]


def conformal_pvalue(s, s_cal, g=None, g_cal=None):
    """pi(x) = (1 + #{i: s_i >= s(x)}) / (n + 1), within the group of x when groups are given."""
    if g is None:
        g, g_cal = np.zeros(len(s), np.int64), np.zeros(len(s_cal), np.int64)
    out = np.empty(len(s))
    for k in np.unique(g):
        m = g == k
        srt = np.sort(s_cal[g_cal == k])
        out[m] = (1 + len(srt) - np.searchsorted(srt, s[m], side="left")) / (len(srt) + 1)
    return out


def pac_bound(n, alpha, delta=DELTA):
    """Calibration-conditional exceedance: P(s > b_hat | cal) ~ Beta(n + 1 - l, l) (Vovk, 2012)."""
    l = math.ceil((n + 1) * (1 - alpha))
    return float(beta_dist.ppf(1 - delta, n + 1 - l, l)) if l <= n else 1.0


# ----------------------------------------------------------------- stage 3: Learn-then-Test
def hb_pvalue(risk_hat, n, eps):
    """Hoeffding-Bentkus p-value for H: R > eps with a [0, 1] loss (Bates et al., 2021)."""
    r = min(risk_hat, eps)
    if r == 0:
        h = math.log(1 / (1 - eps))
    else:
        h = r * math.log(r / eps) + (1 - r) * math.log((1 - r) / (1 - eps))
    return min(math.exp(-n * h), math.e * binom.cdf(math.ceil(n * risk_hat), n, eps), 1.0)


# ----------------------------------------------------------------- main
def run(B, ctx):
    t0 = time.time()
    data = {k: load_backbone(B, k) for k in ["train", *EVAL_SETS]}
    p = {k: softmax_np(v[1]).astype(np.float32) for k, v in data.items()}
    y_train = get_npy("train_labels", mmap=False).astype(np.int64)
    d2 = gmm_distances(B, {k: v[0] for k, v in data.items()}, y_train)
    del data

    perm = np.random.default_rng(0).permutation(ctx.calA)
    A1, A2 = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    calB, test, yv = ctx.calB, ctx.test, ctx.y_val
    p_fm = p
    T_A1 = fit_temperature(torch.tensor(np.log(np.clip(p_fm["val"][A1], 1e-12, None))), torch.tensor(yv[A1]))
    p_ts = {k: softmax_np(np.log(np.clip(v, 1e-12, None)) / T_A1).astype(np.float32) for k, v in p_fm.items()}
    if BASE == "ts":
        p = p_ts
    med ={l: float(np.median(d2[l]["val"][A1])) for l in REGEN_LAYERS}
    sets = ["train", *EVAL_SETS]
    u = {k: np.stack([np.log(np.maximum(d2[l][k], 1e-12) / med[l]) for l in REGEN_LAYERS], 1) for k in sets}
    edges = np.quantile(confidence(p["val"][A1]), np.linspace(0, 1, N_GROUPS + 1)[1:-1])
    g = {k: groups_of(p[k], edges) for k in sets}
    err = p["val"].argmax(1) != yv
    out = {"split": dict(A1=len(A1), A2=len(A2), calB=len(calB), test=len(test)), "group_edges": edges.tolist(),
           "base": dict(name=BASE, T_A1=T_A1), "ltt_mode": LTT_MODE}

    # ---- stage 1: candidate distances, chosen by error AUROC on calB
    pool_idx = np.sort(np.random.default_rng(1).choice(len(p["train"]), min(POOL_LEARN, len(p["train"])), replace=False))
    nll_fit = learn_nll(u["val"][A1], p["val"][A1], yv[A1], u["train"][pool_idx], p["train"][pool_idx])
    thetas = {"unif": [1.0 / len(REGEN_LAYERS)] * len(REGEN_LAYERS), "err": learn_err(u["val"][A1], err[A1]),
              "nll": nll_fit["theta"]}
    diag = {}
    for name, th in thetas.items():
        s_ = {k: u[k] @ np.array(th) for k in sets}
        diag[name] = dict(theta=th, calB_error_auroc=auroc(s_["val"][calB][~err[calB]], s_["val"][calB][err[calB]]),
                          test_error_auroc=auroc(s_["val"][test][~err[test]], s_["val"][test][err[test]]),
                          ood_auroc_score={o: auroc(s_["val"][test], s_[o]) for o in OOD_SETS})
    out["distances"] = dict(candidates=diag, nll_fit=nll_fit,
                            nll_fit_alpha_implied=float(np.mean(u["val"][A1] @ np.array(nll_fit["theta"]) > nll_fit["b"])))
    print(f"[{B}] error AUROC on calB { {n: round(v['calB_error_auroc'], 3) for n, v in diag.items()} }; "
          f"err theta {np.round(thetas['err'], 3)} ({time.time() - t0:.0f}s)", flush=True)

    out["train_vs_heldout_exceedance_per_layer"] = {
        str(l): {"train_fitted": float(np.mean(d2[l]["val"][test] > np.quantile(d2[l]["train"], 0.95))),
                 "heldout_conformal": float(np.mean(d2[l]["val"][test] > conformal_quantile(d2[l]["val"][A2], 0.05)))}
        for l in REGEN_LAYERS}
    w_cache = {}
    shift_groups = {"all": np.ones(len(ctx.y_shift), bool), **{c: ctx.corr == c for c in np.unique(ctx.corr)}}

    def scaffold(a_of, tau):
        """a_of(set) -> jump rates; returns probabilities and jump rates on every evaluation set."""
        if tau not in w_cache:
            w_cache.clear()
            w_cache[tau] = {k: partition(p[k], tau) for k in sets}
        w = w_cache[tau]
        a = {k: a_of(k).astype(np.float32) for k in sets}
        M = fit_M(a["train"], w["train"], p["train"])
        return {k: predict(a[k], w[k], p[k], M) for k in EVAL_SETS}, a

    def report(probs, a, s_used, b_test=None):
        r = ctx.metrics({k: v.astype(np.float64) for k, v in probs.items()}, {k: p[k] for k in EVAL_SETS})
        r["ood_auroc_jump"] = {o: auroc(a["val"][test], a[o]) for o in OOD_SETS}
        r["mean_jump"] = {"test": float(a["val"][test].mean()), **{k: float(a[k].mean()) for k in ["shift", *OOD_SETS]}}
        gt = g["val"][test]
        r["mean_jump_per_group"] = [float(a["val"][test][gt == k].mean()) for k in range(N_GROUPS)]
        if b_test is not None:
            exc = s_used["val"][test] > b_test
            r["exceed"] = float(exc.mean())
            r["exceed_per_group"] = [float(exc[gt == k].mean()) for k in range(N_GROUPS)]
        r["argmax_kept"] = float(np.mean(probs["val"][test].argmax(1) == p["val"][test].argmax(1)))
        r["director"] = director_metrics.evaluate(probs, yv[test], test, ctx.y_shift, shift_groups, OOD_SETS)
        return r

    rows, best_probs = {}, {}
    no_jump = {k: np.zeros(len(p[k]), np.float32) for k in sets}
    rows["fm"] = report({k: p_fm[k] for k in EVAL_SETS}, no_jump, None)
    rows["ts_A1"] = report({k: p_ts[k] for k in EVAL_SETS}, no_jump, None)
    # ---- ablation: everything (incl. the threshold b) learned by A1 NLL, no calibration
    s_nll = {k: u[k] @ np.array(nll_fit["theta"]) for k in sets}
    probs, a = scaffold(lambda k: hinge_rate(s_nll[k], nll_fit["b"], nll_fit["beta"], nll_fit["a0"]), nll_fit["tau"])
    rows["nll_learned_b"] = report(probs, a, s_nll, nll_fit["b"])
    out["ltt"] = {}
    for dist in ("unif", "err"):
        s = {k: u[k] @ np.array(thetas[dist]) for k in sets}
        # ---- marginal vs Mondrian conformal p-values at ALPHA_DEMO (kappa = 1, population target)
        for mode in ("marginal", "mondrian"):
            if mode == "marginal":
                pv = {k: conformal_pvalue(s[k], s["val"][A2]) for k in sets}
                bt = np.full(len(test), conformal_quantile(s["val"][A2], ALPHA_DEMO))
            else:
                pv = {k: conformal_pvalue(s[k], s["val"][A2], g[k], g["val"][A2]) for k in sets}
                bt = mondrian_threshold(s["val"][A2], g["val"][A2], g["val"][test], ALPHA_DEMO)
            probs, a = scaffold(lambda k: pvalue_rate(pv[k], ALPHA_DEMO, 1.0), float("inf"))
            rows[f"{dist}/conformal_{mode}"] = report(probs, a, s, bt)
            rows[f"{dist}/conformal_{mode}"]["budget_bound"] = A0 + (1 - A0) * ALPHA_DEMO / 2

        # ---- point 2: thresholds fitted on the FM's training features instead of held-out data
        out.setdefault("train_vs_heldout_exceedance", {})[dist] = {
            str(al): {"train_fitted": float(np.mean(s["val"][test] > np.quantile(s["train"], 1 - al))),
                      "heldout_conformal": float(np.mean(s["val"][test] > conformal_quantile(s["val"][A2], al))),
                      "pac_bound": pac_bound(len(A2), al)}
            for al in ALPHA_GRID}

        # ---- stage 3: Learn-then-Test over (alpha, kappa, tau) with Mondrian p-values
        pv = {k: conformal_pvalue(s[k], s["val"][A2], g[k], g["val"][A2]) for k in sets}
        cand, n_b = [], len(calB)
        for tau, alpha, kappa in itertools.product(TAUS, ALPHA_GRID, KAPPAS):
            probs, a = scaffold(lambda k: pvalue_rate(pv[k], alpha, kappa), tau)
            qb = probs["val"][calB]
            flip = float(np.mean(qb.argmax(1) != p["val"][calB].argmax(1)))
            jump = float(a["val"][calB].mean())
            cand.append(dict(tau=tau, alpha=alpha, kappa=kappa, calB_flip=flip, calB_jump=jump,
                             p_flip=hb_pvalue(flip, n_b, EPS_FLIP), p_jump=hb_pvalue(jump, n_b, EPS_JUMP),
                             calB_nll=float(-np.log(np.clip(qb[np.arange(n_b), yv[calB]], 1e-12, None)).mean()),
                             budget_bound=A0 + (1 - A0) * alpha * kappa / (kappa + 1)))
        if LTT_MODE == "bonferroni":
            level = DELTA / (2 * len(cand))   # Bonferroni over candidates and the two risks
            cert = [c for c in cand if c["p_flip"] <= level and c["p_jump"] <= level]
        else:
            level, cert = DELTA / len(TAUS), []
            for tau in TAUS:                  # fixed sequence, least aggressive first; stop at the first failure
                for c in sorted((c for c in cand if c["tau"] == tau), key=lambda c: (c["budget_bound"], c["alpha"])):
                    if max(c["p_flip"], c["p_jump"]) > level:
                        break
                    cert.append(c)
        out["ltt"][dist] = dict(candidates=cand, n_certified=len(cert), eps_flip=EPS_FLIP, eps_jump=EPS_JUMP, delta=DELTA,
                                level=level, mode=LTT_MODE)
        for tau in TAUS:
            ct = [c for c in cert if c["tau"] == tau]
            if not ct:
                continue
            for crit, c in (("nll", min(ct, key=lambda c: c["calB_nll"])),
                            ("sensitive", max(ct, key=lambda c: (c["calB_jump"], c["alpha"], c["kappa"])))):
                probs, a = scaffold(lambda k: pvalue_rate(pv[k], c["alpha"], c["kappa"]), c["tau"])
                name = f"{dist}/ltt_tau{tau:g}_{crit}"
                rows[name] = report(probs, a, s, mondrian_threshold(s["val"][A2], g["val"][A2], g["val"][test], c["alpha"]))
                rows[name]["config"] = c
                best_probs[name] = probs
    out["rows"] = rows

    # ---- (D) conformal label sets on calC (a quarter of test), scored on the rest of the test split
    p_tsB, T = ctx.temperature({k: p_fm[k] for k in EVAL_SETS})
    bases = {"fm": {k: p_fm[k] for k in EVAL_SETS}, "ts": p_tsB, **{f"scaffold:{k}": v for k, v in best_probs.items()}}
    tperm = np.random.default_rng(2).permutation(test)
    n_c = min(5000, len(test) // 4)
    calC, rest = np.sort(tperm[:n_c]), np.sort(tperm[n_c:])
    out["split"].update(calC=len(calC), test_rest=len(rest))
    gr = g["val"][rest]
    sets_out = {}
    for name, pr in bases.items():
        for al in SET_ALPHAS:
            sc = 1 - pr["val"][calC][np.arange(len(calC)), yv[calC]]
            for mode in ("marginal", "mondrian"):
                if mode == "marginal":
                    qg = np.full(N_GROUPS, conformal_quantile(sc, al))
                else:
                    qg = np.array([conformal_quantile(sc[g["val"][calC] == j], al) for j in range(N_GROUPS)])
                inset = (1 - pr["val"][rest]) <= qg[gr][:, None]
                cov = inset[np.arange(len(rest)), yv[rest]]
                insh = (1 - pr["shift"]) <= qg[g["shift"]][:, None]
                sets_out[f"{name}/{al}/{mode}"] = dict(
                    coverage=float(cov.mean()), size=float(inset.sum(1).mean()),
                    coverage_per_group=[float(cov[gr == j].mean()) for j in range(N_GROUPS)],
                    size_per_group=[float(inset[gr == j].sum(1).mean()) for j in range(N_GROUPS)],
                    shift_coverage=float(insh[np.arange(len(ctx.y_shift)), ctx.y_shift].mean()),
                    shift_size=float(insh.sum(1).mean()),
                    ood_size={o: float(((1 - pr[o]) <= qg[g[o]][:, None]).sum(1).mean()) for o in OOD_SETS})
    out["label_sets"] = sets_out
    out["temperature"] = T
    print(f"[{B}] done in {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    torch.set_num_threads(32)
    ctx = Ctx()
    for B in sys.argv[1:] or ["fm"]:
        res = run(B, ctx)
        save_json(res, f"conformal_scaffold_{B}{SUFFIX}.json")
        print(f"base={BASE} (T_A1={res['base']['T_A1']:.3f}) ltt={LTT_MODE} certified "
              f"{ {d: v['n_certified'] for d, v in res['ltt'].items()} }", flush=True)
        for k, v in res["rows"].items():
            t = v["test"]
            extra = f" mcc={t['mcc']:.4f}" if "mcc" in t else ""
            print(f"{k:32s} acc={t['acc']:.4f}{extra} nll={t['nll']:.4f} ece={t['ece']:.4f} "
                  f"sh_nll={v['shift']['nll']:.3f} ood={v['ood_auroc']['mean']:.3f} "
                  f"ood_jump={np.mean(list(v['ood_auroc_jump'].values())):.3f} jump={v['mean_jump']['test']:.4f} "
                  f"exceed={v.get('exceed', float('nan')):.3f} "
                  f"per_group={np.round(v.get('exceed_per_group', []), 3).tolist()} flip={v['flip']:.4f}", flush=True)
        for dist, tv in res["train_vs_heldout_exceedance"].items():
            print(f"train vs held-out ({dist}):", {a: (round(v["train_fitted"], 4), round(v["heldout_conformal"], 4))
                                                   for a, v in tv.items()}, flush=True)
        for k, v in res["label_sets"].items():
            print(f"  set {k:44s} cov={v['coverage']:.3f} size={v['size']:.2f} "
                  f"min_group_cov={min(v['coverage_per_group']):.3f} shift_cov={v['shift_coverage']:.3f}", flush=True)
