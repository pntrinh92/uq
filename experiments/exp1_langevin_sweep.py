"""Experiment 1: sweep the marginal-preserving SDE family without learning F_theta or a score.

Lie-Trotter splitting: run the frozen FM block exactly (maps p_l to p_{l+1}),
then take K Langevin steps that target p_{l+1}. The score is analytic from a
class-conditional Gaussian mixture (tied Ledoit-Wolf covariance) fitted to the
training features at that layer. MALA keeps the GMM marginal exactly.

Variant A ("final"): Langevin on the final representation z only (64-d).
Variant B ("all"): Langevin on the full token state after every block (17x64-d).

Swept: diffusion structure D (isotropic, preconditioned D = Sigma, rank-8),
step size h, number of steps K, MALA vs unadjusted Langevin (ULA).
Baselines: deterministic FM, temperature scaling (TS), MC dropout, deep
ensemble, and the GMM's own Mahalanobis OOD score (free once the GMM is fitted).
"""
import itertools
import sys

import numpy as np
import torch

from common import (TiedGMM, auroc, centered_gaussian_w2, classification_metrics, entropy,
                    fit_temperature, get_models, langevin, make_preconditioner, mmd_rbf,
                    save_json, set_mc_dropout, variance_budget)

N_SPLITS = 5
S = 32
H_GRID = [float(h) for h in np.geomspace(1e-3, 2.0, 10)]
CHUNK = 8192


def mutual_info(prob_samples):
    """BALD: H[mean p] - mean H[p]; prob_samples (N, S, C)."""
    return entropy(prob_samples.mean(1)) - entropy(prob_samples.flatten(0, 1)).view(prob_samples.shape[:2]).mean(1)


def margin_bins(p_det, p_new, n_bins=3):
    """Confidence change and flip rate, split by the deterministic top-2 margin (tertiles)."""
    top2 = p_det.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    order = margin.argsort()
    out = []
    for idx in order.tensor_split(n_bins):
        m = torch.zeros(len(margin), dtype=torch.bool)
        m[idx] = True
        out.append(dict(
            margin_lo=float(margin[idx].min()), margin_hi=float(margin[idx].max()),
            d_conf=float((p_new[m].max(1).values - p_det[m].max(1).values).mean()),
            flip=float((p_new[m].argmax(1) != p_det[m].argmax(1)).float().mean()),
        ))
    return out


@torch.no_grad()
def sde_samples(model, X, gmms, cfg, gen):
    """Logit samples (N, S, C) and z samples (N, S, d) under one SDE configuration."""
    Xr = X.repeat_interleave(S, 0)
    zs, accs = [], []
    for idx in torch.arange(len(Xr)).split(CHUNK):
        tokens = model.embed(Xr[idx])
        for li, blk in enumerate(model.blocks):
            tokens = blk(tokens)
            if cfg["variant"] == "all":
                gmm, L, Lp = gmms[li]
                flat, a = langevin(tokens.flatten(1), gmm, L, Lp, cfg["h"], cfg["K"], cfg["mala"], gen)
                tokens = flat.float().view_as(tokens)
                accs.append(a)
        z = model.readout(tokens)
        if cfg["variant"] == "final":
            gmm, L, Lp = gmms["z"]
            z, a = langevin(z, gmm, L, Lp, cfg["h"], cfg["K"], cfg["mala"], gen)
            z = z.float()
            accs.append(a)
        zs.append(z)
    z = torch.cat(zs).view(len(X), S, -1)
    return model.head(z), z, float(np.mean(accs))


def evaluate_probs(p_samples, y=None):
    p = p_samples.mean(1)
    return p, (classification_metrics(p, y) if y is not None else None)


@torch.no_grad()
def run_split(seed, configs):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    data, models = get_models(seed)
    fm = models[0]
    sets = {"val": data["Xva"], "test": data["Xte"], "shift": data["Xsh"], **data["ood"]}
    labels = {"val": data["yva"], "test": data["yte"], "shift": data["yte"]}

    # Deterministic features: fit GMMs on the training split.
    _, ztr, states_tr = fm(data["Xtr"], return_all=True)
    gmms = {"z": TiedGMM(ztr, data["ytr"])}
    for li, st in enumerate(states_tr):
        gmms[li] = TiedGMM(st.flatten(1), data["ytr"])
    pre_cache = {}

    det = {k: fm(X, return_all=True) for k, X in sets.items()}
    p_det = {k: torch.softmax(v[0], 1) for k, v in det.items()}
    z_det_test = det["test"][1]
    out = {"baselines": {}, "configs": []}

    def ood_aurocs(score):
        return {o: auroc(score["test"], score[o]) for o in data["ood"]}

    # --- baselines
    b = out["baselines"]
    b["det"] = dict(test=classification_metrics(p_det["test"], labels["test"]),
                    shift=classification_metrics(p_det["shift"], labels["shift"]),
                    ood_entropy=ood_aurocs({k: entropy(v) for k, v in p_det.items()}))
    T = fit_temperature(det["val"][0], labels["val"])
    p_ts = {k: torch.softmax(v[0] / T, 1) for k, v in det.items()}
    b["ts"] = dict(T=T, test=classification_metrics(p_ts["test"], labels["test"]),
                   shift=classification_metrics(p_ts["shift"], labels["shift"]),
                   ood_entropy=ood_aurocs({k: entropy(v) for k, v in p_ts.items()}))
    maha = {k: gmms["z"].min_mahalanobis(v[1]).numpy() for k, v in det.items()}
    b["ts+gmm"] = dict(test=b["ts"]["test"], shift=b["ts"]["shift"], ood_entropy=ood_aurocs(maha))

    mcd_logits = {}
    for k, X in sets.items():
        set_mc_dropout(fm, True)
        mcd_logits[k] = torch.stack([fm(X) for _ in range(S)], 1)
        set_mc_dropout(fm, False)
    p_mcd_s = {k: torch.softmax(v, -1) for k, v in mcd_logits.items()}
    p_mcd = {k: v.mean(1) for k, v in p_mcd_s.items()}
    T_mcd = fit_temperature(p_mcd["val"].clamp_min(1e-12).log(), labels["val"])
    b["mcd"] = dict(test=classification_metrics(p_mcd["test"], labels["test"]),
                    shift=classification_metrics(p_mcd["shift"], labels["shift"]),
                    ood_entropy=ood_aurocs({k: entropy(v) for k, v in p_mcd.items()}),
                    ood_mi=ood_aurocs({k: mutual_info(v) for k, v in p_mcd_s.items()}))
    b["mcd+ts"] = dict(**{k: classification_metrics(torch.softmax(p_mcd[k].clamp_min(1e-12).log() / T_mcd, 1), labels[k])
                          for k in ("test", "shift")})
    ens_logits = {k: torch.stack([m(X) for m in models], 1) for k, X in sets.items()}
    p_ens_s = {k: torch.softmax(v, -1) for k, v in ens_logits.items()}
    p_ens = {k: v.mean(1) for k, v in p_ens_s.items()}
    b["ensemble"] = dict(test=classification_metrics(p_ens["test"], labels["test"]),
                         shift=classification_metrics(p_ens["shift"], labels["shift"]),
                         ood_entropy=ood_aurocs({k: entropy(v) for k, v in p_ens.items()}),
                         ood_mi=ood_aurocs({k: mutual_info(v) for k, v in p_ens_s.items()}))

    # --- SDE configurations
    for cfg in configs:
        key = (cfg["variant"], cfg["D"])
        if key not in pre_cache:
            if cfg["variant"] == "final":
                g = gmms["z"]
                pre_cache[key] = {"z": (g, *make_preconditioner(g, cfg["D"]))}
            else:
                pre_cache[key] = {li: (gmms[li], *make_preconditioner(gmms[li], cfg["D"])) for li in range(len(fm.blocks))}
        res = dict(cfg)
        logits_s, z_s, p_s, accs = {}, {}, {}, []
        for k, X in sets.items():
            logits_s[k], z_s[k], a = sde_samples(fm, X, pre_cache[key], cfg, gen)
            p_s[k] = torch.softmax(logits_s[k], -1)
            accs.append(a)
        p = {k: v.mean(1) for k, v in p_s.items()}
        res["accept"] = float(np.mean(accs))
        res["val"] = classification_metrics(p["val"], labels["val"])
        res["test"] = classification_metrics(p["test"], labels["test"])
        T_s = fit_temperature(p["val"].clamp_min(1e-12).log(), labels["val"])
        res["T_after"] = T_s
        res["val_ts"] = classification_metrics(torch.softmax(p["val"].clamp_min(1e-12).log() / T_s, 1), labels["val"])
        res["test_ts"] = classification_metrics(torch.softmax(p["test"].clamp_min(1e-12).log() / T_s, 1), labels["test"])
        res["shift"] = classification_metrics(p["shift"], labels["shift"])
        res["shift_ts"] = classification_metrics(torch.softmax(p["shift"].clamp_min(1e-12).log() / T_s, 1), labels["shift"])
        res["ood_entropy"] = ood_aurocs({k: entropy(v) for k, v in p.items()})
        res["ood_mi"] = ood_aurocs({k: mutual_info(v) for k, v in p_s.items()})
        res["flip"] = float((p["test"].argmax(1) != p_det["test"].argmax(1)).float().mean())
        res["per_sample_acc"] = float((logits_s["test"].argmax(-1) == labels["test"][:, None]).float().mean())
        res["budget_z"] = variance_budget(z_det_test, z_s["test"])
        res["mmd_z"] = mmd_rbf(z_det_test, z_s["test"].flatten(0, 1))
        res["w2_mcd"] = centered_gaussian_w2(logits_s["test"], mcd_logits["test"])
        res["w2_ens"] = centered_gaussian_w2(logits_s["test"], ens_logits["test"])
        res["margin_bins"] = margin_bins(p_det["test"], p["test"])
        out["configs"].append(res)
        print(f"  split{seed} {cfg} acc={res['test']['acc']:.3f} nll={res['test']['nll']:.3f} "
              f"ece={res['test']['ece']:.3f} sh_nll={res['shift']['nll']:.3f} flip={res['flip']:.3f} E/det={res['budget_z']['E_share']:.3f} "
              f"accept={res['accept']:.2f} auroc9={res['ood_entropy']['heldout_9']:.3f}", flush=True)
    out["baselines"]["mcd"]["w2_self"] = 0.0
    return out


def build_configs(which):
    cfgs = []
    if which in ("final", "both"):
        for D, h in itertools.product(["iso", "precond", "lowrank"], H_GRID):
            cfgs.append(dict(variant="final", D=D, h=h, K=5, mala=True))
        for h in H_GRID:
            cfgs.append(dict(variant="final", D="precond", h=h, K=25, mala=True))
            cfgs.append(dict(variant="final", D="precond", h=h, K=5, mala=False))
    if which in ("all", "both"):
        for h in H_GRID[::2] + [H_GRID[-1]]:
            cfgs.append(dict(variant="all", D="precond", h=h, K=5, mala=True))
    return cfgs


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "final"
    splits = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(N_SPLITS))
    configs = build_configs(which)
    for s in splits:
        save_json(run_split(s, configs), f"exp1_{which}_split{s}.json")
