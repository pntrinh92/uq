"""Experiment 0: does the reference uncertainty live in the marginal-preserving family?

For every layer, the final representation z and the logits, compare the
deterministic population covariance Sigma_det with the stochastic one Sigma_mc
(MC dropout over inputs and masks) and the average per-input covariance E:

    rho = tr(Sigma_mc - Sigma_det) / tr E = 1 - shrink,
    shrink = (tr Sigma_det - tr Cov_x m(x)) / tr E.

rho ~ 0: the reference only redistributes variance (marginal-preserving).
rho ~ 1: it adds variance on top of the marginal, which no same-marginal SDE can do.
The deep ensemble is used as a second reference in logit space only (members'
feature spaces are not aligned). MMD between the deterministic and stochastic
populations is reported as a direct, non-trace check.
"""
import numpy as np
import torch

from common import (BLUE, GREY, INK, ORANGE, FIGURES, get_models, mmd_rbf, save_json,
                    set_mc_dropout, setup_matplotlib, variance_budget)

N_SPLITS, S = 5, 64


@torch.no_grad()
def collect(model, X, mc):
    set_mc_dropout(model, mc)
    logits, z, states = model(X, return_all=True)
    set_mc_dropout(model, False)
    reps = {f"block{i + 1}": s.flatten(1) for i, s in enumerate(states)}
    reps.update({f"block{i + 1}_cls": s[:, 0] for i, s in enumerate(states)})
    reps["z"] = z
    reps["logits"] = logits
    return reps


@torch.no_grad()
def run_split(seed):
    torch.manual_seed(seed)
    data, models = get_models(seed)
    fm = models[0]
    sets = {"ID": data["Xte"], **data["ood"]}
    out = {}
    for name, X in sets.items():
        det = collect(fm, X, mc=False)
        draws = [collect(fm, X, mc=True) for _ in range(S)]
        res = {}
        for key in det:
            samples = torch.stack([d[key] for d in draws], 1)  # (N, S, dim)
            r = variance_budget(det[key], samples)
            if key in ("z", "logits"):
                r["mmd"] = mmd_rbf(det[key], samples.flatten(0, 1))
                r["mmd_null"] = mmd_rbf(det[key], det[key][torch.randperm(len(X))], seed=1)
            res[key] = r
        # Deep ensemble reference in logit space; member 0 plays the deterministic FM.
        ens = torch.stack([m(X) for m in models], 1)
        r = variance_budget(det["logits"], ens)
        r["mmd"] = mmd_rbf(det["logits"], ens.flatten(0, 1))
        res["logits_ensemble"] = r
        out[name] = res
    return out


def summarize(all_runs):
    keys = all_runs[0]["ID"].keys()
    summary = {}
    for sname in all_runs[0]:
        summary[sname] = {}
        for k in keys:
            summary[sname][k] = {
                stat: [float(np.mean([r[sname][k][stat] for r in all_runs])),
                       float(np.std([r[sname][k][stat] for r in all_runs]))]
                for stat in all_runs[0][sname][k]
            }
    return summary


def plot(summary):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.42))

    ax = axes[0]
    layers = ["block1", "block2", "block3", "block4", "z", "logits", "logits_ensemble"]
    labels = ["blk1", "blk2", "blk3", "blk4", "z", "logits\n(MCD)", "logits\n(ens.)"]
    styles = {"ID": (BLUE, "o", "ID test"), "heldout_9": (ORANGE, "s", "held-out 9"),
              "rotated": (INK, "^", "rotated"), "noise": (GREY, "D", "noise")}
    xs = np.arange(len(layers))
    for j, (sname, (c, mk, lab)) in enumerate(styles.items()):
        mu = [summary[sname][k]["rho"][0] for k in layers]
        sd = [summary[sname][k]["rho"][1] for k in layers]
        ax.errorbar(xs + (j - 1.5) * 0.08, mu, yerr=sd, color=c, marker=mk, ms=5, lw=1.4, capsize=2, label=lab)
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    ax.axhline(1, color=INK, lw=0.8, ls=":")
    ax.text(-0.45, 0.04, "same marginal", fontsize=8.5, color=INK, ha="left", va="bottom")
    ax.text(-0.45, 1.04, "pure added noise", fontsize=8.5, color=INK, ha="left", va="bottom")
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_xticks(xs, labels)
    ax.set_ylabel(r"$\rho=\mathrm{tr}(\Sigma_{\rm mc}-\Sigma_{\rm det})\,/\,\mathrm{tr}\,E$")
    ax.set_title("Where does the reference uncertainty sit?", loc="left", fontsize=12)
    ax.legend(fontsize=8.5, ncol=2, loc="lower left", frameon=False)

    ax = axes[1]
    keys = ["z", "logits", "logits_ensemble"]
    names = ["z (MCD)", "logits (MCD)", "logits (ens.)"]
    s = summary["ID"]
    det = np.array([s[k]["tr_det"][0] for k in keys])
    mean = np.array([s[k]["tr_mean"][0] for k in keys]) / det
    E = np.array([s[k]["tr_E"][0] for k in keys]) / det
    y = np.arange(len(keys))
    ax.barh(y, mean, color=BLUE, label=r"$\mathrm{tr\,Cov}_x\,m(x)$ (explained by input)")
    ax.barh(y, E, left=mean, color=ORANGE, label=r"$\mathrm{tr}\,E$ (per-input uncertainty)")
    ax.axvline(1, color=INK, lw=1.2, ls="--")
    ax.text(1.0, len(keys) - 0.45, r" $\mathrm{tr}\,\Sigma_{\rm det}$", fontsize=9, va="bottom")
    ax.set_yticks(y, names)
    ax.set_xlabel(r"share of $\mathrm{tr}\,\Sigma_{\rm det}$ (ID test)")
    ax.set_title("Variance budget, ID", loc="left", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False)
    fig.savefig(f"{FIGURES}/fig_exp0_rho.png")
    print("saved fig_exp0_rho.png")


if __name__ == "__main__":
    import json, os, sys
    if "--plot" in sys.argv:
        from common import RESULTS
        plot(json.load(open(os.path.join(RESULTS, "exp0_variance_budget.json")))["summary"])
        sys.exit()
    runs = [run_split(s) for s in range(N_SPLITS)]
    summary = summarize(runs)
    save_json({"per_split": runs, "summary": summary}, "exp0_variance_budget.json")
    for sname in summary:
        print(f"\n== {sname}")
        for k, v in summary[sname].items():
            extra = f" mmd={v['mmd'][0]:.4f} (null {v['mmd_null'][0]:.4f})" if "mmd_null" in v else ""
            print(f"  {k:16s} rho={v['rho'][0]:+.3f}±{v['rho'][1]:.3f}  shrink={v['shrink'][0]:+.3f}"
                  f"  E/det={v['E_share'][0]:.4f}{extra}")
    plot(summary)
