"""Aggregate Experiment 1 over splits, answer the decisive question, and plot.

Decisive question: is there a configuration in the marginal-preserving family
that beats temperature scaling on NLL/ECE *and* MC dropout on OOD AUROC? The
configuration is selected on validation NLL (never on test); an "oracle" pick
on test is reported separately as an optimistic upper bound.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from common import BLUE, FIGURES, GREY, INK, ORANGE, RESULTS, setup_matplotlib

OOD = ["heldout_9", "rotated", "noise"]


def load(variant):
    runs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, f"exp1_{variant}_split*.json"))):
        runs.append(json.load(open(f)))
    return runs


def cfg_key(c):
    return (c["variant"], c["D"], c["K"], c["mala"], round(c["h"], 6))


def flat(c):
    """Scalar metrics of one config (or baseline) result."""
    out = {}
    for split in ("val", "test", "test_ts", "shift", "shift_ts"):
        for m, v in c.get(split, {}).items():
            out[f"{split}.{m}"] = v
    for kind in ("ood_entropy", "ood_mi"):
        if kind in c:
            for o in OOD:
                out[f"{kind}.{o}"] = c[kind][o]
            out[f"{kind}.mean"] = float(np.mean([c[kind][o] for o in OOD]))
    for k in ("flip", "accept", "per_sample_acc", "mmd_z", "w2_mcd", "w2_ens", "T_after"):
        if k in c:
            out[k] = c[k]
    if "budget_z" in c:
        out["E_share"] = c["budget_z"]["E_share"]
        out["rho_z"] = c["budget_z"]["rho"]
        out["shrink_z"] = c["budget_z"]["shrink"]
        out["marg_change"] = c["budget_z"]["tr_stoch"] / c["budget_z"]["tr_det"] - 1
    return out


def agg(list_of_flat):
    keys = set().union(*[d.keys() for d in list_of_flat])
    return {k: (float(np.mean([d[k] for d in list_of_flat if k in d])),
                float(np.std([d[k] for d in list_of_flat if k in d]))) for k in keys}


def main():
    final, allv = load("final"), load("all")
    n = len(final)
    base = {b: agg([flat(r["baselines"][b]) for r in final]) for b in final[0]["baselines"]}
    per_cfg = defaultdict(list)
    for r in final + allv:
        for c in r["configs"]:
            per_cfg[cfg_key(c)].append(flat(c))
    cfgs = {k: agg(v) for k, v in per_cfg.items()}

    # Selection on validation NLL, per split, restricted to MALA (exact family members).
    def select(pools, metric):
        picks = []
        for pool in pools:
            best = min([c for c in pool if c["mala"]], key=lambda c: c[metric]["nll"])
            picks.append(flat(best) | {"cfg": cfg_key(best)})
        return agg([{k: v for k, v in d.items() if k != "cfg"} for d in picks]), [str(d["cfg"]) for d in picks]

    pools = {"z only": [r["configs"] for r in final], "all blocks": [r["configs"] for r in allv],
             "either": [f["configs"] + a["configs"] for f, a in zip(final, allv)]}
    selected = {name: select(pl, "val") for name, pl in pools.items()}
    selected_ts = {name: select(pl, "val_ts") for name, pl in pools.items()}
    sel_a, sel_ts_a = selected["either"][0], selected_ts["either"][0]

    ts, mcd, tsg = base["ts"], base["mcd"], base["ts+gmm"]
    def beats(c, nll_key="test.nll", ece_key="test.ece"):
        return (c[nll_key][0] < ts["test.nll"][0] and c[ece_key][0] < ts["test.ece"][0]
                and c["ood_entropy.mean"][0] > mcd["ood_entropy.mean"][0])
    oracle = [k for k, c in cfgs.items() if k[3] and beats(c)]
    oracle_ts = [k for k, c in cfgs.items() if k[3] and "test_ts.nll" in c and beats(c, "test_ts.nll", "test_ts.ece")]
    oracle_vs_gmm = [k for k, c in cfgs.items() if k[3] and c["test.nll"][0] < ts["test.nll"][0]
                     and c["test.ece"][0] < ts["test.ece"][0] and c["ood_entropy.mean"][0] > tsg["ood_entropy.mean"][0]]

    summary = dict(n_splits=n, baselines=base,
                   selected_on_val={k: v[0] for k, v in selected.items()},
                   selected_on_val_with_ts={k: v[0] for k, v in selected_ts.items()},
                   selected_cfgs={k: v[1] for k, v in selected.items()},
                   selected_cfgs_ts={k: v[1] for k, v in selected_ts.items()},
                   oracle_beats_ts_and_mcd=[str(k) for k in oracle],
                   oracle_with_ts_beats_ts_and_mcd=[str(k) for k in oracle_ts],
                   oracle_beats_ts_and_ts_gmm=[str(k) for k in oracle_vs_gmm],
                   configs={str(k): v for k, v in cfgs.items()})
    json.dump(summary, open(os.path.join(RESULTS, "exp1_summary.json"), "w"), indent=1)

    # ---- markdown table
    rows = [("FM (deterministic)", base["det"]), ("Temperature scaling", ts), ("MC dropout", mcd),
            ("MC dropout + TS", base["mcd+ts"]), ("Deep ensemble (5)", base["ensemble"]),
            ("TS + GMM Mahalanobis (OOD)", tsg),
            *[(f"SDE family ({k}), picked on val NLL", v[0]) for k, v in selected.items()],
            *[(f"SDE family ({k}) + TS, picked on val NLL", v[0]) for k, v in selected_ts.items()]]
    fmt = lambda d, k: f"{d[k][0]:.3f} ± {d[k][1]:.3f}" if k in d else "–"
    lines = ["| Method | Acc | NLL | ECE | Shift NLL | Shift ECE | OOD AUROC (mean of 3) | AUROC held-out 9 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, d in rows:
        pre = "test_ts" if "+ TS, picked" in name else "test"
        shp = "shift_ts" if "+ TS, picked" in name else "shift"
        lines.append(f"| {name} | {fmt(d, pre + '.acc')} | {fmt(d, pre + '.nll')} | {fmt(d, pre + '.ece')} | "
                     f"{fmt(d, shp + '.nll')} | {fmt(d, shp + '.ece')} | {fmt(d, 'ood_entropy.mean')} | "
                     f"{fmt(d, 'ood_entropy.heldout_9')} |")
    md = "\n".join(lines)
    md += f"\n\nConfigs picked on val NLL: {summary['selected_cfgs']}\n"
    md += f"Configs picked on val NLL (+TS): {summary['selected_cfgs_ts']}\n"
    md += f"\nOracle (test-picked) MALA configs beating TS on NLL+ECE and MCD on mean OOD AUROC: {len(oracle)} -> {oracle}\n"
    md += f"Same, with TS applied on top of the SDE: {len(oracle_ts)} -> {oracle_ts}\n"
    md += f"Oracle configs beating TS on NLL+ECE and TS+GMM on mean OOD AUROC: {len(oracle_vs_gmm)} -> {oracle_vs_gmm}\n"
    open(os.path.join(RESULTS, "exp1_summary.md"), "w").write(md)
    print(md)

    plot(cfgs, base)


def series(cfgs, variant, D, K, mala, key):
    ks = sorted([k for k in cfgs if k[:4] == (variant, D, K, mala)], key=lambda k: k[4])
    return (np.array([k[4] for k in ks]), np.array([cfgs[k][key][0] for k in ks]),
            np.array([cfgs[k][key][1] for k in ks]))


def plot(cfgs, base):
    plt = setup_matplotlib()
    lines = [("final", "iso", 5, True, BLUE, "-", "o", "z: isotropic"),
             ("final", "precond", 5, True, ORANGE, "-", "s", "z: precond. (K=5)"),
             ("final", "precond", 25, True, ORANGE, "--", "s", "z: precond. (K=25)"),
             ("final", "lowrank", 5, True, INK, "-", "^", "z: rank-8"),
             ("final", "precond", 5, False, GREY, ":", "x", "z: precond., ULA"),
             ("all", "precond", 5, True, "#7FB0EC", "-", "D", "all blocks: precond.")]
    panels = [("E_share", r"$\mathrm{tr}\,E\,/\,\mathrm{tr}\,\Sigma_{\rm det}$ at $z$", "Variance budget spent", None),
              ("marg_change", r"$\mathrm{tr}\,\Sigma_{\rm stoch}/\mathrm{tr}\,\Sigma_{\rm det}-1$ at $z$", "Marginal distortion (ID)", None),
              ("test.nll", "test NLL (ID)", "Calibration (ID)", ("ts", "mcd", "ensemble")),
              ("shift.nll", "test NLL (pixel-noise shift)", "Calibration under shift", ("ts", "mcd", "ensemble")),
              ("ood_entropy.mean", "OOD AUROC (mean of 3 sets)", "OOD detection", ("mcd", "ts+gmm", "ensemble")),
              ("flip", "argmax flip rate vs FM", "Argmax flips vs FM", None)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    axes = axes.ravel()
    bstyle = {"ts": (BLUE, "TS"), "mcd": (ORANGE, "MC dropout"), "ensemble": (INK, "ensemble"),
              "ts+gmm": ("#7FB0EC", "TS+GMM score")}
    for ax, (key, ylab, title, bl) in zip(axes, panels):
        for variant, D, K, mala, c, ls, mk, lab in lines:
            h, mu, sd = series(cfgs, variant, D, K, mala, key)
            if len(h) == 0:
                continue
            ok = np.isfinite(mu) & (np.abs(mu) < 1e3)
            ax.plot(h[ok], mu[ok], color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
            ax.fill_between(h[ok], (mu - sd)[ok], (mu + sd)[ok], color=c, alpha=0.12, lw=0)
        if bl:
            for b in bl:
                c, lab = bstyle[b]
                ax.axhline(base[b][key][0], color=c, lw=1.2, ls=(0, (4, 2)), alpha=0.9)
                ax.text(1.0, base[b][key][0], " " + lab, transform=ax.get_yaxis_transform(), color=c,
                        fontsize=8, va="center", ha="left", clip_on=False)
        ax.axhline(base["det"][key][0] if key in base["det"] else np.nan, color=GREY, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("step size $h$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=11.5)
        if key in ("test.nll", "shift.nll"):
            lo = min(base["ensemble"][key][0], base["ts"][key][0]) * 0.8
            ax.set_ylim(lo, base["det"][key][0] * 2.2)
        if key == "flip":
            ax.set_ylim(-0.005, 0.1)
    axes[0].set_yscale("log")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.5, hspace=0.45)
    fig.savefig(f"{FIGURES}/fig_exp1_sweep.png")

    # Margin split for the preconditioned family at z, across h.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ks = sorted([k for k in cfgs if k[:4] == ("final", "precond", 5, True)], key=lambda k: k[4])
    runs = load("final")
    tert = ["low margin", "mid margin", "high margin"]
    for j, (c, lab) in enumerate(zip([ORANGE, INK, BLUE], tert)):
        dconf, flip = [], []
        for k in ks:
            vals = [cc["margin_bins"][j] for r in runs for cc in r["configs"] if cfg_key(cc) == k]
            dconf.append(np.mean([v["d_conf"] for v in vals]))
            flip.append(np.mean([v["flip"] for v in vals]))
        hs = [k[4] for k in ks]
        axes[0].plot(hs, dconf, color=c, marker="o", ms=4, lw=1.8, label=lab)
        axes[1].plot(hs, flip, color=c, marker="o", ms=4, lw=1.8, label=lab)
    axes[0].axhline(0, color=GREY, lw=1, ls=":")
    for ax, yl, t in ((axes[0], r"$\Delta$ max-prob vs FM", "Confidence change by FM margin"),
                      (axes[1], "argmax flip rate", "Flips by FM margin")):
        ax.set_xscale("log")
        ax.set_xlabel("step size $h$ (z, precond., K=5, MALA)")
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left", fontsize=11.5)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/fig_exp1_margin.png")


if __name__ == "__main__":
    main()
