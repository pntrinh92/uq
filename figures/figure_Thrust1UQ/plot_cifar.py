"""CIFAR ladder figures (DIRECTOR setting, ViT-B/16 fine-tuned on CIFAR-10 / CIFAR-100).

Reads uq/cifar_ladder/results_<dataset>/{ladder_fm,ladder_ode8,ladder_ode,partitioned_regen_fm}.json.
Run from this folder: python plot_cifar.py
"""
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from plot_report import PALETTE, apply_publication_style, finalize_figure, tick_style

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "uq", "cifar_ladder"))
DATASETS = [("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100")]


def load(ds, name):
    path = os.path.join(ROOT, f"results_{ds}", name)
    return json.load(open(path)) if os.path.exists(path) else None


def rows(ds):
    fm, ode8, ode, part = (load(ds, f) for f in ("ladder_fm.json", "ladder_ode8.json", "ladder_ode.json",
                                                   "partitioned_regen_fm.json"))
    spec = [
        ("FM (ViT-B/16 fine-tuned)", fm and fm["rows"].get("fm"), PALETTE["neutral"], None),
        ("FM + temperature scaling", fm and fm["rows"].get("fm+TS"), PALETTE["red_1"], None),
        ("FM + dropout injection", fm and fm["rows"].get("dropout_injection"), PALETTE["red_2"], None),
        ("FM + Brownian SDE (rung 3)", fm and fm["rows"].get("fm+brownian_z"), PALETTE["red_strong"], None),
        ("FM + regeneration (rung 4)", fm and fm["rows"].get("fm+regen"), PALETTE["blue_secondary"], None),
        ("FM + partitioned regeneration", part and part["row"], PALETTE["blue_main"], None),
        ("ODE, last 4 blocks (rung 2)", ode8 and ode8["rows"].get("ode8"), PALETTE["neutral"], ".."),
        ("ODE last 4 + TS", ode8 and ode8["rows"].get("ode8+TS"), PALETTE["red_1"], ".."),
        ("ODE last 4 + regeneration", ode8 and ode8["rows"].get("ode8+regen"), PALETTE["blue_secondary"], ".."),
        ("ODE, all 12 blocks (rung 2)", ode and ode["rows"].get("ode"), PALETTE["neutral"], "xx"),
    ]
    return [(lab, r, c, h) for lab, r, c, h in spec if r], fm


def plot_ladder(ds, title):
    rs, fm = rows(ds)
    if not rs:
        return
    metrics = [(lambda r: 100 * r["test"]["acc"], r"accuracy (%) $\uparrow$"),
               (lambda r: r["test"]["nll"], r"test NLL $\downarrow$"),
               (lambda r: 100 * r["test"]["ece"], r"test ECE (%) $\downarrow$"),
               (lambda r: r["shift"]["nll"], f"{title}-C NLL " + r"$\downarrow$"),
               (lambda r: r["ood_auroc"]["mean"], r"OOD AUROC (mean of 5) $\uparrow$")]
    fig = plt.figure(figsize=(36, 6.6))
    for i, (f, ylab) in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics) + 1, i + 1)
        vals = [f(r) for _, r, _, _ in rs]
        for j, ((lab, r, c, h), v) in enumerate(zip(rs, vals)):
            ax.bar(j, v, color=c, edgecolor="black", linewidth=2, hatch=h, label=lab)
        ml = None
        if "OOD" in ylab and fm and "fm+multilayer_mahalanobis" in fm["info"]:
            ml = float(np.mean(list(fm["info"]["fm+multilayer_mahalanobis"].values())))
        lo, hi = min(vals), max(vals + ([ml] if ml else []))
        pad = 0.15 * (hi - lo + 1e-9)
        ax.set_ylim(max(0, lo - 3 * pad), hi + 1.5 * pad)
        if ml:
            ax.axhline(ml, color=PALETTE["dark"], lw=2.5, ls=":")
            ax.text(-0.5, ml + 0.2 * pad, "multi-layer Mahalanobis (FM)", fontsize=14, va="bottom")
        ax.set_xticks([])
        ax.set_ylabel(ylab, fontsize=22)
        tick_style(ax, 18)
        handles, labels = ax.get_legend_handles_labels()
    ax = fig.add_subplot(1, len(metrics) + 1, len(metrics) + 1)
    ax.legend(handles, labels, fontsize=17, loc="center", title=title, title_fontsize=20)
    ax.set_axis_off()
    finalize_figure(fig, f"fig_{ds}_ladder")


def plot_severity():
    fig, axes = plt.subplots(1, 4, figsize=(30, 6))
    keep = ["FM (ViT-B/16 fine-tuned)", "FM + temperature scaling", "FM + dropout injection",
            "FM + regeneration (rung 4)", "FM + partitioned regeneration", "ODE, last 4 blocks (rung 2)"]
    for di, (ds, title) in enumerate(DATASETS):
        rs, _ = rows(ds)
        for mi, (key, ylab) in enumerate((("ece", "ECE (%)"), ("nll", "NLL"))):
            ax = axes[2 * di + mi]
            for lab, r, c, h in rs:
                if lab not in keep or "shift_per_severity" not in r:
                    continue
                sev = sorted(r["shift_per_severity"], key=int)
                y = [r["shift_per_severity"][s][key] * (100 if key == "ece" else 1) for s in sev]
                ax.plot([int(s) for s in sev], y, color=c, lw=3, marker="o", markersize=9,
                        ls="--" if h else "-", label=lab)
            ax.set_xlabel(f"{title}-C severity", fontsize=18)
            ax.set_ylabel(ylab, fontsize=18)
            ax.set_xticks([1, 2, 3, 4, 5])
            tick_style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=15, loc="lower center", ncols=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=2, rect=(0, 0.1, 1, 1))
    finalize_figure(fig, "fig_cifar_severity", pad=None)


def plot_ood():
    fig, axes = plt.subplots(1, 2, figsize=(26, 6))
    keep = ["FM (ViT-B/16 fine-tuned)", "FM + temperature scaling", "FM + dropout injection",
            "FM + regeneration (rung 4)", "FM + partitioned regeneration", "ODE, last 4 blocks (rung 2)"]
    for ax, (ds, title) in zip(axes, DATASETS):
        rs, fm = rows(ds)
        rs = [r for r in rs if r[0] in keep]
        if not rs:
            continue
        sets = [k for k in rs[0][1]["ood_auroc"] if k != "mean"]
        x = np.arange(len(sets))
        width = 0.8 / (len(rs) + 1)
        for j, (lab, r, c, h) in enumerate(rs):
            ax.bar(x + (j - len(rs) / 2) * width, [r["ood_auroc"][k] for k in sets], width, color=c,
                   edgecolor="black", linewidth=1.5, hatch=h, label=lab)
        if fm and "fm+multilayer_mahalanobis" in fm["info"]:
            ml = fm["info"]["fm+multilayer_mahalanobis"]
            ax.bar(x + (len(rs) - len(rs) / 2) * width, [ml[k] for k in sets], width, color=PALETTE["dark"],
                   edgecolor="black", linewidth=1.5, label="multi-layer Mahalanobis score")
        ax.set_xticks(x)
        ax.set_xticklabels([{"svhn": "SVHN", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "lsun": "LSUN",
                             "dtd": "Textures", "noise": "noise"}[k] for k in sets], fontsize=17)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel(f"OOD AUROC, ID = {title}", fontsize=19)
        tick_style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=15, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=2, rect=(0, 0.14, 1, 1))
    finalize_figure(fig, "fig_cifar_ood", pad=None)


def plot_compile_scaling():
    """Compile fidelity across scales: classes x compiled blocks, and the Euler-step sweep."""
    roots = [("CIFAR-10", 10, os.path.join(ROOT, "results_cifar10")),
             ("CIFAR-100", 100, os.path.join(ROOT, "results_cifar100")),
             ("ImageNet", 1000, os.path.abspath(os.path.join(ROOT, "..", "imagenet_ladder", "results")))]
    fig, axes = plt.subplots(1, 2, figsize=(23, 6))
    ax = axes[0]
    for tag, n_blocks, c, lab in (("ode8", 4, PALETTE["blue_main"], "last 4 blocks compiled"),
                                  ("ode", 12, PALETTE["red_strong"], "all 12 blocks compiled")):
        xs, ys = [], []
        for name, C, root in roots:
            lad = json.load(open(os.path.join(root, f"ladder_{tag}.json")))
            xs.append(C)
            ys.append(100 * lad["info"][f"{tag}_agreement_with_fm_test"])
        ax.plot(xs, ys, color=c, marker="o", markersize=11, lw=3, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2.5, f"{y:.0f}%", ha="center", fontsize=14, color=c)
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["CIFAR-10\n(10)", "CIFAR-100\n(100)", "ImageNet\n(1000)"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("argmax agreement with FM (%)", fontsize=18)
    ax.legend(fontsize=15, loc="lower left")
    tick_style(ax)
    ax = axes[1]
    styles = {"CIFAR-10": PALETTE["green_3"], "CIFAR-100": PALETTE["blue_secondary"], "ImageNet": PALETTE["red_strong"]}
    for name, C, root in roots:
        for tag, n_blocks, ls in (("ode8", 4, "-"), ("ode", 12, "--")):
            sw = json.load(open(os.path.join(root, f"compile_step_sweep_{tag}.json")))
            ks = sorted(int(k) for k in sw)
            ax.plot([k / n_blocks for k in ks], [100 * sw[str(k)]["agree"] for k in ks], color=styles[name], ls=ls,
                    marker="o", markersize=8, lw=3, label=f"{name}, {'last 4' if tag == 'ode8' else 'all 12'}")
    ax.axvline(1, color=PALETTE["grey"], lw=2, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Euler steps per compiled block", fontsize=18)
    ax.set_ylabel("argmax agreement with FM (%)", fontsize=18)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    tick_style(ax)
    finalize_figure(fig, "fig_compile_scaling")


if __name__ == "__main__":
    apply_publication_style()
    for ds, title in DATASETS:
        plot_ladder(ds, title)
    plot_severity()
    plot_ood()
    plot_compile_scaling()
