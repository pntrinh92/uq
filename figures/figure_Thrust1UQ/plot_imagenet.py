"""ImageNet ladder figures (ViT-B/16), figures4papers house style.

Reads uq/imagenet_ladder/results/{ladder_fm,ladder_ode,compile_train_log,compile_step_sweep}.json.
Run from this folder: python plot_imagenet.py
"""
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from plot_report import PALETTE, apply_publication_style, finalize_figure, tick_style

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "uq", "imagenet_ladder", "results"))
OOD = [("imagenet_o", "ImageNet-O"), ("dtd", "Textures (DTD)"), ("noise", "uniform noise")]
CORR = [("gaussian_noise", "Gaussian noise"), ("gaussian_blur", "Gaussian blur"), ("contrast", "contrast"),
        ("jpeg", "JPEG")]


def load(name):
    path = os.path.join(RES, name)
    return json.load(open(path)) if os.path.exists(path) else None


def rows_table():
    fm, ode8, ode = load("ladder_fm.json"), load("ladder_ode8.json"), load("ladder_ode.json")
    spec = [
        ("fm", "FM (ViT-B/16)", PALETTE["neutral"], None, fm),
        ("fm+TS", "FM + temperature scaling", PALETTE["red_1"], None, fm),
        ("dropout_injection", "FM + dropout injection", PALETTE["red_2"], None, fm),
        ("fm+brownian_z", "FM + Brownian SDE (rung 3)", PALETTE["red_strong"], None, fm),
        ("fm+regen", "FM + regeneration (rung 4)", PALETTE["blue_main"], None, fm),
        ("ensemble_3ckpt", "3-checkpoint ensemble", PALETTE["grey"], "//", fm),
        ("ode8", "ODE, last 4 blocks (rung 2)", PALETTE["neutral"], "..", ode8),
        ("ode8+TS", "ODE last 4 + TS", PALETTE["red_1"], "..", ode8),
        ("ode8+brownian_z", "ODE last 4 + Brownian SDE", PALETTE["red_strong"], "..", ode8),
        ("ode8+regen", "ODE last 4 + regeneration", PALETTE["blue_main"], "..", ode8),
    ]
    return [(lab, src["rows"][key], c, h) for key, lab, c, h, src in spec if src and key in src["rows"]], fm, ode8


def plot_ladder():
    rows, fm, ode = rows_table()
    metrics = [(lambda r: 100 * r["test"]["real_acc"], r"ReaL accuracy (%) $\uparrow$"),
               (lambda r: r["test"]["nll"], r"test NLL $\downarrow$"),
               (lambda r: 100 * r["test"]["ece"], r"test ECE (%) $\downarrow$"),
               (lambda r: r["shift"]["nll"], r"shift NLL $\downarrow$"),
               (lambda r: r["ood_auroc"]["mean"], r"OOD AUROC (mean) $\uparrow$")]
    fig = plt.figure(figsize=(36, 6.6))
    for i, (f, ylab) in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics) + 1, i + 1)
        vals = [f(r) for _, r, _, _ in rows]
        for j, ((lab, r, c, h), v) in enumerate(zip(rows, vals)):
            ax.bar(j, v, color=c, edgecolor="black", linewidth=2, hatch=h, label=lab)
        ml = None
        if "OOD" in ylab and fm and "fm+multilayer_mahalanobis" in fm["info"]:
            ml = float(np.mean([fm["info"]["fm+multilayer_mahalanobis"][o] for o, _ in OOD]))
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
    ax.legend(handles, labels, fontsize=17, loc="center")
    ax.set_axis_off()
    finalize_figure(fig, "fig_in_ladder")


def plot_breakdown():
    rows, fm, _ = rows_table()
    keep = [r for r in rows if r[0] in ("FM (ViT-B/16)", "FM + temperature scaling", "FM + dropout injection",
                                         "FM + Brownian SDE (rung 3)", "FM + regeneration (rung 4)",
                                         "3-checkpoint ensemble", "ODE last 4 + regeneration")]
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))
    width = 0.8 / len(keep)
    for ax, groups, getter, ylab in (
            (axes[0], CORR, lambda r, k: r["shift_per_corruption"][k]["nll"], r"NLL per corruption $\downarrow$"),
            (axes[1], OOD, lambda r, k: r["ood_auroc"][k], r"OOD AUROC per set $\uparrow$")):
        x = np.arange(len(groups))
        for j, (lab, r, c, h) in enumerate(keep):
            ax.bar(x + (j - (len(keep) - 1) / 2) * width, [getter(r, k) for k, _ in groups], width, color=c,
                   edgecolor="black", linewidth=1.5, hatch=h, label=lab)
        if ylab.startswith("OOD") and fm:
            ml = fm["info"].get("fm+multilayer_mahalanobis", {})
            for i, (k, _) in enumerate(groups):
                if k in ml:
                    ax.plot([i - 0.42, i + 0.42], [ml[k], ml[k]], color=PALETTE["dark"], lw=2.5, ls=":")
            ax.set_ylim(0.4, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([g for _, g in groups], fontsize=17)
        ax.set_ylabel(ylab, fontsize=19)
        tick_style(ax)
    axes[1].plot([], [], color=PALETTE["dark"], lw=2.5, ls=":", label="multi-layer Mahalanobis score")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=15, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=2, rect=(0, 0.14, 1, 1))
    finalize_figure(fig, "fig_in_breakdown", pad=None)


def plot_compile():
    tags = [("ode8", "last 4 blocks", PALETTE["blue_main"]), ("ode", "all 12 blocks", PALETTE["red_strong"])]
    logs = {t: load(f"compile_train_log_{t}.json") for t, _, _ in tags}
    if not any(logs.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(24, 5.8))
    for t, lab, c in tags:
        log = logs[t]
        if not log:
            continue
        axes[0].semilogy([r["step"] for r in log], [r["fm"] for r in log], color=c, lw=3, label=f"{lab}: flow matching")
        axes[0].semilogy([r["step"] for r in log], [r["kl"] for r in log], color=c, lw=3, ls="--",
                         label=f"{lab}: rollout KL to FM")
        pts = [r for r in log if "agree" in r]
        axes[1].plot([r["step"] for r in pts], [100 * r["agree"] for r in pts], color=c, marker="o", markersize=10,
                     lw=3, label=lab)
        sweep = load(f"compile_step_sweep_{t}.json")
        if sweep:
            ns = sorted(int(k) for k in sweep)
            r_blocks = 12 if t == "ode" else 4
            axes[2].plot([n / r_blocks for n in ns], [100 * sweep[str(n)]["agree"] for n in ns], color=c, marker="o",
                         markersize=10, lw=3, label=lab)
    axes[0].set_xlabel("training step", fontsize=18)
    axes[0].set_ylabel("loss", fontsize=18)
    axes[0].legend(fontsize=13)
    axes[1].set_xlabel("training step", fontsize=18)
    axes[1].set_ylabel("argmax agreement with FM (%)", fontsize=18)
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=14)
    axes[2].set_xscale("log", base=2)
    axes[2].axvline(1, color=PALETTE["grey"], lw=2, ls=":")
    axes[2].set_xlabel("Euler steps per replaced block", fontsize=18)
    axes[2].set_ylabel("argmax agreement with FM (%)", fontsize=18)
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=14)
    for ax in axes:
        tick_style(ax)
    finalize_figure(fig, "fig_in_compile")


if __name__ == "__main__":
    apply_publication_style()
    plot_ladder()
    plot_breakdown()
    plot_compile()


def plot_fidelity_options():
    """Compile-fidelity options on ImageNet: control, per-block adapters, more data, both."""
    arms = [("", "ladder field", PALETTE["neutral"], None), ("_cont", "+6k steps (control)", PALETTE["grey"], None),
            ("_adapt", "+ per-block adapters", PALETTE["blue_secondary"], None),
            ("_data", "+ 3.2x data, 15k steps", PALETTE["green_3"], None),
            ("_both", "adapters + data", PALETTE["blue_main"], "//")]
    metrics = [("agree", "top-1 agreement (%)", 100), ("top5_agree", "top-5 agreement (%)", 100),
               ("acc", "accuracy of the flow (%)", 100), ("kl", r"KL$(p_{\mathrm{FM}}\,\|\,q)$ (nats)", 1)]
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    for ax, (key, ylab, scale) in zip(axes, metrics):
        width = 0.16
        for j, (suf, lab, c, h) in enumerate(arms):
            vals = []
            for base in ("ode8", "ode"):
                r = load(f"compile_final_{base}{suf}.json")
                vals.append(scale * r[key] if r else np.nan)
            ax.bar(np.arange(2) + (j - 2) * width, vals, width, color=c, edgecolor="black", linewidth=1.5, hatch=h,
                   label=lab)
        if key == "acc":
            ax.axhline(81.68, color=PALETTE["dark"], lw=2.5, ls=":")
            ax.text(1.45, 82.5, "FM 81.7%", fontsize=14, ha="right")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["last 4 blocks", "all 12 blocks"], fontsize=17)
        ax.set_ylabel(ylab, fontsize=18)
        tick_style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=15, loc="lower center", ncols=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=2, rect=(0, 0.1, 1, 1))
    finalize_figure(fig, "fig_in_fidelity_options", pad=None)


if __name__ == "__main__":
    apply_publication_style()
    plot_fidelity_options()
