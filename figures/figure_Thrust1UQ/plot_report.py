"""Figures for the Thrust 1 progress report, in the figures4papers house style.

All numbers are read from the pilot experiments in uq/experiments/results
(5 splits, mean and std). Run from this folder:
    python plot_report.py
Outputs go to ./figures/ (png at 300 dpi and pdf).
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "uq", "experiments"))
RES = os.path.join(EXP, "results")
OUT = os.path.join(HERE, "figures")

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "grey": "#767676", "dark": "#272727",
    "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
}
OOD_SETS = ["heldout_9", "rotated", "noise"]


def apply_publication_style(font_size=16, axes_linewidth=2.5):
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    })


def finalize_figure(fig, name, pad=2):
    os.makedirs(OUT, exist_ok=True)
    if pad is not None:
        fig.tight_layout(pad=pad)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("saved", name)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def tick_style(ax, size=16):
    ax.tick_params(labelsize=size, length=6, width=1.5)


# ------------------------------------------------------------------ pipeline
def plot_pipeline():
    fig, ax = plt.subplots(figsize=(26, 6.6))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 6.6)
    ax.set_axis_off()
    stages = [
        ("1. Compile the FM", "kept from Thrust 1",
         [r"$x_{t+1}=x_t+F(x_t,t)\ \Rightarrow\ \dot{x}=F_\theta(x,t)$",
          r"$\|x_\theta(1)-x(1)\|\leq \frac{\eta}{L}\,(e^{L}-1)$",
          "(Gronwall fidelity certificate)"]),
        ("2. Layer distances", "new",
         [r"$d_l^2(x)=\min_c\,(x-\mu_c)^\top\Sigma_l^{-1}(x-\mu_c)$",
          r"$q_l=$ split-conformal $(1-\alpha)$",
          r"quantile of $d_l^2$ on held-out data"]),
        ("3. Regeneration scaffold", "replaces Eq. (5) and (8)",
         [r"$a_l(x)=a_0+(1-a_0)(1-e^{-\beta(d_l^2/q_l-1)_+})$",
          r"$K_l(x,\cdot)=(1-a_l)\,\delta_x+a_l\,\frac{a_l\,p_l}{\mathbb{E}_{p_l}[a_l]}$",
          r"$p_l K_l=p_l$ exactly, no score model"]),
        ("4. Closed-form predictive", "replaces Eq. (6)",
         [r"$q(y|x)=S_L\,p(y|x)+\sum_l S_{l-1}\,a_l\,m_l(y)$",
          r"$S_l=\prod_{k\leq l}(1-a_k)$",
          "one forward pass, no sampling"]),
        ("5. Certificates", "replaces Eq. (9)",
         [r"$P(\mathrm{flip})\leq L\alpha+P(\Delta\leq J/(1-J))$",
          r"$R\leq\hat R+B\sqrt{\log(2G/\delta)/2n}$",
          "(conformal + finite-class Hoeffding)"]),
    ]
    w, h, gap, y0 = 4.7, 4.9, 0.55, 0.8
    for i, (title, tag, lines) in enumerate(stages):
        x0 = 0.25 + i * (w + gap)
        kept = i == 0
        face = PALETTE["green_1"] if kept else "#E7EEF7"
        edge = PALETTE["green_3"] if kept else PALETTE["blue_main"]
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                    facecolor=face, edgecolor=edge, linewidth=3))
        ax.text(x0 + w / 2, y0 + h - 0.55, title, ha="center", va="center", fontsize=21, fontweight="bold",
                color=PALETTE["dark"])
        ax.text(x0 + w / 2, y0 + h - 1.15, tag, ha="center", va="center", fontsize=15, style="italic",
                color=PALETTE["grey"] if kept else PALETTE["blue_secondary"])
        for j, line in enumerate(lines):
            ax.text(x0 + w / 2, y0 + h - 2.05 - j * 0.95, line, ha="center", va="center", fontsize=15.5,
                    color=PALETTE["dark"])
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.04, y0 + h / 2), (x0 + w + gap - 0.04, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=28, linewidth=3,
                                         color=PALETTE["grey"]))
    finalize_figure(fig, "fig_pipeline", pad=0.3)


# ------------------------------------------------------------------ variance budget (analytic OU)
def plot_budget():
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    grid = np.linspace(-4, 4, 600)
    gauss = lambda m, v: np.exp(-(grid - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    x0 = 1.8
    ax.plot(grid, gauss(0, 1), color=PALETTE["grey"], lw=3, ls="--", label=r"marginal $p_1=\mathcal{N}(0,1)$ for every $\sigma$")
    for s, c in ((0.5, PALETTE["green_3"]), (1.0, PALETTE["blue_secondary"]), (2.0, PALETTE["blue_main"])):
        ax.plot(grid, gauss(np.exp(-s ** 2 / 2) * x0, 1 - np.exp(-s ** 2)), color=c, lw=3,
                label=rf"$p(x(1)\,|\,x_0)$, $\sigma={s:g}$")
    ax.axvline(x0, color=PALETTE["dark"], lw=2, ls=":")
    ax.text(x0 + 0.08, 0.95, "ODE output", fontsize=15, color=PALETTE["dark"])
    ax.set_xlabel(r"$x(1)$ for input $x_0=1.8$", fontsize=18)
    ax.set_ylabel("density", fontsize=18)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=14, loc="upper left")
    tick_style(ax)

    ax = axes[1]
    s = np.linspace(0, 3, 300)
    explained = np.exp(-s ** 2)
    ax.fill_between(s, 0, explained, color=PALETTE["blue_secondary"], alpha=0.85, lw=0)
    ax.fill_between(s, explained, 1, color=PALETTE["red_2"], alpha=0.9, lw=0)
    ax.text(0.15, 0.2, r"$\mathrm{Var}\,\mathbb{E}[x(1)|x_0]$" + "\nexplained by input", fontsize=16, color="white")
    ax.text(1.35, 0.6, r"$\mathbb{E}\,\mathrm{Var}[x(1)|x_0]$" + "\nper-input uncertainty", fontsize=16, color=PALETTE["dark"])
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"diffusion scale $\sigma$", fontsize=18)
    ax.set_ylabel(r"share of $\mathrm{Var}[x(1)]=1$", fontsize=18)
    tick_style(ax)
    finalize_figure(fig, "fig_budget")


# ------------------------------------------------------------------ Exp 0
def plot_exp0():
    s = load("exp0_variance_budget.json")["summary"]
    keys = ["block1", "block2", "block3", "block4", "z", "logits", "logits_ensemble"]
    labels = ["block 1", "block 2", "block 3", "block 4", "$z$", "logits\n(MC drop.)", "logits\n(ensemble)"]
    styles = [("ID", PALETTE["blue_main"], "o", "ID test"), ("heldout_9", PALETTE["red_strong"], "s", "held-out digit 9"),
              ("rotated", PALETTE["red_2"], "^", "rotated digits"), ("noise", PALETTE["grey"], "D", "uniform noise")]
    fig, ax = plt.subplots(figsize=(14, 5.6))
    x = np.arange(len(keys))
    for j, (k, c, mk, lab) in enumerate(styles):
        mu = np.array([s[k][q]["rho"][0] for q in keys])
        sd = np.array([s[k][q]["rho"][1] for q in keys])
        ax.errorbar(x + (j - 1.5) * 0.07, mu, yerr=sd, color=c, marker=mk, markersize=10, linewidth=3, capsize=4,
                    label=lab)
    ax.axhline(0, color=PALETTE["dark"], lw=1.5, ls=":")
    ax.axhline(1, color=PALETTE["dark"], lw=1.5, ls=":")
    ax.text(1.01, 0, "same\nmarginal", fontsize=14, va="center", transform=ax.get_yaxis_transform())
    ax.text(1.01, 1, "pure added\nvariance", fontsize=14, va="center", transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel(r"$\rho=\mathrm{tr}(\Sigma_{\mathrm{mc}}-\Sigma_{\mathrm{det}})\,/\,\mathrm{tr}\,E$", fontsize=18)
    ax.set_ylim(-1.6, 1.25)
    ax.legend(fontsize=15, ncols=2, loc="lower left")
    tick_style(ax)
    finalize_figure(fig, "fig_exp0_rho")


# ------------------------------------------------------------------ drift correction
def plot_drift():
    sys.path.insert(0, EXP)
    from q2_drift_correction import simulate, T_END
    sims = {"Eq. (5) as written": (simulate(False), PALETTE["red_strong"]),
            r"with $+\frac{1}{2}\nabla\cdot D$": (simulate(True), PALETTE["blue_main"])}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    grid = np.linspace(-4, 4, 400)
    ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color=PALETTE["grey"], lw=3.5, ls="--",
            label=r"target $p_t=\mathcal{N}(0,1)$")
    bins = np.linspace(-4, 4, 81)
    for name, (snaps, c) in sims.items():
        xs = snaps[T_END]
        hist, e = np.histogram(xs, bins=bins, density=True)
        ax.step(0.5 * (e[1:] + e[:-1]), hist, where="mid", color=c, lw=3, label=f"{name}: Var = {xs.var():.2f}")
    ax.set_xlabel(rf"$x(t)$ at $t={T_END:g}$", fontsize=18)
    ax.set_ylabel("density", fontsize=18)
    ax.legend(fontsize=14, loc="upper left")
    tick_style(ax)
    finalize_figure(fig, "fig_drift")


# ------------------------------------------------------------------ Eq. (6) cost
def plot_cost():
    r = load("q3_eq6_cost.json")
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
    ax = axes[0]
    for K, c in zip((2, 4, 8), (PALETTE["green_3"], PALETTE["blue_secondary"], PALETTE["blue_main"])):
        rows = [x for x in r["tree"] if x["d"] == 1 and x["K"] == K]
        ax.loglog([x["evals"] for x in rows], [x["err_vs_discrete"] for x in rows], color=c, marker="o",
                  markersize=9, lw=3, label=rf"quadrature tree, $1/\epsilon={K}$")
    rows = [x for x in r["mc"] if x["d"] == 1 and x["K"] == 8]
    ax.loglog([x["evals"] for x in rows], [x["rmse_vs_discrete"] for x in rows], color=PALETTE["red_strong"],
              marker="s", markersize=9, lw=3, ls="--", label=r"Monte Carlo paths, $1/\epsilon=8$")
    bias = rows[0]["disc_bias"]
    ax.axhline(bias, color=PALETTE["grey"], lw=2.5, ls=":")
    ax.text(12, bias * 1.4, r"time-discretization bias at $1/\epsilon=8$", fontsize=13, color=PALETTE["grey"])
    ax.set_xlabel("kernel evaluations", fontsize=18)
    ax.set_ylabel("error vs exact discrete chain", fontsize=18)
    ax.legend(fontsize=13, loc="lower left")
    tick_style(ax)

    ax = axes[1]
    Ks = np.arange(2, 21)
    for d, c in ((1, PALETTE["green_3"]), (2, PALETTE["blue_secondary"]), (64, PALETTE["blue_main"]),
                 (768, PALETTE["dark"])):
        ax.plot(Ks, d * Ks * np.log10(3.0), color=c, lw=3, label=rf"tree, $N=3$, $d={d}$")
    ax.plot(Ks, np.log10(1000 * Ks), color=PALETTE["red_strong"], lw=3, ls="--", label=r"Monte Carlo, $M=10^3$, any $d$")
    ax.set_yscale("log")
    ax.set_xlabel(r"steps $1/\epsilon$", fontsize=18)
    ax.set_ylabel(r"$\log_{10}$ kernel evaluations", fontsize=18)
    ax.legend(fontsize=13, loc="center right")
    tick_style(ax)
    finalize_figure(fig, "fig_cost")


# ------------------------------------------------------------------ Eq. (8)
def plot_eq8():
    s = load("q4_eq8_penalty.json")["summary"]
    names = ["T_star", "inv_curv", "score_norm", "entropy", "mahalanobis"]
    labels = [r"$T^\star$ of Eq. (8)", r"$1/c$ (loss curvature)", r"$\|\nabla\log p\|$", "softmax entropy",
              "Mahalanobis distance"]
    colors = [PALETTE["red_strong"], PALETTE["red_2"], PALETTE["blue_secondary"], PALETTE["neutral"], PALETTE["blue_main"]]
    tasks = ["heldout_9", "rotated", "noise", "misclassified"]
    tlabels = ["held-out 9", "rotated", "noise", "misclassified ID"]
    fig = plt.figure(figsize=(20, 5.6))
    ax = fig.add_subplot(1, 4, (1, 3))
    width = 0.16
    x = np.arange(len(tasks))
    for i, (n, c, lab) in enumerate(zip(names, colors, labels)):
        ax.bar(x + (i - 2) * width, [s[n][t][0] for t in tasks], width, yerr=[s[n][t][1] for t in tasks],
               color=c, edgecolor="black", linewidth=1.5, capsize=4, label=lab)
    ax.axhline(0.5, color=PALETTE["dark"], lw=2, ls=":")
    ax.text(len(tasks) - 0.55, 0.52, "chance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tlabels, fontsize=17)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"AUROC (higher $=$ more variance)", fontsize=18)
    tick_style(ax)
    handles, lbls = ax.get_legend_handles_labels()
    ax = fig.add_subplot(1, 4, 4)
    ax.legend(handles, lbls, fontsize=15, loc="center")
    ax.set_axis_off()
    finalize_figure(fig, "fig_eq8")


# ------------------------------------------------------------------ Exp 1 Brownian family sweep
def plot_brownian():
    s = load("exp1_summary.json")
    cfgs, base = s["configs"], s["baselines"]
    lines = [("('final', 'iso', 5, True,", PALETTE["neutral"], "o", r"$z$ only, isotropic $D$"),
             ("('final', 'precond', 5, True,", PALETTE["red_2"], "s", r"$z$ only, preconditioned $D$"),
             ("('all', 'precond', 5, True,", PALETTE["red_strong"], "D", r"every block, preconditioned $D$")]
    panels = [("shift.nll", "NLL under pixel-noise shift", ("ts", "mcd")),
              ("ood_entropy.mean", "OOD AUROC (mean of 3 sets)", ("mcd", "ensemble")),
              ("flip", "argmax flips vs FM", ())]
    bl = {"ts": (PALETTE["blue_secondary"], "temperature scaling"), "mcd": (PALETTE["green_3"], "MC dropout"),
          "ensemble": (PALETTE["dark"], "deep ensemble (5 models)")}
    fig = plt.figure(figsize=(24, 5.6))
    handles = {}
    for i, (key, ylab, bls) in enumerate(panels):
        ax = fig.add_subplot(1, 4, i + 1)
        for prefix, c, mk, lab in lines:
            ks = sorted([k for k in cfgs if k.startswith(prefix)], key=lambda k: float(k.split(",")[-1].strip(" )")))
            h = np.array([float(k.split(",")[-1].strip(" )")) for k in ks])
            mu = np.array([cfgs[k][key][0] for k in ks])
            sd = np.array([cfgs[k][key][1] for k in ks])
            (ln,) = ax.plot(h, mu, color=c, marker=mk, markersize=8, lw=3, label=lab)
            ax.fill_between(h, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
            handles[lab] = ln
        for b in bls:
            c, lab = bl[b]
            handles[lab] = ax.axhline(base[b][key][0], color=c, lw=3, ls="--", label=lab)
        ax.set_xscale("log")
        ax.set_xlabel(r"Langevin step size $h$", fontsize=18)
        ax.set_ylabel(ylab, fontsize=18)
        tick_style(ax)
        if key == "flip":
            ax.set_ylim(-0.002, 0.03)
    ax = fig.add_subplot(1, 4, 4)
    ax.legend(list(handles.values()), list(handles.keys()), fontsize=15, loc="center")
    ax.set_axis_off()
    finalize_figure(fig, "fig_brownian")


# ------------------------------------------------------------------ main comparison
def method_table():
    e1 = load("exp1_summary.json")
    b, sel = e1["baselines"], e1["selected_on_val"]["all blocks"]
    e3 = load("exp3_regeneration.json")

    def ms(vals):
        return float(np.mean(vals)), float(np.std(vals))

    def prop(name):
        S = [r[f"selected_{name}"] for r in e3]
        return {"test.ece": ms([100 * s["test"]["ece"] for s in S]),
                "test.nll": ms([s["test"]["nll"] for s in S]),
                "test.acc": ms([100 * s["test"]["acc"] for s in S]),
                "shift.nll": ms([s["shift"]["nll"] for s in S]),
                "shift.ece": ms([100 * s["shift"]["ece"] for s in S]),
                "ood": ms([np.mean([s["ood_entropy"][o] for o in OOD_SETS]) for s in S])}

    def from_summary(d):
        return {"test.ece": (100 * d["test.ece"][0], 100 * d["test.ece"][1]), "test.nll": tuple(d["test.nll"]),
                "test.acc": (100 * d["test.acc"][0], 100 * d["test.acc"][1]),
                "shift.nll": tuple(d["shift.nll"]),
                "shift.ece": (100 * d["shift.ece"][0], 100 * d["shift.ece"][1]),
                "ood": tuple(d["ood_entropy.mean"])}

    rows = [
        ("FM (deterministic)", from_summary(b["det"]), PALETTE["neutral"], None),
        ("Temperature scaling", from_summary(b["ts"]), PALETTE["red_1"], None),
        ("MC dropout (32 passes)", from_summary(b["mcd"]), PALETTE["red_2"], None),
        ("Brownian SDE, Eq. (5) family", from_summary(sel), PALETTE["red_strong"], None),
        ("Ours, $z$ only", prop("z only"), PALETTE["blue_secondary"], None),
        ("Ours, all layers", prop("all"), PALETTE["blue_main"], None),
        ("Deep ensemble (5 models)", from_summary(b["ensemble"]), PALETTE["grey"], "//"),
    ]
    ml = [np.mean([r["multilayer_mahalanobis_auroc"][o] for o in OOD_SETS]) for r in e3]
    return rows, float(np.mean(ml))


def plot_main():
    rows, ml_auroc = method_table()
    metrics = [("test.ece", r"ID ECE (%) $\downarrow$", None), ("shift.nll", r"shift NLL $\downarrow$", None),
               ("shift.ece", r"shift ECE (%) $\downarrow$", None), ("ood", r"OOD AUROC $\uparrow$", (0.85, 1.0))]
    fig = plt.figure(figsize=(30, 6.4))
    for i, (key, ylab, ylim) in enumerate(metrics):
        ax = fig.add_subplot(1, 5, i + 1)
        for j, (name, d, c, hatch) in enumerate(rows):
            ax.bar(j, d[key][0], yerr=d[key][1], color=c, edgecolor="black", linewidth=2, capsize=5, hatch=hatch,
                   label=name)
        if key == "ood":
            ax.axhline(ml_auroc, color=PALETTE["dark"], lw=2.5, ls=":")
            ax.text(-0.5, ml_auroc + 0.004, "multi-layer Mahalanobis score alone", fontsize=14)
        ax.set_xticks([])
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylab, fontsize=22)
        tick_style(ax, 18)
        handles, labels = ax.get_legend_handles_labels()
    ax = fig.add_subplot(1, 5, 5)
    ax.legend(handles, labels, fontsize=18, loc="center")
    ax.set_axis_off()
    finalize_figure(fig, "fig_main")
    return rows, ml_auroc


# ------------------------------------------------------------------ guarantees of the proposed scaffold
def plot_guarantee():
    e3 = load("exp3_regeneration.json")
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    alphas = [0.01, 0.05, 0.1]
    emp, bound = [], []
    for a in alphas:
        g = [x for r in e3 for x in r["grid"] if x["layers"] == "all" and x["alpha"] == a and x["beta"] == 1.0
             and x["a0"] == 0.001]
        emp.append((np.mean([x["id_exceed"] for x in g]), np.std([x["id_exceed"] for x in g])))
        bound.append(np.mean([x["bound_L_alpha"] for x in g]))
    x = np.arange(len(alphas))
    ax.bar(x - 0.2, [e[0] for e in emp], 0.38, yerr=[e[1] for e in emp], color=PALETTE["blue_main"],
           edgecolor="black", linewidth=2, capsize=5, label="observed on ID test")
    ax.bar(x + 0.2, bound, 0.38, color=PALETTE["neutral"], edgecolor="black", linewidth=2, hatch="//",
           label=r"bound $L\alpha$ ($L=5$)")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\alpha={a:g}$" for a in alphas], fontsize=17)
    ax.set_ylabel(r"$P_{\mathrm{ID}}(\exists l:\ d_l^2>q_l)$", fontsize=18)
    ax.legend(fontsize=15, loc="upper left")
    tick_style(ax)

    ax = axes[1]
    sets = [("test", "ID test", PALETTE["blue_main"]), ("shift", "pixel-noise shift", PALETTE["green_3"]),
            ("heldout_9", "held-out 9", PALETTE["red_2"]), ("rotated", "rotated", PALETTE["red_strong"]),
            ("noise", "uniform noise", PALETTE["grey"])]
    data = [np.concatenate([r["selected_all"]["jump_values"][k] for r in e3]) for k, _, _ in sets]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2.5), whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2))
    for patch, (_, _, c) in zip(bp["boxes"], sets):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
        patch.set_linewidth(2)
    for i, (d, (_, _, c)) in enumerate(zip(data, sets)):
        ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.22, 0.22, len(d)), d, s=6,
                   color=c, alpha=0.25, lw=0)
    ax.set_xticks(np.arange(1, len(sets) + 1))
    ax.set_xticklabels([s[1] for s in sets], fontsize=16)
    ax.set_ylabel(r"jump mass $J(x)=1-S_L(x)$", fontsize=18)
    tick_style(ax)
    finalize_figure(fig, "fig_guarantee")


if __name__ == "__main__":
    apply_publication_style()
    plot_pipeline()
    plot_budget()
    plot_exp0()
    plot_eq8()
    plot_cost()
    plot_brownian()
    rows, ml = plot_main()
    plot_guarantee()
    for name, d, _, _ in rows:
        print(f"{name:32s}", "  ".join(f"{k}={v[0]:.3f}±{v[1]:.3f}" for k, v in d.items()))
    print("multi-layer Mahalanobis OOD AUROC", round(ml, 4))
    plot_drift()
