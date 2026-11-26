"""Final figures for the round-2 report to Prof. Nghia, in the style of BlindAgent.pdf (VisBench).

Every figure is drawn at the exact size it takes on the slide (16:9 beamer, \\linewidth = 14 cm), so the
6.5-7.5 pt text is the size that appears in the PDF; nothing is rescaled by \\includegraphics.
Style: DejaVu Sans, thin dark-grey axes, light grid, legend in a thin box above the panels, five fixed hues
(crimson, amber, purple, green, blue) with darker edges, dot / dumbbell / line / heatmap forms instead of bars.

Run: python plot_final.py   -> figures/fig_fn_*.pdf, copied to uq/feedback_slides/figures/
"""
import glob
import json
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
UQ = os.path.abspath(os.path.join(HERE, "..", "..", "uq"))
OUT = [f"{HERE}/figures", f"{UQ}/feedback_slides/figures"]
CM = 1 / 2.54
W = 14.0 * CM            # \linewidth of the metropolis 16:9 frame

# ---------------------------------------------------------------- style (from BlindAgent.pdf)
C = {"crimson": "#D81B60", "amber": "#E5A000", "purple": "#5E35B1", "green": "#2E7D32", "blue": "#1E88E5",
     "teal": "#00796B", "grey": "#8A8A8A", "ink": "#333333", "axis": "#5C5C5C", "grid": "#E3E3E3",
     "box": "#D6EFFC", "hl": "#FBDF59", "light": "#ECEDF2", "line": "#101214"}


def edge(hexc, f=0.7):
    h = hexc.lstrip("#")
    return "#" + "".join(f"{int(int(h[i:i + 2], 16) * f):02x}" for i in (0, 2, 4))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.2, "axes.linewidth": 0.5,
        "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink"], "xtick.color": C["axis"], "ytick.color": C["axis"],
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.titlepad": 3, "axes.labelpad": 1.5, "lines.linewidth": 1.0, "lines.markersize": 3.6,
        "legend.frameon": True, "legend.edgecolor": "#CCCCCC", "legend.fancybox": False,
        "legend.borderpad": 0.35, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2, "legend.labelspacing": 0.25, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "savefig.dpi": 300})


def top_legend(fig, handles, labels, y=1.0, ncol=None):
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(labels),
                     framealpha=1)
    leg.get_frame().set_linewidth(0.5)
    return leg


def save(fig, name):
    for d in OUT:
        os.makedirs(d, exist_ok=True)
        fig.savefig(f"{d}/{name}.pdf")
    fig.savefig(f"{OUT[0]}/{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def mk(color, hollow=False, marker="o", ms=3.8, mew=0.6):
    return dict(marker=marker, ms=ms, mfc="white" if hollow else color, mec=color if hollow else edge(color),
                mew=mew, ls="none")


def handle(color, hollow=False, marker="o", ls="none", lw=1.0):
    return Line2D([], [], color=color, lw=lw, ls=ls, **{k: v for k, v in mk(color, hollow, marker).items() if k != "ls"})


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def mean_se(v):
    v = np.asarray([x for x in v if x == x], float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


# ---------------------------------------------------------------- data
VISION = [("ImageNet", f"{UQ}/imagenet_ladder/results"), ("CIFAR-100", f"{UQ}/cifar_ladder/results_cifar100"),
          ("CIFAR-10", f"{UQ}/cifar_ladder/results_cifar10")]
NAME = {"cola": "CoLA", "imdb": "IMDB", "sst2": "SST-2", "imagenet": "ImageNet", "cifar100": "CIFAR-100",
        "cifar10": "CIFAR-10"}
OOD_DIRS = {"cola": sorted(glob.glob(f"{UQ}/nlp_ladder/results_cola/seed*")),
            "imdb": sorted(glob.glob(f"{UQ}/nlp_ladder/results_imdb/seed*")),
            "sst2": sorted(glob.glob(f"{UQ}/nlp_ladder/results_sst2/seed*")),
            "imagenet": [f"{UQ}/imagenet_ladder/results"], "cifar100": [f"{UQ}/cifar_ladder/results_cifar100"],
            "cifar10": [f"{UQ}/cifar_ladder/results_cifar10"]}
SUMM = jload(f"{UQ}/director_comparison/summary.json")


def vision_runs():
    return [(n, r) for n, r in ((n, jload(f"{d}/conformal_scaffold_fm.json")) for n, d in VISION) if r]


def ood_runs(lad):
    return [r for r in (jload(f"{d}/conformal_ood.json") for d in OOD_DIRS[lad]) if r]


# ================================================================ pipeline figures (style of VisBench Fig. 1)
MONO = "DejaVu Sans Mono"


def rbox(ax, x, y, w, h, fc="white", ec=C["line"], lw=0.8, r=0.012, ls="-", z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                       zorder=z, transform=ax.transAxes)
    ax.add_patch(p)
    return p


def num(ax, x, y, k, title):
    fw, fh = ax.figure.get_size_inches()
    r = 0.012
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * fw / fh, transform=ax.transAxes, fc=C["line"], ec="none", zorder=3))
    ax.text(x, y, str(k), transform=ax.transAxes, ha="center", va="center", color="white", fontsize=5.6,
            fontweight="bold", family=MONO, zorder=4)
    ax.text(x + 0.02, y, title, transform=ax.transAxes, ha="left", va="center", fontsize=6.6, fontweight="bold",
            family=MONO)


def arrow(ax, x0, y0, x1, y1, lw=0.8, color=C["line"]):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=5,
                                 lw=lw, color=color, zorder=2, shrinkA=0, shrinkB=0))


def txt(ax, x, y, s, size=5.8, mono=True, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=size, family=MONO if mono else "DejaVu Sans", **kw)


def tag(ax, x, y, s, fc=C["hl"], size=5.4):
    return ax.text(x, y, s, transform=ax.transAxes, ha="center", va="center", fontsize=size, family=MONO,
                   fontweight="bold", zorder=5,
                   bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.25", fc=fc, ec=C["line"], lw=0.6))


def plot_pipeline_method():
    """Input -> layer distance -> group conformal p-value -> input-specific jump rate -> closed-form predictive."""
    fig = plt.figure(figsize=(W, 4.1 * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    pw, gap, x0, top, bot = 0.228, 0.019, 0.006, 0.86, 0.035
    xs = [x0 + i * (pw + gap) for i in range(4)]
    titles = ["Layer distance", "Group p-value", "Jump rate", "Predictive"]
    for i, x in enumerate(xs):
        num(ax, x + 0.012, 0.94, i + 1, titles[i])
        rbox(ax, x, bot, pw, top - bot)
        if i < 3:
            arrow(ax, x + pw + 0.002, (top + bot) / 2, x + pw + gap - 0.002, (top + bot) / 2)
    # 1: CLS states of blocks 2..10 and z -> d_l -> s_theta
    x = xs[0]
    labels = ["b2", "b4", "b6", "b8", "b10", "z"]
    for j, l in enumerate(labels):
        bx = x + 0.012 + j * 0.035
        rbox(ax, bx, 0.62, 0.03, 0.12, fc=C["box"], lw=0.6, r=0.006)
        txt(ax, bx + 0.015, 0.68, l, 5.0)
        arrow(ax, bx + 0.015, 0.615, bx + 0.015, 0.545, lw=0.5)
    txt(ax, x + pw / 2, 0.79, "x → CLS state per block", 5.2)
    rbox(ax, x + 0.012, 0.37, pw - 0.024, 0.16, fc=C["box"], lw=0.7)
    txt(ax, x + pw / 2, 0.45, r"$s_\theta(x)=\sum_l\theta_l\log\,d_l^2(x)/\mathrm{med}_l$", 6.0, mono=False)
    txt(ax, x + pw / 2, 0.28, "class Mahalanobis per layer", 5.0, color=C["ink"])
    tag(ax, x + pw / 2, 0.13, "θ learned on A1")
    # 2: score histogram of A2 points in the group of x
    x = xs[1]
    txt(ax, x + pw / 2, 0.79, "G(x): confidence quintile", 5.0)
    rng = np.random.default_rng(3)
    h, e = np.histogram(rng.gamma(4, 1, 400), bins=16, range=(0, 12))
    h = h / h.max() * 0.2
    for k in range(16):
        bx = x + 0.025 + k * 0.0112
        fc = C["crimson"] if e[k] >= 7.5 else "#B9C6D3"
        ax.add_patch(plt.Rectangle((bx, 0.5), 0.0098, h[k], transform=ax.transAxes, fc=fc, ec="none", zorder=2))
    ax.plot([x + 0.022, x + 0.21], [0.5, 0.5], transform=ax.transAxes, color=C["line"], lw=0.5)
    sx = x + 0.025 + 7.5 / 12 * 16 * 0.0112
    ax.plot([sx, sx], [0.49, 0.73], transform=ax.transAxes, color=C["crimson"], lw=0.8, zorder=3)
    txt(ax, sx + 0.004, 0.735, "s(x)", 5.2, color=C["crimson"], ha="left", va="bottom")
    txt(ax, x + pw / 2, 0.39, r"$\pi_G(x)=\frac{1+\#\{i\in G:\ s_i\geq s(x)\}}{1+n_G}$", 6.2, mono=False)
    txt(ax, x + pw / 2, 0.255, "share of A2 scores ≥ s(x)", 5.0, color=C["ink"])
    tag(ax, x + pw / 2, 0.13, "calibrated on A2")
    # 3: h(u) curve
    x = xs[2]
    ox, oy, sw, sh = x + 0.035, 0.45, 0.15, 0.3
    ax.plot([ox, ox + sw], [oy, oy], transform=ax.transAxes, color=C["line"], lw=0.5)
    ax.plot([ox, ox], [oy, oy + sh], transform=ax.transAxes, color=C["line"], lw=0.5)
    u = np.linspace(0, 1, 100)
    al = 0.3
    for kap, col in ((0.5, C["blue"]), (2, C["purple"])):
        hv = 1 - np.minimum(u / al, 1) ** kap
        ax.plot(ox + sw * u, oy + sh * 0.92 * hv, transform=ax.transAxes, color=col, lw=0.9)
    ax.plot([ox + sw * al] * 2, [oy - 0.02, oy + 0.02], transform=ax.transAxes, color=C["line"], lw=0.5)
    txt(ax, ox + sw * al, oy - 0.055, "α", 5.4)
    txt(ax, ox + sw, oy - 0.055, "1", 5.0)
    txt(ax, ox + sw + 0.004, oy + 0.02, "π", 5.4, ha="left")
    txt(ax, ox - 0.005, oy + sh * 0.92, "1", 5.0, ha="right")
    txt(ax, ox + 0.075, oy + sh * 0.55, "h(π)", 5.2, ha="left", color=C["ink"])
    txt(ax, x + pw / 2, 0.79, "no extra jump unless π ≤ α", 5.0)
    txt(ax, x + pw / 2, 0.27, r"$a(x)=a_0+(1-a_0)\,h(\pi_G(x))$", 6.0, mono=False)
    tag(ax, x + pw / 2, 0.13, "(α,κ,τ) certified on calB", size=5.0)
    # 4: mixture of p and pool predictive M
    x = xs[3]
    txt(ax, x + pw / 2, 0.79, "mix FM p with pool M", 5.0)
    bars = {"p": [0.62, 0.22, 0.1, 0.06], "M": [0.3, 0.3, 0.22, 0.18], "q": [0.5, 0.25, 0.14, 0.11]}
    for j, (k, v) in enumerate(bars.items()):
        bx = x + 0.022 + j * 0.068
        col = {"p": "#9AA5B1", "M": "#B9C6D3", "q": C["blue"]}[k]
        for t, hv in enumerate(v):
            ax.add_patch(plt.Rectangle((bx + t * 0.0125, 0.47), 0.011, hv * 0.33, transform=ax.transAxes, fc=col,
                                       ec="none", zorder=2))
        ax.plot([bx - 0.003, bx + 0.051], [0.47, 0.47], transform=ax.transAxes, color=C["line"], lw=0.5)
        txt(ax, bx + 0.024, 0.42, k, 5.6, fontweight="bold" if k == "q" else None)
    txt(ax, x + 0.022 + 0.068 - 0.0085, 0.56, "+", 6.5)
    txt(ax, x + 0.022 + 2 * 0.068 - 0.0085, 0.56, "=", 6.5)
    txt(ax, x + pw / 2, 0.29, r"$q=(1-a)\,p+a\,w(x)M$", 6.0, mono=False)
    tag(ax, x + pw / 2, 0.13, "label sets on calC")
    save(fig, "fig_fn_pipeline_method")


def plot_pipeline_option1():
    """Option 1: TS predictive for probabilities, a conformal flag for atypical inputs."""
    fig = plt.figure(figsize=(W, 2.5 * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    hi, lo = 0.70, 0.25
    rbox(ax, 0.004, 0.30, 0.085, 0.36, fc=C["light"])
    txt(ax, 0.0465, 0.53, "input x", 5.8, fontweight="bold")
    txt(ax, 0.0465, 0.40, "image/text", 4.8, color=C["ink"])
    arrow(ax, 0.091, 0.48, 0.111, 0.48)
    rbox(ax, 0.113, 0.22, 0.125, 0.52, fc=C["box"])
    txt(ax, 0.1755, 0.57, "fine-tuned FM", 5.6, fontweight="bold")
    txt(ax, 0.1755, 0.42, "ViT-B/16 or", 4.8, color=C["ink"])
    txt(ax, 0.1755, 0.31, "Qwen2.5-0.5B", 4.8, color=C["ink"])
    arrow(ax, 0.240, 0.62, 0.268, hi)
    arrow(ax, 0.240, 0.34, 0.268, lo)
    stages_hi = [("logits", None), ("temperature T", "fitted on A1"), ("p_T(·|x)", None)]
    stages_lo = [("score s(x)", "multi-layer Maha."), ("p-value π(x)", "per group, on A2"),
                 ("flag: π(x) ≤ α", None)]
    xs, w = [0.27, 0.435, 0.60], 0.14
    for row, y, stages in ((0, hi, stages_hi), (1, lo, stages_lo)):
        for k, ((t, sub), x) in enumerate(zip(stages, xs)):
            last = k == 2
            rbox(ax, x, y - 0.13, w, 0.26, fc=C["hl"] if (last and row == 1) else (C["box"] if last else "white"),
                 lw=1.0 if last else 0.8)
            txt(ax, x + w / 2, y + (0.05 if sub else 0), t, 5.6, fontweight="bold" if last else None)
            if sub:
                txt(ax, x + w / 2, y - 0.06, sub, 4.8, color=C["ink"])
            if k < 2:
                arrow(ax, x + w + 0.002, y, xs[k + 1] - 0.002, y)
    for y, a, b in ((hi, "probabilities", "keeps accuracy, ID calibration"),
                    (lo, "atypical-input flag", "P(flag | ID, group) ≤ α")):
        arrow(ax, 0.742, y, 0.765, y)
        txt(ax, 0.77, y + 0.06, a, 5.6, ha="left", fontweight="bold")
        txt(ax, 0.77, y - 0.06, b, 4.8, ha="left", color=C["ink"])
    save(fig, "fig_fn_pipeline_option1")


# ================================================================ held-out vs training-feature thresholds (lines)
def plot_train_vs_heldout(dist="err"):
    runs = vision_runs()
    fig, axes = plt.subplots(1, len(runs), figsize=(W, 3.7 * CM))
    for ax, (name, r) in zip(axes, runs):
        tv = r["train_vs_heldout_exceedance"][dist]
        al = 100 * np.array([float(a) for a in tv])
        tr, ho, pac = (100 * np.array([v[k] for v in tv.values()]) for k in ("train_fitted", "heldout_conformal",
                                                                              "pac_bound"))
        ax.fill_between(al, al, pac, color=C["blue"], alpha=0.13, lw=0)
        ax.plot(al, al, color=C["axis"], ls=(0, (3, 2)), lw=0.7)
        ax.plot(al, tr, color=C["crimson"], **{**mk(C["crimson"]), "ls": "-"})
        ax.plot(al, ho, color=C["blue"], **{**mk(C["blue"], marker="s", ms=3.4), "ls": "-"})
        ax.set_xscale("log")
        ax.set_yscale("log")
        t = [1, 2, 5, 10, 20]
        ax.set_xticks(t, [str(v) for v in t])
        ax.set_yticks(t + [30], [str(v) for v in t] + ["30"])
        ax.minorticks_off()
        ax.set_title(name)
        ax.set_xlabel(r"target level $\alpha$ (%)")
        i5 = list(al).index(5.0)
        ax.annotate(f"{tr[i5]:.1f}", (5, tr[i5]), xytext=(-3, 3), textcoords="offset points", ha="right",
                    fontsize=5.6, color=C["ink"])
        ax.annotate(f"{ho[i5]:.1f}", (5, ho[i5]), xytext=(3, -6), textcoords="offset points", ha="left",
                    fontsize=5.6, color=C["ink"])
    axes[0].set_ylabel("test exceedance (%)")
    top_legend(fig, [handle(C["crimson"], ls="-"), handle(C["blue"], marker="s", ls="-"),
                     Line2D([], [], color=C["axis"], ls=(0, (3, 2)), lw=0.7),
                     plt.Rectangle((0, 0), 1, 1, fc=C["blue"], alpha=0.13)],
               ["fitted on FM training features", "split conformal on held-out A2", r"nominal $\alpha$",
                "PAC bound (90%)"])
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.77, wspace=0.22)
    save(fig, "fig_fn_train_vs_heldout")


# ================================================================ group exceedance (dumbbell, VisBench Fig. 3)
def dumbbell(ax, y, a, b, ca, cb, lab=True, fmt="{:.1f}"):
    ax.annotate("", (b, y), (a, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=cb, mutation_scale=5,
                                                    shrinkA=2.2, shrinkB=2.2))
    ax.plot(a, y, **mk(ca, hollow=True))
    ax.plot(b, y, **mk(cb))
    if lab:
        lo, hi = (a, b) if a < b else (b, a)
        ax.text(lo, y, fmt.format(lo) + " ", ha="right", va="center", fontsize=5.4,
                color=C["grey"] if lo == a else C["ink"])
        ax.text(hi, y, " " + fmt.format(hi), ha="left", va="center", fontsize=5.4,
                color=C["grey"] if hi == a else C["ink"])


def plot_group_exceedance(dist="err"):
    runs = vision_runs()
    fig, axes = plt.subplots(1, len(runs), figsize=(W, 3.6 * CM), sharey=True)
    for ax, (name, r) in zip(axes, runs):
        m = 100 * np.array(r["rows"][f"{dist}/conformal_marginal"]["exceed_per_group"])
        g = 100 * np.array(r["rows"][f"{dist}/conformal_mondrian"]["exceed_per_group"])
        for k in range(5):
            dumbbell(ax, k, m[k], g[k], C["crimson"], C["blue"], lab=(k == 0))
        ax.axvline(5, color=C["axis"], ls=(0, (3, 2)), lw=0.7, zorder=0)
        ax.set_xlim(-6, max(m.max(), g.max()) + 8)
        ax.set_ylim(4.6, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(name)
        ax.set_xlabel("test exceedance in the group (%)")
        ax.set_xticks([0, 5, 10, 20, 30] if m.max() > 25 else [0, 5, 10, 20])
    axes[0].set_yticks(range(5), ["Q1 (least conf.)", "Q2", "Q3", "Q4", "Q5 (most conf.)"])
    axes[0].tick_params(axis="y", length=0)
    top_legend(fig, [handle(C["crimson"], hollow=True), handle(C["blue"]),
                     Line2D([], [], color=C["axis"], ls=(0, (3, 2)), lw=0.7)],
               ["marginal threshold", "per-group (Mondrian) threshold", r"target $\alpha=5\%$"])
    fig.subplots_adjust(left=0.135, right=0.99, bottom=0.2, top=0.76, wspace=0.1)
    save(fig, "fig_fn_group_exceedance")


# ================================================================ label sets (lines over quintiles)
def plot_label_sets(alpha="0.1"):
    runs = vision_runs()
    fig, axes = plt.subplots(1, len(runs), figsize=(W, 3.4 * CM))
    q = np.arange(1, 6)
    for ax, (name, r) in zip(axes, runs):
        m = 100 * np.array(r["label_sets"][f"fm/{alpha}/marginal"]["coverage_per_group"])
        g = 100 * np.array(r["label_sets"][f"fm/{alpha}/mondrian"]["coverage_per_group"])
        ax.axhline(100 * (1 - float(alpha)), color=C["axis"], ls=(0, (3, 2)), lw=0.7, zorder=0)
        ax.plot(q, m, color=C["crimson"], **{**mk(C["crimson"], hollow=True), "ls": "-"})
        ax.plot(q, g, color=C["blue"], **{**mk(C["blue"]), "ls": "-"})
        ax.annotate(f"{m[0]:.1f}", (1, m[0]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=5.6,
                    color=C["ink"])
        ax.set_ylim(min(m.min(), g.min()) - 5, 100.5)
        ax.set_xticks(q, ["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_xlim(0.7, 5.3)
        ax.set_title(name)
        ax.set_xlabel("FM-confidence quintile")
    axes[0].set_ylabel("coverage (%)")
    top_legend(fig, [handle(C["crimson"], hollow=True, ls="-"), handle(C["blue"], ls="-"),
                     Line2D([], [], color=C["axis"], ls=(0, (3, 2)), lw=0.7)],
               ["marginal sets", "per-group (Mondrian) sets", "target 90%"])
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.75, wspace=0.2)
    save(fig, "fig_fn_label_sets")


# ================================================================ LTT and the budget bound (scatter)
def plot_ltt(dist="unif"):
    runs = vision_runs()
    fig, axes = plt.subplots(1, len(runs), figsize=(W, 3.9 * CM))
    for ax, (name, r) in zip(axes, runs):
        allc = r["ltt"][dist]["candidates"]
        c = [v for v in allc if not np.isfinite(v["tau"])]
        thr = 0.1 / (2 * len(allc))
        b = 100 * np.array([v["budget_bound"] for v in c])
        j = 100 * np.array([v["calB_jump"] for v in c])
        ok = np.array([v["p_flip"] <= thr and v["p_jump"] <= thr for v in c])
        mx = max(b.max(), j.max()) * 1.08
        ax.fill_between([0, mx], [0, mx], [mx, mx], color=C["crimson"], alpha=0.06, lw=0)
        ax.plot([0, mx], [0, mx], color=C["axis"], ls=(0, (3, 2)), lw=0.7)
        ax.axhline(5, color=C["grey"], ls=":", lw=0.7)
        ax.plot(b[ok], j[ok], **mk(C["blue"]))
        ax.plot(b[~ok], j[~ok], **mk(C["crimson"], hollow=True))
        ax.text(0.04 * mx, 0.93 * mx, "bound violated", fontsize=5.4, color=C["crimson"], va="top")
        ax.text(0.97 * mx, 5, "jump target 5%", fontsize=5.2, color=C["grey"], ha="right", va="bottom")
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_title(f"{name}: {ok.sum()} of {len(c)} certified")
        ax.set_xlabel("Theorem B bound on the jump (%)")
    axes[0].set_ylabel("ID jump on calB (%)")
    top_legend(fig, [handle(C["blue"]), handle(C["crimson"], hollow=True),
                     Line2D([], [], color=C["axis"], ls=(0, (3, 2)), lw=0.7)],
               ["certified by LTT", "not certified", "bound = measured"])
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.77, wspace=0.18)
    save(fig, "fig_fn_ltt_budget")


# ================================================================ like-for-like: change vs FM (dot plot, VisBench Fig. 2a)
METHODS = [("TS", C["amber"], False, "o", "TS"), ("MCD", C["green"], False, "o", "test-time dropout"),
           ("Ensemble", C["purple"], True, "D", "ensemble (not ranked)"),
           ("Scaffold (tau=1)", C["blue"], True, "o", r"scaffold on FM, $\tau=1$"),
           ("TS+scaffold (tau=1)", C["blue"], False, "o", r"TS+scaffold, $\tau=1$"),
           ("TS+scaffold (tau=inf)", C["crimson"], False, "o", r"TS+scaffold, $\tau=\infty$")]


def delta(lad, row, key):
    """mean and s.e. of (row - FM), paired by seed when both have one value per seed."""
    S = SUMM[lad]
    if row not in S or key not in S[row] or key not in S["FM"]:
        return None
    a, f = np.asarray(S[row][key], float), np.asarray(S["FM"][key], float)
    if len(a) == len(f):
        return mean_se(a - f)
    return S and (np.nanmean(a) - np.nanmean(f), 0.0)


def plot_delta(ladders, cols, name, height):
    ladders = [l for l in ladders if l in SUMM]
    fig, axes = plt.subplots(1, len(cols), figsize=(W, height * CM), sharey=True)
    off = np.linspace(-0.3, 0.3, len(METHODS))
    for ax, (key, lab, better) in zip(axes, cols):
        for i, lad in enumerate(ladders):
            k2 = "shift/mcc" if (key == "shift/acc" and lad == "cola") else key
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#F6F7F9", lw=0, zorder=0)
            for o, (row, col, hol, mkr, _) in zip(off, METHODS):
                d = delta(lad, row, k2)
                if d is None:
                    continue
                if d[1] > 0:
                    ax.plot([d[0] - d[1], d[0] + d[1]], [i + o, i + o], color=col, lw=0.7, zorder=2)
                ax.plot(d[0], i + o, zorder=3, **mk(col, hol, mkr, ms=3.3))
        ax.axvline(0, color=C["axis"], lw=0.6, zorder=1)
        ax.set_ylim(len(ladders) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(lab)
        ax.set_xlabel(r"$\Delta$ vs FM, " + ("better $\\rightarrow$" if better > 0 else "$\\leftarrow$ better"), fontsize=6)
    axes[0].set_yticks(range(len(ladders)), [NAME[l] for l in ladders])
    axes[0].tick_params(axis="y", length=0)
    top_legend(fig, [handle(c, h, m) for _, c, h, m, _ in METHODS], [l for *_, l in METHODS], ncol=3)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.25 * 3.6 / height, top=1 - 1.25 / height, wspace=0.1)
    save(fig, name)


DELTA_COLS = [("id/ece", r"ECE (%)", -1), ("id/nll", r"NLL $\times10$", -1), ("id/auroc", "failure AUROC (%)", 1),
              ("shift/nll", r"shift NLL $\times10$", -1), ("ood_entropy/auroc", "OOD AUROC (%)", 1)]
DELTA_COLS_V = [("id/ece", r"ECE (%)", -1), ("id/nll", r"NLL $\times10$", -1), ("shift/acc", "shift accuracy (%)", 1),
                ("shift/nll", r"shift NLL $\times10$", -1), ("ood_entropy/auroc", "OOD AUROC (%)", 1)]


# ================================================================ OOD by entropy: TS -> TS+scaffold (arrows)
def plot_ood_compare():
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    groups = [[l for l in ("cola", "imdb", "sst2") if l in SUMM], [l for l in ("imagenet", "cifar100", "cifar10") if l in SUMM]]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.3 * CM))
    for ax, ladders, title in zip(axes, groups, ["text, Qwen2.5-0.5B (5 seeds)", "vision, ViT-B/16"]):
        lo, hi = 100, 0
        for i, lad in enumerate(ladders):
            S = SUMM[lad]
            g = lambda row: mean_se(S[row]["ood_entropy/auroc"]) if row in S else None
            ts, t1, ti, en = g("TS"), g("TS+scaffold (tau=1)"), g("TS+scaffold (tau=inf)"), g("Ensemble")
            runs = ood_runs(lad)
            dist = np.mean([np.mean(list(r["scores"]["maha_multi_unif"]["auroc"].values())) for r in runs]) if runs else None
            if dist:
                ax.plot([dist, dist], [i - 0.38, i + 0.38], color=C["line"], lw=1.1, solid_capstyle="butt", zorder=1)
            for y, tgt, col in ((i - 0.17, t1, C["blue"]), (i + 0.17, ti, C["crimson"])):
                ax.annotate("", (tgt[0], y), (ts[0], y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=col,
                                                                        mutation_scale=5, shrinkA=2.2, shrinkB=2.2))
                ax.plot(ts[0], y, zorder=3, **mk(C["amber"], hollow=True))
                if tgt[1] > 0:
                    ax.plot([tgt[0] - tgt[1], tgt[0] + tgt[1]], [y, y], color=col, lw=0.7, zorder=2)
                ax.plot(tgt[0], y, zorder=3, **mk(col))
                ax.text(tgt[0], y, f" {tgt[0]:.1f}" if tgt[0] >= ts[0] else f"{tgt[0]:.1f} ", va="center",
                        ha="left" if tgt[0] >= ts[0] else "right", fontsize=5.3, color=C["ink"], path_effects=halo,
                        zorder=4)
            ax.text(ts[0], i, f"{ts[0]:.1f} ", va="center", ha="right", fontsize=5.3, color=C["grey"],
                    path_effects=halo, zorder=4)
            if en:
                ax.plot(en[0], i, zorder=3, **mk(C["purple"], True, "D", ms=3.2))
            if i < len(ladders) - 1:
                ax.axhline(i + 0.5, color=C["grid"], lw=0.5)
            vals = [ts[0], t1[0], ti[0]] + ([dist] if dist else []) + ([en[0]] if en else [])
            lo, hi = min(lo, *vals), max(hi, *vals)
        ax.set_xlim(lo - 0.12 * (hi - lo) - 1.5, min(100.5, hi + 0.08 * (hi - lo) + 1.5))
        ax.set_yticks(range(len(ladders)), [NAME[l] for l in ladders])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(ladders) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    fig.supxlabel("OOD detection AUROC by entropy (%), mean over OOD sets", fontsize=6.5, y=0.01)
    top_legend(fig, [handle(C["amber"], True), handle(C["blue"]), handle(C["crimson"]),
                     handle(C["purple"], True, "D"), Line2D([], [], color=C["line"], lw=1.1)],
               ["TS", r"TS+scaffold, $\tau=1$", r"TS+scaffold, $\tau=\infty$", "ensemble",
                "multi-layer distance alone"])
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.74, wspace=0.17)
    save(fig, "fig_fn_ood_compare")


# ================================================================ Option 1: flag validity (dot plot per dataset)
FLAG_SCORES = [("msp", "MSP"), ("energy", "energy"), ("maha_z", "Maha. $z$"), ("maha_multi_unif", "multi-layer")]


def plot_flag_validity(alpha="0.05"):
    ladders = [l for l in ("cola", "imdb", "sst2", "imagenet", "cifar100", "cifar10") if ood_runs(l)]
    fig, axes = plt.subplots(1, len(ladders), figsize=(W, 3.2 * CM), sharey=True)
    for ax, lad in zip(axes, ladders):
        runs = ood_runs(lad)
        for k, (s, _) in enumerate(FLAG_SCORES):
            wm = np.mean([max(r["scores"][s]["marginal"]["levels"][alpha]["id_far_per_group"]) for r in runs])
            wg = np.mean([max(r["scores"][s]["mondrian"]["levels"][alpha]["id_far_per_group"]) for r in runs])
            mg = np.mean([r["scores"][s]["marginal"]["levels"][alpha]["id_far"] for r in runs])
            ax.annotate("", (wg, k), (wm, k), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C["blue"],
                                                             mutation_scale=5, shrinkA=2.2, shrinkB=2.2))
            ax.plot(wm, k, **mk(C["crimson"], hollow=True))
            ax.plot(wg, k, **mk(C["blue"]))
            ax.plot(mg, k, **mk(C["grey"], marker="D", ms=2.8))
        ax.axvline(100 * float(alpha), color=C["axis"], ls=(0, (3, 2)), lw=0.7, zorder=0)
        ax.set_xlim(0, 36)
        ax.set_xticks([0, 5, 15, 30])
        ax.set_ylim(3.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(NAME[lad])
    axes[0].set_yticks(range(4), [l for _, l in FLAG_SCORES])
    axes[0].tick_params(axis="y", length=0)
    fig.supxlabel(r"ID false alarms at $\alpha=5\%$ (%)", fontsize=6.5, y=0.015)
    top_legend(fig, [handle(C["grey"], marker="D"), handle(C["crimson"], True), handle(C["blue"]),
                     Line2D([], [], color=C["axis"], ls=(0, (3, 2)), lw=0.7)],
               ["marginal, all inputs", "marginal, worst group", "per-group (Mondrian), worst group",
                r"target $\alpha$"])
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.22, top=0.75, wspace=0.1)
    save(fig, "fig_fn_flag_validity")


# ================================================================ Option 1: flag power (heatmap)
def plot_flag_power():
    ladders = [l for l in ("cola", "imdb", "sst2", "imagenet", "cifar100") if ood_runs(l)]
    nice = {"ag_news": "AG News", "newsgroups": "20NG", "noise": "noise", "imagenet_o": "IN-O",
            "dtd": "DTD", "cifar10": "C-10", "svhn": "SVHN", "lsun": "LSUN", "cifar100": "C-100"}
    cols, groups = [], []
    for lad in ladders:
        runs = ood_runs(lad)
        kinds = runs[0]["ood_kind"]
        start = len(cols)
        for o in kinds:
            cols.append((lad, o, kinds[o],
                         [mean_se([r["scores"][s]["auroc"][o] for r in runs])[0] for s, _ in FLAG_SCORES]))
        groups.append((lad, start, len(cols) - 1))
    M = np.array([c[3] for c in cols]).T
    fig, ax = plt.subplots(figsize=(W, 3.4 * CM))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#F4F8FC", "#A9D2F5", "#1E88E5", "#0D47A1"])
    ax.imshow(M, cmap=cmap, vmin=50, vmax=100, aspect="auto")
    best = M.argmax(0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                    color="white" if v > 86 else C["ink"], fontweight="bold" if i == best[j] else None)
    for lad, a, b in groups[:-1]:
        ax.axvline(b + 0.5, color="white", lw=2.2)
    for lad, a, b in groups:
        ax.text((a + b) / 2, -0.85, NAME[lad], ha="center", va="bottom", fontsize=6.4, fontweight="bold")
    ax.set_xticks(range(len(cols)), [f"{nice.get(o, o)}\n{k}" for _, o, k, _ in cols], fontsize=5.3)
    ax.set_yticks(range(len(FLAG_SCORES)), [l for _, l in FLAG_SCORES])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylim(M.shape[0] - 0.5, -0.5)
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.2, top=0.88)
    save(fig, "fig_fn_flag_power")


# ================================================================ Option 1: two flags under shift (dumbbell)
def plot_flag_shift(alpha="0.05"):
    ladders = [l for l in ("imagenet", "cifar100", "cifar10") if ood_runs(l)]
    shown = [("msp", "MSP flag", C["amber"]), ("maha_multi_unif", "distance flag", C["teal"])]
    fig, ax = plt.subplots(figsize=(W * 0.82, 3.0 * CM))
    labels, y = [], 0
    for lad in ladders:
        r = ood_runs(lad)[0]
        for s, lab, col in shown:
            lv = r["scores"][s]["marginal"]["levels"][alpha]
            a, f = lv["shift_acc_accepted"], lv["shift_acc_flagged"]
            ax.plot([f, a], [y, y], color=col, lw=0.8)
            ax.plot(a, y, **mk(col, hollow=True))
            ax.plot(f, y, **mk(col))
            ax.text(103, y, f"{lv['shift_flag']['all']:.0f}% flagged", va="center", fontsize=5.6, color=C["ink"])
            labels.append(f"{NAME[lad]}, {lab}")
            y += 1
        if lad != ladders[-1]:
            ax.axhline(y - 0.5, color=C["grid"], lw=0.5)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("FM accuracy on shifted (corrupted) inputs (%)")
    top_legend(fig, [handle(C["ink"], True), handle(C["ink"])], ["accepted inputs", "flagged inputs"])
    fig.subplots_adjust(left=0.24, right=0.86, bottom=0.22, top=0.8)
    save(fig, "fig_fn_flag_shift")


# ================================================================ full ImageNet compilation (lines over training steps)
def plot_full_compile(prefix, sweep_full, sweep_small, nblocks, name, matched=True):
    res = f"{UQ}/imagenet_ladder/results"
    traj = [(s, jload(f"{res}/compile_final_{prefix}_full_step{s}000.json")) for s in (15, 30, 45)]
    traj += [(60, jload(f"{res}/compile_final_{prefix}_full.json"))]
    traj = [(s, d) for s, d in traj if d]
    small = jload(f"{res}/compile_final_{prefix}.json")
    mid = jload(f"{res}/compile_final_{prefix}_data.json")
    m15 = jload(f"{res}/compile_final_{prefix}_full15k.json") if matched else None
    cols = [("agree", 100, "top-1 agreement (%)"), ("top5_agree", 100, "top-5 agreement (%)"),
            ("kl", 1, r"KL(FM $\|$ flow) $\downarrow$"), ("acc", 100, "accuracy (%)")]
    fig, axes = plt.subplots(1, 5, figsize=(W, 3.1 * CM), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.15]))
    for ax, (key, mult, lab) in zip(axes, cols):
        ax.plot([s for s, _ in traj], [mult * d[key] for _, d in traj], color=C["blue"],
                **{**mk(C["blue"]), "ls": "-"})
        if mid:
            ax.plot(15, mult * mid[key], **mk(C["crimson"], marker="s", ms=3.4))
        if small:
            ax.plot(15, mult * small[key], **mk(C["grey"], marker="^", ms=3.6))
        if m15:
            ax.plot(15, mult * m15[key], **mk(C["amber"], marker="D", ms=3.2))
        last = traj[-1][1][key] * mult
        ax.annotate(f"{last:.1f}" if mult == 100 else f"{last:.2f}", (60, last), xytext=(-3, 3 if key != "kl" else -3),
                    textcoords="offset points", ha="right", va="bottom" if key != "kl" else "top", fontsize=5.6,
                    color=C["ink"])
        if key == "acc":
            ax.axhline(81.68, color=C["line"], ls=(0, (3, 2)), lw=0.7)
            ax.text(13, 81.68, "FM 81.7", fontsize=5.4, va="bottom", color=C["ink"])
        ax.set_xticks([15, 30, 45, 60], ["15k", "30k", "45k", "60k"])
        ax.set_xlim(10, 65)
        ax.set_title(lab)
        ax.margins(y=0.24)
    axes[1].set_xlabel("training steps", labelpad=1)
    axes[1].xaxis.set_label_coords(1.1, -0.2)
    ax = axes[4]
    for sw, col, mkr in ((sweep_full, C["blue"], "o"), (sweep_small, C["grey"], "^")):
        d = jload(f"{res}/{sw}")
        if not d:
            continue
        ks = sorted(d, key=float)
        ax.plot([float(k) / nblocks for k in ks], [100 * d[k]["agree"] for k in ks], color=col,
                **{**mk(col, marker=mkr), "ls": "-"})
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["¼", "½", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_title("top-1 vs Euler steps")
    ax.set_xlabel("steps per block")
    h = [handle(C["blue"], ls="-"), handle(C["crimson"], marker="s"), handle(C["grey"], marker="^")]
    l = ["1.28M images", "318k images", "100k images"]
    if m15:
        h.append(handle(C["amber"], marker="D"))
        l.append("1.28M, 15k-step schedule")
    top_legend(fig, h, l)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.24, top=0.74, wspace=0.42)
    save(fig, name)


if __name__ == "__main__":
    style()
    plot_pipeline_method()
    plot_pipeline_option1()
    plot_train_vs_heldout()
    plot_group_exceedance()
    plot_label_sets()
    plot_ltt()
    plot_delta(["cola", "imdb", "sst2"], DELTA_COLS, "fig_fn_nlp_delta", 3.4)
    plot_delta(["imagenet", "cifar100", "cifar10"], DELTA_COLS_V, "fig_fn_vision_delta", 3.4)
    plot_ood_compare()
    plot_flag_validity()
    plot_flag_power()
    plot_flag_shift()
    plot_full_compile("ode8", "compile_step_sweep_ode8_full.json", "compile_step_sweep_ode8.json", 4,
                      "fig_fn_full_compile")
    plot_full_compile("ode", "compile_step_sweep_ode_full.json", "compile_step_sweep_ode.json", 12,
                      "fig_fn_full_compile_all12", matched=False)
