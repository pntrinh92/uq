"""Figures for the second-round report: held-out calibration, input-level guarantees, NLP ladder, full ImageNet.

Reads uq/*/results*/conformal_scaffold_fm.json, the NLP ladder results (5 seeds per task) and the
full-data compile results. Run from this folder: python plot_feedback.py
"""
import glob
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from plot_report import PALETTE, apply_publication_style, finalize_figure, tick_style

HERE = os.path.dirname(os.path.abspath(__file__))
UQ = os.path.abspath(os.path.join(HERE, "..", "..", "uq"))
VISION = [("CIFAR-10", f"{UQ}/cifar_ladder/results_cifar10"), ("CIFAR-100", f"{UQ}/cifar_ladder/results_cifar100"),
          ("ImageNet", f"{UQ}/imagenet_ladder/results")]
NLP = [("IMDB", "imdb"), ("CoLA", "cola"), ("SST-2", "sst2")]
GROUP_LABELS = ["Q1\n(least\nconfident)", "Q2", "Q3", "Q4", "Q5\n(most\nconfident)"]


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def conformal_runs():
    """(name, result) for every dataset with a conformal scaffold run; NLP tasks use seed 0."""
    out = [(n, jload(f"{d}/conformal_scaffold_fm.json")) for n, d in VISION]
    out += [(n, jload(f"{UQ}/nlp_ladder/results_{t}/seed0/conformal_scaffold_fm.json")) for n, t in NLP]
    return [(n, r) for n, r in out if r]


# ------------------------------------------------------------------ input-level guarantee (A)
def plot_group_exceedance(dist="unif"):
    runs = conformal_runs()
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 6.4), sharey=False)
    axes = np.atleast_1d(axes)
    x = np.arange(5)
    for ax, (name, r) in zip(axes, runs):
        m = r["rows"][f"{dist}/conformal_marginal"]["exceed_per_group"]
        g = r["rows"][f"{dist}/conformal_mondrian"]["exceed_per_group"]
        ax.bar(x - 0.2, 100 * np.array(m), 0.4, color=PALETTE["red_2"], edgecolor="black", linewidth=1.5,
               label="marginal threshold")
        ax.bar(x + 0.2, 100 * np.array(g), 0.4, color=PALETTE["blue_main"], edgecolor="black", linewidth=1.5,
               label="per-group (Mondrian)")
        ax.axhline(5, color=PALETTE["dark"], ls="--", lw=2)
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_title(name, fontsize=20)
        ax.set_xlabel("FM confidence quintile", fontsize=17)
        tick_style(ax)
    axes[0].set_ylabel(r"test exceedance $P(s>\hat b)$ (%)", fontsize=18)
    axes[0].legend(fontsize=15, loc="upper right")
    finalize_figure(fig, f"fig_fb_group_exceedance_{dist}")


# ------------------------------------------------------------------ held-out vs training-feature thresholds
def plot_train_vs_heldout(dist="unif"):
    runs = conformal_runs()
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 5.8))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, runs):
        tv = r["train_vs_heldout_exceedance"][dist]
        al = np.array([float(a) for a in tv])
        tr = np.array([v["train_fitted"] for v in tv.values()])
        ho = np.array([v["heldout_conformal"] for v in tv.values()])
        pac = np.array([v["pac_bound"] for v in tv.values()])
        ax.plot(100 * al, 100 * al, color=PALETTE["dark"], ls="--", lw=2, label="nominal $\\alpha$")
        ax.fill_between(100 * al, 0, 100 * pac, color=PALETTE["green_1"], label="PAC bound (90%)")
        ax.plot(100 * al, 100 * tr, "o-", color=PALETTE["red_strong"], lw=3, ms=9, label="fitted on FM training features")
        ax.plot(100 * al, 100 * ho, "s-", color=PALETTE["blue_main"], lw=3, ms=9, label="split conformal on held-out A2")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
        ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
        ax.set_title(name, fontsize=20)
        ax.set_xlabel(r"target level $\alpha$ (%)", fontsize=17)
        tick_style(ax)
    axes[0].set_ylabel("test exceedance (%)", fontsize=18)
    axes[0].legend(fontsize=13, loc="upper left")
    finalize_figure(fig, f"fig_fb_train_vs_heldout_{dist}")


# ------------------------------------------------------------------ label sets (D)
def plot_label_sets(alpha="0.1"):
    runs = conformal_runs()
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 6.0))
    axes = np.atleast_1d(axes)
    x = np.arange(5)
    for ax, (name, r) in zip(axes, runs):
        m = r["label_sets"][f"fm/{alpha}/marginal"]["coverage_per_group"]
        g = r["label_sets"][f"fm/{alpha}/mondrian"]["coverage_per_group"]
        ax.bar(x - 0.2, 100 * np.array(m), 0.4, color=PALETTE["red_2"], edgecolor="black", linewidth=1.5, label="marginal")
        ax.bar(x + 0.2, 100 * np.array(g), 0.4, color=PALETTE["blue_main"], edgecolor="black", linewidth=1.5,
               label="per-group (Mondrian)")
        ax.axhline(100 * (1 - float(alpha)), color=PALETTE["dark"], ls="--", lw=2)
        lo = min(min(m), min(g))
        ax.set_ylim(max(0, 100 * lo - 8), 101)
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_title(name, fontsize=20)
        ax.set_xlabel("FM confidence quintile", fontsize=17)
        tick_style(ax)
    axes[0].set_ylabel(f"label-set coverage (%), target {100 * (1 - float(alpha)):.0f}%", fontsize=17)
    axes[0].legend(fontsize=15, loc="lower right")
    finalize_figure(fig, "fig_fb_label_sets")


# ------------------------------------------------------------------ Learn-then-Test and the budget bound (B)
def plot_ltt(dist="unif"):
    runs = conformal_runs()
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 5.8))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, runs):
        c = [v for v in r["ltt"][dist]["candidates"] if not np.isfinite(v["tau"])]
        bound = np.array([v["budget_bound"] for v in c])
        jump = np.array([v["calB_jump"] for v in c])
        ok = np.array([v["p_flip"] <= 0.1 / (2 * len(r["ltt"][dist]["candidates"])) and
                       v["p_jump"] <= 0.1 / (2 * len(r["ltt"][dist]["candidates"])) for v in c])
        mx = 100 * max(bound.max(), jump.max()) * 1.05
        ax.plot([0, mx], [0, mx], color=PALETTE["dark"], ls="--", lw=2)
        ax.axhline(5, color=PALETTE["grey"], ls=":", lw=2)
        ax.scatter(100 * bound[ok], 100 * jump[ok], s=110, color=PALETTE["blue_main"], edgecolor="black", label="certified")
        ax.scatter(100 * bound[~ok], 100 * jump[~ok], s=110, color=PALETTE["red_2"], edgecolor="black", label="not certified")
        ax.set_title(name, fontsize=20)
        ax.set_xlabel(r"bound $a_0+(1-a_0)\alpha\kappa/(\kappa+1)$ (%)", fontsize=16)
        tick_style(ax)
    axes[0].set_ylabel("measured ID jump mass on calB (%)", fontsize=17)
    axes[0].legend(fontsize=14, loc="upper left")
    finalize_figure(fig, "fig_fb_ltt_budget")


# ------------------------------------------------------------------ NLP ladder (5 seeds)
NLP_ROWS = [("FM", lambda L, P, C: L["rows"]["fm"]),
            ("FM + TS", lambda L, P, C: L["rows"]["fm+TS"]),
            ("Dropout injection", lambda L, P, C: L["rows"].get("dropout_injection")),
            ("Deep ensemble (5)", lambda L, P, C: L["rows"].get("ensemble_3ckpt")),
            ("Brownian SDE at z", lambda L, P, C: L["rows"]["fm+brownian_z"]),
            ("Regeneration", lambda L, P, C: L["rows"]["fm+regen"]),
            ("Partitioned regen.", lambda L, P, C: P and P["row"]),
            ("Ours: LTT, tau=1", lambda L, P, C: C and C["rows"].get("unif/ltt_tau1_sensitive")),
            ("Ours: LTT, tau=inf", lambda L, P, C: C and C["rows"].get("unif/ltt_tauinf_sensitive"))]


def nlp_table(task):
    """mean and s.e. over seeds of the main metrics, per row."""
    per = {}
    for d in sorted(glob.glob(f"{UQ}/nlp_ladder/results_{task}/seed*")):
        L, P, C = (jload(f"{d}/{f}") for f in ("ladder_fm.json", "partitioned_regen_fm.json", "conformal_scaffold_fm.json"))
        if not L:
            continue
        for name, f in NLP_ROWS:
            r = f(L, P, C)
            if not r:
                continue
            t = r["test"]
            per.setdefault(name, []).append([100 * t.get("mcc", t["acc"]) if task == "cola" else 100 * t["acc"],
                                             10 * t["nll"], 100 * t["ece"], 100 * t["brier"],
                                             10 * r["shift"]["nll"], 100 * r["shift"]["ece"], 100 * r["ood_auroc"]["mean"]])
    return {k: (np.mean(v, 0), np.std(v, 0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros(7), len(v))
            for k, v in per.items()}


def plot_nlp():
    tabs = [(n, nlp_table(t), t) for n, t in NLP]
    tabs = [x for x in tabs if x[1]]
    if not tabs:
        return
    cols = [(0, "ACC / MCC (%) $\\uparrow$"), (1, "NLL x10 $\\downarrow$"), (2, "ECE (%) $\\downarrow$"),
            (4, "shift NLL x10 $\\downarrow$"), (6, "OOD AUROC (%) $\\uparrow$")]
    colors = [PALETTE["neutral"], PALETTE["red_1"], PALETTE["red_2"], PALETTE["violet"], PALETTE["red_strong"],
              PALETTE["blue_secondary"], PALETTE["teal"], PALETTE["blue_main"], PALETTE["green_3"]]
    fig, axes = plt.subplots(len(tabs), len(cols), figsize=(6.4 * len(cols), 5.2 * len(tabs)), squeeze=False)
    for i, (name, tab, task) in enumerate(tabs):
        names = [n for n, _ in NLP_ROWS if n in tab]
        for j, (c, lab) in enumerate(cols):
            ax = axes[i, j]
            mu = np.array([tab[n][0][c] for n in names])
            se = np.array([tab[n][1][c] for n in names])
            for k, n in enumerate(names):
                ax.bar(k, mu[k], yerr=se[k], color=colors[[m for m, _ in NLP_ROWS].index(n)], edgecolor="black",
                       linewidth=1.5, capsize=4, label=n)
            lo, hi = (mu - se).min(), (mu + se).max()
            ax.set_ylim(lo - 0.15 * (hi - lo + 1e-9), hi + 0.1 * (hi - lo + 1e-9))
            ax.set_xticks([])
            ax.set_title(f"{name}: {lab}" if j == 0 else lab, fontsize=17)
            tick_style(ax, 14)
    axes[0, -1].legend(fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5))
    finalize_figure(fig, "fig_fb_nlp")


# ------------------------------------------------------------------ like-for-like comparison (DIRECTOR's metrics)
# Reads uq/director_comparison/summary.json (director_eval.py table) and */conformal_ood.json.
CMP_ROWS = [("FM", PALETTE["neutral"], None), ("TS", PALETTE["red_2"], None), ("MCD", PALETTE["red_1"], None),
            ("Ensemble", PALETTE["violet"], "//"), ("Scaffold (tau=1)", PALETTE["blue_secondary"], None),
            ("TS+scaffold (tau=1)", PALETTE["blue_main"], None), ("TS+scaffold (tau=inf)", PALETTE["teal"], None)]
ROW_LABEL = {"Scaffold (tau=1)": r"Scaffold, $\tau=1$", "TS+scaffold (tau=1)": r"TS+scaffold, $\tau=1$",
             "TS+scaffold (tau=inf)": r"TS+scaffold, $\tau=\infty$", "Ensemble": "Ensemble (not ranked)"}
LADDER_NAME = {"cola": "CoLA (5 seeds)", "imagenet": "ImageNet", "cifar100": "CIFAR-100", "cifar10": "CIFAR-10",
               "sst2": "SST-2 (5 seeds)", "imdb": "IMDB (5 seeds)"}
OOD_DIRS = {"cola": sorted(glob.glob(f"{UQ}/nlp_ladder/results_cola/seed*")),
            "imdb": sorted(glob.glob(f"{UQ}/nlp_ladder/results_imdb/seed*")),
            "sst2": sorted(glob.glob(f"{UQ}/nlp_ladder/results_sst2/seed*")), "imagenet": [f"{UQ}/imagenet_ladder/results"],
            "cifar100": [f"{UQ}/cifar_ladder/results_cifar100"], "cifar10": [f"{UQ}/cifar_ladder/results_cifar10"]}
FLAG_SCORES = [("msp", r"$1-\max p_T$", PALETTE["red_2"]), ("energy", "energy", PALETTE["red_1"]),
               ("maha_z", r"Mahalanobis at $z$", PALETTE["teal"]),
               ("maha_multi_unif", "multi-layer (primary)", PALETTE["blue_main"])]


def mean_se(v):
    v = np.asarray([x for x in v if x == x], float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def legend_panel(ax, handles, labels, size=16):
    ax.set_axis_off()
    ax.legend(handles, labels, loc="center left", fontsize=size)


def plot_compare(ladders, name):
    summ = jload(f"{UQ}/director_comparison/summary.json")
    ladders = [l for l in ladders if summ and l in summ]
    if not ladders:
        return
    cols = [("id/ece", "ECE (%) $\\downarrow$"), ("id/nll", "NLL $\\times10$ $\\downarrow$"),
            ("shift/acc", "shift ACC / MCC $\\uparrow$"), ("shift/nll", "shift NLL $\\times10$ $\\downarrow$"),
            ("ood_entropy/auroc", "OOD AUROC (%) $\\uparrow$")]
    fig, axes = plt.subplots(len(ladders), len(cols) + 1, figsize=(5.4 * len(cols) + 4.6, 5.0 * len(ladders)),
                             squeeze=False)
    handles = {}
    for i, lad in enumerate(ladders):
        S = summ[lad]
        for j, (key, lab) in enumerate(cols):
            ax = axes[i, j]
            if key == "shift/acc" and lad == "cola":   # DIRECTOR scores only CoLA by MCC
                key = "shift/mcc"
            vals = []
            for k, (row, color, hatch) in enumerate(CMP_ROWS):
                if row not in S or key not in S[row]:
                    continue
                m, se = mean_se(S[row][key])
                b = ax.bar(k, m, yerr=se if se > 0 else None, color=color, hatch=hatch, edgecolor="black",
                           linewidth=1.5, capsize=4)
                handles[row] = b
                vals += [m - se, m + se]
            lo, hi = min(vals), max(vals)
            ax.set_ylim(lo - 0.25 * (hi - lo + 1e-9), hi + 0.12 * (hi - lo + 1e-9))
            ax.set_xticks([])
            ax.set_title(f"{LADDER_NAME[lad]}: {lab}" if j == 0 else lab, fontsize=17)
            tick_style(ax, 14)
        axes[i, -1].set_axis_off()
    rows = [r for r, _, _ in CMP_ROWS if r in handles]
    legend_panel(axes[0, -1], [handles[r] for r in rows], [ROW_LABEL.get(r, r) for r in rows])
    finalize_figure(fig, name)


def ood_runs(lad):
    return [r for r in (jload(f"{d}/conformal_ood.json") for d in OOD_DIRS[lad]) if r]


def plot_ood_compare():
    summ = jload(f"{UQ}/director_comparison/summary.json")
    ladders = [l for l in ("cola", "imdb", "sst2", "imagenet", "cifar100", "cifar10") if summ and l in summ]
    rows = [r for r in CMP_ROWS if r[0] != "Scaffold (tau=1)"]
    fig, axes = plt.subplots(1, len(ladders) + 1, figsize=(5.6 * len(ladders) + 4.6, 5.6))
    handles = {}
    for ax, lad in zip(axes, ladders):
        vals = []
        for k, (row, color, hatch) in enumerate(rows):
            if row not in summ[lad]:
                continue
            m, se = mean_se(summ[lad][row]["ood_entropy/auroc"])
            handles[row] = ax.bar(k, m, yerr=se if se > 0 else None, color=color, hatch=hatch, edgecolor="black",
                                  linewidth=1.5, capsize=4)
            vals += [m - se, m + se]
        runs = ood_runs(lad)
        if runs:
            dist = np.mean([np.mean(list(r["scores"]["maha_multi_unif"]["auroc"].values())) for r in runs])
            handles["dist"] = ax.axhline(dist, color=PALETTE["dark"], ls="--", lw=2.5)
            vals.append(dist)
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - 0.25 * (hi - lo + 1e-9), hi + 0.12 * (hi - lo + 1e-9))
        ax.set_xticks([])
        ax.set_title(LADDER_NAME[lad], fontsize=19)
        tick_style(ax, 14)
    axes[0].set_ylabel("OOD AUROC by entropy (%)", fontsize=17)
    keys = [r for r, _, _ in rows if r in handles] + ["dist"]
    legend_panel(axes[-1], [handles[k] for k in keys],
                 [ROW_LABEL.get(k, k) for k in keys[:-1]] + ["multi-layer distance alone"])
    finalize_figure(fig, "fig_fb_ood_compare")


# ------------------------------------------------------------------ Option 1: conformal OOD flag
def plot_flag_validity(alpha="0.05"):
    ladders = [l for l in ("cola", "imdb", "sst2", "imagenet", "cifar100", "cifar10") if ood_runs(l)]
    fig, axes = plt.subplots(1, len(ladders) + 1, figsize=(5.4 * len(ladders) + 4.4, 5.6))
    x = np.arange(len(FLAG_SCORES))
    for ax, lad in zip(axes, ladders):
        runs = ood_runs(lad)
        for k, (s, _, _) in enumerate(FLAG_SCORES):
            wm = mean_se([max(r["scores"][s]["marginal"]["levels"][alpha]["id_far_per_group"]) for r in runs])
            wg = mean_se([max(r["scores"][s]["mondrian"]["levels"][alpha]["id_far_per_group"]) for r in runs])
            mg = mean_se([r["scores"][s]["marginal"]["levels"][alpha]["id_far"] for r in runs])
            hm = ax.bar(k - 0.2, wm[0], 0.4, yerr=wm[1] or None, color=PALETTE["red_2"], edgecolor="black", lw=1.5,
                        capsize=4)
            hg = ax.bar(k + 0.2, wg[0], 0.4, yerr=wg[1] or None, color=PALETTE["blue_main"], edgecolor="black", lw=1.5,
                        capsize=4)
            hd = ax.scatter(k - 0.2, mg[0], s=110, marker="D", color=PALETTE["highlight"], edgecolor="black", zorder=5)
        ha = ax.axhline(100 * float(alpha), color=PALETTE["dark"], ls="--", lw=2.5)
        ax.set_xticks(x, ["MSP", "energy", "$z$", "multi"])
        ax.set_title(LADDER_NAME[lad], fontsize=19)
        tick_style(ax, 14)
    axes[0].set_ylabel(f"ID false alarms at $\\alpha$={100 * float(alpha):.0f}% (%)", fontsize=17)
    legend_panel(axes[-1], [hd, hm, hg, ha], ["marginal, all inputs", "marginal, worst group",
                                              "per-group (Mondrian), worst group", r"target $\alpha$"])
    finalize_figure(fig, "fig_fb_flag_validity")


def plot_flag_power():
    ladders = [l for l in ("cola", "imdb", "sst2", "imagenet", "cifar100") if ood_runs(l)]
    fig, axes = plt.subplots(1, len(ladders) + 1, figsize=(6.6 * len(ladders) + 4.4, 5.6))
    handles = []
    for ax, lad in zip(axes, ladders):
        runs = ood_runs(lad)
        kinds = runs[0]["ood_kind"]
        sets = list(kinds)
        x = np.arange(len(sets))
        w = 0.8 / len(FLAG_SCORES)
        vals, handles = [], []
        for k, (s, _, color) in enumerate(FLAG_SCORES):
            ms = [mean_se([r["scores"][s]["auroc"][o] for r in runs]) for o in sets]
            handles.append(ax.bar(x - 0.4 + w * (k + 0.5), [m for m, _ in ms], w, yerr=[e or 0 for _, e in ms],
                                  color=color, edgecolor="black", lw=1.2, capsize=3))
            vals += [m for m, _ in ms]
        nice = {"ag_news": "AG News", "newsgroups": "20NG", "noise": "noise", "imagenet_o": "ImageNet-O",
                "dtd": "Textures", "cifar10": "CIFAR-10", "svhn": "SVHN", "lsun": "LSUN"}
        ax.set_xticks(x, [f"{nice.get(o, o)}\n({kinds[o]})" for o in sets], fontsize=13)
        ax.set_ylim(max(40, min(vals) - 5), 101)
        ax.set_title(LADDER_NAME[lad], fontsize=19)
        tick_style(ax, 13)
    axes[0].set_ylabel("detection AUROC (%)", fontsize=17)
    legend_panel(axes[-1], handles, [lab for _, lab, _ in FLAG_SCORES])
    finalize_figure(fig, "fig_fb_flag_power")


def plot_flag_shift(alpha="0.05"):
    ladders = [l for l in ("cifar100", "imagenet", "cifar10") if ood_runs(l)]
    shown = [s for s in FLAG_SCORES if s[0] in ("msp", "maha_multi_unif")]
    fig, axes = plt.subplots(1, len(ladders) + 1, figsize=(4.8 * len(ladders) + 4.4, 5.4))
    for ax, lad in zip(axes, ladders):
        r = ood_runs(lad)[0]
        for k, (s, _, color) in enumerate(shown):
            lv = r["scores"][s]["marginal"]["levels"][alpha]
            ha = ax.bar(k - 0.2, lv["shift_acc_accepted"], 0.4, color=color, edgecolor="black", lw=1.5)
            hf = ax.bar(k + 0.2, lv["shift_acc_flagged"], 0.4, color=color, hatch="//", edgecolor="black", lw=1.5)
            ax.text(k, 103, f"{lv['shift_flag']['all']:.0f}% flagged", ha="center", fontsize=13)
        ax.set_xticks([0, 1], ["MSP", "multi-layer"])
        ax.set_ylim(0, 112)
        ax.set_title(f"{LADDER_NAME[lad]} shift", fontsize=18)
        tick_style(ax, 14)
    axes[0].set_ylabel("FM accuracy on shifted inputs (%)", fontsize=16)
    import matplotlib.patches as mpatches
    legend_panel(axes[-1], [mpatches.Patch(facecolor="white", edgecolor="black"),
                            mpatches.Patch(facecolor="white", edgecolor="black", hatch="//")],
                 ["accepted (not flagged)", "flagged"])
    finalize_figure(fig, "fig_fb_flag_shift")


# ------------------------------------------------------------------ last 4 blocks compiled on all of ImageNet-1k
LAST4_ARMS = [("100k images", "compile_final_ode8.json", PALETTE["neutral"], None),
            ("318k images, 15k steps", "compile_final_ode8_data.json", PALETTE["red_2"], None),
            ("1.28M, 15k steps, matched LR", "compile_final_ode8_full15k.json", PALETTE["highlight"], None),
            ("1.28M, snapshot at 15k", "compile_final_ode8_full_step15000.json", PALETTE["blue_secondary"], "//"),
            ("1.28M, snapshot at 30k", "compile_final_ode8_full_step30000.json", PALETTE["blue_secondary"], ".."),
            ("1.28M, snapshot at 45k", "compile_final_ode8_full_step45000.json", PALETTE["blue_secondary"], None),
            ("1.28M, 60k steps", "compile_final_ode8_full.json", PALETTE["blue_main"], None)]
ALL12_ARMS = [("100k images, 15k steps", "compile_final_ode.json", PALETTE["neutral"], None),
              ("318k images, 15k steps", "compile_final_ode_data.json", PALETTE["red_2"], None),
              ("1.28M, snapshot at 15k", "compile_final_ode_full_step15000.json", PALETTE["blue_secondary"], "//"),
              ("1.28M, snapshot at 30k", "compile_final_ode_full_step30000.json", PALETTE["blue_secondary"], ".."),
              ("1.28M, snapshot at 45k", "compile_final_ode_full_step45000.json", PALETTE["blue_secondary"], None),
              ("1.28M, 60k steps", "compile_final_ode_full.json", PALETTE["blue_main"], None)]


def plot_full_compile(arms=LAST4_ARMS, name="fig_fb_full_compile"):
    res = f"{UQ}/imagenet_ladder/results"
    arms = [a + (jload(f"{res}/{a[1]}"),) for a in arms]
    arms = [a for a in arms if a[4]]
    cols = [("agree", 100, "top-1 agreement with the FM (%) $\\uparrow$"), ("top5_agree", 100, "top-5 agreement (%) $\\uparrow$"),
            ("kl", 1, "KL(FM $\\|$ flow) $\\downarrow$"), ("acc", 100, "accuracy (%) $\\uparrow$")]
    fig, axes = plt.subplots(1, len(cols) + 1, figsize=(5.4 * len(cols) + 5.2, 5.4))
    handles = []
    for ax, (key, mult, lab) in zip(axes, cols):
        handles = []
        vals = [mult * a[4][key] for a in arms]
        for k, (a, v) in enumerate(zip(arms, vals)):
            handles.append(ax.bar(k, v, color=a[2], hatch=a[3], edgecolor="black", linewidth=1.5))
        if key == "acc":
            ax.axhline(81.68, color=PALETTE["dark"], ls="--", lw=2.5)
            vals = vals + [81.68]
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - 0.2 * (hi - lo), hi + 0.12 * (hi - lo))
        ax.set_xticks([])
        ax.set_title(lab, fontsize=16)
        tick_style(ax, 14)
    legend_panel(axes[-1], handles, [a[0] for a in arms], size=15)
    finalize_figure(fig, name)


if __name__ == "__main__":
    import sys
    apply_publication_style()
    if sys.argv[1:2] == ["round2"]:   # figures added for the round-2 report; the earlier ones are left as they are
        # SST-2 joins the first figure once run_sst2_fix.sh has rebuilt its results
        nlp = ["cola", "imdb", "sst2"] if "--with-sst2" in sys.argv else ["cola", "imdb"]
        plot_compare(nlp, "fig_fb_nlp_compare")
        plot_compare(["imagenet", "cifar100"], "fig_fb_vision_compare")
        plot_ood_compare()
        plot_flag_validity()
        plot_flag_power()
        plot_flag_shift()
        plot_full_compile()
        plot_full_compile(ALL12_ARMS, "fig_fb_full_compile_all12")
    else:
        for dist in ("unif", "err"):
            plot_group_exceedance(dist)
            plot_train_vs_heldout(dist)
        plot_label_sets()
        plot_ltt()
        plot_nlp()
