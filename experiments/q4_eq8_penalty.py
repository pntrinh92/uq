"""Quick test for Eq. (8): where does the score-weighted penalty put the variance?

On the final representation z of the trained digits ViT, with the analytic
GMM score, the quadratic approximation of Eq. (8) gives
    T*(z) = sqrt(lambda * ||grad log p(z)|| / c(z)),
    c(z)  = tr(W^T (diag p - p p^T) W)   (CE curvature in z; W = head weights).
Behaviour (b) asks for large T* OOD; behaviour (a) for small T* on
well-represented inputs. We score T* (lambda drops out of AUROC) as a detector
of misclassified ID inputs and of each OOD set, next to the score norm alone,
1/c alone, the deterministic entropy, and the Mahalanobis distance.
"""
import numpy as np
import torch

from common import (BLUE, FIGURES, GREY, INK, ORANGE, TiedGMM, auroc, entropy, get_models,
                    save_json, setup_matplotlib)

N_SPLITS = 5


@torch.no_grad()
def scores_for(fm, gmm, X):
    logits, z, _ = fm(X, return_all=True)
    p = torch.softmax(logits, 1).double()
    W = fm.head.weight.double()
    # c = tr(W^T (diag p - p p^T) W) = sum_c p_c ||w_c||^2 - ||W^T p||^2
    c = p @ (W ** 2).sum(1) - ((p @ W) ** 2).sum(1)
    s = gmm.score(z.double()).norm(dim=1)
    top2 = p.topk(2, 1).values
    return dict(
        T_star=torch.sqrt(s / c).numpy(), score_norm=s.numpy(), inv_curv=(1 / c).numpy(),
        entropy=entropy(p.float()).numpy(), mahalanobis=gmm.min_mahalanobis(z).numpy(),
        margin=(top2[:, 0] - top2[:, 1]).numpy(),
    )


def run_split(seed):
    data, models = get_models(seed)
    fm = models[0]
    with torch.no_grad():
        ztr = fm(data["Xtr"], return_all=True)[1]
    gmm = TiedGMM(ztr, data["ytr"])
    sc = {"ID": scores_for(fm, gmm, data["Xte"])}
    with torch.no_grad():
        sc["ID"]["wrong"] = (fm(data["Xte"]).argmax(1) != data["yte"]).numpy()
    for k, X in data["ood"].items():
        sc[k] = scores_for(fm, gmm, X)
    out = {}
    for name in ("T_star", "score_norm", "inv_curv", "entropy", "mahalanobis"):
        r = {o: auroc(sc["ID"][name], sc[o][name]) for o in data["ood"]}
        w = sc["ID"]["wrong"]
        r["misclassified"] = auroc(sc["ID"][name][~w], sc["ID"][name][w]) if w.any() else float("nan")
        out[name] = r
    return out, sc


def main():
    runs, scs = zip(*[run_split(s) for s in range(N_SPLITS)])
    names = list(runs[0].keys())
    tasks = list(runs[0][names[0]].keys())
    summary = {n: {t: [float(np.nanmean([r[n][t] for r in runs])), float(np.nanstd([r[n][t] for r in runs]))]
                   for t in tasks} for n in names}
    save_json({"per_split": list(runs), "summary": summary}, "q4_eq8_penalty.json")
    for n in names:
        print(f"{n:12s}", "  ".join(f"{t}={summary[n][t][0]:.3f}±{summary[n][t][1]:.3f}" for t in tasks))

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    sc = scs[0]
    for k, c, mk, lab in (("ID", BLUE, "o", "ID test"), ("heldout_9", ORANGE, "s", "held-out 9"),
                          ("rotated", INK, "^", "rotated"), ("noise", GREY, "D", "noise")):
        ax.scatter(sc[k]["margin"], sc[k]["T_star"], s=10, alpha=0.55, color=c, marker=mk, label=lab, lw=0)
    w = sc["ID"]["wrong"]
    ax.scatter(sc["ID"]["margin"][w], sc["ID"]["T_star"][w], s=60, facecolor="none", edgecolor=ORANGE, lw=1.5,
               label="ID misclassified")
    ax.set_yscale("log")
    ax.set_xlabel("top-2 softmax margin of the FM")
    ax.set_ylabel(r"$T^\star\propto\sqrt{\|\nabla\log p\|/c}$")
    ax.set_title("Eq. (8) optimum vs. classifier margin (split 0)", loc="left", fontsize=11.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)

    ax = axes[1]
    labels = {"T_star": r"$T^\star$ (Eq. 8)", "score_norm": r"$\|\nabla\log p\|$", "inv_curv": r"$1/c$",
              "entropy": "entropy", "mahalanobis": "Mahalanobis"}
    cols = [BLUE, "#7FB0EC", GREY, INK, ORANGE]
    width = 0.16
    x = np.arange(len(tasks))
    for i, (n, c) in enumerate(zip(names, cols)):
        ax.bar(x + (i - 2) * width, [summary[n][t][0] for t in tasks], width, yerr=[summary[n][t][1] for t in tasks],
               color=c, label=labels[n], capsize=1.5, error_kw=dict(lw=0.8))
    ax.axhline(0.5, color=INK, lw=0.8, ls=":")
    ax.set_xticks(x, ["held-out 9", "rotated", "noise", "misclassified"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC (higher = flags it)")
    ax.set_title("Does the variance go where it should?", loc="left", fontsize=11.5)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.33))
    fig.savefig(f"{FIGURES}/fig_q4_eq8_penalty.png")


if __name__ == "__main__":
    main()
