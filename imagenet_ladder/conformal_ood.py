"""Option 1: temperature-scaled predictive + a conformal OOD flag with a guaranteed ID false-alarm rate.

Output for an input x: the TS predictive p_T(.|x) (T fitted on A1) and a conformal p-value pi(x) of an
atypicality score s(x), calibrated on A2; x is flagged at level alpha when pi(x) <= alpha.
Guarantees (exchangeable ID inputs, s and the groups fixed before A2 is seen):
  marginal   P(flag | ID) <= alpha
  Mondrian   P(flag | ID, G(x) = k) <= alpha for every FM-confidence quintile k
  PAC        P(flag | ID, A2) <= Beta_{1-delta}(n + 1 - l, l) with probability >= 1 - delta over A2
  BH         Benjamini-Hochberg on conformal p-values controls the FDR of a flagged batch at q
             (the p-values are PRDS: Bates et al., 2023, "Testing for outliers with conformal p-values")
Scores (fixed in advance; none uses OOD data): 1 - max p_T (MSP), entropy of p_T, energy -logsumexp(logits),
min-class Mahalanobis at z, relative Mahalanobis at z (Ren et al., 2021), multi-layer Mahalanobis with
uniform weights ("ours" by default, the scaffold's distance) and with weights learned on A1 errors.

OOD protocol: "far" / "near" OOD are inputs outside the task; for NLP only random-token sequences are
outside the task, AG News and 20 Newsgroups are out-of-domain text that can still be labelled.
Shift sets are reported separately: flag rate and TS accuracy on accepted versus flagged inputs.

Usage: LADDER=<ladder> [NLP_SEED=s] python conformal_ood.py      -> results/conformal_ood.json
       python conformal_ood.py table                            -> ../conformal_ood/tables.md
"""
import glob
import json
import os
import sys

import numpy as np

ALPHAS = [0.01, 0.05, 0.1]
FDR_Q = 0.1
SCORES = ["msp", "entropy", "energy", "maha_z", "rmd_z", "maha_multi_unif", "maha_multi_err"]
PRIMARY = "maha_multi_unif"
OOD_KIND = {"cifar10": {"cifar100": "near", "svhn": "far", "lsun": "far", "dtd": "far", "noise": "far"},
            "cifar100": {"cifar10": "near", "svhn": "far", "lsun": "far", "dtd": "far", "noise": "far"},
            "imagenet": {"imagenet_o": "near", "dtd": "far", "noise": "far"},
            **{t: {"ag_news": "domain", "newsgroups": "domain", "noise": "far"} for t in ("cola", "sst2", "imdb")}}
HERE = os.path.dirname(os.path.abspath(__file__))


def bh(p, q):
    """Benjamini-Hochberg rejections at level q."""
    n = len(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, n + 1) / n
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    return rej


def run():
    import torch
    from conformal_scaffold import N_GROUPS, conformal_pvalue, groups_of, learn_err, pac_bound, confidence
    from inl_common import LADDER, OOD_SETS, REGEN_LAYERS, TiedGMM, auroc, fit_temperature, get_npy, save_json
    from ladder_eval import EVAL_SETS, Ctx, gmm_distances, load_backbone, softmax_np
    from sklearn.metrics import average_precision_score

    torch.set_num_threads(int(os.environ.get("NUM_THREADS", "16")))
    ctx = Ctx()
    sets = ["train", *EVAL_SETS]
    data = {k: load_backbone("fm", k) for k in sets}
    y_train = get_npy("train_labels", mmap=False).astype(np.int64)
    d2 = gmm_distances("fm", {k: v[0] for k, v in data.items()}, y_train)
    bg = TiedGMM(torch.tensor(data["train"][0]["z"]), torch.zeros(len(y_train), dtype=torch.long))
    d0 = {k: np.concatenate([bg.min_mahalanobis(torch.tensor(data[k][0]["z"][i:i + 20_000])).numpy()
                             for i in range(0, len(data[k][0]["z"]), 20_000)]) for k in sets}
    logits = {k: v[1].astype(np.float64) for k, v in data.items()}
    del data

    perm = np.random.default_rng(0).permutation(ctx.calA)
    A1, A2 = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    test, yv = ctx.test, ctx.y_val
    T = fit_temperature(torch.tensor(logits["val"][A1]), torch.tensor(yv[A1]))
    p = {k: softmax_np(v / T) for k, v in logits.items()}
    med = {l: float(np.median(d2[l]["val"][A1])) for l in REGEN_LAYERS}
    u = {k: np.stack([np.log(np.maximum(d2[l][k], 1e-12) / med[l]) for l in REGEN_LAYERS], 1) for k in sets}
    err = p["val"].argmax(1) != yv
    th_err = np.array(learn_err(u["val"][A1], err[A1]))
    lse = lambda l: l.max(1) + np.log(np.exp(l - l.max(1, keepdims=True)).sum(1))  # noqa: E731
    score = {
        "msp": {k: 1 - p[k].max(1) for k in sets},
        "entropy": {k: -(np.clip(p[k], 1e-12, None) * np.log(np.clip(p[k], 1e-12, None))).sum(1) for k in sets},
        "energy": {k: -lse(logits[k]) for k in sets},
        "maha_z": {k: d2["z"][k] for k in sets},
        "rmd_z": {k: d2["z"][k] - d0[k] for k in sets},
        "maha_multi_unif": {k: u[k].mean(1) for k in sets},
        "maha_multi_err": {k: u[k] @ th_err for k in sets},
    }
    edges = np.quantile(confidence(p["val"][A1]), np.linspace(0, 1, N_GROUPS + 1)[1:-1])
    g = {k: groups_of(p[k], edges) for k in sets}
    shift_groups = {"all": np.ones(len(ctx.y_shift), bool), **{c: ctx.corr == c for c in np.unique(ctx.corr)}}
    pred_ok = {"test": p["val"][test].argmax(1) == yv[test], "shift": p["shift"].argmax(1) == ctx.y_shift}
    gt, n_a2 = g["val"][test], len(A2)

    out = {"T_A1": T, "split": dict(A1=len(A1), A2=n_a2, test=len(test)), "theta_err": th_err.tolist(),
           "group_sizes_A2": np.bincount(g["val"][A2], minlength=N_GROUPS).tolist(),
           "pac": {str(a): pac_bound(n_a2, a) for a in ALPHAS}, "ood_kind": OOD_KIND[LADDER], "scores": {}}
    for name, s in score.items():
        r = {"auroc": {}, "aupr": {}, "fpr95": {}}
        for o in OOD_SETS:   # threshold-free detection, OOD positive
            si, so = s["val"][test], s[o]
            r["auroc"][o] = 100 * auroc(si, so)
            lab = np.r_[np.zeros(len(si)), np.ones(len(so))]
            r["aupr"][o] = 100 * float(average_precision_score(lab, np.r_[si, so]))
            r["fpr95"][o] = 100 * float(np.mean(si >= np.quantile(so, 0.05)))
        for mode in ("marginal", "mondrian"):
            if mode == "marginal":
                pv = {k: conformal_pvalue(s[k], s["val"][A2]) for k in ["val", "shift", *OOD_SETS]}
            else:
                pv = {k: conformal_pvalue(s[k], s["val"][A2], g[k], g["val"][A2]) for k in ["val", "shift", *OOD_SETS]}
            res = {}
            for a in ALPHAS:
                ft = pv["val"][test] <= a
                fs = pv["shift"] <= a
                res[str(a)] = dict(
                    id_far=100 * float(ft.mean()), id_far_per_group=[100 * float(ft[gt == k].mean()) for k in range(N_GROUPS)],
                    id_acc_accepted=100 * float(pred_ok["test"][~ft].mean()),
                    id_acc_flagged=100 * float(pred_ok["test"][ft].mean()) if ft.any() else float("nan"),
                    tpr={o: 100 * float((pv[o] <= a).mean()) for o in OOD_SETS},
                    shift_flag={c: 100 * float(fs[m].mean()) for c, m in shift_groups.items()},
                    shift_acc_accepted=100 * float(pred_ok["shift"][~fs].mean()) if (~fs).any() else float("nan"),
                    shift_acc_flagged=100 * float(pred_ok["shift"][fs].mean()) if fs.any() else float("nan"))
            fdr = {}
            for o in OOD_SETS:   # a batch of ID test inputs and one OOD set; BH at FDR_Q
                pb = np.r_[pv["val"][test], pv[o]]
                is_ood = np.r_[np.zeros(len(test), bool), np.ones(len(pv[o]), bool)]
                rej = bh(pb, FDR_Q)
                fdr[o] = dict(fdr=100 * float((rej & ~is_ood).sum() / max(rej.sum(), 1)), power=100 * float(rej[is_ood].mean()))
            r[mode] = dict(levels=res, bh=fdr)
        out["scores"][name] = r
        print(f"{name:16s} AUROC " + " ".join(f"{o}={r['auroc'][o]:.1f}" for o in OOD_SETS) +
              f" | alpha=0.05 marginal FAR={r['marginal']['levels']['0.05']['id_far']:.2f} "
              f"worst-group={max(r['mondrian']['levels']['0.05']['id_far_per_group']):.2f} (Mondrian)", flush=True)
    save_json(out, "conformal_ood.json")


# ----------------------------------------------------------------- tables
LADDER_DIRS = {"cifar10": [f"{HERE}/../cifar_ladder/results_cifar10"],
               "cifar100": [f"{HERE}/../cifar_ladder/results_cifar100"],
               "imagenet": [f"{HERE}/results"],
               **{t: sorted(glob.glob(f"{HERE}/../nlp_ladder/results_{t}/seed*")) for t in ("cola", "sst2", "imdb")}}


def _ms(vals, d=1):
    vals = [v for v in vals if v == v]
    if not vals:
        return "–"
    m = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return f"{m:.{d}f}" if se < 0.05 * 10 ** (-d + 1) / 10 else f"{m:.{d}f}±{se:.{d}f}"


def table():
    L = ["# Option 1: TS predictive + conformal OOD flag", "",
         "AUROC / AUPR / FPR95 in %, OOD positive; FAR = ID false-alarm rate on the test split; TPR = flagged "
         "fraction of an OOD set. Mean ± s.e. over seeds (NLP 5 seeds, vision one checkpoint). "
         f"Primary score fixed in advance: `{PRIMARY}`. No score uses OOD data.", ""]
    for ladder, dirs in LADDER_DIRS.items():
        runs = [json.load(open(f"{d}/conformal_ood.json")) for d in dirs if os.path.exists(f"{d}/conformal_ood.json")]
        if not runs:
            continue
        kinds = runs[0]["ood_kind"]
        osets = list(kinds)
        L += [f"## {ladder} (n={len(runs)}; A2 = {runs[0]['split']['A2']}, per group ≈ "
              f"{min(runs[0]['group_sizes_A2'])}–{max(runs[0]['group_sizes_A2'])}; T = {_ms([r['T_A1'] for r in runs], 3)})", "",
              "### Threshold-free detection: AUROC (mean over sets of each kind)", "",
              "| Score | " + " | ".join(f"{o} ({kinds[o]})" for o in osets) + " |", "|---" * (len(osets) + 1) + "|"]
        for s in SCORES:
            L.append(f"| {'**' + s + '**' if s == PRIMARY else s} | " +
                     " | ".join(_ms([r["scores"][s]["auroc"][o] for r in runs]) for o in osets) + " |")
        for a in ("0.05", "0.01"):
            L += ["", f"### Conformal flag at α = {a}: validity and power", "",
                  "| Score | mode | ID FAR | worst-group FAR | " + " | ".join(f"TPR {o}" for o in osets) +
                  " | shift flagged | shift acc accepted / flagged | ID acc accepted / flagged |",
                  "|---" * (len(osets) + 7) + "|"]
            for s in SCORES:
                for mode in ("marginal", "mondrian"):
                    lv = [r["scores"][s][mode]["levels"][a] for r in runs]
                    L.append(f"| {s} | {mode} | {_ms([x['id_far'] for x in lv], 2)} | "
                             f"{_ms([max(x['id_far_per_group']) for x in lv], 2)} | " +
                             " | ".join(_ms([x["tpr"][o] for x in lv]) for o in osets) +
                             f" | {_ms([x['shift_flag']['all'] for x in lv])} | "
                             f"{_ms([x['shift_acc_accepted'] for x in lv])} / {_ms([x['shift_acc_flagged'] for x in lv])} | "
                             f"{_ms([x['id_acc_accepted'] for x in lv])} / {_ms([x['id_acc_flagged'] for x in lv])} |")
        L += ["", f"### BH at FDR q = {FDR_Q} on (ID test ∪ one OOD set), marginal p-values: realized FDR / power", "",
              "| Score | " + " | ".join(osets) + " |", "|---" * (len(osets) + 1) + "|"]
        for s in SCORES:
            L.append(f"| {s} | " + " | ".join(
                f"{_ms([r['scores'][s]['marginal']['bh'][o]['fdr'] for r in runs])} / "
                f"{_ms([r['scores'][s]['marginal']['bh'][o]['power'] for r in runs])}" for o in osets) + " |")
        L.append("")
    os.makedirs(f"{HERE}/../conformal_ood", exist_ok=True)
    with open(f"{HERE}/../conformal_ood/tables.md", "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    table() if sys.argv[1:] == ["table"] else run()
