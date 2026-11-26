"""Like-for-like comparison in DIRECTOR's metrics (director_metrics.py) on our backbones.

Same backbone, same splits and the same metrics for every row:
  FM            the fine-tuned model (ViT-B/16 or Qwen2.5-0.5B)
  TS            temperature scaling, T fitted on the whole held-out set (calA + calB)
  MCD           test-time dropout, 8 passes, p chosen on calB NLL (nlp_prepare.py / dropout_baseline.py)
  Ensemble      deep ensemble of the available checkpoints (NLP: the 5 seeds; ImageNet: 3 ViT-B/16), 5x / 3x cost
  Scaffold      certified path scaffold on the raw FM (conformal_scaffold_fm_fst.json)
  TS+scaffold   the same on the A1-tempered FM (conformal_scaffold_fm_ts_fst.json)
The scaffold rows are fixed in advance: uniform distance, most sensitive LTT-certified candidate, tau = 1 and inf.

Usage:
  LADDER=<ladder> [NLP_SEED=s] python director_eval.py eval    # one ladder / seed -> results/director_eval.json
  python director_eval.py table                                # every ladder, mean +- s.e. over seeds
"""
import glob
import json
import os
import sys

import numpy as np

SCAFFOLD_ROWS = {"Scaffold (tau=1)": ("fm_fst", "unif/ltt_tau1_sensitive"),
                 "Scaffold (tau=inf)": ("fm_fst", "unif/ltt_tauinf_sensitive"),
                 "TS+scaffold (tau=1)": ("fm_ts_fst", "unif/ltt_tau1_sensitive"),
                 "TS+scaffold (tau=inf)": ("fm_ts_fst", "unif/ltt_tauinf_sensitive")}
ROW_ORDER = ["FM", "TS", "MCD", "Ensemble", *SCAFFOLD_ROWS]
ID_COLS = ["acc", "aurc", "auroc", "fpr95", "ece", "nll", "brier"]
SHIFT_COLS = ["acc", "aurc", "auroc", "fpr95", "ece", "nll", "brier"]
LOWER = {"aurc", "fpr95", "ece", "nll", "brier"}
HERE = os.path.dirname(os.path.abspath(__file__))
LADDER_DIRS = {"imagenet": [f"{HERE}/results"],
               "cifar10": [f"{HERE}/../cifar_ladder/results_cifar10"],
               "cifar100": [f"{HERE}/../cifar_ladder/results_cifar100"],
               **{t: sorted(glob.glob(f"{HERE}/../nlp_ladder/results_{t}/seed*")) for t in ("cola", "sst2", "imdb")}}


def run_eval():
    import director_metrics
    from inl_common import FEAT, OOD_SETS, fit_temperature, get_npy, save_json
    from ladder_eval import EVAL_SETS, Ctx, softmax_np
    import torch

    ctx = Ctx()
    held = np.sort(np.concatenate([ctx.calA, ctx.calB]))
    shift_groups = {"all": np.ones(len(ctx.y_shift), bool), **{c: ctx.corr == c for c in np.unique(ctx.corr)}}

    def ev(probs):
        return director_metrics.evaluate(probs, ctx.y_val[ctx.test], ctx.test, ctx.y_shift, shift_groups, OOD_SETS)

    lg = {k: np.asarray(get_npy(f"{k}_logits"), dtype=np.float32) for k in EVAL_SETS}
    p_fm = {k: softmax_np(v[:, 0] if v.ndim == 3 else v) for k, v in lg.items()}
    rows = {"FM": ev(p_fm)}
    lp = {k: np.log(np.clip(v, 1e-12, None)) for k, v in p_fm.items()}
    T = fit_temperature(torch.tensor(lp["val"][held]), torch.tensor(ctx.y_val[held]))
    rows["TS"] = ev({k: softmax_np(v / T) for k, v in lp.items()})
    if os.path.exists(f"{FEAT}/dropout_val_logits.npy"):
        rows["MCD"] = ev({k: softmax_np(np.load(f"{FEAT}/dropout_{k}_logits.npy").astype(np.float32)).mean(1)
                          for k in EVAL_SETS})
    if lg["val"].ndim == 3 and lg["val"].shape[1] > 1:
        rows["Ensemble"] = ev({k: softmax_np(v).mean(1) for k, v in lg.items()})
        rows["Ensemble"]["members"] = int(lg["val"].shape[1])
    save_json({"rows": rows, "T_heldout": T, "n_test": int(len(ctx.test))}, "director_eval.json")
    print(f"director_eval: rows {list(rows)}; T={T:.3f}", flush=True)


def load_seed(d):
    with open(f"{d}/director_eval.json") as f:
        rows = json.load(f)["rows"]
    for name, (suffix, key) in SCAFFOLD_ROWS.items():
        path = f"{d}/conformal_scaffold_{suffix}.json"
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)["rows"]
            if key in r and "director" in r[key]:
                rows[name] = r[key]["director"]
    return rows


def flatten(r):
    """Row dict -> {section/metric: value}; OOD detection averaged over the OOD sets (and kept per set)."""
    out = {f"id/{k}": v for k, v in r["id"].items()}
    out.update({f"shift/{k}": v for k, v in r["shift"]["all"].items()})
    for score in ("msp", "entropy"):
        for m in ("auroc", "aupr"):
            vals = [o[score][m] for o in r["ood"].values()]
            out[f"ood_{score}/{m}"] = float(np.mean(vals))
            out.update({f"ood_{score}/{m}/{name}": o[score][m] for name, o in r["ood"].items()})
    return out


def fmt(vals, best):
    m = float(np.mean(vals))
    se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    s = f"{m:.2f}" if se < 5e-3 else f"{m:.2f}±{se:.2f}"
    return f"**{s}**" if best else s


def table():
    lines = ["# Like-for-like comparison in DIRECTOR's metrics", "",
             "Units as in DIRECTOR: ACC/MCC, AURC, AUROC, FPR95, ECE, Brier in %; NLL x10. "
             "Mean ± s.e. over seeds (NLP: 5 seeds; vision: one fine-tuned checkpoint). "
             "Bold: best single-model row with every seed (the ensemble is shown but not ranked; "
             "for NLP it is the same 5-seed ensemble in every seed, so it has no s.e.).", ""]
    summary = {}
    for ladder, dirs in LADDER_DIRS.items():
        seeds = [load_seed(d) for d in dirs if os.path.exists(f"{d}/director_eval.json")]
        if not seeds:
            continue
        flat = {name: [flatten(s[name]) for s in seeds if name in s] for name in ROW_ORDER}
        flat = {k: v for k, v in flat.items() if v}
        summary[ladder] = {k: {m: [x[m] for x in v] for m in v[0]} for k, v in flat.items()}
        acc_key = "mcc" if ladder == "cola" else "acc"   # DIRECTOR scores only CoLA by MCC
        for section, cols, title in (("id", [acc_key] + ID_COLS[1:], "in-distribution (main axis)"),
                                     ("shift", [acc_key] + SHIFT_COLS[1:], "distribution shift"),
                                     ("ood", None, "OOD detection, mean over OOD sets")):
            if section == "ood":
                keys = ["ood_msp/auroc", "ood_msp/aupr", "ood_entropy/auroc", "ood_entropy/aupr"]
                header = ["MSP AUROC", "MSP AUPR", "Entropy AUROC", "Entropy AUPR"]
            else:
                keys = [f"{section}/{c}" for c in cols]
                header = [c.upper() + (" ↓" if c in LOWER else " ↑") for c in cols]
            lines += [f"## {ladder}: {title} (n_seeds={len(seeds)})", "",
                      "| Method | " + " | ".join(header) + " |", "|---" * (len(keys) + 1) + "|"]
            ranked = [k for k in flat if k != "Ensemble" and len(flat[k]) == len(seeds)]
            for name in flat:
                cells = []
                for k in keys:
                    vals = summary[ladder][name][k]
                    means = {r: np.mean(summary[ladder][r][k]) for r in ranked}
                    lower = k.split("/")[-1] in LOWER
                    best = name in ranked and np.mean(vals) == (min(means.values()) if lower else max(means.values()))
                    cells.append(fmt(vals, best))
                n = len(flat[name])
                tag = f" ({n} seeds)" if n != len(seeds) else ""
                lines.append(f"| {name}{tag} | " + " | ".join(cells) + " |")
            lines.append("")
    out = f"{HERE}/../director_comparison"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/tables.md", "w") as f:
        f.write("\n".join(lines))
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f)
    print("\n".join(lines))


if __name__ == "__main__":
    {"eval": run_eval, "table": table}[sys.argv[1]]()
