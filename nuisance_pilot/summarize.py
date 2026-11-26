"""Mean +- s.e. over seeds of pilot2 results -> markdown tables (stdout and tables.md)."""
import glob
import json
import os

import numpy as np

ROOT = "/ephemeral/trinhpn_uq/nuisance_pilot2"
IN_G = ["space_toggle", "double_space", "newline", "period_toggle"]
OUT_G = ["capitalize_toggle", "upper", "prefix", "quotes"]
SHORT = {"orig": "orig", "space_toggle": "space", "double_space": "2space", "newline": "\\n", "period_toggle": "period",
         "capitalize_toggle": "Cap", "upper": "UPPER", "prefix": "prefix", "quotes": "quotes"}
ROWS = [("raw_last", "maha_multi"), ("raw_last", "maha_z"), ("raw_last", "msp"), ("raw_last", "energy"),
        ("raw_lastc", "maha_multi"), ("raw_lastc", "maha_z"), ("raw_mean", "maha_multi"), ("raw_mean", "maha_z"),
        ("canon_last", "maha_multi"), ("canon_last", "maha_z"), ("canon_last", "msp"),
        ("orbit_last", "maha_multi"), ("orbit_last", "maha_z"), ("orbit_last", "msp")]


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.1f}" if len(v) == 1 else f"{v.mean():.1f}±{v.std(ddof=1) / np.sqrt(len(v)):.1f}"


lines = []
for task in ["sst2", "cola"]:
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/{task}/seed*/results.json"))]
    if not rs:
        continue
    ood = list(rs[0]["raw_last"]["scores"]["maha_z"]["tpr"])
    lines += [f"\n## {task} (n_seeds={len(rs)}); ID false-alarm rate at alpha=5%, mean over 200 calibration re-draws",
              "", "| pipeline | score | " + " | ".join(SHORT[k] for k in ["orig", *IN_G, *OUT_G]) +
              " | " + " | ".join(f"TPR {o}" for o in ood) + " | " + " | ".join(f"AUROC {o}" for o in ood) + " |",
              "|" + "---|" * (2 + 9 + 2 * len(ood))]
    for p, s in ROWS:
        cells = [ms([r[p]["scores"][s]["far_rs"][k] for r in rs]) for k in ["orig", *IN_G, *OUT_G]]
        cells += [ms([r[p]["scores"][s]["tpr_rs"][o] for r in rs]) for o in ood]
        cells += [ms([r[p]["scores"][s]["auroc"][o] for r in rs]) for o in ood]
        lines.append(f"| {p} | {s} | " + " | ".join(cells) + " |")
    lines += ["", "Accuracy of the classifier on each input format (the pipeline decides what the classifier reads):", "",
              "| pipeline | " + " | ".join(SHORT[k] for k in ["orig", *IN_G, *OUT_G]) + " |", "|" + "---|" * 10]
    for p in ["raw_last", "canon_last"]:
        lines.append(f"| {p} | " + " | ".join(ms([r[p]["acc"][k] for r in rs]) for k in ["orig", *IN_G, *OUT_G]) + " |")
txt = "\n".join(lines)
print(txt)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.md"), "w").write(
    "# Pilot 2: format-robust conformal OOD flags (Qwen2.5-0.5B)\n\nIn-group transforms: space, 2space, \\n, period. "
    "Out-of-group: Cap, UPPER, prefix, quotes.\n" + txt + "\n")
