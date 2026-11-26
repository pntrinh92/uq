"""Aggregate the DIRECTOR reproduction (authors' code, README commands) and set it against the paper's tables.

Sources
  paper : paper/neurips_2026.tex (arXiv 2602.08920 source), Tables 15-18 and the OOD-detection table.
  ours  : the LAST attempt of state/<job>.log ("---> Test ... results" lines) for in-distribution, CoLA-OOD and
          OOD-detection numbers. The repo's test_results.csv files are not used: TS / MCD / KFLLA write into the
          ViT's folder and the OOD tests write into the base model's folder, so those CSVs get overwritten.
          CIFAR-10-C comes from cifar10c_results_*.csv (mean over 15 corruptions x 5 severities per seed);
          ViT / TS / MCD / KFLLA use the separate re-tests in results/cifarc_sep (jobs c10c_*).
  Scale : the code logs AURC x1000, the paper reports x100, so AURC is divided by 10. "FPR" in the logs is FPR95.
  Std   : sample std (ddof=1), as in the authors' utils.compute_statistics.

Usage: python aggregate.py  ->  tables.md, summary.json (next to this file)
"""
import csv
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE = f"{ROOT}/state"
CIFAR = f"{ROOT}/TransDiff/CIFAR/results"
TEX = f"{ROOT}/paper/neurips_2026.tex"
SEEDS = range(5)

# ----------------------------------------------------------------------------------------------- paper tables
def tex_table(caption_start):
    """Return the source of the (uncommented) table whose caption starts with caption_start."""
    lines = [l.split("%")[0] if not l.lstrip().startswith("%") else "" for l in open(TEX, encoding="utf-8")]
    src = "\n".join(lines)
    i = src.find("\\caption{" + caption_start)
    assert i >= 0, caption_start
    a, b = src.rfind("\\begin{table", 0, i), src.find("\\end{table", i)
    return src[a:b]


NUM = re.compile(r"(-?\d+\.\d+)[^\d±&]*±\s*(\d+\.\d+)")


def clean(cell):
    cell = cell.replace("$\\pm$", "±").replace("\\pm", "±").replace("$", "")
    for _ in range(3):
        cell = re.sub(r"\\(?:textbf|textit|rotatebox\{\d+\}|parbox\{[^}]*\})\{([^{}]*)\}", r"\1", cell)
    cell = re.sub(r"\\(?:mid|top|bottom)rule|\\cmidrule(?:\([a-z]+\))?\{[^}]*\}|\\centering|\\hline", " ", cell)
    return cell


def parse_rows(src, ncols):
    """Yield (labels, values) for every data row; values = [(mean, std)] * ncols."""
    body = src[src.find("\\midrule"):]
    for row in body.split("\\\\"):
        cells = [clean(c) for c in row.split("&")]
        vals = [m for c in cells for m in NUM.findall(c)]
        if len(vals) < ncols:
            continue
        labels = [re.sub(r"\s+", " ", re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}|[{}*]", " ", c)).strip()
                  for c in cells if not NUM.search(c)]
        yield [l for l in labels if l], [(float(a), float(b)) for a, b in vals[:ncols]]


ALIAS = {"MC Dropout": "MCD", "SVDKL": "SV-DKL"}


def name_rows(rows, datasets=None):
    """Attach method names; \\ourmethod rows become DIRECTOR(<base above>)."""
    out, base, ds = {}, None, None
    for labels, vals in rows:
        for l in labels:
            for d in datasets or []:
                if re.fullmatch(r"(\\centering )?" + re.escape(d), l.split("}")[-1].strip()) or l.endswith(" " + d):
                    ds = d
        meth = [l for l in labels if not any(l.endswith(d) for d in datasets or [])]
        m = meth[-1] if meth else "?"
        if "ourmethod" in m:
            m = f"DIRECTOR({base})"
        else:
            m = ALIAS.get(m, m)
            base = m
        out[(ds, m) if datasets else m] = vals
    return out


def paper_tables():
    t16 = name_rows(parse_rows(tex_table("Performance comparison for in-distribution classification across four"), 7),
                    datasets=["CIFAR-100", "CIFAR-10", "IMDB", "CoLA"])
    t17 = name_rows(parse_rows(tex_table("Performance comparison on CIFAR-10-C, with 15"), 7))
    t18 = name_rows(parse_rows(tex_table("Performance comparison on CoLA OOD over five"), 7))
    ood = name_rows(parse_rows(tex_table("Comparison of average OOD detection performance in AUROC"), 6))
    kd = {}
    for labels, vals in parse_rows(tex_table("Performance comparison between knowledge distillation"), 8):
        lab = " ".join(labels)
        key = ("Teacher ViT (7 layers)" if "Teacher" in lab else "Student ViT (3 layers)" if "3 layers" in lab
               else "Student DiT" if "DiT" in lab else "DIRECTOR(ViT)")
        kd[key] = vals
    return t16, t17, t18, ood, kd


# ----------------------------------------------------------------------------------------------- our results
def last_attempt(jid):
    p = f"{STATE}/{jid}.log"
    if not os.path.exists(p) or not os.path.exists(f"{STATE}/{jid}.done"):
        return None
    t = open(p, errors="ignore").read()
    return t[t.rfind("### attempt"):]


def result_line(jid, kind):
    t = last_attempt(jid)
    if t is None:
        return None
    hits = re.findall(r"---> Test " + re.escape(kind) + r" results：\t(.*)", t)
    if not hits:
        return None
    return {k.strip(): float(v) for k, v in (kv.split(": ") for kv in hits[-1].strip().split("\t"))}


ID_COLS = ["ACC", "AURC", "AUROC", "FPR95", "ECE", "NLL", "Brier"]


def id_metrics(r, mcc):
    return [r["MCC"] if mcc else r["Acc."], r["AURC"] / 10, r["AUROC"], r["FPR"], r["ECE"], r["NLL"], r["Brier"]]


def stats(per_seed):
    a = np.array([v for v in per_seed if v is not None], dtype=float)
    if len(a) == 0:
        return None
    sd = a.std(0, ddof=1) if len(a) > 1 else np.full(a.shape[1], np.nan)
    return {"mean": a.mean(0).tolist(), "std": sd.tolist(), "n": len(a)}


def methods(ds):
    v = "vit" if ds in ("c10", "c100") else "tf"
    base_lbl = "ViT" if v == "vit" else "Transformer"
    keps = ["kep1", "kep2", "kep7"] if v == "vit" else ["kep1", "kep2", "kep5"]
    n = 7 if v == "vit" else 5
    rows = [("temperature_scaling", "TS"), ("mc_dropout", "MCD"), ("kflla", "KFLLA"), ("svdkl", "SV-DKL"),
            ("sgpa", "SGPA"), (v, base_lbl), (f"dit_{v}", f"DIRECTOR({base_lbl})")]
    for k in keps:
        lbl = f"KEP-{k[3:]}/{n}"
        rows += [(k, lbl), (f"dit_{k}", f"DIRECTOR({lbl})")]
    return rows


def our_id(ds):
    kind = "MSP" if ds in ("c10", "c100") else "Test Evaluation"
    out = {}
    for grp, lbl in methods(ds):
        per = []
        for s in SEEDS:
            r = result_line(f"{ds}_{grp}_s{s}", kind)
            per.append(id_metrics(r, ds == "cola") if r else None)
        out[lbl] = stats(per)
    return out


def our_cola_ood():
    out = {}
    for grp, lbl in methods("cola"):
        per = []
        for s in SEEDS:
            r = result_line(f"cola_{grp}_s{s}", "OOD Robustness")
            per.append(id_metrics(r, True) if r else None)
        out[lbl] = stats(per)
    return out


def our_ood():
    out = {}
    for grp, lbl in methods("c10"):
        if grp == "temperature_scaling":
            continue
        per = []
        for s in SEEDS:
            row = []
            for d in ("svhn", "cifar100", "lsun"):
                r = result_line(f"c10_ood_{d}_{grp}_s{s}", "MSP")
                row += [r["AUROC"], r["AUPR"]] if r else [None, None]
            per.append(row if None not in row else None)
        out[lbl] = stats(per)
    return out


def cifarc_csv(lbl, s):
    k = {"KEP-1/7": 1, "KEP-2/7": 2, "KEP-7/7": 7}
    sep = {"ViT": "vit_cifar", "TS": "temperature_scaling", "MCD": "mc_dropout", "KFLLA": "kflla"}
    if lbl in sep:
        return f"{CIFAR}/cifarc_sep/{sep[lbl]}/cifar10_softmax_vit_cifar_{s}/cifar10c_results_softmax.csv", f"c10c_{sep[lbl]}_s{s}"
    if lbl == "SV-DKL":
        return f"{CIFAR}/vit_out/cifar10_softmax_svdkl_{s}/cifar10c_results_softmax.csv", f"c10_svdkl_s{s}"
    if lbl == "SGPA":
        return f"{CIFAR}/vit_out/cifar10_sgpa_vit_cifar_{s}/cifar10c_results_sgpa.csv", f"c10_sgpa_s{s}"
    if lbl in k:
        return (f"{CIFAR}/vit_out_cat/cifar10_kep_svgp_vit_cifar_ksvdlayer{k[lbl]}_ksvd10.0_kl1.0_{s}/"
                f"cifar10c_results_kep_svgp.csv", f"c10_kep{k[lbl]}_s{s}")
    if lbl == "DIRECTOR(ViT)":
        return (f"{CIFAR}/diffusion/cifar10_softmax_diffusion_{s}_transformer_1_12_1.0_0.1_0.001_100/"
                f"cifar10c_results_softmax.csv", f"c10_dit_vit_s{s}")
    b = re.match(r"DIRECTOR\(KEP-(\d)/7\)", lbl)
    if b:
        return (f"{CIFAR}/diffusion/cifar10_kep_svgp_diffusion_ksvdlayer{b[1]}_ksvd10.0_kl1.0_{s}_transformer_1_12_1.0"
                f"_0.1_0.001_100/cifar10c_results_kep_svgp.csv", f"c10_dit_kep{b[1]}_s{s}")
    return None, None


def read_cifarc(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if len(rows) != 75:
        return None
    col = {"ACC": "Acc.", "AURC": "AURC", "AUROC": "AUROC", "FPR95": "FPR", "ECE": "ECE", "NLL": "NLL", "Brier": "Brier"}
    return [np.mean([float(r[col[c]]) for r in rows]) / (10 if c == "AURC" else 1) for c in ID_COLS]


def our_cifarc():
    out = {}
    for _, lbl in methods("c10"):
        per = []
        for s in SEEDS:
            p, jid = cifarc_csv(lbl, s)
            ok = p and os.path.exists(p) and os.path.exists(f"{STATE}/{jid}.done")
            per.append(read_cifarc(p) if ok else None)
        out[lbl] = stats(per)
    return out


def our_kd():
    out = {}
    for lbl, grp in [("Teacher ViT (7 layers)", "vit"), ("Student ViT (3 layers)", "distill_vit3"),
                     ("Student DiT", "distill_dit"), ("DIRECTOR(ViT)", "dit_vit")]:
        row = []
        for ds in ("c10", "c100"):
            per = []
            for s in SEEDS:
                r = result_line(f"{ds}_{grp}_s{s}", "MSP")
                per.append([r["Acc."], r["ECE"], r["NLL"], r["Brier"]] if r else None)
            row.append(stats(per))
        out[lbl] = row
    return out


# ----------------------------------------------------------------------------------------------- report
def fmt(m, s, n=5):
    if m is None:
        return "–"
    t = f"{m:.2f}" + ("" if np.isnan(s) else f"±{s:.2f}")
    return t + (f" (n={n})" if n < 5 else "")


def table(title, cols, ours, paper, note=""):
    lines = [f"### {title}", "", "Each cell: **ours** mean±std over seeds / paper. " + note, "",
             "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for lbl in list(dict.fromkeys(list(ours) + list(paper))):
        o, p = ours.get(lbl), paper.get(lbl)
        cells = []
        for i in range(len(cols)):
            a = fmt(o["mean"][i], o["std"][i], o["n"]) if o else "–"
            b = f"{p[i][0]:.2f}" if p else "–"
            cells.append(f"**{a}** / {b}")
        lines.append(f"| {lbl} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


HIGHER = {"ACC": 1, "MCC": 1, "AUROC": 1, "AUPR": 1, "AURC": -1, "FPR95": -1, "ECE": -1, "NLL": -1, "Brier": -1}


def claim_check(name, cols, ours, paper):
    """For every base -> DIRECTOR(base) pair, does the sign of the change agree between paper and ours?"""
    agree = total = 0
    rows = []
    for lbl in paper:
        b = re.fullmatch(r"DIRECTOR\((.+)\)", lbl)
        if not b or b[1] not in paper or not ours.get(lbl) or not ours.get(b[1]):
            continue
        base = b[1]
        for i, c in enumerate(cols):
            sgn = HIGHER[c.split()[-1]]
            dp = (paper[lbl][i][0] - paper[base][i][0]) * sgn
            do = (ours[lbl]["mean"][i] - ours[base]["mean"][i]) * sgn
            total += 1
            agree += (dp > 0) == (do > 0)
            rows.append((base, c, dp * sgn, do * sgn))
    return {"table": name, "agree": agree, "total": total, "pairs": rows}


def main():
    t16, t17, t18, ood_p, kd_p = paper_tables()
    md = ["# DIRECTOR reproduction vs paper (arXiv 2602.08920)", "",
          "Authors' code (TransDiff, branch `supplementary`), README commands with the fixes listed in "
          "`jobs.json` (`fix` fields). Generated by `aggregate.py`.", ""]
    summary = {"paper": {}, "ours": {}, "claims": []}
    dsn = {"c10": "CIFAR-10", "c100": "CIFAR-100", "imdb": "IMDB", "cola": "CoLA"}
    for ds, name in dsn.items():
        ours = our_id(ds)
        paper = {m: v for (d, m), v in t16.items() if d == name}
        cols = ["MCC" if ds == "cola" else "ACC"] + ID_COLS[1:]
        md.append(table(f"Table 16 — in-distribution, {name}", cols, ours, paper))
        summary["ours"][f"t16_{ds}"], summary["paper"][f"t16_{ds}"] = ours, {k: v for k, v in paper.items()}
        summary["claims"].append(claim_check(f"T16 {name}", cols, ours, paper))
    ours = our_cifarc()
    md.append(table("Table 17 — CIFAR-10-C (mean over 15 corruptions × 5 severities)", ID_COLS, ours, t17))
    summary["ours"]["t17"], summary["paper"]["t17"] = ours, t17
    summary["claims"].append(claim_check("T17 CIFAR-10-C", ID_COLS, ours, t17))
    ours = our_cola_ood()
    cols = ["MCC"] + ID_COLS[1:]
    md.append(table("Table 18 — CoLA OOD", cols, ours, t18))
    summary["ours"]["t18"], summary["paper"]["t18"] = ours, t18
    summary["claims"].append(claim_check("T18 CoLA OOD", cols, ours, t18))
    ours = our_ood()
    cols = ["SVHN AUROC", "SVHN AUPR", "C100 AUROC", "C100 AUPR", "LSUN AUROC", "LSUN AUPR"]
    md.append(table("OOD detection, CIFAR-10 in-distribution (Table 5 / appendix)", cols, ours, ood_p,
                    "AUROC/AUPR = mean of the softmax- and entropy-score values, as in the authors' val_ood.py."))
    summary["ours"]["ood"], summary["paper"]["ood"] = ours, ood_p
    summary["claims"].append(claim_check("OOD detection", cols, ours, ood_p))
    kd = our_kd()
    cols = ["C10 ACC", "C10 ECE", "C10 NLL", "C10 Brier", "C100 ACC", "C100 ECE", "C100 NLL", "C100 Brier"]
    kd_flat = {}
    for lbl, (a, b) in kd.items():
        if a or b:
            m = (a["mean"] if a else [None] * 4) + (b["mean"] if b else [None] * 4)
            s = (a["std"] if a else [np.nan] * 4) + (b["std"] if b else [np.nan] * 4)
            kd_flat[lbl] = {"mean": m, "std": s, "n": min(x["n"] for x in (a, b) if x)}
    md.append(table("Table 15 — knowledge distillation", cols, kd_flat, kd_p,
                    "The README has CIFAR-10 distillation commands only. The ViT-3 student uses 100 epochs "
                    "(README: 0, which trains nothing; the paper gives no value)."))
    summary["ours"]["t15"], summary["paper"]["t15"] = kd_flat, kd_p
    md.append("### Direction of DIRECTOR's change vs its base model (paper vs ours)\n")
    md.append("| Table | same direction | pairs × metrics |\n|---|---|---|")
    for c in summary["claims"]:
        md.append(f"| {c['table']} | {c['agree']} | {c['total']} |")
    md.append("")
    open(f"{ROOT}/tables.md", "w").write("\n".join(md) + "\n")
    json.dump(summary, open(f"{ROOT}/summary.json", "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
