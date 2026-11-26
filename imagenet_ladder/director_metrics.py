"""DIRECTOR's evaluation metrics (arXiv 2602.08920, Sec. 3.1), in DIRECTOR's units.

In-distribution and shift sets (Tables 1-4, 17, 18):
  acc (%), mcc (x100, binary tasks), aurc (%), auroc (%, failure prediction), fpr95 (%),
  ece (%, 15 bins), nll (x10), brier (%, summed over classes).
Failure prediction ranks inputs by the maximum softmax probability: a correct prediction is the
positive class, so AUROC is P(conf(correct) > conf(wrong)) and FPR95 is the fraction of wrong
predictions kept at 95% recall of the correct ones. AURC is the area under the risk-coverage curve.
OOD detection (Table 5): AUROC and AUPR (%) of the in-distribution test set against each OOD set,
scored by 1 - max softmax (MSP) and by predictive entropy; AUPR takes the OOD inputs as positives.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

N_BINS = 15


def _mcc(pred, y):
    tp, tn = np.sum((pred == 1) & (y == 1)), np.sum((pred == 0) & (y == 0))
    fp, fn = np.sum((pred == 1) & (y == 0)), np.sum((pred == 0) & (y == 1))
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def _ece(conf, correct):
    edges = np.linspace(0, 1, N_BINS + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def classification(probs, y):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    y = np.asarray(y, dtype=np.int64)
    pred, conf = p.argmax(1), p.max(1)
    correct = pred == y
    risk = np.cumsum(~correct[np.argsort(-conf, kind="stable")]) / np.arange(1, len(y) + 1)
    out = dict(acc=100 * correct.mean(), aurc=100 * risk.mean(),
               ece=100 * _ece(conf, correct), nll=10 * float(-np.log(p[np.arange(len(y)), y]).mean()),
               brier=100 * float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()))
    if correct.all() or not correct.any():
        out.update(auroc=float("nan"), fpr95=float("nan"))
    else:
        out["auroc"] = 100 * roc_auc_score(correct, conf)
        fpr, tpr, _ = roc_curve(correct, conf)
        out["fpr95"] = 100 * float(fpr[np.searchsorted(tpr, 0.95, side="left")])
    if p.shape[1] == 2:
        out["mcc"] = 100 * _mcc(pred, y)
    return {k: float(v) for k, v in out.items()}


def ood_detection(p_id, p_ood):
    """AUROC / AUPR (%) of ID vs OOD, OOD positive, for the MSP and entropy scores."""
    out = {}
    for name, f in (("msp", lambda p: 1 - p.max(1)),
                    ("entropy", lambda p: -(np.clip(p, 1e-12, None) * np.log(np.clip(p, 1e-12, None))).sum(1))):
        s = np.concatenate([f(np.asarray(p_id, np.float64)), f(np.asarray(p_ood, np.float64))])
        lab = np.concatenate([np.zeros(len(p_id)), np.ones(len(p_ood))])
        out[name] = dict(auroc=100 * float(roc_auc_score(lab, s)), aupr=100 * float(average_precision_score(lab, s)))
    return out


def evaluate(probs, y_test, test_idx, y_shift, shift_groups, ood_sets):
    """probs: dict set -> (N, K) with "val" on the full held-out + test array (indexed by test_idx),
    "shift" and every OOD set. shift_groups: name -> boolean mask over the shift set ("all" included)."""
    pt = probs["val"][test_idx]
    return dict(id=classification(pt, y_test),
                shift={g: classification(probs["shift"][m], y_shift[m]) for g, m in shift_groups.items()},
                ood={o: ood_detection(pt, probs[o]) for o in ood_sets})
