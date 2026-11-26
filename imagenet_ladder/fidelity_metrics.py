"""Option 3: softer fidelity metrics for the compiled flows, from cached logits (CPU only).

For each compile (last 4 / all 12 blocks) on each ladder, on the test split:
  top-1 agreement  argmax q = argmax p_FM
  top-5 agreement  argmax p_FM is in the top 5 of q
  KL(p_FM || q)    mean over inputs, in nats
  accuracy         of the compiled flow, and of the FM for reference
Usage: python fidelity_metrics.py   (writes results/fidelity_metrics.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LADDERS = {
    "imagenet": ("/ephemeral/trinhpn_uq/features", os.path.join(HERE, "results")),
    "cifar10": ("/ephemeral/trinhpn_uq/cifar10/features", os.path.join(HERE, "..", "cifar_ladder", "results_cifar10")),
    "cifar100": ("/ephemeral/trinhpn_uq/cifar100/features", os.path.join(HERE, "..", "cifar_ladder", "results_cifar100")),
}


def log_softmax(lg):
    lg = lg.astype(np.float64)
    lg = lg - lg.max(-1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(-1, keepdims=True))


def test_index(ladder):
    if ladder == "imagenet":
        return np.sort(np.random.default_rng(0).permutation(50_000)[10_000:])
    return np.arange(5000, 15_000)


def metrics(fm_logits, logits, labels):
    lp_fm, lq = log_softmax(fm_logits), log_softmax(logits)
    fm_top1 = lp_fm.argmax(1)
    top5 = np.argsort(-lq, axis=1)[:, :5]
    return {
        "top1_agree": float((lq.argmax(1) == fm_top1).mean()),
        "top5_agree": float((top5 == fm_top1[:, None]).any(1).mean()),
        "kl_fm_to_flow": float((np.exp(lp_fm) * (lp_fm - lq)).sum(1).mean()),
        "acc_flow": float((lq.argmax(1) == labels).mean()),
        "acc_fm": float((fm_top1 == labels).mean()),
    }


def flow_logits(feat, tag, chunk=10_000):
    lg = np.load(f"{feat}/{tag}_val_logits.npy", mmap_mode="r")
    return lg[:, 0] if lg.ndim == 3 else lg


if __name__ == "__main__":
    out = {}
    for ladder, (feat, res) in LADDERS.items():
        idx = test_index(ladder)
        fm = np.load(f"{feat}/val_logits.npy", mmap_mode="r")[:, 0][idx].astype(np.float32)
        y = np.load(f"{feat}/val_labels.npy")[idx].astype(np.int64)
        for tag in ("ode8", "ode"):
            path = f"{feat}/{tag}_val_logits.npy"
            if os.path.exists(path):
                m = metrics(fm, np.asarray(flow_logits(feat, tag))[idx].astype(np.float32), y)
                out[f"{ladder}/{tag}"] = m
                print(f"{ladder:9s} {tag:5s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
    with open(os.path.join(HERE, "results", "fidelity_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
