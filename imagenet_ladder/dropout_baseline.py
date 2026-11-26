"""Baseline: test-time dropout injection on the FM (the checkpoint was trained without dropout).

Dropout with rate p is enabled after the attention projection and inside the MLP of every
block; x0 caches are reused, so no image is decoded again. p is chosen on calB NLL
(S passes each), then S passes are run on every evaluation set with that p.
"""
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

from inl_common import FEAT, FM_NAME, OOD_SETS, get_npy, load_fm, pick_gpu, readout, save_json, val_split

P_GRID = [0.02, 0.05, 0.1]
S = 8


@torch.no_grad()
def mc_logits(model, x0, device, passes=S, bs=128):
    out = np.zeros((len(x0), passes, model.head.out_features), np.float16)
    for i in range(0, len(x0), bs):
        xb = torch.from_numpy(np.asarray(x0[i:i + bs], dtype=np.float32)).to(device)
        for s in range(passes):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = xb
                for blk in model.blocks:
                    tok = blk(tok)
                _, lg = readout(model, tok)
            out[i:i + len(xb), s] = lg.float().cpu().numpy()
    return out


def make(p, device):
    m = load_fm(device, proj_drop_rate=p)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout) and mod.p > 0:
            mod.train()
    return m


if __name__ == "__main__":
    device = pick_gpu()
    calB = np.sort(val_split()["calB"])
    val_x0 = get_npy("val_x0")
    yb = torch.from_numpy(get_npy("val_labels", mmap=False)[calB])
    xb = np.asarray(val_x0[calB])
    nll = {}
    for p in P_GRID:
        lg = torch.from_numpy(mc_logits(make(p, device), xb, device).astype(np.float32))
        prob = F.softmax(lg, -1).mean(1)
        nll[p] = -prob[torch.arange(len(yb)), yb].clamp_min(1e-12).log().mean().item()
        print("p", p, "calB NLL", nll[p], flush=True)
    best = min(nll, key=nll.get)
    model = make(best, device)
    for name in ["val", "shift", *OOD_SETS]:
        t0 = time.time()
        np.save(f"{FEAT}/dropout_{name}_logits.npy", mc_logits(model, get_npy(f"{name}_x0"), device))
        print(f"[dropout {name}] {time.time() - t0:.0f}s", flush=True)
    save_json({"calB_nll": {str(k): v for k, v in nll.items()}, "p": best, "passes": S}, "dropout_selection.json")
