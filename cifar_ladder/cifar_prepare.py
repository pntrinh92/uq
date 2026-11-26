"""CIFAR ladder, DIRECTOR setting: fine-tune ViT-B/16 on CIFAR and cache features for the ladder.

Same checkpoint as the ImageNet ladder (timm vit_base_patch16_224.orig_in21k_ft_in1k, the
weights of HF google/vit-base-patch16-224 used by DIRECTOR), new C-way head, 32 px images
upsampled to 224 (bicubic). Split: 45k fine-tune / 5k held out (calA 2.5k, calB 2.5k) from
the official training set, official 10k test set.

Feature caches use the ImageNet layout, so compile_flow.py, dropout_baseline.py,
ladder_eval.py and partitioned_regen.py run unchanged with LADDER=cifar10 or cifar100:
  train (45k), val (5k held out + 10k test), shift (CIFAR-C: 15 corruptions x 5 severities
  x 400 test images), svhn, cifar10/cifar100 (the other one), lsun (LSUN_resize), dtd, noise.

Usage (LADDER=cifar10|cifar100 in the environment):
  python cifar_prepare.py finetune [epochs]
  python cifar_prepare.py extract [set ...]
"""
import glob
import io
import math
import os
import sys
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imagenet_ladder"))
from inl_common import (CKPT, CORRUPTIONS, FEAT, FM_NAME, LADDER, NUM_CLASSES, OOD_SETS, embed, load_fm,  # noqa: E402
                        pick_gpu, readout, save_json, set_npy)

CDATA = "/ephemeral/trinhpn_uq/cifar_data"
DTD = "/ephemeral/trinhpn_uq/data/dtd/images"
SPLIT_SEED = 0
N_PER_CORRUPTION_SEVERITY = 400
N_OOD = 5000


# ----------------------------------------------------------------- raw 32x32 uint8 arrays
def decode(col):
    return np.stack([np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB").resize((32, 32), Image.BICUBIC))
                     for c in col]).transpose(0, 3, 1, 2)


def cifar(name, split):
    import pyarrow.parquet as pq
    sub = "plain_text" if name == "cifar10" else "cifar100"
    t = pq.read_table(f"{CDATA}/{name}/{sub}/{split}-00000-of-00001.parquet")
    label = "label" if name == "cifar10" else "fine_label"
    return decode(t.column("img").to_pylist()), np.array(t.column(label).to_pylist())


def splits():
    x, y = cifar(LADDER, "train")
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(x))
    held, tr = perm[:5000], perm[5000:]
    xt, yt = cifar(LADDER, "test")
    return (x[tr], y[tr]), (np.concatenate([x[held], xt]), np.concatenate([y[held], yt])), (xt, yt)


def cifar_c(xt_labels):
    """CIFAR-C subset: for every corruption and severity, the same 400 random test indices."""
    root = f"{CDATA}/CIFAR-10-C" if LADDER == "cifar10" else f"{CDATA}/CIFAR-100-C"
    labels = np.load(f"{root}/labels.npy")
    idx = np.sort(np.random.default_rng(1).choice(10_000, N_PER_CORRUPTION_SEVERITY, replace=False))
    xs, ys, cs, sv = [], [], [], []
    for c in CORRUPTIONS:
        arr = np.load(f"{root}/{c}.npy", mmap_mode="r")
        for s in range(5):
            xs.append(np.asarray(arr[s * 10_000 + idx]).transpose(0, 3, 1, 2))
            ys.append(labels[s * 10_000 + idx])
            cs += [c] * len(idx)
            sv += [s + 1] * len(idx)
    return np.concatenate(xs), np.concatenate(ys), np.array(cs), np.array(sv)


def ood(name):
    rng = np.random.default_rng(2)
    if name == "svhn":
        import pyarrow.parquet as pq
        t = pq.read_table(f"{CDATA}/svhn/cropped_digits/test-00000-of-00001.parquet")
        col = t.column("image").to_pylist()
        keep = np.sort(rng.choice(len(col), N_OOD, replace=False))
        return decode([col[i] for i in keep])
    if name in ("cifar10", "cifar100"):
        x, _ = cifar(name, "test")
        return x[np.sort(rng.choice(len(x), N_OOD, replace=False))]
    if name == "lsun":
        files = sorted(glob.glob(f"{CDATA}/LSUN_resize/LSUN_resize/*.jpg"))
        files = [files[i] for i in np.sort(rng.choice(len(files), N_OOD, replace=False))]
        return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]).transpose(0, 3, 1, 2)
    if name == "dtd":
        files = sorted(glob.glob(f"{DTD}/*/*.jpg"))
        files = [files[i] for i in np.sort(rng.choice(len(files), N_OOD, replace=False))]
        # Same protocol as LSUN_resize: downsample to 32 px, then the shared 32 -> 224 upsampling.
        return np.stack([np.asarray(Image.open(f).convert("RGB").resize((32, 32), Image.BICUBIC))
                         for f in files]).transpose(0, 3, 1, 2)
    if name == "noise":
        return (rng.uniform(0, 1, (2000, 3, 32, 32)) * 255).astype(np.uint8)
    raise ValueError(name)


# ----------------------------------------------------------------- 32 -> 224 on the GPU
def to_input(x_uint8, device, augment=False):
    x = torch.from_numpy(x_uint8).to(device).float() / 255.0
    if augment:
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        i, j = np.random.randint(0, 9, 2)
        x = x[:, :, i:i + 32, j:j + 32]
        flip = torch.rand(x.shape[0], device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = F.interpolate(x, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - 0.5) / 0.5


# ----------------------------------------------------------------- fine-tuning
def finetune(epochs=3, batch=64, lr=5e-5, wd=0.05):
    device = pick_gpu(min_free_mb=4500)
    (xtr, ytr), (xval, yval), _ = splits()
    model = timm.create_model(FM_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)
    model.set_grad_checkpointing(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * (len(xtr) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ytr_t = torch.from_numpy(ytr).to(device)
    log, t0, step = [], time.time(), 0
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(len(xtr))
        for i in range(0, len(perm) - batch + 1, batch):
            idx = perm[i:i + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(to_input(xtr[idx], device, augment=True)), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        acc = evaluate(model, xval[:5000], yval[:5000], device)
        log.append(dict(epoch=ep, heldout_acc=acc, sec=time.time() - t0))
        print(log[-1], flush=True)
        os.makedirs(CKPT, exist_ok=True)
        torch.save(model.state_dict(), f"{CKPT}/finetuned.pt")
    save_json(log, "finetune_log.json")


@torch.no_grad()
def evaluate(model, x, y, device, bs=256):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correct += (model(to_input(x[i:i + bs], device)).argmax(-1).cpu().numpy() == y[i:i + bs]).sum()
    return float(correct / len(x))


# ----------------------------------------------------------------- feature extraction
@torch.no_grad()
def extract_set(name, x, labels, model, device, extra=None, bs=128):
    n = len(x)
    x0 = open_memmap(f"{FEAT}/{name}_x0.npy", mode="w+", dtype=np.float16, shape=(n, 197, 768))
    cls = np.zeros((n, 12, 768), np.float16)
    z = np.zeros((n, 768), np.float32)
    logits = np.zeros((n, 1, NUM_CLASSES), np.float16)
    t0 = time.time()
    for i in range(0, n, bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = embed(model, to_input(x[i:i + bs], device))
            x0[i:i + len(tok)] = tok.float().cpu().numpy().astype(np.float16)
            for li, blk in enumerate(model.blocks):
                tok = blk(tok)
                cls[i:i + len(tok), li] = tok[:, 0].float().cpu().numpy()
            zz, lg = readout(model, tok)
        z[i:i + len(tok)] = zz.float().cpu().numpy()
        logits[i:i + len(tok), 0] = lg.float().cpu().numpy()
    x0.flush()
    set_npy(f"{name}_cls", cls)
    set_npy(f"{name}_z", z)
    set_npy(f"{name}_logits", logits)
    set_npy(f"{name}_labels", labels)
    for k, v in (extra or {}).items():
        set_npy(f"{name}_{k}", v)
    print(f"[{name}] {n} images in {time.time() - t0:.0f}s", flush=True)


def extract(sets):
    device = pick_gpu()
    os.makedirs(FEAT, exist_ok=True)
    model = load_fm(device)
    (xtr, ytr), (xval, yval), (xt, yt) = splits()
    for name in sets:
        if name == "train":
            extract_set("train", xtr, ytr, model, device)
        elif name == "val":
            extract_set("val", xval, yval, model, device)
        elif name == "shift":
            xs, ys, cs, sv = cifar_c(yt)
            extract_set("shift", xs, ys, model, device, extra={"corruption": cs, "severity": sv})
        else:
            xo = ood(name)
            extract_set(name, xo, np.full(len(xo), -1), model, device)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "finetune":
        finetune(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == "extract":
        extract(sys.argv[2:] or ["val", "train", *OOD_SETS, "shift"])
