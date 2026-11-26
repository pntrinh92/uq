"""Run the frozen ViT-B/16 backbones once over every image set and cache what the ladder needs.

Per set: input embeddings x0 (for the compiled flow), CLS token after every block,
final representation z, FM logits, and logits of the two extra checkpoints that
form the 3-model ensemble. Sets: train (23 shards), val (50k), shift (ImageNet-C-style
corruptions of 5k test images, 4 types), imagenet_o, dtd, noise.

Usage: python extract_features.py [set ...]
"""
import glob
import sys
import time

import numpy as np
import timm
import torch
from numpy.lib.format import open_memmap

from inl_common import (CORRUPTIONS, ENSEMBLE_NAMES, FEAT, IMAGENET, N_SHIFT_PER_CORRUPTION, N_TRAIN_SHARDS, ImageSet,
                        embed, load_parquet_rows, loader, ood_files, pick_gpu, readout, real_labels, set_npy,
                        val_split)


def build_set(name):
    if name == "train":
        shards = sorted(glob.glob(f"{IMAGENET}/train-*-of-00294.parquet"))
        assert len(shards) == N_TRAIN_SHARDS, f"expected {N_TRAIN_SHARDS} train shards, found {len(shards)}"
        imgs, labels, _ = load_parquet_rows("train-*-of-00294.parquet")
        return ImageSet(imgs), labels, {}
    if name == "train2":   # option 2: 50 more class-shuffled train shards (23 to 72), stored on /ephemeral
        imgs, labels, _ = load_parquet_rows("train-*-of-00294.parquet", root="/ephemeral/trinhpn_uq/imagenet_extra/data")
        return ImageSet(imgs), labels, {}
    if name.startswith("train3"):   # full-data compile: shards 73-293 in three parts (a: 73-146, b: 147-220, c: 221-293)
        lo, hi = {"train3a": (73, 147), "train3b": (147, 221), "train3c": (221, 294)}[name]
        root = "/ephemeral/trinhpn_uq/imagenet_full/data"
        imgs, labels = [], []
        for i in range(lo, hi):
            im, lb, _ = load_parquet_rows(f"train-{i:05d}-of-00294.parquet", root=root)
            imgs += im
            labels.append(lb)
        return ImageSet(imgs), np.concatenate(labels), {}
    if name in ("val", "shift"):
        imgs, labels, names = load_parquet_rows("validation-*.parquet")
        if name == "val":
            real = real_labels(names)
            return ImageSet(imgs), labels, {"names": np.array(names), "real": real}
        test = val_split()["test"][:N_SHIFT_PER_CORRUPTION]
        src = [imgs[i] for _ in CORRUPTIONS for i in test]
        cor = [c for c in CORRUPTIONS for _ in test]
        return ImageSet(src, cor), np.concatenate([labels[test]] * len(CORRUPTIONS)), \
            {"corruption": np.array(cor), "val_index": np.concatenate([test] * len(CORRUPTIONS))}
    if name in ("imagenet_o", "dtd"):
        files = ood_files()[name]
        return ImageSet(files), np.full(len(files), -1), {}
    if name == "noise":
        return ImageSet([f"noise:{i}" for i in range(2000)]), np.full(2000, -1), {}
    raise ValueError(name)


@torch.no_grad()
def run(name, models, device, store_x0=True):
    ds, labels, extra = build_set(name)
    n = len(ds)
    print(f"[{name}] {n} images", flush=True)
    x0 = open_memmap(f"{FEAT}/{name}_x0.npy", mode="w+", dtype=np.float16, shape=(n, 197, 768)) if store_x0 else None
    cls = np.zeros((n, 12, 768), np.float16)
    z = np.zeros((n, 768), np.float32)
    logits = np.zeros((n, len(models), 1000), np.float16)
    i, t0 = 0, time.time()
    for batch in loader(ds):
        b = batch.shape[0]
        img = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fm = models[0]
            tok = embed(fm, img)
            if store_x0:
                x0[i:i + b] = tok.float().cpu().numpy().astype(np.float16)
            for li, blk in enumerate(fm.blocks):
                tok = blk(tok)
                cls[i:i + b, li] = tok[:, 0].float().cpu().numpy()
            zz, lg = readout(fm, tok)
            z[i:i + b] = zz.float().cpu().numpy()
            logits[i:i + b, 0] = lg.float().cpu().numpy()
            for mi, m in enumerate(models[1:], start=1):
                logits[i:i + b, mi] = m(img).float().cpu().numpy()
        i += b
        if (i // b) % 50 == 0:
            print(f"  {i}/{n}  {i / (time.time() - t0):.0f} img/s", flush=True)
    if store_x0:
        x0.flush()
    set_npy(f"{name}_cls", cls)
    set_npy(f"{name}_z", z)
    set_npy(f"{name}_logits", logits)
    set_npy(f"{name}_labels", labels)
    for k, v in extra.items():
        if k == "real":
            np.save(f"{FEAT}/{name}_real.npy", np.array(v, dtype=object), allow_pickle=True)
        else:
            set_npy(f"{name}_{k}", v)
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sets = sys.argv[1:] or ["val", "shift", "imagenet_o", "dtd", "noise", "train"]
    device = pick_gpu()
    models = [timm.create_model(n, pretrained=True).to(device).eval() for n in ENSEMBLE_NAMES]
    for s in sets:
        # The train pool only needs FM features; ensemble logits are for evaluation sets.
        run(s, models[:1] if s.startswith("train") else models, device, store_x0=True)
