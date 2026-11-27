"""Compile all 12 ViT-B/16 blocks on CIFAR under DIRECTOR's training setting (2026-11-26).

Goal: a like-for-like comparison with DIRECTOR's large-scale experiment (arXiv 2602.08920, Table 6:
ViT-B-16 fine-tuned on CIFAR-10, full reconfiguration, 97.17% acc, ECE 1.61). Their ViT-B code is not
released, so the recipe below is taken from the released CIFAR code (TransDiff/CIFAR main.py,
train.py, README commands) and the paper's Table "Hyperparameters for unified transition model":

  data       45k train images, RandomCrop(32, padding=4) + horizontal flip per image, then our
             32 -> 224 bicubic upsampling; 5k held-out images for model selection
  budget     100 epochs, batch 128, drop_last (351 steps per epoch, 35,100 steps)
  optimizer  Adam, betas (0.9, 0.999), weight decay 5e-5 (README commands; the paper table says 1e-5)
  schedule   GradualWarmupScheduler(5 epochs) + CosineAnnealingLR(T_max=100, eta_min=1e-5), stepped
             once per epoch; the exact per-epoch values (epoch 0 runs at lr 0, as in their code) are
             read from director_lr_per_epoch.json, produced by their scheduler classes
  selection  checkpoint with the best held-out accuracy (their best_acc_net); the last one is kept too

Two losses (DIR_LOSS), everything else identical:
  director   0.8 * L_mean + 0.2 * CE(rollout logits, label)       (their lambda_mean, lambda_ce for ViT)
             L_mean: one-step teacher-forced prediction s_k + dt F(s_k, t_k) vs s_{k+1}, squared error in
             per-channel normalized units, one random k per image (an unbiased estimate of their
             average over k; raw MSE would be dominated by ViT-B's few huge residual channels)
  ours       our compile_flow.py objective: flow matching on interpolants + lam_state * rollout state
             error + lam_KL * KL(FM logits || rollout logits); no labels

Deviations from DIRECTOR (stated, not hidden): our transition field (FlowNet, 6 DiT blocks, 66M
parameters) instead of their unpublished ViT-B transition model (the paper gives 50M in total);
deterministic Euler rollout (their CE rollout samples mean + std * noise); patch embedding, final norm
and head stay frozen (theirs are trainable); gradient-norm clipping at 1.0 (theirs has none) as a
safeguard at lr 1e-3.

Usage (LADDER=cifar10|cifar100, COMPILE_TAG, COMPILE_FROM=0, DIR_LOSS=director|ours):
  python compile_director.py train
Afterwards `compile_flow.py final|sweep|rollout` and `ladder_eval.py <tag>` use <tag>_flownet.pt.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "imagenet_ladder"))
sys.path.insert(0, HERE)
from compile_flow import (CKPT, K0, LAM_KL, N_BLOCKS, TAG, FlowNet, fidelity, load_models,  # noqa: E402
                          teacher_states, velocity_stats)
from inl_common import embed, get_npy, pick_gpu, readout, save_json, val_split  # noqa: E402
from cifar_prepare import splits, to_input  # noqa: E402

LOSS = os.environ.get("DIR_LOSS", "director")
EPOCHS = int(os.environ.get("DIR_EPOCHS", "100"))
BATCH = 128
MICRO = int(os.environ.get("DIR_MICRO", "32"))       # gradient accumulation only (memory); same update
WD = float(os.environ.get("DIR_WD", "5e-5"))
LAM_MEAN, LAM_CE = 0.8, 0.2
LAM_STATE = 1.0
ROLLOUT_START = int(os.environ.get("COMPILE_ROLLOUT_START", "1000"))   # "ours" only (compile_flow default)
LRS = json.load(open(os.path.join(HERE, "director_lr_per_epoch.json")))
if os.environ.get("DIR_LR_CONST"):   # smoke tests only: constant lr instead of DIRECTOR's schedule
    LRS = [float(os.environ["DIR_LR_CONST"])] * EPOCHS
PRINT_EVERY = int(os.environ.get("DIR_PRINT", "50"))
assert K0 == 0, "DIRECTOR setting = full reconfiguration: set COMPILE_FROM=0"
assert LOSS in ("director", "ours"), LOSS
DT = 1.0 / N_BLOCKS
# Step times exactly as euler_rollout computes them (k * dt in float32), so teacher-forced and rolled-out
# calls of the field see the same t (floor(12 t) picks the per-block velocity statistics).
T_GRID = torch.tensor([k * DT for k in range(N_BLOCKS)], dtype=torch.float32)


def augment(x, rng):
    """torchvision RandomCrop(32, padding=4) (zero fill) + RandomHorizontalFlip, per image."""
    n = len(x)
    pad = np.zeros((n, 3, 40, 40), np.uint8)
    pad[:, :, 4:36, 4:36] = x
    i, j = rng.integers(0, 9, n), rng.integers(0, 9, n)
    out = np.stack([pad[b, :, i[b]:i[b] + 32, j[b]:j[b] + 32] for b in range(n)])
    flip = rng.random(n) < 0.5
    out[flip] = out[flip][..., ::-1]
    return np.ascontiguousarray(out)


def rollout_ckpt(net, x):
    """Euler rollout over all 12 blocks with activation checkpointing per step."""
    states = []
    for k in range(N_BLOCKS):
        t = torch.full((x.shape[0],), k * DT, device=x.device)
        x = x + DT * checkpoint(net, x, t, use_reentrant=False)
        states.append(x)
    return x, states


def micro_loss(net, vit, img, y, state_sigma, gen, ramp):
    """Loss on one micro-batch (already augmented images); returns (loss, parts)."""
    n = img.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            x0 = embed(vit, img)
            s = teacher_states(vit, x0)
            lg_t = readout(vit, s[-1])[1].float() if LOSS == "ours" else None
            S = torch.stack(s, 1)
            kappa = torch.randint(0, N_BLOCKS, (n,), device=img.device, generator=gen)
            ar = torch.arange(n, device=img.device)
            sk, sk1 = S[ar, kappa].float(), S[ar, kappa + 1].float()
            del S
        parts = {}
        if LOSS == "director":
            pred = sk + DT * net(sk, T_GRID.to(img.device)[kappa])
            l_mean = (((pred.float() - sk1) / state_sigma[kappa][:, None]) ** 2).mean()
            xr, _ = rollout_ckpt(net, x0.float())
            ce = F.cross_entropy(readout(vit, xr)[1].float(), y)
            loss = LAM_MEAN * l_mean + LAM_CE * ce
            parts.update(mean=l_mean.item(), ce=ce.item())
        else:
            u = torch.rand(n, device=img.device, generator=gen)
            xt = sk + u[:, None, None] * (sk1 - sk)
            target = (N_BLOCKS * (sk1 - sk) - net.v_mu[kappa][:, None]) / net.v_sigma[kappa][:, None]
            fm = ((net.raw(xt, (kappa + u) / N_BLOCKS).float() - target) ** 2).mean()
            loss = fm
            parts["fm"] = fm.item()
            if ramp > 0:
                xr, states = rollout_ckpt(net, x0.float())
                lg_s = readout(vit, xr)[1].float()
                kl = F.kl_div(F.log_softmax(lg_s, -1), F.log_softmax(lg_t, -1), log_target=True,
                              reduction="batchmean")
                st = torch.stack([(((a.float() - b.float()) / state_sigma[k]) ** 2).mean()
                                  for k, (a, b) in enumerate(zip(states, s[1:]))]).mean()
                loss = loss + ramp * (LAM_KL * kl + LAM_STATE * st)
                parts.update(kl=kl.item(), state=st.item())
    return loss, parts


@torch.no_grad()
def heldout_eval(net, vit, device, val_x0, val_y):
    """Agreement with the FM, KL, accuracy on the 5k held-out images (DIRECTOR's validation set)."""
    net.eval()
    outs = []
    for c in range(0, len(val_y), 1000):
        xb = torch.from_numpy(np.asarray(val_x0[c:c + 1000], dtype=np.float32))
        outs.append(fidelity(net, vit, xb, device, labels=torch.from_numpy(val_y[c:c + 1000].astype(np.int64))))
    net.train()
    return {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}


def train():
    device = pick_gpu(min_free_mb=6000)
    vit = load_models(device)
    net = FlowNet().to(device)
    net.init_from_vit(vit)
    mu, sigma, state_sigma = velocity_stats(vit, get_npy("train_x0"), device)
    net.v_mu.copy_(mu)
    net.v_sigma.copy_(sigma)
    (xtr, ytr), _, _ = splits()
    held = np.concatenate([val_split()["calA"], val_split()["calB"]])
    assert held.tolist() == list(range(5000))
    val_x0 = get_npy("val_x0")[:5000]
    val_y = get_npy("val_labels", mmap=False)[:5000]
    steps_per_epoch = int(os.environ.get("DIR_STEPS_PER_EPOCH", len(xtr) // BATCH))   # override: smoke tests only
    print(f"[{TAG}] DIRECTOR setting, loss={LOSS}, {len(xtr)} train images, {EPOCHS} epochs x {steps_per_epoch} "
          f"steps, batch {BATCH} (micro {MICRO}), Adam wd {WD}, field {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LRS[0], betas=(0.9, 0.999), weight_decay=WD)
    state_path = f"{CKPT}/{TAG}_flownet_state.pt"
    epoch0, step_in0, gstep, log, best = 0, 0, 0, [], -1.0
    if os.path.exists(state_path):
        st0 = torch.load(state_path, map_location=device)
        net.load_state_dict(st0["net"])
        opt.load_state_dict(st0["opt"])
        epoch0, step_in0, gstep, log, best = st0["epoch"], st0["step_in"], st0["gstep"], st0["log"], st0["best"]
        print(f"resumed at epoch {epoch0} step {step_in0} (global {gstep}), best held-out acc {best:.4f}", flush=True)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    t0 = time.time()
    for epoch in range(epoch0, EPOCHS):
        for g in opt.param_groups:
            g["lr"] = LRS[epoch]
        perm = np.random.default_rng(10_000 + epoch).permutation(len(xtr))
        for si in range(step_in0 if epoch == epoch0 else 0, steps_per_epoch):
            idx = perm[si * BATCH:(si + 1) * BATCH]
            rng = np.random.default_rng([epoch, si])
            gen = torch.Generator(device=device).manual_seed(epoch * 100_000 + si)
            img_all = augment(xtr[idx], rng)
            ramp = 0.0 if gstep < ROLLOUT_START else min(1.0, (gstep - ROLLOUT_START) / 1000)
            opt.zero_grad(set_to_none=True)
            agg = {}
            for m in range(0, BATCH, MICRO):
                img = to_input(img_all[m:m + MICRO], device)
                y = ytr_t[idx[m:m + MICRO]].to(device)
                loss, parts = micro_loss(net, vit, img, y, state_sigma, gen, ramp)
                (loss * len(y) / BATCH).backward()
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + v * len(y) / BATCH
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            gstep += 1
            if gstep % PRINT_EVERY == 0:
                print({"epoch": epoch, "step": gstep, "lr": LRS[epoch], **{k: round(v, 5) for k, v in agg.items()},
                       "sec": round(time.time() - t0)}, flush=True)
            if gstep % 200 == 0:
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), epoch=epoch, step_in=si + 1,
                                gstep=gstep, log=log, best=best), state_path + ".tmp")
                os.replace(state_path + ".tmp", state_path)
        rec = dict(epoch=epoch, step=gstep, lr=LRS[epoch], sec=time.time() - t0,
                   **{f"held_{k}": v for k, v in heldout_eval(net, vit, device, val_x0, val_y).items()})
        log.append(rec)
        print(rec, flush=True)
        torch.save(net.state_dict(), f"{CKPT}/{TAG}_last_flownet.pt")
        if rec["held_acc"] > best:
            best = rec["held_acc"]
            torch.save(net.state_dict(), f"{CKPT}/{TAG}_flownet.pt")
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), epoch=epoch + 1, step_in=0, gstep=gstep,
                        log=log, best=best), state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        save_json(log, f"compile_train_log_{TAG}.json")
    print(f"[{TAG}] done: best held-out acc {best:.4f} "
          f"(epoch {max(log, key=lambda r: r['held_acc'])['epoch']}), last {log[-1]['held_acc']:.4f}", flush=True)


if __name__ == "__main__":
    os.makedirs(CKPT, exist_ok=True)
    if sys.argv[1] == "train":
        train()
