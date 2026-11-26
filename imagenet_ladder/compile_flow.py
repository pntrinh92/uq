"""Rung 2: compile ViT-B/16 into one velocity field F_theta(x, t) (Thrust 1, Task 1, Eq. 7).

Blocks are mapped to times t = k/12. Between consecutive block outputs s_k, s_{k+1}
we use the linear interpolant psi(r) = s_k + 12 r (s_{k+1} - s_k), r ~ U(0, 1/12),
whose velocity is 12 (s_{k+1} - s_k), and regress F_theta onto it (amortized flow
matching). The lambda term of Eq. 7 is a rollout loss: integrate the ODE with 12
Euler steps and match the FM's logits (KL) and block states. Matching the FM (not
the labels) makes rung 2 a fidelity test of the compilation.

Usage:
  python compile_flow.py train [steps]
  python compile_flow.py rollout [set ...]     # ODE features for the ladder
"""
import math
import os
import sys
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from inl_common import CKPT, FEAT, FM_NAME, get_npy, load_fm, pick_gpu, readout, save_json, set_npy, val_split

N_BLOCKS = 12
DEPTH = int(os.environ.get("COMPILE_DEPTH", "6"))
# COMPILE_FROM = k0 > 0 compiles only blocks k0+1..12 (blocks 1..k0 stay exact), as DIRECTOR does at scale.
K0 = int(os.environ.get("COMPILE_FROM", "0"))
TAG = os.environ.get("COMPILE_TAG") or ("ode" if K0 == 0 else f"ode{K0}")
# Fidelity options (see README): per-block low-rank adapters (option 1), more training data (option 2),
# continuing from a trained shared field.
ADAPTER_RANK = int(os.environ.get("COMPILE_ADAPTER_RANK", "0"))
DATA_SETS = os.environ.get("COMPILE_DATA", "train").split(",")
INIT = os.environ.get("COMPILE_INIT", "")
LR = float(os.environ.get("COMPILE_LR", "5e-4"))
ROLLOUT_START = int(os.environ.get("COMPILE_ROLLOUT_START", "1000"))
# Weight of the rollout KL to the FM logits (lambda_KL; 0.1 in every run before ode8_full_kl1).
LAM_KL = float(os.environ.get("COMPILE_LAM_KL", "0.1"))
# Extra checkpoints kept at these steps (full-data run: compare with the 318k-image arm at equal steps).
SNAPSHOTS = {int(v) for v in os.environ.get("COMPILE_SNAPSHOTS", "").split(",") if v}
# `final` scores this checkpoint instead of <TAG>_flownet.pt when set (results go to compile_final_<TAG>_<suffix>.json).
EVAL_CKPT = os.environ.get("COMPILE_EVAL_CKPT", "")


def timestep_embedding(t, dim=256, max_period=10_000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t[:, None] * 1000) * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class BlockLoRA(nn.Module):
    """Per-ViT-block low-rank update W_k = A_k B_k of a shared linear layer (B_k starts at zero).

    Each sample uses the adapter of the ViT block it is currently imitating, k = floor(12 t), so
    the field keeps one shared backbone plus a small block-indexed part (option 1).
    """

    def __init__(self, d_in, d_out, rank, n=N_BLOCKS):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n, d_in, rank) / math.sqrt(d_in))
        self.B = nn.Parameter(torch.zeros(n, rank, d_out))

    def forward(self, h, k):
        return torch.bmm(torch.bmm(h, self.A[k].to(h.dtype)), self.B[k].to(h.dtype))


class DiTBlock(nn.Module):
    """Pre-LN transformer block with adaLN-Zero time conditioning (Peebles and Xie, 2023)."""

    def __init__(self, d=768, heads=12, mlp_ratio=4, rank=0):
        super().__init__()
        self.lora = nn.ModuleDict({name: BlockLoRA(i, o, rank) for name, (i, o) in
                                   {"qkv": (d, 3 * d), "proj": (d, d), "fc1": (d, mlp_ratio * d),
                                    "fc2": (mlp_ratio * d, d)}.items()}) if rank > 0 else None
        self.heads = heads
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def lin(self, name, h, kb):
        out = getattr(self, name)(h)
        return out if self.lora is None else out + self.lora[name](h, kb)

    def forward(self, x, c, kb=None):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(c))[:, None].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1) + sh1
        B, N, D = h.shape
        q, k, v = self.lin("qkv", h, kb).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + g1 * self.lin("proj", a, kb)
        h = self.norm2(x) * (1 + sc2) + sh2
        return x + g2 * self.lin("fc2", F.gelu(self.lin("fc1", h, kb)), kb)


class FlowNet(nn.Module):
    """F_theta(x, t) = sigma_k(t) * net(x, t) + mu_k(t): velocity of the token state at time t in [0, 1].

    mu_k, sigma_k are fixed per-block, per-channel statistics of the teacher velocity
    12 (s_{k+1} - s_k) (k = floor(12 t)). ViT residual streams have a few channels with
    very large values; without this fixed rescaling a zero-initialized output layer needs
    tens of thousands of steps just to reach the target scale (first run: no learning).
    """

    def __init__(self, d=768, depth=DEPTH, heads=12, rank=ADAPTER_RANK):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([DiTBlock(d, heads, rank=rank) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        for m in (self.ada, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.register_buffer("v_mu", torch.zeros(N_BLOCKS, d))
        self.register_buffer("v_sigma", torch.ones(N_BLOCKS, d))

    def block_index(self, t):
        return (t * N_BLOCKS).floor().long().clamp(0, N_BLOCKS - 1)

    def raw(self, x, t):
        """Normalized velocity (the regression target is (v - mu_k) / sigma_k)."""
        c = self.t_mlp(timestep_embedding(t))
        kb = self.block_index(t)
        for blk in self.blocks:
            x = blk(x, c, kb)
        sh, sc = self.ada(F.silu(c))[:, None].chunk(2, dim=-1)
        return self.out(self.norm(x) * (1 + sc) + sh)

    def init_from_vit(self, vit, block_index=6):
        """Copy attention and MLP weights of a middle ViT block into every DiT block."""
        src = vit.blocks[block_index]
        for blk in self.blocks:
            blk.qkv.load_state_dict(src.attn.qkv.state_dict())
            blk.proj.load_state_dict(src.attn.proj.state_dict())
            blk.fc1.load_state_dict(src.mlp.fc1.state_dict())
            blk.fc2.load_state_dict(src.mlp.fc2.state_dict())

    def forward(self, x, t):
        k = self.block_index(t)
        return self.v_sigma[k][:, None] * self.raw(x, t) + self.v_mu[k][:, None]


@torch.no_grad()
def velocity_stats(vit, x0_all, device, n=2000, bs=50):
    """Per-block, per-channel mean and std of 12 (s_{k+1} - s_k) over images and tokens."""
    idx = np.sort(np.random.default_rng(7).choice(len(x0_all), n, replace=False))
    s1 = torch.zeros(N_BLOCKS, 768, device=device, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    q1, q2 = torch.zeros_like(s1), torch.zeros_like(s1)
    count = 0
    for i in range(0, n, bs):
        x0 = torch.from_numpy(np.asarray(x0_all[idx[i:i + bs]], dtype=np.float32)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = teacher_states(vit, x0)
        for k in range(N_BLOCKS):
            v = (N_BLOCKS * (s[k + 1].float() - s[k].float())).double().reshape(-1, 768)
            s1[k] += v.sum(0)
            s2[k] += (v ** 2).sum(0)
            st = s[k + 1].float().double().reshape(-1, 768)
            q1[k] += st.sum(0)
            q2[k] += (st ** 2).sum(0)
        count += v.shape[0]
    mu = s1 / count
    sigma = (s2 / count - mu ** 2).clamp_min(1e-8).sqrt()
    state_sigma = (q2 / count - (q1 / count) ** 2).clamp_min(1e-8).sqrt()
    return mu.float(), sigma.float(), state_sigma.float()


def euler_rollout(net, x0, n_steps=None, keep_states=False):
    """Integrate from t = K0/12 (state s_K0) to t = 1."""
    n_steps = n_steps or N_BLOCKS - K0
    x, states = x0, []
    t_start = K0 / N_BLOCKS
    dt = (1.0 - t_start) / n_steps
    for k in range(n_steps):
        t = torch.full((x.shape[0],), t_start + k * dt, device=x.device)
        x = x + dt * net(x, t)
        if keep_states:
            states.append(x)
    return x, states


@torch.no_grad()
def teacher_states(vit, x0):
    s = [x0]
    for blk in vit.blocks:
        s.append(blk(s[-1]))
    return s


def load_models(device):
    vit = load_fm(device)
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit


class X0Pool:
    """Several x0 caches seen as one indexable pool (option 2: more training images)."""

    def __init__(self, names):
        self.parts = [get_npy(f"{n}_x0") for n in names]
        self.offsets = np.cumsum([0] + [len(p) for p in self.parts])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), 197, 768), np.float32)
        for j, p in enumerate(self.parts):
            m = (idx >= self.offsets[j]) & (idx < self.offsets[j + 1])
            if m.any():
                out[m] = p[np.sort(idx[m] - self.offsets[j])]
        return out


def train(steps=10_000, batch=32, roll_batch=4, lr=LR, lam_kl=LAM_KL, lam_state=1.0, rollout_start=ROLLOUT_START):
    """Flow matching (Eq. 7) from step 0; rollout terms (state and KL to the FM) from rollout_start."""
    device = pick_gpu()
    vit = load_models(device)
    net = FlowNet().to(device)
    net.init_from_vit(vit)
    x0_all = X0Pool(DATA_SETS)
    n = len(x0_all)
    print(f"[{TAG}] data {DATA_SETS} = {n} images, adapter rank {ADAPTER_RANK}, init {INIT or 'ViT block 6'}, "
          f"steps {steps}, lr {lr}, lam_kl {lam_kl}", flush=True)
    mu, sigma, state_sigma = velocity_stats(vit, x0_all, device)
    net.v_mu.copy_(mu)
    net.v_sigma.copy_(sigma)
    if INIT:   # continue from a trained shared field; adapters (if any) start at zero
        missing, unexpected = net.load_state_dict(torch.load(INIT, map_location=device), strict=False)
        assert not unexpected and all(".lora." in m for m in missing), (missing, unexpected)
    val_x0 = get_npy("val_x0")
    probe = np.sort(val_split()["calB"][:500])
    probe_x0 = torch.from_numpy(np.asarray(val_x0[probe], dtype=np.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    # Resume after an interruption (shared GPUs: out-of-memory kills are expected).
    state_path, start, log = f"{CKPT}/{TAG}_flownet_state.pt", 0, []
    if os.path.exists(state_path):
        st0 = torch.load(state_path, map_location=device)
        net.load_state_dict(st0["net"])
        opt.load_state_dict(st0["opt"])
        sched.load_state_dict(st0["sched"])
        start, log = st0["step"] + 1, st0["log"]
        print(f"resumed at step {start}", flush=True)
    rng = np.random.default_rng(1000 + start)
    t0 = time.time()
    for step in range(start, steps + 1):
        idx = np.sort(rng.choice(n, batch + roll_batch, replace=False))
        x0 = torch.from_numpy(x0_all[idx]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = teacher_states(vit, x0)
            # Amortized flow matching on linear interpolants between consecutive blocks.
            kappa = torch.from_numpy(rng.integers(K0, N_BLOCKS, batch)).to(device)
            u = torch.rand(batch, device=device)
            S = torch.stack([si[:batch] for si in s], 1)
            sk = S[torch.arange(batch), kappa].float()
            sk1 = S[torch.arange(batch), kappa + 1].float()
            del S
            xt = sk + u[:, None, None] * (sk1 - sk)
            v = N_BLOCKS * (sk1 - sk)
            # Regression in normalized units: (v - mu_k) / sigma_k, so every block and channel counts.
            target = (v - net.v_mu[kappa][:, None]) / net.v_sigma[kappa][:, None]
            fm_loss = ((net.raw(xt, (kappa + u) / N_BLOCKS).float() - target) ** 2).mean()
            # Rollout term (lambda of Eq. 7): match block states (per-channel normalized) and FM logits
            # along the integrated ODE path; switched on after the flow-matching warm-up.
            xr, states = euler_rollout(net, s[K0][batch:].float(), keep_states=True)
            _, lg_s = readout(vit, xr)
            _, lg_t = readout(vit, s[-1][batch:])
            kl = F.kl_div(F.log_softmax(lg_s.float(), -1), F.log_softmax(lg_t.float(), -1), log_target=True,
                          reduction="batchmean")
            st = torch.stack([(((a.float() - b[batch:].float()) / state_sigma[K0 + k]) ** 2).mean()
                              for k, (a, b) in enumerate(zip(states, s[K0 + 1:]))]).mean()
            ramp = 0.0 if step < rollout_start else min(1.0, (step - rollout_start) / 1000)
            loss = fm_loss + ramp * (lam_kl * kl + lam_state * st)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            rec = dict(step=step, fm=fm_loss.item(), kl=kl.item(), state=st.item(), sec=time.time() - t0)
            if step % 2000 == 0:
                rec.update(fidelity(net, vit, probe_x0, device))
                torch.save(net.state_dict(), f"{CKPT}/{TAG}_flownet.pt")
            log.append(rec)
            print(rec, flush=True)
        if step in SNAPSHOTS:
            torch.save(net.state_dict(), f"{CKPT}/{TAG}_step{step}_flownet.pt")
        if step % 500 == 0:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, log=log),
                       state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
    torch.save(net.state_dict(), f"{CKPT}/{TAG}_flownet.pt")
    save_json(log, f"compile_train_log_{TAG}.json")


@torch.no_grad()
def fidelity(net, vit, x0, device, n_steps=None, labels=None):
    """Top-1 and top-5 agreement with the FM, KL(p_FM || q), relative final-state error, accuracy."""
    agree, top5, kl, rel, correct = [], [], [], [], []
    for i, xb in enumerate(x0.split(100)):
        xb = xb.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = teacher_states(vit, xb)
            xr, _ = euler_rollout(net, s[K0].float(), n_steps)
            _, lg_s = readout(vit, xr)
            _, lg_t = readout(vit, s[-1])
        lq, lp = F.log_softmax(lg_s.float(), -1), F.log_softmax(lg_t.float(), -1)
        fm_top1 = lp.argmax(-1)
        agree.append((lq.argmax(-1) == fm_top1).float().cpu())
        top5.append((lq.topk(5, -1).indices == fm_top1[:, None]).any(-1).float().cpu())
        kl.append((lp.exp() * (lp - lq)).sum(-1).cpu())
        rel.append(((xr.float() - s[-1].float()).norm(dim=(1, 2)) / s[-1].float().norm(dim=(1, 2))).cpu())
        if labels is not None:
            correct.append((lq.argmax(-1).cpu() == labels[i * 100:(i + 1) * 100]).float())
    out = {"agree": torch.cat(agree).mean().item(), "top5_agree": torch.cat(top5).mean().item(),
           "kl": torch.cat(kl).mean().item(), "rel_e1": torch.cat(rel).mean().item()}
    if labels is not None:
        out["acc"] = torch.cat(correct).mean().item()
    return out


@torch.no_grad()
def final_eval(n=10_000):
    """Fidelity of the trained field on the first n test images (Euler step per compiled block)."""
    device = pick_gpu()
    vit = load_models(device)
    net = FlowNet().to(device)
    net.load_state_dict(torch.load(EVAL_CKPT or f"{CKPT}/{TAG}_flownet.pt", map_location=device))
    net.eval()
    idx = np.sort(val_split()["test"][:n])
    x0 = torch.from_numpy(np.asarray(get_npy("val_x0")[idx], dtype=np.float32))
    y = torch.from_numpy(get_npy("val_labels", mmap=False)[idx].astype(np.int64))
    out = fidelity(net, vit, x0, device, labels=y)
    out.update(tag=TAG, adapter_rank=ADAPTER_RANK, data=DATA_SETS, init=INIT, lam_kl=LAM_KL,
               params_M=sum(p.numel() for p in net.parameters()) / 1e6)
    suffix = "_" + os.path.basename(EVAL_CKPT).split("_flownet")[0].split("_")[-1] if EVAL_CKPT else ""
    out["ckpt"] = EVAL_CKPT or f"{CKPT}/{TAG}_flownet.pt"
    save_json(out, f"compile_final_{TAG}{suffix}.json")
    print(out, flush=True)


@torch.no_grad()
def rollout(sets):
    """ODE features for the ladder: CLS at t = k/12, z and logits, mirroring the FM caches.

    Blocks 1..K0 run exactly (teacher); the compiled flow integrates the rest with one
    Euler step per replaced block, so CLS indices line up with the FM caches.
    """
    device = pick_gpu()
    vit = load_models(device)
    net = FlowNet().to(device)
    net.load_state_dict(torch.load(f"{CKPT}/{TAG}_flownet.pt", map_location=device))
    net.eval()
    for name in sets:
        x0 = get_npy(f"{name}_x0")
        n = len(x0)
        cls = np.zeros((n, N_BLOCKS, 768), np.float16)
        z = np.zeros((n, 768), np.float32)
        logits = np.zeros((n, vit.head.out_features), np.float16)
        t0 = time.time()
        for i in range(0, n, 128):
            xb = torch.from_numpy(np.asarray(x0[i:i + 128], dtype=np.float32)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok, states = xb, []
                for blk in vit.blocks[:K0]:
                    tok = blk(tok)
                    states.append(tok)
                xr, ode_states = euler_rollout(net, tok.float(), keep_states=True)
                zz, lg = readout(vit, xr)
            cls[i:i + len(xb)] = torch.stack([s[:, 0] for s in states + ode_states], 1).float().cpu().numpy()
            z[i:i + len(xb)] = zz.float().cpu().numpy()
            logits[i:i + len(xb)] = lg.float().cpu().numpy()
        set_npy(f"{TAG}_{name}_cls", cls)
        set_npy(f"{TAG}_{name}_z", z)
        set_npy(f"{TAG}_{name}_logits", logits)
        print(f"[{TAG} {name}] {n} in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def step_sweep():
    """Fidelity of the compiled ODE as a function of the number of Euler steps (the epsilon knob)."""
    device = pick_gpu()
    vit = load_models(device)
    net = FlowNet().to(device)
    net.load_state_dict(torch.load(f"{CKPT}/{TAG}_flownet.pt", map_location=device))
    net.eval()
    idx = np.sort(val_split()["test"][:5000])
    x0 = torch.from_numpy(np.asarray(get_npy("val_x0")[idx], dtype=np.float32))
    r = N_BLOCKS - K0
    steps_list = sorted({max(1, r // 4), max(1, r // 2), r, 2 * r, 4 * r})
    out = {n: fidelity(net, vit, x0, device, n) for n in steps_list}
    save_json(out, f"compile_step_sweep_{TAG}.json")
    print(out)


if __name__ == "__main__":
    os.makedirs(CKPT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "train":
        train(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 10_000)
    elif cmd == "rollout":
        rollout(sys.argv[2:] or ["val", "shift", "imagenet_o", "dtd", "noise", "train"])
    elif cmd == "sweep":
        step_sweep()
    elif cmd == "final":
        final_eval()
