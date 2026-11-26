"""Shared pieces for the Thrust 1 pilot experiments.

Pilot setup: sklearn digits (8x8), a 4-block pre-LN ViT trained from scratch,
MC dropout and a small deep ensemble as references. Classes 0-8 are
in-distribution (ID); class 9, 90-degree rotations and uniform noise are OOD.
Everything runs on CPU in a few minutes.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

torch.set_num_threads(int(os.environ.get("NUM_THREADS", "16")))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "..", "thrust1_critique_overleaf", "figures")
ID_CLASSES = 9  # digits 0-8; digit 9 is held out as near-OOD


# ----------------------------------------------------------------- data
def load_data(seed):
    """ID train/val/test split plus three OOD sets, as float tensors in [0, 1]."""
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32).reshape(-1, 8, 8)
    idm = y < ID_CLASSES
    Xid, yid = X[idm], y[idm]
    Xtr, Xte, ytr, yte = train_test_split(Xid, yid, test_size=0.2, stratify=yid, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=seed)
    rng = np.random.default_rng(seed)
    # Covariate shift with labels: ID test images plus pixel noise (accuracy drops, so calibration matters).
    shift_rng = np.random.default_rng(10_000 + seed)
    Xsh = np.clip(Xte + shift_rng.normal(0, 0.3, size=Xte.shape), 0, 1).astype(np.float32)
    ood = {
        "heldout_9": X[~idm],
        "rotated": np.rot90(Xte, k=1, axes=(1, 2)).copy(),
        "noise": rng.uniform(0, 1, size=(len(Xte), 8, 8)).astype(np.float32),
    }
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    tl = lambda a: torch.tensor(a, dtype=torch.long)
    return dict(
        Xtr=t(Xtr), ytr=tl(ytr), Xva=t(Xva), yva=tl(yva), Xte=t(Xte), yte=tl(yte), Xsh=t(Xsh),
        ood={k: t(v) for k, v in ood.items()},
    )


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, heads, mlp, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.drop1, self.drop2 = nn.Dropout(p), nn.Dropout(p)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.drop1(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop2(self.mlp(self.ln2(x)))


class TinyViT(nn.Module):
    """2x2 patches -> 16 tokens + CLS, `depth` pre-LN blocks, LN, linear head on CLS."""

    def __init__(self, d=64, depth=4, heads=4, mlp=128, p=0.1, n_classes=ID_CLASSES):
        super().__init__()
        self.patch = nn.Linear(4, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 17, d))
        self.drop = nn.Dropout(p)
        self.blocks = nn.ModuleList([Block(d, heads, mlp, p) for _ in range(depth)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def embed(self, img):
        B = img.shape[0]
        patches = img.reshape(B, 4, 2, 4, 2).permute(0, 1, 3, 2, 4).reshape(B, 16, 4)
        x = torch.cat([self.cls.expand(B, -1, -1), self.patch(patches)], dim=1)
        return self.drop(x + self.pos)

    def readout(self, tokens):
        """Final representation z (post-LN CLS) from the last token state."""
        return self.ln(tokens[:, 0])

    def forward(self, img, return_all=False):
        x = self.embed(img)
        states = []
        for blk in self.blocks:
            x = blk(x)
            states.append(x)
        z = self.readout(x)
        logits = self.head(z)
        if return_all:
            return logits, z, states
        return logits


def set_mc_dropout(model, on):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(on)


def train_model(data, seed, epochs=150, lr=1e-3, wd=0.05):
    torch.manual_seed(seed)
    model = TinyViT()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X, y = data["Xtr"], data["ytr"]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(X), generator=g).split(64):
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def get_models(seed, n_members=5):
    """Member 0 is 'the FM'; members 1.. form the deep-ensemble reference. Cached on disk."""
    data = load_data(seed)
    os.makedirs(os.path.join(RESULTS, "ckpt"), exist_ok=True)
    models = []
    for m in range(n_members):
        path = os.path.join(RESULTS, "ckpt", f"vit_split{seed}_member{m}.pt")
        model = TinyViT()
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            model.eval()
        else:
            model = train_model(data, seed=1000 * seed + m)
            torch.save(model.state_dict(), path)
        models.append(model)
    return data, models


# ----------------------------------------------------------------- metrics
def ece_score(probs, y, n_bins=15):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return ece


def classification_metrics(probs, y):
    probs = probs.clamp_min(1e-12)
    onehot = F.one_hot(y, probs.shape[1]).float()
    return dict(
        acc=(probs.argmax(1) == y).float().mean().item(),
        nll=-probs[torch.arange(len(y)), y].log().mean().item(),
        ece=ece_score(probs, y),
        brier=((probs - onehot) ** 2).sum(1).mean().item(),
    )


def entropy(probs):
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(1)


def auroc(score_id, score_ood):
    """Higher score = more OOD."""
    s = np.concatenate([np.asarray(score_id), np.asarray(score_ood)])
    lab = np.concatenate([np.zeros(len(score_id)), np.ones(len(score_ood))])
    return float(roc_auc_score(lab, s))


def fit_temperature(logits, y):
    def nll(logt):
        return F.cross_entropy(logits / np.exp(logt), y).item()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def centered_gaussian_w2(a, b):
    """Mean over inputs of W2 between Gaussian fits of two sample sets.

    a, b: (N, S, K) logit samples per input. Logits are centred over classes
    first, since softmax ignores a common shift.
    """
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    ma, mb = a.mean(1), b.mean(1)
    ca = torch.einsum("nsk,nsl->nkl", a - ma[:, None], a - ma[:, None]) / (a.shape[1] - 1)
    cb = torch.einsum("nsk,nsl->nkl", b - mb[:, None], b - mb[:, None]) / (b.shape[1] - 1)
    ca, cb = ca.double().numpy(), cb.double().numpy()

    def psd_sqrt(c):
        w, v = np.linalg.eigh(c)
        return (v * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.swapaxes(v, -1, -2)

    sb = psd_sqrt(cb)
    cross = psd_sqrt(sb @ ca @ sb)
    tr = np.trace(ca, axis1=1, axis2=2) + np.trace(cb, axis1=1, axis2=2) - 2 * np.trace(cross, axis1=1, axis2=2)
    w2sq = ((ma - mb) ** 2).sum(1).double().numpy() + np.clip(tr, 0, None)
    return float(np.sqrt(w2sq).mean())


def mmd_rbf(x, y, max_n=1500, seed=0):
    """Unbiased MMD^2 with a median-heuristic RBF kernel (subsampled)."""
    g = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(len(x), generator=g)[:max_n]].double()
    y = y[torch.randperm(len(y), generator=g)[:max_n]].double()
    z = torch.cat([x, y])
    d2 = torch.cdist(z, z) ** 2
    bw = d2[d2 > 0].median()
    k = torch.exp(-d2 / bw)
    n, m = len(x), len(y)
    kxx = (k[:n, :n].sum() - k[:n, :n].diag().sum()) / (n * (n - 1))
    kyy = (k[n:, n:].sum() - k[n:, n:].diag().sum()) / (m * (m - 1))
    kxy = k[:n, n:].mean()
    return float(kxx + kyy - 2 * kxy)


# ----------------------------------------------------------------- GMM score + MALA
class TiedGMM:
    """Class-conditional Gaussians with one shared Ledoit-Wolf covariance.

    log p(x) = logsumexp_c [log pi_c - 0.5 (x-mu_c)^T P (x-mu_c)] + const
    grad log p(x) = -P (x - sum_c r_c(x) mu_c)
    """

    def __init__(self, feats, labels):
        feats = feats.double()
        C = int(labels.max()) + 1
        self.mu = torch.stack([feats[labels == c].mean(0) for c in range(C)])
        self.log_pi = torch.log(torch.bincount(labels, minlength=C).double() / len(labels))
        resid = feats - self.mu[labels]
        self.cov = torch.tensor(LedoitWolf().fit(resid.numpy()).covariance_)
        self.prec = torch.linalg.inv(self.cov)
        self.muP = self.mu @ self.prec
        self.muPmu = (self.muP * self.mu).sum(1)
        # Shrunk population covariance (full rank even when d > n); used for preconditioners.
        self.pop_cov = torch.tensor(LedoitWolf().fit(feats.numpy()).covariance_)

    def _maha(self, x, xP):
        return (xP * x).sum(1, keepdim=True) - 2 * xP @ self.mu.T + self.muPmu

    def logp_and_score(self, x):
        xP = x @ self.prec
        comp = self.log_pi - 0.5 * self._maha(x, xP)
        r = torch.softmax(comp, dim=1)
        return torch.logsumexp(comp, dim=1), -(xP - r @ self.muP)

    def log_prob(self, x):
        return self.logp_and_score(x)[0]

    def score(self, x):
        return self.logp_and_score(x)[1]

    def min_mahalanobis(self, x):
        x = x.double()
        return self._maha(x, x @ self.prec).min(1).values


def make_preconditioner(gmm, kind, k=8):
    """Constant diffusion D = L L^T, returned as (L, L_pinv).

    Built from the GMM's within-class covariance, so for "precond" D P = I and the
    drift is a plain pull towards the responsibility-weighted class mean. (The
    population covariance makes the dynamics stiff: between-class variance is
    much larger than within-class variance, and MALA then rejects everything.)
    """
    cov = gmm.cov
    d = cov.shape[0]
    if kind == "iso":
        s = torch.sqrt(torch.trace(cov) / d)
        L = s * torch.eye(d, dtype=torch.float64)
    elif kind == "precond":
        w, v = torch.linalg.eigh(cov)
        L = v * w.clamp_min(1e-10).sqrt()
    elif kind == "lowrank":
        w, v = torch.linalg.eigh(cov)
        L = v[:, -k:] * w[-k:].sqrt()
    else:
        raise ValueError(kind)
    return L, torch.linalg.pinv(L)


def langevin(x, gmm, L, Lpinv, h, steps, mala=True, generator=None):
    """(MA)LA steps targeting the GMM with constant preconditioner D = L L^T.

    Proposal x' = x + (h/2) D grad log p(x) + sqrt(h) L xi. With MALA the target is
    preserved exactly (on the affine slice x + range(D) when D is low rank).
    Returns the new state and the mean acceptance rate.
    """
    x = x.double()
    D = L @ L.T
    acc_rates = []
    lp, sc = gmm.logp_and_score(x)
    for _ in range(steps):
        xi = torch.randn(x.shape[0], L.shape[1], dtype=torch.float64, generator=generator)
        xp = x + 0.5 * h * sc @ D + np.sqrt(h) * xi @ L.T
        if not mala:
            x = xp
            lp, sc = gmm.logp_and_score(x)
            continue
        lpp, scp = gmm.logp_and_score(xp)
        fwd = 0.5 * (xi ** 2).sum(1)
        rev_xi = (x - xp - 0.5 * h * scp @ D) @ Lpinv.T / np.sqrt(h)
        rev = 0.5 * (rev_xi ** 2).sum(1)
        log_a = lpp - lp + fwd - rev
        accept = torch.log(torch.rand(x.shape[0], dtype=torch.float64, generator=generator)) < log_a
        x = torch.where(accept[:, None], xp, x)
        lp = torch.where(accept, lpp, lp)
        sc = torch.where(accept[:, None], scp, sc)
        acc_rates.append(accept.double().mean().item())
    return x, (float(np.mean(acc_rates)) if acc_rates else 1.0)


def variance_budget(det, samples):
    """Law-of-total-variance decomposition.

    det: (N, d) deterministic features; samples: (N, S, d) stochastic features.
    Returns traces of Sigma_det, Cov_x(m(x)), E_x Cov(.|x), Sigma_stoch and rho.
    """
    det, samples = det.double(), samples.double()
    m = samples.mean(1)
    tr_det = torch.trace(torch.cov(det.T)).item()
    tr_mean = torch.trace(torch.cov(m.T)).item()
    tr_E = ((samples - m[:, None]) ** 2).sum(-1).mean().item() * samples.shape[1] / (samples.shape[1] - 1)
    tr_stoch = torch.trace(torch.cov(samples.reshape(-1, samples.shape[-1]).T)).item()
    return dict(
        tr_det=tr_det, tr_mean=tr_mean, tr_E=tr_E, tr_stoch=tr_stoch,
        rho=(tr_stoch - tr_det) / tr_E if tr_E > 0 else float("nan"),
        shrink=(tr_det - tr_mean) / tr_E if tr_E > 0 else float("nan"),
        E_share=tr_E / tr_det,
    )


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "axes.edgecolor": "#52514E",
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })
    return plt


BLUE, ORANGE, INK, GREY = "#2A78D6", "#EB6834", "#2B2B2B", "#8C8C8C"
