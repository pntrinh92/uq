"""Ablations of DIRECTOR(ViT) on CIFAR-10 in the authors' own code (2026-11-27).

Question: which part of DIRECTOR's recipe lets the reconfigured model keep (and raise) the ViT's accuracy? Each
ablation changes one thing of job c10_dit_vit_s0 (main.py --model diffusion ... --lambda_mean 0.8 --lambda_var 0
--lambda_ce 0.2, seed 0; reproduced 5-seed DIRECTOR(ViT) 86.26 +- 0.35, ViT 84.08 +- 0.61):
  freeze_io      patch embedding, positional embedding, last-block LN + MLP and classifier head are copied from the
                 ViT as before but frozen (in their code everything trains); only the DiT transition and its
                 mean/var branches learn
  det            deterministic rollout at train and test: each step returns its mean (theirs: mean + std * N(0, I))
  det_mean_only  det plus lambda_ce = 0 (teacher-state matching only, no labels). Deterministic, because with
                 lambda_ce = lambda_var = 0 nothing trains the std branch and its untrained noise would confound
                 the result. Compare with det. The paper's loss-weight table has no lambda_ce = 0 row; its
                 (lambda_mean, lambda_ce) = (0, 1) row (labels only) gives 84.03 vs 84.22 for the ViT.
The authors' files are not modified: this wrapper patches the model classes at import time, then runs main.py with
runpy (main.py trains, keeps the best-held-out-accuracy checkpoint and runs test.test_diffusion).
Usage (cwd = TransDiff/CIFAR, CUDA_VISIBLE_DEVICES set): python ablate_director.py <ablation> main.py <args...>
"""
import os
import runpy
import sys

ABL = sys.argv[1]
sys.argv = sys.argv[2:]   # [main.py, args...]
sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

torch.cuda.init()         # before main.py rewrites CUDA_VISIBLE_DEVICES from --gpu (as runner.py's LAUNCH)
import models.diffusion as D  # noqa: E402
import models.get_model as G  # noqa: E402


def _arg(name):
    return sys.argv[sys.argv.index(name) + 1]


def det_forward_step(self, x, t):
    x = self.share_params(x, t)
    mean_x_t = self.mean_model(x) + x
    std = self.var_model(x)
    return mean_x_t, std, mean_x_t


if ABL in ("det", "det_mean_only"):
    D.Diffusion_Transformer.forward_step = det_forward_step
if ABL == "det_mean_only":
    assert float(_arg("--lambda_ce")) == 0.0, "pass --lambda_ce 0"
if ABL == "freeze_io":
    _get_model = G.get_model

    def get_model(name, *a, **k):
        net = _get_model(name, *a, **k)
        if name == "diffusion":
            for m in (net.emb, net.ln, net.solution_head_1, net.solution_head_2):
                for p in m.parameters():
                    p.requires_grad_(False)
            net.pos_emb.requires_grad_(False)
            print("freeze_io: trainable", sum(p.numel() for p in net.parameters() if p.requires_grad), flush=True)
        return net

    G.get_model = get_model
if ABL not in ("freeze_io", "det", "det_mean_only"):
    raise SystemExit(f"unknown ablation {ABL}")
print(f"ablation {ABL}: {' '.join(sys.argv)}", flush=True)
runpy.run_path(sys.argv[0], run_name="__main__")
