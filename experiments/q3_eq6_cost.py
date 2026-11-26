"""Quick test for Eq. (6): quadrature recursion tree vs Monte Carlo paths.

d-dimensional OU from the critique: F_theta = 0, p_0 = N(0, I), g = sigma I,
G(x) = -sigma^2/2 x. Euler kernel x' ~ N(x + eps G(x), eps sigma^2 I).
Target: P(y | x0) = E[sigmoid(w^T x(1))], a smooth predictive probability.

Three references separate the two error sources:
  exact discrete chain (Gaussian in closed form) -> numerical error of the tree / MC
  exact continuous OU                            -> + time-discretization error O(eps)
Cost = number of transition-kernel evaluations: (N^d)^(1/eps) for a tensor
Gauss-Hermite tree, M/eps for M Monte Carlo paths (dimension free).
"""
import itertools

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from common import BLUE, FIGURES, GREY, INK, ORANGE, save_json, setup_matplotlib

SIGMA = 1.2


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def gauss_expect_sigmoid(mean, var, n=200):
    t, w = hermegauss(n)
    return float((w * sigmoid(mean + np.sqrt(var) * t)).sum() / w.sum())


def problem(d):
    x0 = np.full(d, 0.9 / np.sqrt(d))
    w = np.full(d, 1.5 / np.sqrt(d))
    return x0, w


def exact_discrete(d, K):
    eps = 1 / K
    a = 1 - eps * SIGMA ** 2 / 2
    x0, w = problem(d)
    var = eps * SIGMA ** 2 * sum(a ** (2 * j) for j in range(K))
    return gauss_expect_sigmoid(a ** K * (w @ x0), var * (w @ w))


def exact_continuous(d):
    x0, w = problem(d)
    return gauss_expect_sigmoid(np.exp(-SIGMA ** 2 / 2) * (w @ x0), (1 - np.exp(-SIGMA ** 2)) * (w @ w))


def tree(d, K, N):
    """Backward Chapman-Kolmogorov recursion with a tensor Gauss-Hermite rule per step."""
    eps = 1 / K
    t, wq = hermegauss(N)
    wq = wq / wq.sum()
    nodes = np.array(list(itertools.product(t, repeat=d)))  # (N^d, d)
    weights = np.prod(np.array(list(itertools.product(wq, repeat=d))), axis=1)
    x0, w = problem(d)
    evals = 0

    def U(x, k):
        nonlocal evals
        if k == K:
            return sigmoid(w @ x)
        evals += 1
        mean = x + eps * (-SIGMA ** 2 / 2) * x
        children = mean + np.sqrt(eps) * SIGMA * nodes
        return float(sum(wt * U(c, k + 1) for wt, c in zip(weights, children)))

    return U(x0, 0), evals


def mc(d, K, M, seed):
    rng = np.random.default_rng(seed)
    eps = 1 / K
    x0, w = problem(d)
    x = np.tile(x0, (M, 1))
    for _ in range(K):
        x = x + eps * (-SIGMA ** 2 / 2) * x + np.sqrt(eps) * SIGMA * rng.standard_normal(x.shape)
    return float(sigmoid(x @ w).mean()), M * K


def main():
    res = {"tree": [], "mc": [], "exact_continuous": {}}
    for d in (1, 2):
        res["exact_continuous"][d] = exact_continuous(d)
        for K in (2, 4, 8):
            ref_disc = exact_discrete(d, K)
            for N in (2, 3, 4):
                if (N ** d) ** K > 3e6:
                    continue
                val, evals = tree(d, K, N)
                res["tree"].append(dict(d=d, K=K, N=N, value=val, evals=evals,
                                        err_vs_discrete=abs(val - ref_disc),
                                        err_vs_continuous=abs(val - res["exact_continuous"][d])))
            for M in (10, 100, 1000, 10_000, 100_000):
                errs = [abs(mc(d, K, M, s)[0] - ref_disc) for s in range(20)]
                res["mc"].append(dict(d=d, K=K, M=M, evals=M * K, rmse_vs_discrete=float(np.sqrt(np.mean(np.square(errs)))),
                                      disc_bias=abs(ref_disc - res["exact_continuous"][d])))
    save_json(res, "q3_eq6_cost.json")

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for d, mk in ((1, "o"), (2, "s")):
        for K, c in zip((2, 4, 8), (BLUE, ORANGE, INK)):
            rows = [r for r in res["tree"] if r["d"] == d and r["K"] == K]
            if rows:
                ax.loglog([r["evals"] for r in rows], [max(r["err_vs_discrete"], 1e-17) for r in rows],
                          marker=mk, color=c, lw=1.4, label=f"tree d={d}, 1/ε={K}")
        rows = [r for r in res["mc"] if r["d"] == d and r["K"] == 8]
        ax.loglog([r["evals"] for r in rows], [r["rmse_vs_discrete"] for r in rows], marker=mk, color=GREY,
                  ls="--", lw=1.4, label=f"MC paths d={d}, 1/ε=8")
    ax.set_xlabel("kernel evaluations")
    ax.set_ylabel("|error| vs exact discrete chain")
    ax.set_title("Numerical error: the tree is accurate but explodes", loc="left", fontsize=11.5)
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="lower left")

    bias8 = abs(exact_discrete(1, 8) - exact_continuous(1))
    axes[0].axhline(bias8, color=ORANGE, lw=1.4, ls=":")
    axes[0].text(1.5, bias8 * 1.25, "discretization bias at 1/ε=8 (vs continuous SDE)", color=ORANGE, fontsize=8)

    ax = axes[1]
    Ks = np.arange(2, 21)
    for d, c in ((1, BLUE), (2, ORANGE), (64, INK), (768, GREY)):
        ax.plot(Ks, d * Ks * np.log10(3.0), color=c, lw=2, label=f"tree, N=3, d={d}")
    ax.plot(Ks, np.log10(1000 * Ks), color=BLUE, ls="--", lw=2, label="MC, M=1000 paths (any d)")
    ax.set_yscale("log")
    ax.set_xlabel(r"steps $1/\epsilon$")
    ax.set_ylabel(r"$\log_{10}$(kernel evaluations)")
    ax.legend(fontsize=8.5, frameon=False, loc="center right")
    ax.set_title(r"Cost $(N^d)^{1/\epsilon}$ vs $M/\epsilon$", loc="left", fontsize=11.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/fig_q3_eq6_cost.png")
    for r in res["tree"]:
        print("tree", r)
    for r in res["mc"]:
        if r["K"] == 8:
            print("mc", r)


if __name__ == "__main__":
    main()
