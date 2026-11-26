"""Quick test for the Eq. (5) correction: a state-dependent scaffold needs +1/2 div(g g^T).

1-D, F_theta = 0, p_0 = N(0, 1), g(x) = sigma * sqrt(1 + x^2), so D = sigma^2 (1 + x^2).
  proposal drift (Eq. 5):   G = 1/2 D d/dx log p        = -1/2 sigma^2 (1 + x^2) x
  corrected drift:          G = 1/2 D d/dx log p + 1/2 D' = G_prop + sigma^2 x
Only the corrected SDE keeps p_t = N(0, 1). The proposal's version has stationary
density proportional to exp(-x^2/2) / (1 + x^2), which is narrower.
"""
import numpy as np
from scipy import integrate, stats

from common import BLUE, FIGURES, INK, ORANGE, save_json, setup_matplotlib

SIGMA, N_PATHS, DT, T_END = 1.5, 400_000, 1e-3, 3.0


def simulate(corrected, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N_PATHS)
    snapshots = {}
    n_steps = int(round(T_END / DT))
    for k in range(1, n_steps + 1):
        D = SIGMA ** 2 * (1 + x ** 2)
        drift = -0.5 * D * x + (SIGMA ** 2 * x if corrected else 0.0)
        x = x + drift * DT + np.sqrt(D * DT) * rng.standard_normal(N_PATHS)
        if k in (int(1 / DT), n_steps):
            snapshots[round(k * DT, 3)] = x.copy()
    return snapshots


def main():
    res = {}
    sims = {"proposal (Eq. 5)": simulate(False), "corrected (+½∇·D)": simulate(True)}
    for name, snaps in sims.items():
        res[name] = {str(t): dict(var=float(x.var()), ks_vs_N01=float(stats.kstest(x, "norm").statistic))
                     for t, x in snaps.items()}
    z = integrate.quad(lambda u: np.exp(-u ** 2 / 2) / (1 + u ** 2), -np.inf, np.inf)[0]
    stat_var = integrate.quad(lambda u: u ** 2 * np.exp(-u ** 2 / 2) / (1 + u ** 2), -np.inf, np.inf)[0] / z
    res["stationary_var_without_correction"] = stat_var
    save_json(res, "q2_drift_correction.json")
    print(res)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    grid = np.linspace(-4, 4, 400)
    ax.plot(grid, stats.norm.pdf(grid), color=INK, ls="--", lw=2, label=r"target $p_t=\mathcal{N}(0,1)$")
    bins = np.linspace(-4, 4, 81)
    for (name, snaps), c in zip(sims.items(), [ORANGE, BLUE]):
        x = snaps[T_END]
        h, e = np.histogram(x, bins=bins, density=True)
        ax.step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=c, lw=2,
                label=f"{name}: Var$={x.var():.2f}$")
    ax.set_xlabel(r"$x(t)$ at $t=%g$" % T_END)
    ax.set_ylabel("density")
    ax.set_title(r"State-dependent $g(x)=\sigma\sqrt{1+x^2}$: is $p_t$ preserved?", loc="left", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.savefig(f"{FIGURES}/fig_q2_drift_correction.png")


if __name__ == "__main__":
    main()
