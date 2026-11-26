# Thrust 1 pilot experiments

Experiments behind the "Quick test" column of the critique deck
(`../thrust1_critique_overleaf/main.tex`) and Experiments 0–1 from the plan.
Figures are written straight into `../thrust1_critique_overleaf/figures/`.

**Pilot setup.** sklearn digits (8×8), with classes 0–8 in-distribution (ID).
- **Model:** a 4-block pre-LN ViT (2×2 patches, d = 64, dropout 0.1) trained from scratch, giving about 98% ID accuracy.
- **Splits:** 5 random splits (train/val/test = 60/20/20).
- **References:** MC dropout (64 masks in Exp 0, 32 in Exp 1) and a 5-member deep ensemble. Member 0 of the ensemble is "the FM".
- **OOD sets:** held-out digit 9, 90°-rotated test digits, and uniform noise.
- **Shift set (labels kept):** test digits plus pixel noise with σ = 0.3.
- **Hardware:** everything runs on CPU.

Reproduce with `bash run_all.sh` (set `PY` to a Python with torch, scikit-learn, scipy and matplotlib).

| Script | Tests | Figure |
|---|---|---|
| `exp0_variance_budget.py` | Is the reference uncertainty inside the marginal-preserving family? Computes ρ = tr(Σ_mc − Σ_det)/tr E per layer. | `fig_exp0_rho.png` |
| `exp1_langevin_sweep.py`, `exp1_report.py` | Sweeps the marginal-preserving SDE family (Lie–Trotter: exact FM block, then MALA targeting a class-conditional GMM) against TS, MC dropout, an ensemble, and the GMM's own Mahalanobis score. | `fig_exp1_sweep.png`, `fig_exp1_margin.png` |
| `q2_drift_correction.py` | Is +½∇·(ggᵀ) needed for a state-dependent g? | `fig_q2_drift_correction.png` |
| `q3_eq6_cost.py` | Eq. (6): Gauss–Hermite recursion tree vs Monte Carlo paths, error vs cost. | `fig_q3_eq6_cost.png` |
| `q4_eq8_penalty.py` | Eq. (8): where does the optimal variance T* = √(λ‖∇log p‖/c) go? | `fig_q4_eq8_penalty.png` |
| `exp3_regeneration.py` | Proposed regeneration scaffold: jump to a training feature with distance-aware, conformally thresholded probability. Preserves every layer marginal exactly and has a closed-form predictive. | `figures4papers/figure_Thrust1UQ/figures/fig_main.png`, `fig_guarantee.png` |

## Results (mean ± sd over 5 splits)

**Q2: the drift correction is needed.** With g(x) = 1.5·√(1+x²), the drift as written in Eq. (5) shrinks the marginal to Var 0.53 (analytic stationary value 0.525, KS distance 0.088). Adding ½∇·D keeps N(0,1): Var 1.00, KS distance 0.001.

**Q3: the Eq. (6) tree is accurate but exponential.**
- In 1–2 dimensions, Gauss–Hermite reaches an error of 1e-6 against the discrete chain.
- Cost grows as (N^d)^(1/ε): about 10^7300 evaluations for d = 768, N = 3, 1/ε = 20.
- The error against the continuous SDE is set by discretization (0.0055 at 1/ε = 8), which neither the tree nor Monte Carlo removes.
- Monte Carlo cost is M/ε regardless of dimension.

**Exp 0: does the reference stay inside the same-marginal family?**
- **ID, final representation z:** ρ = −0.04 ± 0.004. MC dropout's population variance matches the deterministic one at trace level; the added per-input variance (7.6% of the budget) is offset by shrinking the means. This doesn't support the premise that the reference "adds variance on top of the marginal" in-distribution. Since ρ only matches traces, ρ ≈ 0 is a necessary condition, not proof of an equal marginal.
- **ID, deeper token states and logits:** ρ is strongly negative (logits −1.34 ± 0.10). MC dropout *removes* population variance, which is also outside the family, from the other side.
- **Deep ensemble, logits, ID:** ρ = −0.13 ± 0.23.
- **OOD:** ρ = 0.2–0.4 at z, so the reference adds variance there. The marginal constraint only binds under p₀, so this doesn't rule the family out.

**Q4: Eq. (8) puts variance in the wrong places.**
- T* flags OOD and misclassified inputs with AUROC 0.14–0.15. That is well below chance, meaning it gives *less* variance to exactly the inputs that need more.
- The cause is the loss curvature c, which is largest on ambiguous inputs (1/c alone: AUROC 0.07–0.10).
- The score norm alone is a good OOD detector (0.91–0.97), essentially the Mahalanobis distance, because the score model here is Gaussian (no heavy tails).

**Exp 1: the decisive question.** In the table, ECE is in %, and the SDE rows use the configuration with the best validation NLL per split.

| Method | Acc | NLL | ECE | Shift NLL | Shift ECE | OOD AUROC |
|---|---|---|---|---|---|---|
| FM | .978 | .099 ± .022 | 1.8 | 1.61 ± .17 | 23.7 | .915 |
| Temperature scaling | .978 | .093 ± .016 | 1.7 | 1.49 ± .18 | 22.6 | .916 |
| MC dropout | .979 | .075 ± .015 | 1.5 | 1.21 ± .13 | 14.8 | .925 |
| Deep ensemble (5) | .990 | .050 ± .014 | 1.5 | 0.80 ± .06 | 6.4 | .971 |
| TS + GMM Mahalanobis score | .978 | .093 | 1.7 | 1.49 | 22.6 | .943 |
| SDE, Langevin on z only | .978 | .100 ± .023 | 1.9 | 1.63 ± .18 | 24.5 | .915 |
| SDE, Langevin after every block | .984 | .080 ± .032 | 1.5 | 1.31 ± .13 | 17.6 | .905 |

- **No MALA configuration beats TS on NLL+ECE and MC dropout on OOD AUROC at the same time.** That holds even when picked on test, with or without TS on top. The same is true against TS + the GMM score.
- **Langevin on z only** never helps. Raising h hurts NLL, shift NLL and OOD AUROC. Low-margin inputs get *more* confident (+2.7 pp max-prob): the mixture score pulls them toward a class mode.
- **Langevin after every block** at small h (≈0.03) beats TS on calibration and slightly on accuracy. It still trails MC dropout, is far behind the ensemble, and *lowers* OOD AUROC.
- **The isotropic diffusion's flat curve isn't stability.** At large h, MALA rejects every proposal, so the model is left unchanged.
- **Preconditioning:** D must use the GMM's within-class covariance. With the population covariance, MALA rejects everything and ULA diverges.

**Exp 3: the proposed regeneration scaffold** (all layers, picked on half of the validation set; see `uq/proposal_slides`).
- Accuracy equals the FM (97.84%), with zero argmax flips in all 5 splits. ID ECE is 1.42 ± 0.15%, the lowest single-model value.
- Shift NLL is 1.25 ± 0.17 (TS 1.49, MC dropout 1.21, ensemble 0.80).
- OOD AUROC is 0.970 ± 0.013 (ensemble 0.971). The multi-layer Mahalanobis score alone also gives 0.971, so the OOD *ranking* comes from the distances; the scaffold turns them into a predictive distribution.
- Conformal check: P(any threshold exceeded) is 0.024, 0.125 and 0.203, against bounds Lα = 0.05, 0.25 and 0.5.
- Near OOD (held-out 9, AUROC 0.910) is the weak case.

## Caveats

- **Easy pilot.** The ID test set has about 7 errors per split, so ID ECE/NLL are noisy. The pixel-noise shift set carries most of the calibration signal.
- **GMM stand-in for the score.** The Langevin target is a tied-covariance Gaussian mixture, not the true p_l. Marginal distortion at z stays within about 1% for MALA.
- **ρ only compares traces.** It is a necessary condition. The RBF-MMD check is included but had no power here (all values ≈ 0).
- **Small scale.** Results are for a 4-block ViT on 8×8 digits. The ImageNet ablation ladder (Exp 2) is still to do.
