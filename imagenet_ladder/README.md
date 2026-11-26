# ImageNet ablation ladder (ViT-B/16)

This tests Thrust 1 and the proposed regeneration scaffold at foundation-model scale. The four rungs:

1. **FM:** `vit_base_patch16_224.orig_in21k_ft_in1k` (timm; the same weights as HF `google/vit-base-patch16-224`, used by DIRECTOR).
2. **Compiled flow (Thrust 1 Task 1, Eq. 7):** one velocity field F_θ(x, t) is trained by amortized flow matching on linear interpolants between consecutive blocks. A rollout term matches the FM's block states and logits. Two variants:
   - `ode`: all 12 blocks compiled.
   - `ode8`: only blocks 9 to 12 compiled; blocks 1 to 8 run exactly. This mirrors DIRECTOR's large-scale protocol.
3. **Brownian same-marginal SDE at the final representation z:** Lie-Trotter splitting with MALA targeting a 1000-class tied GMM, so the marginal is preserved exactly.
4. **Regeneration scaffold:** CLS after blocks 2, 4, 6, 8 and 10, plus z. Split-conformal thresholds, closed-form predictive.

Rungs 3 and 4 run on top of both the FM and the compiled ODE, so errors can be attributed to the right component.

## Data

All stored under `/data/trinhpn/imagenet-1k`, `/ephemeral/trinhpn_uq/data`; features in `/ephemeral/trinhpn_uq/features`.

| Set | Source | Size |
|---|---|---|
| Regeneration pool and compile data | ImageNet-1k train, shards 0 to 22 (class-shuffled, about 100 per class) | 100,234 |
| calA (conformal thresholds) / calB (selection, temperature) / test | ImageNet-1k validation, random split (seed 0) | 5k / 5k / 40k |
| ReaL labels | google-research/reassessed-imagenet | 46,837 of 50k labelled |
| Shift | ImageNet-C definitions, severity 3, our re-implementation: Gaussian noise σ = 0.18, Gaussian blur σ = 3, contrast 0.2, JPEG quality 15; 5k test images each | 20,000 |
| OOD | ImageNet-O (2,000), DTD textures (5,640), uniform noise (2,000) | 9,640 |

**Baselines:** temperature scaling; test-time dropout injection (p picked on calB from {0.02, 0.05, 0.1}, 8 passes); a 3-checkpoint ensemble (orig, augreg and augreg2 ViT-B/16 fine-tuned on in1k; independently trained checkpoints, not a same-recipe deep ensemble); the multi-layer Mahalanobis OOD score.

## Running (tmux session `pnt`)

```
bash run_ladder.sh fm            # FM backbone: rungs 1, 3, 4 and baselines (CPU)
bash run_ladder.sh gpu 8 6000    # compile last 4 blocks, epsilon sweep, rollouts, evaluation
bash run_ladder.sh gpu 0 15000   # compile all 12 blocks, same chain
```

**Shared GPUs:**
- GPU stages retry automatically after an out-of-memory error.
- Compile training resumes from `/ephemeral/trinhpn_uq/ckpt/<tag>_flownet_state.pt`.
- `pick_gpu` keeps our own jobs on different cards.

**Feature extraction (already done):** `extract_features.py`, `dropout_baseline.py`.

**Figures:** `figures4papers/figure_Thrust1UQ/plot_imagenet.py`.

## Results so far (FM backbone, test split of 40k)

| Row | Acc | ReaL | NLL | ECE | Shift NLL | Shift ECE | OOD AUROC (mean of 3) |
|---|---|---|---|---|---|---|---|
| FM | 81.68 | 86.65 | 0.695 | 4.00 | 1.401 | 6.3 | 0.915 |
| FM + TS | 81.68 | 86.65 | 0.692 | 2.84 | 1.397 | 4.5 | 0.921 |
| FM + Brownian SDE at z | 81.67 | 86.67 | 0.695 | 4.08 | 1.402 | 6.4 | 0.915 |
| FM + regeneration | 81.66 | 86.64 | 0.700 | 3.41 | 1.432 | 3.7 | 0.929 |
| Dropout injection (p = 0.1) | 82.11 | 87.20 | 0.658 | 1.90 | 1.380 | 3.0 | 0.915 |
| 3-checkpoint ensemble | 85.50 | 89.49 | 0.563 | 5.12 | 1.031 | 10.8 | 0.923 |
| Multi-layer Mahalanobis (OOD only) | | | | | | | 0.957 |

**What holds and what does not:**
- **The guarantees hold.**
  - Conformal: ID exceedance is 0.034 against the bound Lα = 0.06.
  - Flips: 0.27%, of which 0.07% turned a correct ReaL label into a wrong one.
- **The regeneration scaffold does not reproduce its pilot gains with 1,000 classes.**
  - Selection on calB NLL picks the gentlest setting (α = 0.01, β = 0.1, a₀ = 10⁻³).
  - Mean jump mass is small: 0.007 ID, 0.09 DTD, 0.03 ImageNet-O.
  - The OOD signal is mostly lost: 0.929 against 0.957 for the distances alone.
- **Why:** with q = (1 − J)·p + J·m, the true class gains probability exactly when m_y > p_y, because q_y − p_y = J·(m_y − p_y). The population predictive is nearly uniform (m_y ≈ 1/1000), so it almost never helps, and costs up to −log(1 − J) when the FM is confident and right. With 9 classes (the pilot) m_y ≈ 1/9, which is closer to p_y on hard inputs.
- **The Brownian family at z is a no-op once selected:** h = 0.003 wins on calB, and larger h strictly increases NLL.
- **Dropout injection is the best single-model calibrator here; the ensemble is best on shift NLL.**

### Partitioned regeneration (`partitioned_regen.py`)

Regenerate inside a soft partition w_c(x) = softmax(log p(·|x)/τ), drawing the target from a·w_c·p / E_p[a·w_c]. The marginal is still exact for any partition. The selected setting is τ = 1, α = 0.01, β = 0.1, a₀ = 0.01.

Results on the FM: NLL 0.691, ECE 3.30, shift NLL 1.393, shift ECE 5.5, OOD AUROC 0.918, flips 0.21%.
- **NLL:** the cost of the population version disappears, and NLL is now below both the FM and TS.
- **OOD:** the gain shrinks, because reverting toward the predicted class adds little entropy.
- **Separate detector:** the jump mass J(x) on its own gets AUROC 0.713 on ImageNet-O, 0.838 on DTD and 1.0 on noise.

### Rung 2: compiled flow

**Last 4 blocks (`ode8`, 6,000 steps, 6-block DiT field):**
- Argmax agreement with the FM is 64.1% on test.
- Accuracy: top-1 60.6% against 81.7% for the FM (ReaL 68.1% against 86.7%).
- Calibration: ECE 11.2% (1.5% after TS), NLL 1.598.
- **Euler-step sweep:** agreement is 43.0% at 1/4 step per replaced block, 53.1% at 1/2, 64.9% at 1, 64.7% at 2 and 63.7% at 4. Finer integration doesn't help, so the learned field is not a convergent ODE.

| Row on `ode8` | ReaL | NLL | ECE | Shift NLL | OOD AUROC |
|---|---|---|---|---|---|
| ODE | 68.05 | 1.598 | 11.19 | 2.515 | 0.851 |
| ODE + TS | 68.05 | 1.528 | 1.51 | 2.423 | 0.823 |
| ODE + Brownian at z | 69.03 | 1.597 | 11.85 | 2.499 | 0.850 |
| ODE + regeneration | 67.82 | 1.605 | 11.34 | 2.553 | 0.863 |

**All 12 blocks (`ode`, 15,000 steps):** argmax agreement on the 500-image probe grows slowly: 0.2% at 2k steps, 4.2% at 6k, 9.2% at 8k, 16.2% at 12k and 15.6% at 14k. The relative final-state error is 0.78.
  - **Final, on the 40k test split:** agreement with the FM 16.3%, top-1 15.8%, ReaL 18.0%, NLL 4.30, ECE 2.4%, shift NLL 6.22, OOD AUROC 0.463 (below chance).
  - **Euler-step sweep:** 3.8%, 10.7%, 16.5%, 14.8% and 10.2% at 1/4, 1/2, 1, 2 and 4 steps per block.
  - **Rungs 3 and 4 on this backbone:** TS, Brownian and regeneration all leave NLL at about 4.30, so they can't repair a broken flow.

**Reading of the ladder:** rung 2 is already far below rung 1, so the compiled flow, not the uncertainty layer, is the dominant error. This is the "(2) worse than (1)" case of the ablation plan.

### Can compile fidelity be improved? Three options (`run_fidelity.sh`)

Every arm continues from the trained ladder field and is scored on the same 10k test images (`compile_flow.py final`):
- **control:** +6k steps on the same 100k images.
- **option 1:** per-ViT-block rank-32 adapters on every linear layer of the shared field (+28M parameters, zero-initialized), +6k steps.
- **option 2:** 50 more train shards (318k images in total, 3.2×) and +15k steps.
- **option 3:** top-5 agreement and KL(p_FM ‖ q), alongside top-1 agreement and accuracy. Scores for all six ladders are in `results/fidelity_metrics.json`.

| Arm | Last 4: top-1 | top-5 | KL | acc | All 12: top-1 | top-5 | KL | acc |
|---|---|---|---|---|---|---|---|---|
| ladder field | 64.4 | 89.1 | 1.06 | 61.0 | 16.4 | 37.8 | 3.75 | 16.0 |
| control | 68.0 | 90.9 | 0.89 | 64.4 | 17.4 | 38.9 | 3.72 | 16.6 |
| + adapters (option 1) | 67.8 | 91.1 | 0.89 | 64.3 | 17.5 | 39.2 | 3.70 | 16.8 |
| + data, longer (option 2) | 74.0 | 93.5 | 0.66 | 70.0 | 26.4 | 52.4 | 3.00 | 25.2 |
| adapters + data | 74.2 | 93.5 | 0.65 | 70.0 | 26.6 | 53.0 | 2.98 | 25.5 |

The FM's accuracy is 81.7%.

**Reading:**
- **Per-block adapters add nothing measurable** (≤ 0.3 points in every setting). The field is not limited by block-specific capacity, at least at low rank.
- **More data and longer training is the only lever that moved fidelity.**
  - Last 4 blocks: +6 points top-1 over control, accuracy 64 → 70%.
  - All 12 blocks: +9 points top-1.
  - Data and steps change together in this arm, so their individual effects can't be separated.
- **Still not usable:** the best last-4 flow agrees on 74% (top-1) and 93.5% (top-5) and loses 11.7 accuracy points. A 95% target would need much more data (the full 1.28M images) and compute, with no guarantee of reaching it.

## Held-out calibration and input-level guarantees (`conformal_scaffold.py`)

This section is the second-round answer to the meeting feedback. It runs on every ladder (`LADDER=imagenet|cifar10|cifar100|imdb|cola|sst2`). Results: `results*/conformal_scaffold_fm.json`. Slides: `../feedback_slides`.

**Path scaffold.**
- One regeneration decision is made after the last layer, with a rate that depends on the whole feature path.
- Drawing a pool path with weight a·w_c keeps every layer marginal, and q stays closed form.
- One combined score replaces the L per-layer thresholds, so the flip bound is α instead of Lα.

**Protocol.** Each split has its own role:

| Split | Role |
|---|---|
| A1 (half of calA) | Learn the distance θ: non-negative logistic regression of FM errors on log layer distances (`err`), or uniform weights (`unif`). |
| A2 | Per-group (Mondrian) conformal thresholds and p-values. Groups are FM confidence quintiles. |
| calB | Learn-then-Test over (α, κ, τ). Hoeffding-Bentkus p-values, Bonferroni, flip ≤ 1% and ID jump ≤ 5% at δ = 0.1. |
| calC (¼ of test) | Calibrates the label sets on q. |
| Rest of test | Reporting only. |

The FM's training features are never used to calibrate.

**Findings (FM backbone):**
- **Training-feature thresholds overshoot their level; held-out thresholds hit it.**
  - ImageNet, error distance, α = 5%: 9.1% exceedance with training-feature thresholds, 4.8% with held-out.
  - CIFAR-100, same setting: 9.2% versus 4.4%.
- **Learning the threshold b by in-distribution NLL drives it to +∞** (implied α = 0 on CIFAR-100 and ImageNet), as Gibbs' inequality predicts for a calibrated FM. So the threshold is not fitted. Its level is set by conformal calibration, and its aggressiveness is certified by Learn-then-Test.
- **A marginal threshold at α = 5% hides large violations on the least confident inputs.** Exceedance in that quintile is 19.2% on ImageNet, 21.3% on CIFAR-100 and 30.6% on CIFAR-10. Per-group thresholds bring every group to 3.4–5.9% on ImageNet. For label sets at a 90% target, worst-group coverage rises from 78.5% to 90.3% on ImageNet.
- **The certified scaffold improves OOD AUROC.**
  - With τ = ∞ (population target): ImageNet 0.929 → 0.955 and CIFAR-100 0.972 → 0.976, with accuracy unchanged and flips ≤ 0.31%. The cost is NLL under shift: ImageNet shift NLL 1.40 → 2.13.
  - With τ = 1, the cost mostly disappears. CIFAR-100 reaches 0.967 OOD AUROC at NLL 0.266 (FM 0.265).
- **The error-trained distance puts all its weight on z.** That makes it much better at near-OOD detection (CIFAR-100 vs CIFAR-10: 0.961 versus 0.702 for uniform weights) and worse on textures.

## Full ImageNet-1k compile (`run_full.sh`)

- **Data:** shards 73–293 are cached as `train3a/b/c`, on top of the 73 shards already cached.
- **Training:** both ladder fields continue for 60k steps on all 1,281,167 images.
- **Snapshots:** at 15k, 30k and 45k steps. They are not comparable with the 318k-image arm: the cosine schedule runs over 60k steps, so the learning rate at step 15k is still 85% of its peak. `run_full_match15k.sh` trains a separate 15k-step arm on all images with its own schedule for that comparison.
- **Data access:** the gated dataset needs a token with ImageNet access in `/ephemeral/trinhpn_uq/hf_token`. The tmux window `pnt:full` starts the download, then the whole chain, as soon as that file exists.
