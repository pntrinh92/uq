# CIFAR ablation ladder (DIRECTOR setting)

This is the same ladder as `../imagenet_ladder`, on the setting of the DIRECTOR paper (Appendix A.6.1): ViT-B/16 fine-tuned on CIFAR at 224 px.

- **Checkpoint:** timm `vit_base_patch16_224.orig_in21k_ft_in1k`, the weights of HF `google/vit-base-patch16-224`, with a new head.
- **Fine-tuning:** 3 epochs, AdamW, lr 5e-5, weight decay 0.05, batch 64, random crop and flip at 32 px, then bicubic upsampling to 224. No label smoothing.
- **Splits:** 45k fine-tune / 5k held out (calA 2.5k conformal thresholds, calB 2.5k selection and temperature) from the official train set; the official 10k test set is the test split.
- **Shift:** CIFAR-10-C / CIFAR-100-C, all 15 corruptions × 5 severities, the same 400 test images per cell (30k).
- **OOD (5k each):** SVHN (test), the other CIFAR (test), LSUN_resize (the ODIN version, as in DIRECTOR), DTD textures downsampled to 32 px, uniform noise (2k).
- **Rungs:**
  1. the fine-tuned FM;
  2. compiled flow (Eq. 7 with a rollout term), last 4 blocks (`ode8`) and all 12 (`ode`);
  3. Brownian same-marginal SDE at z;
  4. regeneration scaffold, population and partitioned.
- **Baselines:** temperature scaling, dropout injection (8 passes, p chosen on calB), the multi-layer Mahalanobis OOD score. No ensemble here: only one fine-tuned checkpoint exists.

**Run** (tmux `pnt`, with `LADDER=cifar10|cifar100` set by the script):

```
bash run_cifar.sh gpu cifar10    # fine-tune, features, dropout, compiles + ODE evaluation
bash run_cifar.sh cpu cifar10    # FM-backbone evaluation and partitioned regeneration
```

**Code reuse:** `cifar_prepare.py` writes the ImageNet feature layout, so `compile_flow.py`, `dropout_baseline.py`, `ladder_eval.py` and `partitioned_regen.py` from `../imagenet_ladder` run unchanged. `inl_common.LADDER` switches paths, class count, corruption and OOD sets. **Results:** `results_cifar10/`, `results_cifar100/`. **Figures:** `figures4papers/figure_Thrust1UQ/plot_cifar.py`.

## Results (FM backbone, 10k test)

| | CIFAR-10 Acc | NLL | ECE % | C-NLL | C-ECE % | OOD | CIFAR-100 Acc | NLL | ECE % | C-NLL | C-ECE % | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FM | 98.79 | 0.044 | 0.63 | 0.312 | 4.3 | 0.993 | 92.53 | 0.265 | 1.17 | 1.017 | 5.2 | 0.945 |
| + TS | 98.79 | 0.043 | 0.50 | 0.297 | 3.8 | 0.993 | 92.53 | 0.262 | 1.74 | 1.042 | 8.0 | 0.942 |
| + Brownian at z | 98.80 | 0.045 | 0.65 | 0.313 | 4.3 | 0.993 | 92.49 | 0.265 | 1.13 | 1.019 | 5.3 | 0.945 |
| + regeneration | 98.79 | 0.049 | 0.42 | 0.374 | 6.6 | **0.996** | 92.53 | 0.271 | 1.67 | 1.092 | 4.1 | **0.972** |
| + partitioned regen. | 98.79 | 0.044 | 0.62 | 0.308 | 4.3 | 0.994 | 92.52 | 0.265 | 1.22 | 1.016 | 4.6 | 0.953 |
| Dropout injection | 98.81 | 0.040 | 0.43 | 0.311 | 3.6 | 0.993 | 92.45 | 0.263 | 1.23 | 1.023 | 4.0 | 0.946 |
| Multi-layer Mahalanobis | | | | | | 0.992 | | | | | | 0.965 |

"C-" means the CIFAR-C shift set; OOD is the mean AUROC over the 5 OOD sets.

**What holds:**
- **Our fine-tune is well calibrated.** ECE is 0.63% on CIFAR-10, whereas DIRECTOR's ViT-B/16 baseline (Table 6) has ECE 18.0% at 97.9% accuracy. Their large baseline ECE is a property of their fine-tuning recipe, not of ViT-B.
- **Regeneration gives the best OOD AUROC on both datasets, above the distance score it uses.** On CIFAR-100: 0.972 vs 0.965. The distance score fails on near-OOD (CIFAR-10 vs CIFAR-100: 0.842; the reverse: 0.962). The scaffold keeps the softmax level there (0.920, 0.984) and still gains on far OOD (CIFAR-100: LSUN 0.892 → 0.995, SVHN 0.944 → 0.966).
- **That OOD gain costs NLL, in distribution and under shift.** CIFAR-100: NLL 0.265 → 0.271, C-NLL 1.017 → 1.092. Partitioned regeneration removes the cost (0.265, 1.016) but keeps only part of the OOD gain (0.953).
- **The guarantees hold:** ID threshold exceedance is 0.028 (CIFAR-10) and 0.032 (CIFAR-100), against the bound Lα = 0.06. Flips are 0.00% and 0.05%.

## Rung 2: compiled flow

| Backbone | CIFAR-10 Acc | NLL | ECE % | C-NLL | OOD | CIFAR-100 Acc | NLL | ECE % | C-NLL | OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| FM | 98.79 | 0.044 | 0.63 | 0.312 | 0.993 | 92.53 | 0.265 | 1.17 | 1.017 | 0.945 |
| ODE, last 4 blocks (6k steps) | 97.97 | 0.073 | 1.12 | 0.365 | 0.983 | 87.26 | 0.459 | 2.30 | 1.236 | 0.902 |
| + TS | 97.97 | 0.071 | 0.79 | 0.347 | 0.984 | 87.26 | 0.459 | 3.08 | 1.242 | 0.901 |
| + regeneration | 97.97 | 0.077 | 0.55 | 0.433 | 0.990 | 87.26 | 0.465 | 2.60 | 1.317 | 0.936 |
| ODE, all 12 blocks (10k steps) | 89.43 | 0.383 | 5.86 | 1.012 | 0.872 | 59.09 | 1.665 | 12.86 | 2.655 | 0.733 |
| + TS | 89.43 | 0.343 | 2.16 | 0.833 | 0.881 | 59.09 | 1.625 | 5.84 | 2.533 | 0.743 |
| + regeneration | 89.43 | 0.378 | 5.09 | 0.961 | 0.880 | 59.06 | 1.666 | 12.43 | 2.650 | 0.733 |

**Argmax agreement with the FM** (test split):

| | Last 4 blocks | All 12 blocks |
|---|---|---|
| CIFAR-10 | 98.6% | 89.7% |
| CIFAR-100 | 90.4% | 60.0% |
| ImageNet | 64.1% | 16.3% |

**Euler-step sweep:** at 1/4, 1/2, 1, 2 and 4 steps per compiled block, every one of the six compiles stops improving at 1 step per block (at most +0.1 point beyond it, usually a loss):
- CIFAR-10, all 12 blocks: 46.2, 75.4, 89.5, 89.6, 87.8%.
- CIFAR-100, all 12 blocks: 17.2, 36.5, 59.2, 55.5, 49.7%.

**Reading:**
- **Compilation is the dominant error, and it grows with the number of classes and of compiled blocks.** On CIFAR-10 the last-4 compile costs 0.8 accuracy points, comparable to DIRECTOR's reported 0.7-point drop (97.88 → 97.17%) for a full reconfiguration. Their objective also includes a label loss, which can buy accuracy without fidelity.
- **The ε knob of Eq. (6) does not converge.** The learned field behaves like a 1-step-per-block map, not an ODE.
- **On broken flows, temperature scaling is the only thing that helps calibration.** The scaffolds keep their OOD gain only where the backbone is still good (last 4 blocks, CIFAR-100: 0.902 → 0.936).
