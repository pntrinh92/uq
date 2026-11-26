# UQ: Thrust 1 critique and follow-up to DIRECTOR

This is research code and slides for the uncertainty-quantification project: a critique of "Thrust 1" (an SDE scaffold over compiled ViT flows) and its replacement by a regeneration scaffold with conformal guarantees. It also holds a like-for-like reproduction of DIRECTOR (arXiv 2602.08920).

## Layout

| Folder | Contents |
|---|---|
| `imagenet_ladder/` | ViT-B/16 ladder on ImageNet-1k: compiled flows (`compile_flow.py`), the ladder evaluation, the partitioned and conformal scaffolds, and DIRECTOR-metric comparisons. See its README. |
| `cifar_ladder/`, `nlp_ladder/` | The same ladder on CIFAR-10/100 and on IMDB, CoLA and SST-2 (Qwen2.5-0.5B, 5 seeds). |
| `conformal_ood/`, `director_comparison/` | Conformal OOD flag tables and our metrics in DIRECTOR's format. |
| `nuisance_pilot/` | Pilot: certified OOD flags on LLM hidden states break under harmless format changes. |
| `experiments/` | First-round experiments (small checkpoints included). |
| `director_repro_runner/` | DIRECTOR reproduction with the authors' code: job list with README fixes (`jobs.json`), the GPU runner, and `aggregate.py`, which writes `tables.md` and `summary.json`. |
| `figures/figure_Thrust1UQ/` | Plot scripts and figures used in the slides. |
| `proposal_slides/`, `feedback_slides/`, `round2_slides/`, `thrust1_critique_overleaf/` | Beamer decks (compiled on Overleaf). |

## Setup

See [SETUP.md](SETUP.md) for running on a new server: environments (`envs/`), path configuration, GPU selection and which data to copy.

## Not in this repository

- **Checkpoints, features and environments** stay on the compute box under `/ephemeral/trinhpn_uq`, about 870 GB.
- **The DIRECTOR authors' code** is not included. It is `TransDiff`, branch `supplementary`.
- **Inputs to `aggregate.py`:** it also needs the paper's arXiv source (`paper/neurips_2026.tex`) and the per-job logs, and neither is included.

## Updating

`director_repro_runner/` and `figures/figure_Thrust1UQ/` are copies of folders that live elsewhere on the compute box. Run `./sync_external.sh` before committing to refresh them.

All GPU launchers pick a card through `imagenet_ladder/inl_common.py:pick_gpu`, which skips GPU 3 by default (`GPU_EXCLUDE`).
