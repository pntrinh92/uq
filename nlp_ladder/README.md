# NLP ladder (DIRECTOR text benchmarks)

This is the ladder of `../imagenet_ladder` on the NLP datasets of DIRECTOR (Sec. 3.1, App. A.6.1), with Qwen2.5-0.5B as the FM. Qwen2.5 is the language model DIRECTOR reconfigures at scale.

- **Model:** Qwen2.5-0.5B (24 blocks, d = 896), linear head on the last non-pad token, full fine-tuning.
  - Optimizer: AdamW, lr 2e-5 (IMDB 1e-5: 2e-5 diverged on 512-token inputs), weight decay 0.01, 6% warm-up, batch 16, bf16 autocast.
  - Epochs: IMDB 2, CoLA 4, SST-2 2. The last epoch is kept, so the held-out set is never used for early stopping.
  - 5 seeds per task. The 5 seeds also form the deep-ensemble baseline.
- **Splits (held out = calA + calB halves):**

| Task | Fine-tune | Held out | Test | Shift (labels kept) | Max tokens |
|---|---|---|---|---|---|
| IMDB | 20k of train | 5k of train | 25k official test | SST-2 test (1,821), Yelp (5k) | 512 |
| CoLA | 6,355 | 907 | 1,816 (random split of in-domain train + dev, DIRECTOR's sizes) | CoLA out-of-domain dev (516) | 64 |
| SST-2 | GLUE train phrases (67,349) | GLUE dev (872) | SST-2 test sentences (1,821) | IMDB (5k), Yelp (5k) | 128 |

- **OOD sets (every task):** AG News (5k), 20 Newsgroups (5k), random-token sequences (2k).
- **Scaffold layers:** last-token state after blocks 4, 8, 12, 16 and 20, plus z (final norm).
- **Metrics:** CoLA reports MCC as well as accuracy, as DIRECTOR does.

**Run** (tmux `pnt`):

```
bash run_nlp.sh gpu <imdb|cola|sst2>   # 5 seeds: data, fine-tune, features, dropout baseline, then ensemble caches
bash run_nlp.sh cpu <imdb|cola|sst2>   # per seed: ladder_eval, partitioned_regen, conformal_scaffold (FM backbone)
```

`nlp_prepare.py data` tokenizes every set once per task. The features use the ImageNet layout, so the ladder scripts in `../imagenet_ladder` run unchanged with `LADDER=<task> NLP_SEED=<s>`.

- **Results:** `results_<task>/seed<s>/`.
- **Figures:** `figures4papers/figure_Thrust1UQ/plot_feedback.py` (mean ± s.e. over seeds).
- **Raw data:** `/ephemeral/trinhpn_uq/nlp_data`. Public datasets are downloaded without a token. CoLA's out-of-domain dev comes from the original release, `cola_public_1.1.zip`.
