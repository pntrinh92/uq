# Running on a new server

This repo was developed on an 8×H100 box: driver 595, CUDA 12.8, with data under `/ephemeral/trinhpn_uq`. Below is how to set up the same environment elsewhere.

## 1. Check the machine

```bash
nvidia-smi        # driver version and "CUDA Version" (top right)
df -h /mnt/disk1  # wherever the data will live
```

- **Driver:** the pinned PyTorch build is CUDA 12.8 and wants a recent driver (570 or newer). With an older driver, install the build that matches the "CUDA Version" `nvidia-smi` shows (see step 3).
- **Package manager:** any conda works (Miniforge or Miniconda).

## 2. Get the code and choose a data folder

```bash
git clone https://github.com/pntrinh92/uq.git
export UQ=/mnt/disk1/trinhpn/trinhpn_uq        # data, checkpoints, features, logs
mkdir -p $UQ/tmp $UQ/logs $UQ/hf_cache
```

The scripts use absolute paths: `/ephemeral/trinhpn_uq` for everything, and `/data/trinhpn/imagenet-1k` for the first ImageNet shards. Pick **one** of these two ways to point them at the new server.

**A. With root: symlink the old paths**
This leaves the code unchanged.

```bash
sudo mkdir -p /ephemeral /data/trinhpn
sudo ln -s $UQ /ephemeral/trinhpn_uq
sudo ln -s $UQ/imagenet-1k /data/trinhpn/imagenet-1k
```

**B. Without root: rewrite the paths in your clone**

```bash
cd uq
grep -rlE "/ephemeral/trinhpn_uq|/data/trinhpn/imagenet-1k" --include=*.py --include=*.sh . \
  | xargs sed -i -e "s#/ephemeral/trinhpn_uq#$UQ#g" -e "s#/data/trinhpn/imagenet-1k#$UQ/imagenet-1k#g"
```

## 3. Main environment

This env runs `imagenet_ladder`, `cifar_ladder`, `nlp_ladder`, `nuisance_pilot` and the figures.

```bash
conda create -n uq python=3.11 -y
conda activate uq
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r envs/requirements-uq.txt
python -c "import torch, timm, transformers; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

- **Older driver:** replace `cu128` with the index that matches `nvidia-smi`, for example `cu126` or `cu118`. If that index has no build of 2.11.0, use the newest torch it offers. Results were produced with 2.11.0.

Then write `$UQ/env.sh`. Every run script sources it.

```bash
cat > $UQ/env.sh <<EOF
export HF_HUB_CACHE=$UQ/hf_cache
export HF_DATASETS_CACHE=$UQ/hf_cache/datasets
export TORCH_HOME=$UQ/hf_cache/torch
export TMPDIR=$UQ/tmp
export HF_HUB_ENABLE_HF_TRANSFER=0
export PY=$(which python)
EOF
```

## 4. GPU selection

`imagenet_ladder/inl_common.py:pick_gpu` takes the GPU with the most free memory. It **skips GPU 3 by default**, because GPU 3 was reserved on the old box.

- **On a server where GPU 3 is free to use:** set `export GPU_EXCLUDE=""`.
- **To skip other cards:** list them, for example `export GPU_EXCLUDE="0,5"`.

The DIRECTOR runner (`director_repro_runner/runner.py`) reads the same variable.

## 5. Data

Copy only what the step you want to run needs. Sizes are from the old box.

| Folder under `$UQ` | Size | Needed for |
|---|---|---|
| `imagenet-1k/data` (ImageNet train shards 0–22, validation) | 17 GB | `extract_features.py` (ImageNet ladder from scratch) |
| `imagenet_extra`, `imagenet_full` (train shards 23–293) | 24 + 103 GB | full-ImageNet compile (`run_full.sh`) |
| `features` | 435 GB | everything after feature extraction: compile, rollout, `ladder_eval`, scaffolds |
| `ckpt` | 20 GB (7.6 GB without `*_state.pt`) | rollout and scoring of the compiled flows (`run_score.sh`) |
| `data` (DTD etc.) | 1.6 GB | OOD sets |
| `cifar_data`, `cifar10`, `cifar100` | 12 + 41 + 41 GB | CIFAR ladder |
| `nlp_data`, `nlp` | 0.4 + 104 GB | NLP ladder: data, then fine-tuned Qwen checkpoints and features |

- **ImageNet download:** ImageNet-1k on Hugging Face is gated. Put your own token in `$UQ/hf_token`, which `download_full_train.py` reads.
- **Other models:** public models (ViT-B/16, Qwen2.5-0.5B) download on first use into `$UQ/hf_cache`.

Example copy, run on the old server. `-R` keeps the folder layout, and `--partial` lets an interrupted copy resume.

```bash
cd /ephemeral/trinhpn_uq
rsync -aR --partial --exclude '*_state.pt' ./ckpt <new-server>:/mnt/disk1/trinhpn/trinhpn_uq/
```

## 6. Quick test

```bash
cd imagenet_ladder
source $UQ/env.sh
LADDER=imagenet $PY -c "import inl_common as c; print(c.WORK, c.pick_gpu())"
```

## 7. DIRECTOR reproduction envs (optional)

These are only for re-running the authors' code (`TransDiff`, branch `supplementary`) with `director_repro_runner/`. The package sets are exact `pip freeze` lists. Install torch first, then the list with `--no-deps`, so pip doesn't re-resolve old pins.

**`director` env** (CIFAR, IMDB):

```bash
conda create -p $UQ/envs/director python=3.10 -y
$UQ/envs/director/bin/pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
$UQ/envs/director/bin/pip install --no-deps -r envs/director.freeze.txt
cp envs/sitecustomize_director.py $UQ/envs/director/lib/python3.10/site-packages/sitecustomize.py
```

**`cola` env** (CoLA; allennlp 2.10.1):

```bash
conda create -p $UQ/envs/cola python=3.10 -y
$UQ/envs/cola/bin/pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
$UQ/envs/cola/bin/pip install --no-deps -r envs/cola.freeze.txt
cp envs/sitecustomize_cola.py $UQ/envs/cola/lib/python3.10/site-packages/sitecustomize.py
```

- **Why torch 2.0.1 for CoLA:** allennlp pins torch 1.12, which has no kernels for H100 (sm_90).
- **What `sitecustomize_cola.py` does:** it turns off torch 2.0.1's flash and memory-efficient attention, which crash on H100. It also moves temporary files off `/tmp`.
- **On non-H100 GPUs:** the stock torch 1.12 from the freeze resolution would also work, but the pins above are what produced `director_repro_runner/tables.md`.
- **Paths:** the `sitecustomize_*.py` files hard-code `/ephemeral/trinhpn_uq/tmp`. Option B of step 2 rewrites them along with everything else, as long as you run it before copying them.
