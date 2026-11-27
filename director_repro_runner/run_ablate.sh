#!/usr/bin/env bash
# One DIRECTOR(ViT) CIFAR-10 ablation (see ablate_director.py), seed 0, on a shared GPU with retry on out-of-memory.
#   bash run_ablate.sh freeze_io|det|det_mean_only        # run inside tmux "pnt"
# Results: TransDiff/CIFAR/results/ablate_<ablation>/.../test_results_diffusion.csv and run.log; our log:
# state/ablate_<ablation>.log. GPU 3 is never used.
set -uo pipefail
ROOT=/ephemeral/trinhpn_uq/director_repro
ABL=${1:?usage: run_ablate.sh freeze_io|det|det_mean_only}
PY=/ephemeral/trinhpn_uq/envs/director/bin/python
LOG=$ROOT/state/ablate_$ABL.log
LAM_CE=0.2; [ "$ABL" = det_mean_only ] && LAM_CE=0
ARGS="main.py --model diffusion --seed 0 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128
 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/ablate_$ABL --backbone transformer
 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --pretrained_dir ./results/vit_out
 --pretrained_seed 0 --lambda_mean 0.8 --lambda_var 0 --lambda_ce $LAM_CE --run_name ablate-$ABL --gpu 0 Cifar10"
cd $ROOT/TransDiff/CIFAR
for attempt in $(seq 1 200); do
  # GPU with the most free memory, excluding GPU 3 and cards with fewer than 3000 MB free
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F', ' '$1!=3 && $2>=3000 {print $2, $1}' | sort -rn | head -1 | awk '{print $2}')
  if [ -z "$GPU" ]; then echo "[$(date +%H:%M:%S)] $ABL: no GPU with 3000 MB free, waiting" | tee -a $LOG; sleep 120; continue; fi
  echo "[$(date +%H:%M:%S)] $ABL attempt $attempt on GPU $GPU" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled WANDB_SILENT=true CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    $PY $ROOT/ablate_director.py $ABL $ARGS >> $LOG 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "[$(date +%H:%M:%S)] ABLATE_${ABL}_DONE" | tee -a $LOG; exit 0; fi
  if tail -n 30 $LOG | grep -qiE "out of memory|OutOfMemoryError"; then
    echo "[$(date +%H:%M:%S)] $ABL: out of memory, retrying in 120 s" | tee -a $LOG; sleep 120
  else
    echo "[$(date +%H:%M:%S)] ABLATE_${ABL}_FAILED (rc=$rc)" | tee -a $LOG; exit 1
  fi
done
