"""Job list for reproducing DIRECTOR (arXiv 2602.08920) with the authors' code (TransDiff, branch supplementary).

Commands follow the repo README; deviations, all fixing evident typos, are marked FIX:
  - CIFAR SV-DKL: README calls --model temperature_scaling; the svdkl model (as in its OOD command) is used.
  - CoLA / IMDB SGPA: README repeats the softmax pre-training command; --attn-type sgpa is used.
  - IMDB DiT on the Transformer lists seed 3 twice and no seed 2; seeds 0-4 are used.
  - IMDB DiT on KEP-SVGP points --pretrained_dir at vit_out; the KEP checkpoints are saved in vit_out_sum.
  - KEP-SVGP ksvd-layers in {1, 2, 7} (CIFAR) and {1, 2, 5} (CoLA, IMDB), as the README text says.
  - CIFAR-100 commands come from CIFAR/scripts/vit_cifar100.sh, run for 5 seeds.
Every job gets --gpu <id> from the runner. Output: jobs.json (ordered by priority; deps are job ids).
"""
import json

R = "/ephemeral/trinhpn_uq/director_repro/TransDiff"
SEEDS = range(5)
jobs = []


def add(jid, task, args, deps=(), env="director"):
    jobs.append(dict(id=jid, cwd=f"{R}/{task}", env=env, args=args, deps=list(deps)))


# ---------------------------------------------------------------- CIFAR-10 / CIFAR-100
for ds, tag in (("Cifar10", "c10"), ("Cifar100", "c100")):
    for s in SEEDS:
        add(f"{tag}_vit_s{s}", "CIFAR", f"main.py --seed {s} --attn-type softmax --batch-size 128 --nb-epochs 300 --nb-run 1 "
            f"--model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out {ds}")
        for k in (1, 2, 7):
            add(f"{tag}_kep{k}_s{s}", "CIFAR", f"main.py --seed {s} --attn-type kep_svgp --concate --ksvd-layers {k} --eta-ksvd 10 "
                f"--batch-size 128 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 "
                f"--save-dir ./results/vit_out_cat {ds}")
    lam_vit = "--lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2" if ds == "Cifar10" else "--lambda_mean 0.5 --lambda_var 0 --lambda_ce 0.5"
    run_name = "DiT-5-seed" if ds == "Cifar10" else "DiT-CIFAR100"
    dit = ("--num_heads 12 --hdim 384 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 "
           "--save-dir ./results/diffusion --backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 "
           "--trans_dropout 0.1")
    for s in SEEDS:
        add(f"{tag}_dit_vit_s{s}", "CIFAR", f"main.py --model diffusion --seed {s} --depth 7 --attn-type softmax {dit} "
            f"--pretrained_dir ./results/vit_out --pretrained_seed {s} {lam_vit} --run_name {run_name} {ds}", [f"{tag}_vit_s{s}"])
        for k in (1, 2, 7):
            add(f"{tag}_dit_kep{k}_s{s}", "CIFAR", f"main.py --model diffusion --seed {s} --depth 7 --attn-type kep_svgp --concate "
                f"--ksvd-layers {k} --eta-ksvd 10 {dit} --pretrained_dir ./results/vit_out_cat --pretrained_seed {s} "
                f"--lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name {run_name} {ds}", [f"{tag}_kep{k}_s{s}"])
    for s in SEEDS:   # post-hoc baselines on the pre-trained ViT
        base = f"--seed {s} --attn-type softmax --batch-size 128 --nb-run 1 --weight-decay 5e-5 --save-dir ./results/vit_out"
        for m in ("temperature_scaling", "mc_dropout", "kflla"):
            add(f"{tag}_{m}_s{s}", "CIFAR", f"main.py {base} --nb-epochs 0 --model {m} --lr 1e-3 {ds}", [f"{tag}_vit_s{s}"])
        add(f"{tag}_svdkl_s{s}", "CIFAR", f"main.py {base} --nb-epochs 50 --model svdkl --lr 1e-2 "   # FIX: svdkl
            f"--pretrained_dir ./results/vit_out --pretrained_seed {s} {ds}", [f"{tag}_vit_s{s}"])
        add(f"{tag}_sgpa_s{s}", "CIFAR", f"main.py --seed {s} --attn-type sgpa --batch-size 128 --nb-epochs 300 --nb-run 1 "
            f"--model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out {ds}")

# ---------------------------------------------------------------- distillation (CIFAR-10, Table 15)
for s in SEEDS:
    add(f"c10_distill_vit3_s{s}", "CIFAR", f"main.py --depth 3 --attn-type softmax --batch-size 128 --nb-epochs 0 --nb-run 1 "
        f"--model vit_cifar_distillation --lr 1e-3 --weight-decay 5e-5 --seed {s} --save-dir ./results/vit_out_distillation "
        f"--pretrained_seed {s} --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0.0 --lambda_ce 0.2 "
        f"--temperature 3 --run_name ViT-distillation Cifar10", [f"c10_vit_s{s}"])
    add(f"c10_distill_dit_s{s}", "CIFAR", f"main.py --model diffusion_distillation --seed {s} --depth 7 --attn-type softmax "
        f"--num_heads 12 --hdim 384 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 "
        f"--save-dir ./results/diffusion_distillation --backbone transformer --trans_depth 1 --trans_num_heads 12 "
        f"--trans_mlp_ratio 1 --trans_dropout 0.1 --pretrained_seed {s} --pretrained_dir ./results/vit_out --lambda_mean 0.8 "
        f"--lambda_var 0 --lambda_ce 0.2 --temperature 3 --run_name DiT-5-seed-distillation Cifar10", [f"c10_vit_s{s}"])

# ---------------------------------------------------------------- CoLA (own env: allennlp / ELMo)
for s in SEEDS:
    add(f"cola_tf_s{s}", "CoLA", f"main.py --depth 5 --attn-type softmax --batch-size 32 --nb-epochs 100 --nb-run 1 "
        f"--model vit_cola --lr 5e-4 --seed {s} --save-dir ./results/vit_out", env="cola")
    for k in (1, 2, 5):
        add(f"cola_kep{k}_s{s}", "CoLA", f"main.py --depth 5 --attn-type kep_svgp --ksvd-layers {k} --eta-ksvd 1 --batch-size 32 "
            f"--nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --weight-decay 5e-5 --seed {s} --save-dir ./results/vit_out_sum",
            env="cola")
    add(f"cola_dit_tf_s{s}", "CoLA", f"main.py --model diffusion --seed {s} --depth 5 --attn-type softmax --batch-size 32 "
        f"--nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer "
        f"--pretrained_dir ./results/vit_out --pretrained_seed {s} --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 "
        f"--trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2", [f"cola_tf_s{s}"], env="cola")
    for k in (1, 2, 5):
        add(f"cola_dit_kep{k}_s{s}", "CoLA", f"main.py --model diffusion --seed {s} --depth 5 --attn-type kep_svgp --ksvd-layers {k} "
            f"--eta-ksvd 1 --batch-size 32 --nb-epochs 50 --nb-run 1 --lr 5e-3 --weight-decay 5e-5 --save-dir ./results/diffusion "
            f"--backbone transformer --pretrained_dir ./results/vit_out_sum --pretrained_seed {s} --trans_depth 1 "
            f"--trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3",
            [f"cola_kep{k}_s{s}"], env="cola")
    base = f"--depth 5 --attn-type softmax --batch-size 32 --nb-run 1 --seed {s} --save-dir ./results/vit_out"
    for m in ("temperature_scaling", "mc_dropout", "kflla"):
        add(f"cola_{m}_s{s}", "CoLA", f"main.py {base} --nb-epochs 0 --model {m} --lr 5e-4", [f"cola_tf_s{s}"], env="cola")
    add(f"cola_svdkl_s{s}", "CoLA", f"main.py {base} --nb-epochs 50 --model svdkl --lr 1e-2", [f"cola_tf_s{s}"], env="cola")
    add(f"cola_sgpa_s{s}", "CoLA", f"main.py --depth 5 --attn-type sgpa --batch-size 32 --nb-epochs 100 --nb-run 1 "   # FIX: sgpa
        f"--model vit_cola --lr 5e-4 --seed {s} --save-dir ./results/vit_out", env="cola")

# ---------------------------------------------------------------- IMDB
for s in SEEDS:
    add(f"imdb_tf_s{s}", "IMDB", f"main.py --depth 5 --attn-type softmax --batch-size 32 --nb-epochs 50 --nb-run 1 "
        f"--model transformer_imdb --lr 1e-3 --seed {s} --save-dir ./results/vit_out")
    for k in (1, 2, 5):
        add(f"imdb_kep{k}_s{s}", "IMDB", f"main.py --depth 5 --attn-type kep_svgp --ksvd-layers {k} --eta-ksvd 10 --batch-size 32 "
            f"--nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed {s} --save-dir ./results/vit_out_sum")
    dit = ("--batch-size 32 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --save-dir ./results/diffusion "
           "--backbone transformer --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1")
    add(f"imdb_dit_tf_s{s}", "IMDB", f"main.py --depth 5 --attn-type softmax {dit} --seed {s} --pretrained_dir ./results/vit_out "
        f"--pretrained_seed {s} --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2", [f"imdb_tf_s{s}"])   # FIX: seeds 0-4
    for k in (1, 2, 5):
        add(f"imdb_dit_kep{k}_s{s}", "IMDB", f"main.py --depth 5 --attn-type kep_svgp --ksvd-layers {k} --eta-ksvd 10 {dit} "
            f"--weight-decay 5e-5 --seed {s} --pretrained_dir ./results/vit_out_sum --pretrained_seed {s} "   # FIX: vit_out_sum
            f"--lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3", [f"imdb_kep{k}_s{s}"])
    base = f"--depth 5 --attn-type softmax --batch-size 32 --nb-run 1 --seed {s} --save-dir ./results/vit_out"
    for m in ("temperature_scaling", "mc_dropout", "kflla"):
        add(f"imdb_{m}_s{s}", "IMDB", f"main.py {base} --nb-epochs 0 --model {m} --lr 1e-3", [f"imdb_tf_s{s}"])
    add(f"imdb_svdkl_s{s}", "IMDB", f"main.py {base} --nb-epochs 50 --model svdkl --lr 1e-2", [f"imdb_tf_s{s}"])
    add(f"imdb_sgpa_s{s}", "IMDB", f"main.py --depth 5 --attn-type sgpa --batch-size 32 --nb-epochs 100 --nb-run 1 "   # FIX: sgpa
        f"--model transformer_imdb --lr 1e-3 --seed {s} --save-dir ./results/vit_out")


# ---------------------------------------------------------------- OOD detection (Table 5): CIFAR-10 in distribution
# main_ood.py only loads the trained checkpoints and scores SVHN, CIFAR-100 and LSUN.
OOD_SETS = {"svhn": "./data", "cifar100": "./data/CIFAR100/test", "lsun": "./data/LSUN"}
DIT = ("--num_heads 12 --hdim 384 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 "
       "--save-dir ./results/diffusion --backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 "
       "--trans_dropout 0.1 --run_name DiT-5-seed")
for s in SEEDS:
    vit = (f"--seed {s} --attn-type softmax --batch-size 128 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 "
           f"--weight-decay 5e-5 --save-dir ./results/vit_out")
    ood_models = {"vit": (vit, [f"c10_vit_s{s}"]),
                  "sgpa": (vit.replace("softmax", "sgpa"), [f"c10_sgpa_s{s}"]),
                  "dit_vit": (f"--model diffusion --seed {s} --depth 7 --attn-type softmax {DIT} --pretrained_dir ./results/vit_out "
                              f"--pretrained_seed {s} --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2", [f"c10_dit_vit_s{s}"]),
                  "svdkl": (f"--seed {s} --attn-type softmax --batch-size 128 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-3 "
                            f"--weight-decay 5e-5 --save-dir ./results/vit_out --pretrained_dir ./results/vit_out "
                            f"--pretrained_seed {s}", [f"c10_svdkl_s{s}"])}
    for m in ("kflla", "mc_dropout"):
        ood_models[m] = (f"--seed {s} --attn-type softmax --batch-size 128 --nb-epochs 0 --nb-run 1 --model {m} --lr 1e-3 "
                         f"--weight-decay 5e-5 --save-dir ./results/vit_out", [f"c10_vit_s{s}"])
    for k in (1, 2, 7):
        ood_models[f"kep{k}"] = (f"--attn-type kep_svgp --concate --ksvd-layers {k} --eta-ksvd 10 --batch-size 128 --nb-epochs 300 "
                                 f"--nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --seed {s} "
                                 f"--save-dir ./results/vit_out_cat", [f"c10_kep{k}_s{s}"])
        ood_models[f"dit_kep{k}"] = (f"--model diffusion --seed {s} --depth 7 --attn-type kep_svgp --concate --ksvd-layers {k} "
                                     f"--eta-ksvd 10 {DIT} --pretrained_dir ./results/vit_out_cat --pretrained_seed {s} "
                                     f"--lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3", [f"c10_dit_kep{k}_s{s}"])
    for name, (a, deps) in ood_models.items():
        for o, d in OOD_SETS.items():
            add(f"c10_ood_{o}_{name}_s{s}", "CIFAR", f"main_ood.py {a} --ood_data {o} --ood_test_dir {d} Cifar10", deps)


def priority(j):
    """Tier 1: the backbones of Tables 1-2 (ViT/Transformer, KEP-1, KEP-last), their DiTs and TS/MCD/KFLLA;
    tier 2: SV-DKL, SGPA; tier 3: KEP-2 (Table 16) and distillation (Table 15). Within a tier: pre-training
    before dependents, then dataset, then seed."""
    jid = j["id"]
    ds, rest = jid.split("_", 1)
    if any(k in rest for k in ("kep2", "distill")):
        tier = 3
    elif rest.startswith(("svdkl", "sgpa")):
        tier = 2
    else:
        tier = 1
    stage = 0 if not j["deps"] else (1 if "dit" in rest else 2)
    # dependents first: when a backbone finishes, its DiT and baselines take the freed slot, so each dataset
    # completes early instead of waiting for every pre-training run to have started
    return tier, -stage, ["cola", "imdb", "c10", "c100"].index(ds), int(jid.rsplit("_s", 1)[1])


jobs.sort(key=priority)
json.dump(jobs, open("/ephemeral/trinhpn_uq/director_repro/jobs.json", "w"), indent=1)
print(len(jobs), "jobs;", sum(1 for j in jobs if not j["deps"]), "without dependencies")
