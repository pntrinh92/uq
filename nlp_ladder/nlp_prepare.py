"""NLP ladder, DIRECTOR's text benchmarks: fine-tune Qwen2.5-0.5B and cache features for the ladder.

Tasks and splits (DIRECTOR, Sec. 3.1 and App. A.6.1):
  imdb  train 25k -> 20k fine-tune / 5k held out (random, seed 0); official 25k test. Max 512 tokens.
        Shift (labels kept): SST-2 test sentences (1,821), Yelp polarity test (5k).
  cola  in-domain train + in-domain dev (9,078) -> 6,355 / 907 / 1,816 (random, seed 0). Max 64 tokens.
        Shift: CoLA out-of-domain dev (516), DIRECTOR's "CoLA OOD".
  sst2  GLUE SST-2 train phrases (67,349) to fine-tune; held out = GLUE dev (872 sentences); test = SST-2
        test sentences (1,821, labelled copy from SetFit/sst2). Dev and test sentences never occur in
        the training phrases. Max 128 tokens. Shift: IMDB test (5k), Yelp polarity test (5k).
  OOD for every task: AG News test (5k), 20 Newsgroups test (5k), random-token sequences (2k).

The held-out set is split in two halves, calA and calB, exactly like the vision ladders.

Model: Qwen2.5-0.5B (24 blocks, d = 896) with a linear head on the last non-pad token, the
backbone DIRECTOR reconfigures in Table 7. Full fine-tuning, AdamW, lr 2e-5 (IMDB 1e-5), weight decay 0.01,
6% warm-up then linear decay, batch 16, bf16 autocast; the last epoch is kept (the held-out set
is never used for early stopping). Five seeds per task give mean +- s.e. and a 5-member ensemble.

Feature caches follow the ImageNet layout (inl_common, LADDER=<task>, NLP_SEED=<s>):
  <set>_cls (n, 24, 896)  last-token state after every block (index 23 is after the final norm)
  <set>_z (n, 896), <set>_logits (n, members, 2), <set>_labels; shift_corruption names the shift set.

Usage (LADDER and NLP_SEED in the environment):
  python nlp_prepare.py data                 # tokenize every set once per task
  python nlp_prepare.py finetune
  python nlp_prepare.py extract              # features of every set for this seed
  python nlp_prepare.py dropout              # test-time dropout baseline (p on calB NLL, 8 passes)
  python nlp_prepare.py ensemble             # stack the other seeds' logits as ensemble members
"""
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imagenet_ladder"))
from inl_common import (CKPT, CORRUPTIONS, FEAT, LADDER, NLP_SEED, NLP_TASKS, OOD_SETS, WORK, pick_gpu,  # noqa: E402
                        save_json, set_npy, val_split)

assert LADDER in NLP_TASKS, f"set LADDER to one of {NLP_TASKS}"
RAW = f"{WORK}/nlp_data"
TASK_DIR = f"{WORK}/nlp/{LADDER}"
MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LEN = {"imdb": 512, "cola": 64, "sst2": 128}[LADDER]
EPOCHS = {"imdb": 2, "cola": 4, "sst2": 2}[LADDER]
LR = {"imdb": 1e-5, "cola": 2e-5, "sst2": 2e-5}[LADDER]   # imdb diverged at 2e-5 (512-token inputs)
WD, BATCH, WARMUP = 0.01, 16, 0.06
N_SEEDS = 5
SPLIT_SEED = 0
N_SUB = 5000
P_GRID, PASSES = [0.02, 0.05, 0.1, 0.2, 0.3], 8


# ----------------------------------------------------------------- raw text
def parquet(path, text="text", label="label"):
    import pyarrow.parquet as pq
    t = pq.read_table(f"{RAW}/{path}")
    return [s.replace("<br />", " ") for s in t.column(text).to_pylist()], np.array(t.column(label).to_pylist())


def jsonl(path):
    rows = [json.loads(l) for l in open(f"{RAW}/{path}")]
    return [r["text"] for r in rows], np.array([r.get("label", -1) for r in rows])


def cola_tsv(name):
    rows = [l.rstrip("\n").split("\t") for l in open(f"{RAW}/cola_public/raw/{name}.tsv") if l.strip()]
    return [r[3] for r in rows], np.array([int(r[1]) for r in rows])


def subsample(texts, labels, n, seed):
    idx = np.sort(np.random.default_rng(seed).choice(len(texts), min(n, len(texts)), replace=False))
    return [texts[i] for i in idx], labels[idx]


def raw_sets():
    """Text and labels of every set: train, held (calA + calB), test, shift parts, OOD sets."""
    imdb_test = parquet("stanfordnlp__imdb/plain_text/test-00000-of-00001.parquet")
    sst2_test = jsonl("SetFit__sst2/test.jsonl")
    yelp = subsample(*parquet("fancyzhx__yelp_polarity/plain_text/test-00000-of-00001.parquet"), N_SUB, 11)
    out = {}
    if LADDER == "imdb":
        x, y = parquet("stanfordnlp__imdb/plain_text/train-00000-of-00001.parquet")
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(x))
        out["train"] = ([x[i] for i in perm[5000:]], y[perm[5000:]])
        out["held"] = ([x[i] for i in perm[:5000]], y[perm[:5000]])
        out["test"] = imdb_test
        out["shift_parts"] = {"sst2": sst2_test, "yelp": yelp}
    elif LADDER == "sst2":
        out["train"] = parquet("nyu-mll__glue/sst2/train-00000-of-00001.parquet", text="sentence")
        out["held"] = parquet("nyu-mll__glue/sst2/validation-00000-of-00001.parquet", text="sentence")
        out["test"] = sst2_test
        out["shift_parts"] = {"imdb": subsample(*imdb_test, N_SUB, 12), "yelp": yelp}
    else:
        xa, ya = cola_tsv("in_domain_train")
        xb, yb = cola_tsv("in_domain_dev")
        x, y = xa + xb, np.concatenate([ya, yb])
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(x))
        tr, he, te = perm[:6355], perm[6355:6355 + 907], perm[6355 + 907:]
        out["train"] = ([x[i] for i in tr], y[tr])
        out["held"] = ([x[i] for i in he], y[he])
        out["test"] = ([x[i] for i in te], y[te])
        out["shift_parts"] = {"cola_ood": cola_tsv("out_of_domain_dev")}
    ag = parquet("fancyzhx__ag_news/data/test-00000-of-00001.parquet")
    ng = jsonl("SetFit__20_newsgroups/test.jsonl")
    ng = ([t for t in ng[0] if len(t.split()) >= 5], None)
    out["ood"] = {"ag_news": subsample(ag[0], np.full(len(ag[0]), -1), N_SUB, 13)[0],
                  "newsgroups": subsample(ng[0], np.full(len(ng[0]), -1), N_SUB, 14)[0]}
    return out


def build_data():
    """Tokenize once per task; every seed reads the same token ids (data.pkl) and split sizes."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = lambda texts: [ids[:MAX_LEN] for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    r = raw_sets()
    # SST-2: every GLUE train and dev sentence ends with a space, and the head reads the last token. All other
    # sets are put in that format (also after truncation), or they differ from the training data by that token alone.
    end = tok(" ", add_special_tokens=False)["input_ids"] if LADDER == "sst2" else []
    enc_new = lambda texts: [ids[:MAX_LEN - len(end)] + end
                             for ids in tok([t.rstrip() for t in texts] if end else texts, add_special_tokens=False)["input_ids"]]
    sets = {"train": (enc(r["train"][0]), r["train"][1])}
    held_ids = enc(r["held"][0])
    if end:
        assert all(i[-len(end):] == end for i in held_ids + sets["train"][0][:1000]), "GLUE format changed"
    sets["val"] = (held_ids + enc_new(r["test"][0]), np.concatenate([r["held"][1], r["test"][1]]))
    sh_ids, sh_y, sh_c = [], [], []
    for name in CORRUPTIONS:
        t, y = r["shift_parts"][name]
        sh_ids += enc_new(t)
        sh_y.append(y)
        sh_c += [name] * len(t)
    sets["shift"] = (sh_ids, np.concatenate(sh_y))
    for name, texts in r["ood"].items():
        sets[name] = (enc_new(texts), np.full(len(texts), -1))
    # Random-token sequences, lengths drawn from the test-set lengths.
    rng = np.random.default_rng(15)
    lens = rng.choice([len(i) for i in sets["val"][0][len(r["held"][0]):]], 2000)
    sets["noise"] = ([rng.integers(0, 150_000, max(int(n), 4)).tolist() + end for n in lens], np.full(2000, -1))
    os.makedirs(TASK_DIR, exist_ok=True)
    with open(f"{TASK_DIR}/data.pkl", "wb") as f:
        pickle.dump({"sets": sets, "shift_corruption": np.array(sh_c)}, f)
    json.dump([len(r["held"][0]), len(r["test"][0])], open(f"{TASK_DIR}/split_sizes.json", "w"))
    print({k: (len(v[0]), float(np.mean([len(i) for i in v[0]]))) for k, v in sets.items()}, flush=True)


def load_data():
    with open(f"{TASK_DIR}/data.pkl", "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------- model
def load_model(device, path=None):
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(path or MODEL, num_labels=2, torch_dtype=torch.float32)
    m.config.pad_token_id = PAD
    return m.to(device)


PAD = 151643   # <|endoftext|>, Qwen2.5's pad token


def collate(seqs):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), PAD, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


def finetune():
    torch.manual_seed(NLP_SEED)
    device = pick_gpu(min_free_mb=14_000)
    ids, y = load_data()["sets"]["train"]
    model = load_model(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(len(ids) / BATCH)
    warm = int(WARMUP * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    rng = np.random.default_rng(100 + NLP_SEED)
    log, step, skipped, t0 = [], 0, 0, time.time()
    model.train()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(ids))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x, m = collate([ids[j] for j in b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=x.to(device), attention_mask=m.to(device))
            loss = F.cross_entropy(out.logits.float(), torch.from_numpy(y[b]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(gnorm):   # skip the rare bf16 overflow step
                opt.step()
            else:
                skipped += 1
            sched.step()
            step += 1
            if step % 100 == 0:
                log.append(dict(epoch=ep, step=step, loss=loss.item(), skipped=skipped, sec=time.time() - t0))
                print(log[-1], flush=True)
    os.makedirs(CKPT, exist_ok=True)
    model.save_pretrained(f"{CKPT}/model", safe_serialization=True)
    torch.save({"weight": model.score.weight.detach().cpu(), "bias": torch.zeros(2)}, f"{CKPT}/head.pt")
    save_json(log, "finetune_log.json")


# ----------------------------------------------------------------- features
def batches(seqs, max_tokens=16_384):
    """Length-sorted batches of indices with at most max_tokens padded tokens each."""
    order = np.argsort([len(s) for s in seqs])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(seqs[i]) > max_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


@torch.no_grad()
def forward_states(model, seqs, device):
    """Last-token state after every block (n, 24, d), z = final-norm last token, logits."""
    n, L, d = len(seqs), model.config.num_hidden_layers, model.config.hidden_size
    cls = np.zeros((n, L, d), np.float32)
    z = np.zeros((n, d), np.float32)
    logits = np.zeros((n, 2), np.float32)
    for b in batches(seqs):
        x, m = collate([seqs[j] for j in b])
        last = m.sum(1) - 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x.to(device), attention_mask=m.to(device), output_hidden_states=True)
        hs = torch.stack([h[torch.arange(len(b)), last.to(device)] for h in out.hidden_states[1:]], 1).float()
        cls[b] = hs.cpu().numpy()
        z[b] = hs[:, -1].cpu().numpy()
        logits[b] = out.logits.float().cpu().numpy()
    return cls, z, logits


def extract():
    device = pick_gpu(min_free_mb=6000)
    model = load_model(device, f"{CKPT}/model").eval()
    data = load_data()
    for name, (seqs, y) in data["sets"].items():
        t0 = time.time()
        cls, z, lg = forward_states(model, seqs, device)
        set_npy(f"{name}_cls", cls.astype(np.float32))
        set_npy(f"{name}_z", z)
        set_npy(f"{name}_logits", lg[:, None])
        set_npy(f"{name}_labels", y)
        print(f"[{LADDER} seed {NLP_SEED} {name}] {len(seqs)} in {time.time() - t0:.0f}s, "
              f"acc {np.mean(lg.argmax(1) == y) if y[0] >= 0 else float('nan'):.4f}", flush=True)
    set_npy("shift_corruption", data["shift_corruption"])


# ----------------------------------------------------------------- baselines
@torch.no_grad()
def mc_logits(model, seqs, device, p):
    """Test-time dropout with rate p on the attention and MLP outputs of every block (8 passes)."""
    hooks = [mod.register_forward_hook(lambda _m, _i, o: (F.dropout(o[0], p, True),) + tuple(o[1:])
                                       if isinstance(o, tuple) else F.dropout(o, p, True))
             for layer in model.model.layers for mod in (layer.self_attn, layer.mlp)]
    out = np.zeros((len(seqs), PASSES, 2), np.float32)
    for s in range(PASSES):
        torch.manual_seed(s)
        for b in batches(seqs):
            x, m = collate([seqs[j] for j in b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[b, s] = model(input_ids=x.to(device), attention_mask=m.to(device)).logits.float().cpu().numpy()
    for h in hooks:
        h.remove()
    return out


def dropout():
    device = pick_gpu(min_free_mb=6000)
    model = load_model(device, f"{CKPT}/model").eval()
    sets = load_data()["sets"]
    calB = np.sort(val_split()["calB"])
    seqs, y = sets["val"]
    nll = {}
    for p in P_GRID:
        prob = torch.softmax(torch.from_numpy(mc_logits(model, [seqs[i] for i in calB], device, p)), -1).mean(1)
        nll[p] = -prob[torch.arange(len(calB)), torch.from_numpy(y[calB])].clamp_min(1e-12).log().mean().item()
        print("p", p, "calB NLL", nll[p], flush=True)
    best = min(nll, key=nll.get)
    for name in ["val", "shift", *OOD_SETS]:
        np.save(f"{FEAT}/dropout_{name}_logits.npy", mc_logits(model, sets[name][0], device, best).astype(np.float16))
    save_json({"calB_nll": {str(k): v for k, v in nll.items()}, "p": best, "passes": PASSES}, "dropout_selection.json")


def ensemble():
    """Logit caches of shape (n, 5, 2): this seed first, then the other four seeds (deep ensemble)."""
    for name in ["val", "shift", *OOD_SETS]:
        own = np.load(f"{FEAT}/{name}_logits.npy")[:, :1]
        others = [np.load(f"{TASK_DIR}/seed{s}/features/{name}_logits.npy")[:, :1] for s in range(N_SEEDS) if s != NLP_SEED]
        np.save(f"{FEAT}/{name}_logits.npy", np.concatenate([own, *others], 1))


if __name__ == "__main__":
    {"data": build_data, "finetune": finetune, "extract": extract, "dropout": dropout, "ensemble": ensemble}[sys.argv[1]]()
