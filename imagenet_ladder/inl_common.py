"""Shared pieces for the ImageNet ablation ladder (ViT-B/16).

Ladder: (1) FM, (2) compiled ODE flow F_theta (Thrust 1 Task 1, Eq. 7), (3) Brownian
same-marginal SDE at the final representation, (4) regeneration scaffold.

Data (see README.md): ImageNet-1k validation (50k) split into calA (5k, conformal
thresholds), calB (5k, selection and temperature) and test (40k); a 23-shard train
subset (about 100 images per class) as the regeneration pool and compilation data;
ImageNet-C-style corruptions (severity 3) of test images; ImageNet-O, DTD and
uniform noise as OOD sets. Big arrays live under /ephemeral/trinhpn_uq.
"""
import glob
import io
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from common import (TiedGMM, auroc, classification_metrics, ece_score, entropy,  # noqa: E402,F401
                    fit_temperature)

IMAGENET = "/data/trinhpn/imagenet-1k/data"
WORK = "/ephemeral/trinhpn_uq"
DATA = f"{WORK}/data"
FM_NAME = "vit_base_patch16_224.orig_in21k_ft_in1k"
ENSEMBLE_NAMES = [FM_NAME, "vit_base_patch16_224.augreg_in21k_ft_in1k", "vit_base_patch16_224.augreg2_in21k_ft_in1k"]
N_TRAIN_SHARDS = 23
N_SHIFT_PER_CORRUPTION = 5000
SPLIT_SEED = 0

# Which ladder: "imagenet" (default) or "cifar10" / "cifar100" (DIRECTOR setting: the same
# ViT-B/16 checkpoint fine-tuned on CIFAR, 224 px inputs). Set LADDER in the environment.
LADDER = os.environ.get("LADDER", "imagenet")
# NLP ladders (DIRECTOR's text benchmarks: IMDB, CoLA, SST-2) use a fine-tuned Qwen2.5-0.5B, one run
# per seed (NLP_SEED); see ../nlp_ladder/nlp_prepare.py.
NLP_TASKS = ("imdb", "cola", "sst2")
NLP_SEED = int(os.environ.get("NLP_SEED", "0"))
# Layers that carry the regeneration scaffold, as indices into the cached per-block CLS states.
REGEN_LAYERS = [1, 3, 5, 7, 9, "z"]          # ViT-B/16: CLS after blocks 2, 4, 6, 8, 10, then z
if LADDER == "imagenet":
    FEAT = f"{WORK}/features"
    CKPT = f"{WORK}/ckpt"
    RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    NUM_CLASSES = 1000
    CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "contrast", "jpeg"]
    OOD_SETS = ["imagenet_o", "dtd", "noise"]
    HAS_REAL = True
elif LADDER in NLP_TASKS:
    FEAT = f"{WORK}/nlp/{LADDER}/seed{NLP_SEED}/features"
    CKPT = f"{WORK}/nlp/{LADDER}/seed{NLP_SEED}/ckpt"
    RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlp_ladder", f"results_{LADDER}",
                           f"seed{NLP_SEED}")
    NUM_CLASSES = 2
    # Shift sets keep the task labels; their names play the role of corruption types.
    CORRUPTIONS = {"imdb": ["sst2", "yelp"], "sst2": ["imdb", "yelp"], "cola": ["cola_ood"]}[LADDER]
    OOD_SETS = ["ag_news", "newsgroups", "noise"]
    HAS_REAL = False
    REGEN_LAYERS = [3, 7, 11, 15, 19, "z"]   # Qwen2.5-0.5B (24 blocks): last token after blocks 4, 8, ..., 20, then z
else:
    FEAT = f"{WORK}/{LADDER}/features"
    CKPT = f"{WORK}/{LADDER}/ckpt"
    RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cifar_ladder", f"results_{LADDER}")
    NUM_CLASSES = {"cifar10": 10, "cifar100": 100}[LADDER]
    # The 15 standard CIFAR-C corruptions (Hendrycks and Dietterich, 2019), all 5 severities.
    CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"]
    OOD_SETS = ["svhn", "cifar100" if LADDER == "cifar10" else "cifar10", "lsun", "dtd", "noise"]
    HAS_REAL = False


# GPUs our jobs must never use; override with GPU_EXCLUDE="3,5" (or GPU_EXCLUDE="" for none).
EXCLUDED_GPUS = {int(g) for g in os.environ.get("GPU_EXCLUDE", "3").replace(" ", "").split(",") if g}


def pick_gpu(min_free_mb=3000):
    """Use the GPU with the most free memory (the box is shared).

    Our own jobs record a claim (pid -> GPU) under a file lock and skip GPUs claimed by
    another live job of ours, so two of our jobs never start on the same card at once.
    GPU_ALLOW_SHARED=1 (opt-in) lets this job pick a card our other jobs already hold when it
    has the most free memory; the claim is still recorded.
    """
    import fcntl
    claims_dir = f"{WORK}/tmp/gpu_claims"
    os.makedirs(claims_dir, exist_ok=True)
    with open(f"{claims_dir}/.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        claimed = set()
        for f in os.listdir(claims_dir):
            if f.startswith("."):
                continue
            pid = int(f)
            if pid != os.getpid() and os.path.exists(f"/proc/{pid}"):
                claimed.add(int(open(f"{claims_dir}/{f}").read()))
            elif pid != os.getpid():
                os.remove(f"{claims_dir}/{f}")
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        skip = set() if os.environ.get("GPU_ALLOW_SHARED") == "1" else set(claimed)
        skip |= EXCLUDED_GPUS  # never ours to use (2026-11-26: GPU 3 is off-limits)
        free = sorted(((int(f), int(i)) for i, f in (l.split(",") for l in out) if int(i) not in skip),
                      reverse=True)
        if not free or free[0][0] < min_free_mb:
            raise RuntimeError(f"no GPU with {min_free_mb} MB free (excluding our claims {claimed} and {sorted(EXCLUDED_GPUS)}): {free}")
        with open(f"{claims_dir}/{os.getpid()}", "w") as f:
            f.write(str(free[0][1]))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    print(f"using GPU {free[0][1]} ({free[0][0]} MB free)", flush=True)
    return torch.device("cuda")


# ----------------------------------------------------------------- image sources
def val_split():
    """Indices into the "val" feature set: calA (thresholds), calB (selection, temperature), test.

    ImageNet: a random split of the 50k validation images. CIFAR: "val" holds the 5k images held
    out from the training split (calA, calB) followed by the 10k official test images (test).
    """
    if LADDER == "imagenet":
        perm = np.random.default_rng(SPLIT_SEED).permutation(50_000)
        return {"calA": perm[:5000], "calB": perm[5000:10_000], "test": perm[10_000:]}
    if LADDER in NLP_TASKS:   # "val" = held-out set followed by the test set (nlp_prepare.py writes the sizes)
        n_held, n_test = json.load(open(f"{FEAT}/../../split_sizes.json"))
        h = n_held // 2
        return {"calA": np.arange(0, h), "calB": np.arange(h, n_held), "test": np.arange(n_held, n_held + n_test)}
    return {"calA": np.arange(0, 2500), "calB": np.arange(2500, 5000), "test": np.arange(5000, 15_000)}


def head_weights():
    """Weight and bias of the FM's linear classification head (maps z to logits)."""
    if LADDER in NLP_TASKS:
        st = torch.load(f"{CKPT}/head.pt", map_location="cpu")
        return st["weight"].float(), st["bias"].float()
    head = load_fm("cpu").head
    return head.weight.detach().float(), head.bias.detach().float()


def load_fm(device, **kwargs):
    """The FM: the ImageNet checkpoint, or the same checkpoint fine-tuned on CIFAR (finetune.py)."""
    import timm
    if LADDER == "imagenet":
        return timm.create_model(FM_NAME, pretrained=True, **kwargs).to(device).eval()
    m = timm.create_model(FM_NAME, pretrained=False, num_classes=NUM_CLASSES, **kwargs)
    m.load_state_dict(torch.load(f"{CKPT}/finetuned.pt", map_location="cpu"))
    return m.to(device).eval()


def load_parquet_rows(pattern, root=IMAGENET):
    import pyarrow.parquet as pq
    images, labels, names = [], [], []
    for f in sorted(glob.glob(f"{root}/{pattern}")):
        t = pq.read_table(f)
        col = t.column("image").to_pylist()
        images += [c["bytes"] for c in col]
        names += [c["path"] for c in col]
        labels += t.column("label").to_pylist()
    return images, np.array(labels), names


def real_labels(names):
    """ReaL label sets aligned with the given validation file names (empty list = no ReaL label)."""
    real = json.load(open(f"{DATA}/real.json"))
    return [real[int(n.split("_")[2].split(".")[0]) - 1] for n in names]


def ood_files():
    io_files = sorted(glob.glob(f"{DATA}/imagenet-o/n*/*"))
    dtd_files = sorted(glob.glob(f"{DATA}/dtd/images/*/*.jpg"))
    return {"imagenet_o": io_files, "dtd": dtd_files}


# ----------------------------------------------------------------- preprocessing and corruptions
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def to_unit_tensor(img):
    """timm eval transform up to [0, 1]: bicubic resize to 248 (crop_pct 0.9), center crop 224."""
    img = img.convert("RGB")
    w, h = img.size
    s = 248 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def corrupt(x, kind, rng):
    """ImageNet-C definitions at severity 3 (Hendrycks and Dietterich, 2019), on a [0, 1] crop."""
    if kind == "gaussian_noise":
        return (x + torch.from_numpy(rng.normal(0, 0.18, x.shape).astype(np.float32))).clamp(0, 1)
    if kind == "gaussian_blur":
        sigma = 3.0
        k = int(2 * round(4 * sigma) + 1)
        g = torch.exp(-((torch.arange(k) - k // 2) ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).float()
        y = F.conv2d(F.pad(x[:, None], (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
        y = F.conv2d(F.pad(y, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
        return y[:, 0].clamp(0, 1)
    if kind == "contrast":
        means = x.mean(dim=(1, 2), keepdim=True)
        return ((x - means) * 0.2 + means).clamp(0, 1)
    if kind == "jpeg":
        img = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=15)
        return torch.from_numpy(np.asarray(Image.open(buf), dtype=np.float32) / 255.0).permute(2, 0, 1)
    raise ValueError(kind)


def normalize(x):
    return (x - MEAN) / STD


class ImageSet(torch.utils.data.Dataset):
    """Items are (source, corruption) pairs; source is jpeg bytes, a file path, or 'noise:<seed>'."""

    def __init__(self, sources, corruptions=None):
        self.sources = sources
        self.corruptions = corruptions or [None] * len(sources)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        src = self.sources[i]
        if isinstance(src, str) and src.startswith("noise:"):
            rng = np.random.default_rng(int(src[6:]))
            x = torch.from_numpy(rng.uniform(0, 1, (3, 224, 224)).astype(np.float32))
        else:
            img = Image.open(io.BytesIO(src)) if isinstance(src, bytes) else Image.open(src)
            x = to_unit_tensor(img)
        if self.corruptions[i] is not None:
            x = corrupt(x, self.corruptions[i], np.random.default_rng(i))
        return normalize(x)


def loader(dataset, batch_size=64, workers=16):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                                       pin_memory=True, persistent_workers=False, prefetch_factor=4)


# ----------------------------------------------------------------- backbone access
def embed(model, img):
    x = model.patch_embed(img)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    return model.norm_pre(x)


def readout(model, tokens):
    """Final representation z (post-norm CLS) and logits from a token state."""
    x = model.norm(tokens)
    z = model.fc_norm(model.pool(x))
    return z, model.head(model.head_drop(z))


def set_npy(name, arr):
    os.makedirs(FEAT, exist_ok=True)
    np.save(f"{FEAT}/{name}.npy", arr)


def get_npy(name, mmap=True):
    return np.load(f"{FEAT}/{name}.npy", mmap_mode="r" if mmap else None)


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)
