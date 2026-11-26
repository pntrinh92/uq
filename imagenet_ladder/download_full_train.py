"""Download the ImageNet-1k train shards not yet on disk (shards 73-293) for the full-data compile.

Shards 0-22 live under /data/trinhpn/imagenet-1k (train), 23-72 under imagenet_extra (train2);
the rest go to /ephemeral/trinhpn_uq/imagenet_full/data (train3a/b/c). The gated dataset needs a
Hugging Face token with ImageNet access in /ephemeral/trinhpn_uq/hf_token (the shared login of this
account belongs to someone else).
"""
from huggingface_hub import snapshot_download

token = open("/ephemeral/trinhpn_uq/hf_token").read().strip()
patterns = [f"data/train-{i:05d}-of-00294.parquet" for i in range(73, 294)]
snapshot_download("ILSVRC/imagenet-1k", repo_type="dataset", allow_patterns=patterns,
                  local_dir="/ephemeral/trinhpn_uq/imagenet_full", max_workers=8, token=token)
print("done", flush=True)
