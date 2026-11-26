# Added 2026-11-03 for the DIRECTOR reproduction (CoLA env, torch 2.0.1+cu118 on H100).
import os
# /tmp sits on the nearly full root disk; keep temporary files (W&B artifacts etc.) on /ephemeral.
if os.path.isdir("/ephemeral/trinhpn_uq/tmp"):
    os.environ["TMPDIR"] = "/ephemeral/trinhpn_uq/tmp"
# torch 2.0.1's memory-efficient / flash SDPA backward hits "CUDA error: an illegal instruction" on sm_90
# (CoLA DiT). Use the math kernel instead: same result, slower. Applied lazily so importing torch stays optional.
import builtins
_orig_import = builtins.__import__
def _hook(name, *a, **k):
    m = _orig_import(name, *a, **k)
    if name == "torch" or name.startswith("torch."):
        torch = __import__("sys").modules.get("torch")
        fn = getattr(getattr(getattr(torch, "backends", None), "cuda", None), "enable_mem_efficient_sdp", None)
        if fn is not None:   # torch fully initialised: apply once and remove the hook
            builtins.__import__ = _orig_import
            fn(False)
            torch.backends.cuda.enable_flash_sdp(False)
    return m
builtins.__import__ = _hook
