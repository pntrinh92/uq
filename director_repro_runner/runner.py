"""Bounded parallel runner for jobs.json on shared GPUs.

A job starts when its dependencies are done and a GPU has at least NEED_MB free and fewer than PER_GPU of our
jobs; it gets --gpu <id> (the repo scripts set CUDA_VISIBLE_DEVICES from it). Out-of-memory failures are
retried after a pause; other failures are recorded and their dependents are skipped. State lives in state/:
<id>.done, <id>.fail, <id>.log. Safe to restart: finished jobs are not rerun.

Usage: python runner.py [max_concurrent] [per_gpu]
"""
import json
import os
import subprocess
import sys
import time

ROOT = "/ephemeral/trinhpn_uq/director_repro"
ENVS = {"director": "/ephemeral/trinhpn_uq/envs/director/bin/python", "cola": "/ephemeral/trinhpn_uq/envs/cola/bin/python"}
MAX_CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
PER_GPU = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NEED_MB = 4000
OOM = ("out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_NOT_INITIALIZED")
LAUNCH = ("import sys, runpy, torch; torch.cuda.init(); sys.argv = sys.argv[1:]; "
          "runpy.run_path(sys.argv[0], run_name='__main__')")
STATE = f"{ROOT}/state"
os.makedirs(STATE, exist_ok=True)
WANDB_ROOT = "/ephemeral/trinhpn_uq/tmp/wandb"
os.makedirs(WANDB_ROOT, exist_ok=True)
jobs = json.load(open(f"{ROOT}/jobs.json"))
by_id = {j["id"]: j for j in jobs}


def status(jid):
    for s in ("done", "fail"):
        if os.path.exists(f"{STATE}/{jid}.{s}"):
            return s
    return None


def data_ready(j):
    """CIFAR jobs wait until download_cifar.sh has written the splits and removed the raw batches."""
    d = f"{ROOT}/TransDiff/CIFAR/data"
    if "Cifar100" in j["args"]:
        return os.path.isdir(f"{d}/CIFAR100/test") and not os.path.exists(f"{d}/cifar-100-python")
    if "Cifar10" in j["args"]:
        return os.path.isdir(f"{d}/CIFAR10/test") and not os.path.exists(f"{d}/cifar-10-batches-py")
    return True


def free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    excluded = {int(g) for g in os.environ.get("GPU_EXCLUDE", "3").replace(" ", "").split(",") if g}  # GPU 3 off-limits
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines()) if int(a) not in excluded}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Adopted:
    """A job process left running by an earlier runner that died (ppid 1). We cannot get its exit code, so when it
    disappears we read the log of its last attempt: a Traceback means failure (OOM is then retried), else success."""

    def __init__(self, pid, jid):
        self.pid, self.jid = pid, jid

    def poll(self):
        if os.path.exists(f"/proc/{self.pid}"):
            return None
        t = open(f"{STATE}/{self.jid}.log", errors="ignore").read()
        return 1 if "Traceback" in t[t.rfind("### attempt"):] else 0


def adopt_orphans():
    """Find live processes whose cwd + argv match a job without a status file, so a restart does not rerun them."""
    want = {}
    for j in jobs:
        if status(j["id"]) is None:
            a = j["args"].split()
            want[(j["cwd"], tuple([a[0], "--gpu", "0", *a[1:]]))] = j["id"]
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    uuid2idx = {u.strip(): int(i) for i, u in (l.split(",") for l in smi.strip().splitlines())}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    pid2gpu = {int(p): uuid2idx.get(u.strip(), -1) for p, u in (l.split(",") for l in apps.strip().splitlines() if l)}
    match = {}   # pid -> (jid, ppid)
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().decode(errors="ignore").split("\0")[:-1]
            cwd = os.readlink(f"/proc/{d}/cwd")
            ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if len(argv) > 3 and argv[1] == "-c" and argv[2] == LAUNCH and (cwd, tuple(argv[3:])) in want:
            match[int(d)] = (want[(cwd, tuple(argv[3:]))], ppid)
    # 2026-11-20: DataLoader workers carry the same argv and cwd as their job; adopt only the main process (its
    # parent is not itself a matching process), else a short-lived worker's exit looks like the job ending.
    found = {}
    for pid, (jid, ppid) in match.items():
        if ppid in match:
            continue
        found[jid] = (Adopted(pid, jid), pid2gpu.get(pid, -1))
        attempts[jid] = 1
        log(f"adopt {jid} (pid {pid}, gpu {found[jid][1]})")
    return found


running, retry_at, attempts = {}, {}, {}
running.update(adopt_orphans())
while True:
    jobs = json.load(open(f"{ROOT}/jobs.json"))   # re-read, so jobs can be appended while the runner is up
    for jid, (p, gpu) in list(running.items()):   # reap finished jobs
        rc = p.poll()
        if rc is None:
            continue
        del running[jid]
        full = open(f"{STATE}/{jid}.log", errors="ignore").read()
        tail = full[-6000:]
        # 2026-11-20: c100_kflla_s2/s3 and imdb_kflla_s3 exited 0 mid Laplace fit without printing results and were
        # marked done. Every job's test prints "---> Test ... results："; require it in the last attempt.
        if rc == 0 and "results：" not in full[full.rfind("### attempt"):]:
            if attempts[jid] < 5:
                retry_at[jid] = time.time() + 180
                log(f"nores {jid} (gpu {gpu}) exit 0 without results, retry {attempts[jid]}")
            else:
                open(f"{STATE}/{jid}.fail", "w").write("exit 0 without a results line\n" + tail[-2000:])
                log(f"FAIL  {jid} exit 0 without results")
        elif rc == 0:
            open(f"{STATE}/{jid}.done", "w").close()
            log(f"done  {jid} (gpu {gpu})")
        elif any(s in tail for s in OOM) and attempts[jid] < 30:
            retry_at[jid] = time.time() + 180
            log(f"oom   {jid} (gpu {gpu}), retry {attempts[jid]}")
        else:
            open(f"{STATE}/{jid}.fail", "w").write(tail[-2000:])
            log(f"FAIL  {jid} rc={rc}")
    for j in jobs:   # dependents of failed jobs fail too
        if status(j["id"]) is None and any(status(d) == "fail" for d in j["deps"]):
            open(f"{STATE}/{j['id']}.fail", "w").write("dependency failed")
            log(f"skip  {j['id']} (dependency failed)")
    pending = [j for j in jobs if status(j["id"]) is None and j["id"] not in running]
    if not pending and not running:
        log("ALL FINISHED")
        break
    if len(running) < MAX_CONC:
        fm = free_mb()
        ours = {}
        for _, g in running.values():
            ours[g] = ours.get(g, 0) + 1
        for j in pending:
            if len(running) >= MAX_CONC:
                break
            if (time.time() < retry_at.get(j["id"], 0) or any(status(d) != "done" for d in j["deps"])
                    or not data_ready(j)):
                continue
            cands = [g for g, m in sorted(fm.items(), key=lambda x: -x[1]) if m >= NEED_MB and ours.get(g, 0) < PER_GPU]
            if not cands:
                break
            gpu = cands[0]
            attempts[j["id"]] = attempts.get(j["id"], 0) + 1
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE="disabled", WANDB_SILENT="true",
                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
                       # 2026-11-20: even disabled W&B stages artifact files under ~/.local/share/wandb on the full
                       # root disk (c10_kep7_s4 died there with ENOSPC); keep W&B and temp files on /ephemeral.
                       WANDB_DIR=WANDB_ROOT, WANDB_DATA_DIR=WANDB_ROOT, WANDB_CACHE_DIR=f"{WANDB_ROOT}/cache",
                       WANDB_CONFIG_DIR=f"{WANDB_ROOT}/config", WANDB_ARTIFACT_DIR=f"{WANDB_ROOT}/artifacts",
                       TMPDIR="/ephemeral/trinhpn_uq/tmp")
            # The repo scripts were run with --gpu 0 on one-GPU nodes; they set CUDA_VISIBLE_DEVICES from --gpu and
            # some use cuda:{gpu}. Initialising CUDA with the physical GPU visible first pins every device index to it.
            args = j["args"].split()
            cmd = [ENVS[j["env"]], "-c", LAUNCH, args[0], "--gpu", "0", *args[1:]]
            with open(f"{STATE}/{j['id']}.log", "a") as f:
                f.write(f"\n### attempt {attempts[j['id']]} gpu {gpu} {time.ctime()}\n### {' '.join(cmd)}\n")
            p = subprocess.Popen(cmd, cwd=j["cwd"], env=env, stdout=open(f"{STATE}/{j['id']}.log", "a"),
                                 stderr=subprocess.STDOUT)
            running[j["id"]] = (p, gpu)
            ours[gpu] = ours.get(gpu, 0) + 1
            fm[gpu] -= NEED_MB
            log(f"start {j['id']} on gpu {gpu}")
            time.sleep(20)   # let the job allocate before reading free memory again
    time.sleep(30)
