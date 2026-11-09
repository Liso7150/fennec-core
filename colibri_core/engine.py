from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

COLIBRI_ROOT = "D:/Colibri"
PY = os.path.join(COLIBRI_ROOT, ".venv", "Scripts", "python.exe")
COLI = os.path.join(COLIBRI_ROOT, "c", "coli")
MODEL_DIR = "D:/colibri-models/qwen3-coder-30b-colibri"
PORT = int(os.environ.get("NCM_COLIBRI_PORT", "8181"))
BASE = f"http://127.0.0.1:{PORT}"
SERVE_LOG = "D:/NCMUI/_colibri_engine.log"
API_KEY = os.environ.get("COLI_API_KEY", "")

NPU_PORT = int(os.environ.get("NCM_NPU_PORT", "8182"))
NPU_BASE = f"http://127.0.0.1:{NPU_PORT}"
NPU_LOG = "D:/NCMUI/_npu_embed.log"
NPU_SERVER = os.path.join(COLIBRI_ROOT, "scripts", "npu_embed_server.py")


def _headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def _get(path, timeout=4, base=None):
    req = urllib.request.Request((base or BASE) + path, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(path, body, timeout=600, base=None):
    h = {"Content-Type": "application/json", **_headers()}
    req = urllib.request.Request(
        (base or BASE) + path, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _engine_env():
    return dict(os.environ, COLI_VULKAN="1", COLI_VK_DEV2="auto",
                HF_HOME="D:/.cache/huggingface")


def is_running() -> bool:
    try:
        _get("/health", timeout=2)
        return True
    except Exception:
        return False


_model_id = None


def model_id() -> str:
    global _model_id
    if _model_id:
        return _model_id
    try:
        d = _get("/v1/models")
        _model_id = d["data"][0]["id"]
    except Exception:
        _model_id = "qwen3-coder-30b-a3b-colibri"
    return _model_id


def _coli(args, timeout=120):
    env = _engine_env()
    p = subprocess.run([PY, COLI] + args, cwd=COLIBRI_ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return {"exit": p.returncode, "out": (p.stdout + p.stderr)[-8000:]}


def start(wait_s=180) -> dict:
    if is_running():
        return {"running": True, "already": True, "model": model_id()}
    env = _engine_env()
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with open(SERVE_LOG, "w", encoding="utf-8") as log:
        subprocess.Popen(
            [PY, COLI, "serve", "--auto-tier", "--model", MODEL_DIR,
             "--port", str(PORT), "--ngen", "4096"],
            cwd=COLIBRI_ROOT, env=env, stdout=log, stderr=log, creationflags=flags)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if is_running():
            return {"running": True, "started": True, "model": model_id(),
                    "warmup_s": round(time.time() - t0, 1)}
        time.sleep(2)
    return {"running": False, "error": "engine did not become healthy in time",
            "log": SERVE_LOG}


def stop() -> dict:
    return _coli(["stop", "--port", str(PORT)], timeout=90)


def telemetry() -> dict:
    """Live Colibri feature surface for the dashboard."""
    out = {"running": is_running(), "port": PORT, "model_dir": MODEL_DIR}
    if not out["running"]:
        return out
    out["model"] = model_id()
    for key, path in (("health", "/health"), ("metrics", "/metrics"),
                      ("experts", "/experts"), ("profile", "/profile")):
        try:
            out[key] = _get(path, timeout=5)
        except Exception as e:
            out[key] = {"error": repr(e)}
    return out


def plan() -> dict:
    return _coli(["plan", "--model", MODEL_DIR], timeout=120)


def doctor() -> dict:
    return _coli(["doctor"], timeout=120)


def status() -> dict:
    return _coli(["status"], timeout=60)


def logs(n=60) -> dict:
    try:
        with open(SERVE_LOG, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return {"lines": lines[-n:], "path": SERVE_LOG}
    except OSError as e:
        return {"error": repr(e)}


def chat(messages, tools=None, tool_choice="auto", max_tokens=2048,
         temperature=0.2, timeout=600):
    body = {"model": model_id(), "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    d = _post("/v1/chat/completions", body, timeout=timeout)
    return d["choices"][0]["message"], d.get("usage", {})


# ---- Intel NPU (OpenVINO embedding service) ------------------------------
def npu_health(timeout=2):
    try:
        return _get("/health", timeout=timeout, base=NPU_BASE)
    except Exception as e:
        return {"status": "down", "error": repr(e)}


def ensure_npu(wait_s=60) -> dict:
    h = npu_health()
    if h.get("status") == "ok":
        return {"running": True, "already": True, **h}
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with open(NPU_LOG, "w", encoding="utf-8") as log:
        subprocess.Popen(
            [PY, NPU_SERVER, "--port", str(NPU_PORT), "--device", "NPU"],
            cwd=COLIBRI_ROOT, env=dict(os.environ, HF_HOME="D:/.cache/huggingface"),
            stdout=log, stderr=log, creationflags=flags)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        h = npu_health()
        if h.get("status") == "ok":
            return {"running": True, "started": True,
                    "warmup_s": round(time.time() - t0, 1), **h}
        time.sleep(1)
    return {"running": False, "error": "NPU embedding service did not start",
            "log": NPU_LOG}


def embed(texts, timeout=120):
    if isinstance(texts, str):
        texts = [texts]
    d = _post("/v1/embeddings", {"input": texts}, timeout=timeout, base=NPU_BASE)
    return [row["embedding"] for row in d.get("data", [])]


# ---- hardware snapshot ---------------------------------------------------
def _run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "").strip()
    except Exception:
        return ""


_VIDEO_NAMES = None


def _video_names():
    global _VIDEO_NAMES
    if _VIDEO_NAMES is None:
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        _VIDEO_NAMES = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return _VIDEO_NAMES


def hardware() -> dict:
    import psutil
    vm = psutil.virtual_memory()
    du = shutil.disk_usage("D:/")
    out = {
        "ram": {"total_gb": round(vm.total / 2**30, 1),
                "used_gb": round(vm.used / 2**30, 1),
                "available_gb": round(vm.available / 2**30, 1)},
        "disk_d": {"total_gb": round(du.total / 2**30, 1),
                   "free_gb": round(du.free / 2**30, 1)},
        "npu": npu_health(),
        "engine_running": is_running(),
    }
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"])
    gpus = []
    for ln in smi.splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) == 4 and parts[1].isdigit():
            gpus.append({"name": parts[0], "mem_used_mb": int(parts[1]),
                         "mem_total_mb": int(parts[2]), "util_pct": int(parts[3])})
    out["gpu"] = gpus
    ps = ("$n=@{}; Get-CimInstance Win32_VideoController | ForEach-Object { "
          "if ($_.AdapterLuid) { $n[$_.AdapterLuid.ToLower()] = $_.Name } }; "
          "Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory "
          "| ForEach-Object { $l = ($_.Name -replace '^luid_','' -replace '_phys_0$','').ToLower(); "
          "\"{0}|{1}|{2}|{3}\" -f $_.Name,$n[$l],$_.DedicatedUsage,$_.SharedUsage }")
    adapters, tot_ded, tot_shared = [], 0, 0
    names = _video_names()
    intel = next((n for n in names if "Intel" in n), "")
    for ln in _run(["powershell", "-NoProfile", "-Command", ps], timeout=25).splitlines():
        parts = ln.split("|")
        if len(parts) != 4 or not parts[2].isdigit() or not parts[3].isdigit():
            continue
        ded, shared = int(parts[2]) / 2**20, int(parts[3]) / 2**20
        if ded < 1 and shared < 1:
            continue
        label = parts[1] or parts[0]
        if label.startswith("luid_"):
            label = "shared-memory adapter" if ded < 50 else "GPU"
        adapters.append({"name": label, "luid": parts[0], "dedicated_mb": round(ded),
                         "shared_mb": round(shared)})
        tot_ded += ded
        tot_shared += shared
    if adapters:
        top = max(adapters, key=lambda a: a["dedicated_mb"])
        if gpus and top["dedicated_mb"] > 100:
            top["name"] = gpus[0]["name"]
        others = [a for a in adapters if a is not top and a["dedicated_mb"] < 50]
        if intel and others:
            big = max(others, key=lambda a: a["shared_mb"])
            if big["shared_mb"] > 200:
                big["name"] = f"{intel} (shared RAM)"
    out["gpu_adapters"] = adapters
    out["gpu_total"] = {"dedicated_mb": round(tot_ded), "shared_mb": round(tot_shared)}
    try:
        with open(SERVE_LOG, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-400:]
        out["tier"] = [ln.strip() for ln in lines
                       if "[VK] tier" in ln and ("second device" in ln or " on," in ln)][-3:]
    except OSError:
        out["tier"] = []
    return out