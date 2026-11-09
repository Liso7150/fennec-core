from __future__ import annotations

import os
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ncm.config import CKPT, RESULTS, ROOT, Config
from . import engine, loop
from .tools import ToolContext, list_dir, read_file

ACTIVE_WORKSPACE = os.environ.get("NCM_WORKSPACE", "D:/NCMUI/workspace")
app = FastAPI(title="NCM Coder — local coding agent on Colibri")
NCM = {"ready": False, "cfg": None, "gen": None, "error": None}
_engine_starting = {"active": False}


def _load_ncm():
    if NCM["ready"] or NCM["error"]:
        return
    try:
        from generate import load_models
        if not os.path.exists(os.path.join(CKPT, "tokenizer.pt")):
            NCM["error"] = "NCM tokenizer not trained yet"
            return
        cfg, tok, gen = load_models("cpu")
        NCM.update(ready=True, cfg=cfg, tok=tok, gen=gen)
    except Exception as e:
        NCM["error"] = repr(e)


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "agent.html"))


@app.get("/api/health")
def health():
    return {"app": "NCM Coder", "engine_running": engine.is_running(),
            "workspace": ACTIVE_WORKSPACE, "ncm_ready": NCM["ready"]}


# ---- engine (all Colibri features) --------------------------------------
@app.get("/api/engine")
def engine_telemetry():
    return engine.telemetry()


@app.post("/api/engine/start")
def engine_start():
    if engine.is_running():
        return {"running": True, "already": True}
    if _engine_starting["active"]:
        return {"starting": True}
    _engine_starting["active"] = True

    def _bg():
        try:
            engine.start(wait_s=240)
        finally:
            _engine_starting["active"] = False
    threading.Thread(target=_bg, daemon=True).start()
    return {"starting": True}


@app.post("/api/engine/stop")
def engine_stop():
    return engine.stop()


@app.get("/api/engine/plan")
def engine_plan():
    return engine.plan()


@app.get("/api/engine/doctor")
def engine_doctor():
    return engine.doctor()


@app.get("/api/engine/status")
def engine_status():
    return engine.status()


@app.get("/api/engine/logs")
def engine_logs():
    return engine.logs(80)


# ---- hardware (Intel NPU + RTX 4060 + shared RAM + SSD) -----------------
@app.get("/api/hardware")
def hardware():
    return engine.hardware()


@app.post("/api/npu/start")
def npu_start():
    return engine.ensure_npu()


class EmbedReq(BaseModel):
    input: str | list[str]


@app.post("/api/npu/embed")
def npu_embed(req: EmbedReq):
    try:
        return {"device": engine.npu_health(timeout=1).get("device", "?"),
                "embeddings": engine.embed(req.input)}
    except Exception as e:
        raise HTTPException(503, f"NPU embedding service unavailable: {e}")


@app.on_event("startup")
def _startup_npu():
    threading.Thread(target=engine.ensure_npu, kwargs={"wait_s": 90},
                     daemon=True).start()


# ---- agent ---------------------------------------------------------------
class AgentReq(BaseModel):
    task: str
    workspace: str | None = None
    allow_shell: bool = False
    max_steps: int = 14


@app.post("/api/agent/start")
def agent_start(req: AgentReq):
    if not engine.is_running():
        raise HTTPException(503, "Colibri engine is not running; start it first")
    ws = req.workspace or ACTIVE_WORKSPACE
    if not os.path.isdir(ws):
        os.makedirs(ws, exist_ok=True)
    sid = loop.new_session(req.task, ws, req.allow_shell)
    loop.start_thread(sid, req.max_steps)
    return {"id": sid, "workspace": ws}


@app.get("/api/agent/poll")
def agent_poll(id: str, after: int = 0):
    s = loop.get_steps(id, after)
    if s is None:
        raise HTTPException(404, "unknown session")
    return s


@app.get("/api/agent/sessions")
def agent_sessions():
    return {"sessions": [{"id": k, "task": v["task"], "status": v["status"],
                          "steps": len(v["steps"])} for k, v in loop.SESSIONS.items()]}


# ---- workspace files -----------------------------------------------------
@app.get("/api/files")
def files(path: str = "."):
    ctx = ToolContext(workspace=ACTIVE_WORKSPACE)
    return list_dir(ctx, path)


@app.get("/api/file")
def file_read(path: str):
    ctx = ToolContext(workspace=ACTIVE_WORKSPACE)
    return read_file(ctx, path)


class WorkspaceReq(BaseModel):
    path: str


@app.post("/api/workspace")
def set_workspace(req: WorkspaceReq):
    global ACTIVE_WORKSPACE
    p = os.path.abspath(req.path)
    if not os.path.isdir(p):
        raise HTTPException(400, "not a directory")
    ACTIVE_WORKSPACE = p
    return {"workspace": p}


# ---- NCM concept engine --------------------------------------------------
@app.get("/api/ncm/health")
def ncm_health():
    _load_ncm()
    cfg = NCM["cfg"] or Config()
    return {"ready": NCM["ready"], "error": NCM["error"], "device": "cpu",
            "K": cfg.K, "levels": list(cfg.levels), "seq_len": cfg.seq_len,
            "codebooks": list(cfg.codebook_sizes), "nfe": cfg.K,
            "tokens_per_pass": round(cfg.seq_len / cfg.K, 1)}


class NcmReq(BaseModel):
    n: int = 3
    temp: float = 0.9
    top_p: float = 0.9


@app.post("/api/ncm/generate")
def ncm_generate(req: NcmReq):
    _load_ncm()
    if not NCM["ready"]:
        raise HTTPException(503, NCM["error"] or "NCM not ready")
    from generate import run
    res = run(NCM["cfg"], NCM["gen"], "cpu", n=max(1, min(req.n, 6)),
              temp=req.temp, top_p=req.top_p, partials=True)
    return JSONResponse(res)


@app.get("/api/ncm/metrics")
def ncm_metrics():
    import json
    out = {}
    for name in ("tok_train", "gen_train"):
        p = os.path.join(RESULTS, f"{name}.json")
        out[name] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
    ev = os.path.join(RESULTS, "evaluation.json")
    out["evaluation"] = json.load(open(ev, encoding="utf-8")) if os.path.exists(ev) else None
    return out


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")