"""Slopsmith Demucs Separation Service.

Lightweight HTTP server that runs demucs source separation.
Designed to run on a desktop with GPU/RAM while Slopsmith runs on a NAS.

Usage:
    python server.py --port 7865
    python server.py --port 7865 --device cuda
    python server.py --port 7865 --model mdx_extra --api-key mysecret
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── Configuration ───────────────────────────────────────────────────────

DEMUCS_MODEL = os.environ.get("SLOPSMITH_DEMUCS_MODEL", "htdemucs_ft")
DEMUCS_DEVICE = os.environ.get("SLOPSMITH_DEMUCS_DEVICE", "")
API_KEY = os.environ.get("SLOPSMITH_API_KEY", "")
CACHE_DIR = Path(os.environ.get(
    "SLOPSMITH_DEMUCS_CACHE",
    Path.home() / ".cache" / "slopsmith-demucs",
))
MAX_CONCURRENT = 2

# ── State ───────────────────────────────────────────────────────────────

app = FastAPI(title="Slopsmith Demucs Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jobs: job_id -> {status, progress, stems, error, created_at, audio_hash, model}
jobs: OrderedDict[str, dict] = OrderedDict()
jobs_lock = threading.Lock()
active_count = 0
active_lock = threading.Lock()

# WebSocket subscribers: job_id -> set of WebSocket
ws_subscribers: dict[str, set] = {}

# Resolved config (set at startup)
_model = DEMUCS_MODEL
_device = ""
_gpu_available = False


# ── Auth middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def check_api_key(request, call_next):
    if API_KEY and request.url.path not in ("/health", "/docs", "/openapi.json"):
        key = request.headers.get("X-API-Key", request.query_params.get("api_key", ""))
        if key != API_KEY:
            return JSONResponse({"error": "Unauthorized"}, 401)
    return await call_next(request)


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "demucs_model": _model,
        "gpu": _gpu_available,
        "device": _device,
        "cache_dir": str(CACHE_DIR),
    }


# ── Separation via file upload ──────────────────────────────────────────

@app.post("/separate")
async def separate_upload(
    file: UploadFile = File(...),
    stems: str = Query("drums,bass,vocals,other"),
    model: str = Query(""),
):
    use_model = model or _model
    stem_list = [s.strip() for s in stems.split(",") if s.strip()]

    # Save upload to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "audio.mp3").suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    # Hash for cache key
    audio_hash = hashlib.sha256(content).hexdigest()[:16]
    job_id = audio_hash

    # Check cache
    cached = _check_cache(job_id, stem_list, use_model)
    if cached:
        os.unlink(tmp.name)
        return {"job_id": job_id, "stems": cached, "cached": True}

    # Queue the job
    result = _enqueue_job(job_id, tmp.name, stem_list, use_model)
    if result.get("error"):
        return JSONResponse(result, 503)
    return result


# ── Separation via URL ──────────────────────────────────────────────────

@app.post("/separate-url")
async def separate_url(
    data: dict,
    stems: str = Query("drums,bass,vocals,other"),
    model: str = Query(""),
):
    url = data.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, 400)

    use_model = model or _model
    stem_list = [s.strip() for s in stems.split(",") if s.strip()]

    # Hash the URL for cache key
    audio_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    job_id = audio_hash

    # Check cache
    cached = _check_cache(job_id, stem_list, use_model)
    if cached:
        return {"job_id": job_id, "stems": cached, "cached": True}

    # Download the file first
    import urllib.request
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        urllib.request.urlretrieve(url, tmp.name)
    except Exception as e:
        os.unlink(tmp.name)
        return JSONResponse({"error": f"Failed to download audio: {e}"}, 400)

    result = _enqueue_job(job_id, tmp.name, stem_list, use_model)
    if result.get("error"):
        return JSONResponse(result, 503)
    return result


# ── Download stems ──────────────────────────────────────────────────────

@app.get("/download/{job_id}/{stem}")
def download_stem(job_id: str, stem: str):
    # stem can be "drums.mp3", "drums.wav", or just "drums"
    stem_name = Path(stem).stem

    # Try multiple extensions
    for ext in (".mp3", ".wav", ".flac"):
        path = CACHE_DIR / job_id / f"{stem_name}{ext}"
        if path.exists():
            media = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
            return FileResponse(str(path), media_type=media.get(ext[1:], "application/octet-stream"))

    return JSONResponse({"error": "Stem not found"}, 404)


# ── Jobs list ───────────────────────────────────────────────────────────

@app.get("/jobs")
def list_jobs():
    with jobs_lock:
        return list(jobs.values())[-50:]  # last 50


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, 404)
    return job


# ── Cache management ────────────────────────────────────────────────────

@app.delete("/cache/{job_id}")
def delete_cache(job_id: str):
    cache_path = CACHE_DIR / job_id
    if cache_path.exists():
        shutil.rmtree(cache_path, ignore_errors=True)
    with jobs_lock:
        jobs.pop(job_id, None)
    return {"ok": True}


# ── WebSocket for job progress ──────────────────────────────────────────

@app.websocket("/ws/jobs/{job_id}")
async def ws_job_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in ws_subscribers:
        ws_subscribers[job_id] = set()
    ws_subscribers[job_id].add(websocket)
    try:
        # Send current state immediately
        with jobs_lock:
            job = jobs.get(job_id)
        if job:
            await websocket.send_json(job)
        # Keep connection open until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_subscribers.get(job_id, set()).discard(websocket)


# ── Internal helpers ────────────────────────────────────────────────────

def _check_cache(job_id, stem_list, model):
    """Return stem download URLs if all requested stems are cached."""
    cache_path = CACHE_DIR / job_id
    if not cache_path.exists():
        return None

    stems_found = {}
    for stem_name in stem_list:
        for ext in (".mp3", ".wav", ".flac"):
            p = cache_path / f"{stem_name}{ext}"
            if p.exists():
                stems_found[stem_name] = f"/download/{job_id}/{stem_name}{ext}"
                break

    if len(stems_found) == len(stem_list):
        return stems_found
    return None


def _enqueue_job(job_id, audio_path, stem_list, model):
    """Create a job and start processing in background."""
    global active_count

    with jobs_lock:
        # If job already exists and is processing/complete, return it
        existing = jobs.get(job_id)
        if existing and existing["status"] in ("processing", "complete"):
            if existing["status"] == "complete":
                return {"job_id": job_id, "stems": existing["stems"], "cached": True}
            return {"job_id": job_id, "status": "processing"}

    with active_lock:
        if active_count >= MAX_CONCURRENT:
            return {"error": "Server busy — max concurrent separations reached", "job_id": job_id}

    job = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0,
        "stems": {},
        "error": None,
        "model": model,
        "created_at": time.time(),
    }
    with jobs_lock:
        jobs[job_id] = job
        # Trim old jobs
        while len(jobs) > 200:
            jobs.popitem(last=False)

    thread = threading.Thread(
        target=_run_demucs,
        args=(job_id, audio_path, stem_list, model),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "processing"}


def _run_demucs(job_id, audio_path, stem_list, model):
    """Run demucs separation in a background thread."""
    global active_count

    with active_lock:
        active_count += 1

    tmp_out = tempfile.mkdtemp(prefix="demucs_out_")
    try:
        _update_job(job_id, status="processing", progress=10)

        # Build demucs command
        run_demucs = str(Path(__file__).parent / "run_demucs.py")
        cmd = [sys.executable, run_demucs, "--shifts", "2"]
        if model:
            cmd.extend(["-n", model])
        if _device:
            cmd.extend(["-d", _device])
        cmd.extend(["-o", tmp_out, audio_path])

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        # Read stderr for progress (demucs outputs progress there)
        _update_job(job_id, progress=20)
        _, stderr = proc.communicate(timeout=600)

        if proc.returncode != 0:
            # Strip tqdm progress bars — real errors are at the end
            err_lines = [l for l in stderr.splitlines() if l and '%|' not in l and 'B/s]' not in l]
            err_msg = '\n'.join(err_lines[-20:]) if err_lines else stderr[-1000:]
            _update_job(job_id, status="failed", error=err_msg[:1000])
            return

        _update_job(job_id, progress=80)

        # Find output stems
        # Demucs outputs to: {out_dir}/{model}/{track_name}/{stem}.wav
        audio_stem = Path(audio_path).stem
        out_model_dir = Path(tmp_out) / model
        if not out_model_dir.exists():
            # Try finding any model directory
            subdirs = list(Path(tmp_out).iterdir())
            out_model_dir = subdirs[0] if subdirs else Path(tmp_out)

        out_track_dir = out_model_dir / audio_stem
        if not out_track_dir.exists():
            # Try finding any track directory
            subdirs = list(out_model_dir.iterdir())
            out_track_dir = subdirs[0] if subdirs else out_model_dir

        # Copy stems to cache — keep as lossless WAV for quality
        cache_path = CACHE_DIR / job_id
        cache_path.mkdir(parents=True, exist_ok=True)

        stems_result = {}
        for stem_name in stem_list:
            src = out_track_dir / f"{stem_name}.wav"
            if not src.exists():
                continue

            wav_dest = cache_path / f"{stem_name}.wav"
            shutil.copy2(src, wav_dest)
            stems_result[stem_name] = f"/download/{job_id}/{stem_name}.wav"

        _update_job(job_id, status="complete", progress=100, stems=stems_result)

    except subprocess.TimeoutExpired:
        proc.kill()
        _update_job(job_id, status="failed", error="Separation timed out (10 min limit)")
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e))
    finally:
        with active_lock:
            active_count -= 1
        # Cleanup
        shutil.rmtree(tmp_out, ignore_errors=True)
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def _update_job(job_id, **kwargs):
    """Update job state and notify WebSocket subscribers."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(kwargs)

    # Notify WebSocket subscribers
    subs = ws_subscribers.get(job_id, set()).copy()
    for ws in subs:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(ws.send_json(job))
        except Exception:
            ws_subscribers.get(job_id, set()).discard(ws)


# ── GPU detection ───────────────────────────────────────────────────────

def _detect_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    global _model, _device, _gpu_available, API_KEY

    parser = argparse.ArgumentParser(description="Slopsmith Demucs Separation Service")
    parser.add_argument("--port", type=int, default=7865, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", default="", help="Demucs model (htdemucs, mdx_extra)")
    parser.add_argument("--device", default="", help="Device (cpu, cuda)")
    parser.add_argument("--api-key", default="", help="API key for auth")
    args = parser.parse_args()

    if args.model:
        _model = args.model
    if args.device:
        _device = args.device
    if args.api_key:
        API_KEY = args.api_key

    _gpu_available = _detect_gpu()
    if not _device:
        _device = "cuda" if _gpu_available else "cpu"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Slopsmith Demucs Server starting on {args.host}:{args.port}")
    print(f"  Model: {_model}")
    print(f"  Device: {_device} (GPU: {_gpu_available})")
    print(f"  Cache: {CACHE_DIR}")
    if API_KEY:
        print("  API key: enabled")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
