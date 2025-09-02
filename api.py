# ──────────────────────────────────────────────────────────────────────────────
# api.py — 只管连通各模块：下载/解码 → 调图像分析 → 静态出图 →（可选）LLM 异步任务
# ──────────────────────────────────────────────────────────────────────────────

import os
import re
import ipaddress
import socket
import uuid
import base64
import binascii
import shutil
import tempfile
import logging
import threading
import asyncio
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, unquote
from pathlib import Path
from typing import Dict

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from starlette.background import BackgroundTask

from settings import BASE_DIR, PUBLIC_DIR, ALLOWED_EXT, MAX_DOWNLOAD_BYTES, FONT_PATH
from face_img import analyze_image
from llm import call_llm_for_face_analysis

# ---------------- 日志 ----------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("facemesh_api")

# ---------------- 应用与静态 ----------------
app = FastAPI(title="FaceMesh 12 Palaces API", version="3.0.0")
app.mount("/files", StaticFiles(directory=str(PUBLIC_DIR)), name="files")

# ---------------- 任务内存存储 ----------------
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = timedelta(hours=1)

def _utcnow():
    return datetime.now(timezone.utc)

def _new_job(status: str, result_url: str, filename: str, metrics: dict | None) -> dict:
    job_id = uuid.uuid4().hex
    job = dict(
        job_id=job_id,
        status=status,                 # running/completed/failed
        result_url=result_url,
        filename=filename,
        metrics=metrics,
        face_analysis=None,
        error=None,
        created_at=_utcnow().isoformat(),
        expires_at=(_utcnow() + JOB_TTL).isoformat(),
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job

def _get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        try:
            if _utcnow() > datetime.fromisoformat(job["expires_at"]):
                JOBS.pop(job_id, None)
                return None
        except Exception:
            pass
        return job

def _update_job(job_id: str, **fields):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)

# ---------------- 模型 ----------------
class AnalyzeReq(BaseModel):
    image_url: str                # http/https 或 data:image/...;base64,
    return_data_url: bool = False
    with_ai: bool = False

# ---------------- 工具 ----------------
def decode_base64_forgiving(b64: str) -> bytes:
    s = unquote(b64)
    s = re.sub(r"\s+", "", s)
    if len(s) % 4 != 0:
        s += "=" * (4 - (len(s) % 4))
    try:
        if "-" in s or "_" in s:
            return base64.urlsafe_b64decode(s)
        try:
            return base64.b64decode(s, validate=True)
        except binascii.Error:
            return base64.b64decode(s)
    except Exception as e:
        raise HTTPException(400, f"invalid base64 data: {e}")

def _is_private_host(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except Exception:
        return True

# ---------------- 全局异常 ----------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        raw = await request.body()
        preview = raw[:200].decode("utf-8", "ignore")
        size = len(raw)
    except Exception:
        preview, size = "", None
    logger.error("422 ValidationError: errors=%s | ct=%s | body_size=%s | preview=%r",
                 exc.errors(), request.headers.get("content-type"), size, preview)
    return JSONResponse(status_code=422, content={
        "detail": exc.errors(),
        "body_preview": preview,
        "body_size": size,
        "content_type": request.headers.get("content-type"),
        "hint": "检查字段名/类型与 JSON 语法；image_url 需为 http(s) 或 data:image/...;base64,..."
    })

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error("HTTPException %s on %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "status_code": exc.status_code, "path": request.url.path})

# ---------------- 基础路由 ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return JSONResponse({"msg": "POST /analyze → 返回 result_url (+ 可选 ai_job)。随后 GET /jobs/{job_id}?wait=1&timeout=55 轮询 face_analysis。"})

# ---------------- 后台 LLM 任务 ----------------
def _run_llm_job(job_id: str):
    job = _get_job(job_id)
    if not job:
        return
    metrics = job.get("metrics")
    if not metrics:
        _update_job(job_id, status="failed", error="metrics not found")
        return
    try:
        _update_job(job_id, status="running")
        face_analysis = call_llm_for_face_analysis(metrics)
        _update_job(job_id, status="completed", face_analysis=face_analysis, error=None)
    except HTTPException as e:
        _update_job(job_id, status="failed", error=f"{e.status_code}: {e.detail}")
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e))

def _start_llm_background(job_id: str):
    t = threading.Thread(target=_run_llm_job, args=(job_id,), daemon=True)
    t.start()

# ---------------- 主流程：下载 → 分析 → 静态出图 →（可选）LLM ----------------
@app.post("/analyze")
def analyze(req: AnalyzeReq, request: Request):
    client_ip = getattr(request.client, "host", "?")
    ct = request.headers.get("content-type")
    url_in = (req.image_url or "").strip()
    url_preview = (url_in[:80] + "...") if len(url_in) > 80 else url_in
    logger.info("Incoming /analyze from %s | CT=%s | image_url(len=%d, preview=%r) | return_data_url=%s | with_ai=%s",
                client_ip, ct, len(url_in), url_preview, req.return_data_url, req.with_ai)

    if not url_in:
        raise HTTPException(400, "image_url is required")

    workdir = Path(tempfile.mkdtemp(prefix="facemesh_api_"))
    logger.info("Workdir: %s", workdir)

    try:
        # 解析输入 → 保存图片到临时文件
        if url_in.startswith("data:"):
            if "," not in url_in:
                raise HTTPException(400, "invalid data URL")
            header, b64 = url_in.split(",", 1)
            mime = "image/jpeg"
            if ":" in header:
                h1 = header.split(":", 1)[1]
                mime = (h1.split(";")[0] or "image/jpeg").lower()
            ext_map = {"image/jpeg": ".jpg","image/jpg": ".jpg","image/png": ".png","image/webp": ".webp","image/bmp": ".bmp"}
            ext = ext_map.get(mime, ".jpg")
            raw = decode_base64_forgiving(b64)
            if len(raw) > MAX_DOWNLOAD_BYTES:
                raise HTTPException(413, "image too large (>20MB)")
            input_path = workdir / f"{uuid.uuid4().hex}{ext}"
            input_path.write_bytes(raw)
            logger.info("Saved decoded data URL to %s (%d bytes)", input_path, len(raw))
        else:
            parsed = urlparse(url_in)
            scheme = (parsed.scheme or "").lower()
            if scheme not in ("http", "https"):
                raise HTTPException(400, "image_url must be http(s) or data URL")
            host = parsed.hostname
            if not host or _is_private_host(host):
                raise HTTPException(400, "refuse to fetch from private/loopback addresses")
            url_path = parsed.path or ""
            ext = (os.path.splitext(url_path)[1] or ".jpg").lower()
            if ext not in ALLOWED_EXT:
                ext = ".jpg"
            input_path = workdir / f"{uuid.uuid4().hex}{ext}"
            try:
                with requests.get(url_in, stream=True, timeout=20) as r:
                    r.raise_for_status()
                    ctype = (r.headers.get("content-type", "").split(";")[0] or "").lower()
                    if ctype not in {"image/jpeg","image/jpg","image/png","image/webp","image/bmp"}:
                        raise HTTPException(415, f"unsupported content-type: {ctype}")
                    downloaded = 0
                    with open(input_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            downloaded += len(chunk)
                            if downloaded > MAX_DOWNLOAD_BYTES:
                                raise HTTPException(413, "image too large (>20MB)")
                            f.write(chunk)
                logger.info("Downloaded URL to %s (%d bytes)", input_path, input_path.stat().st_size)
            except Exception as e:
                raise HTTPException(502, f"failed to download image: {e}")

        # 调 face_img.analyze_image
        rec = analyze_image(
            img_path=str(input_path),
            out_dir=str(workdir),
            max_width=1600,
            max_height=1600,
            font_path=(FONT_PATH if FONT_PATH.exists() else None),
        )

        # 找输出图
        output_path = Path(rec.get("output") or "")
        if not output_path.exists():
            cands = sorted(workdir.glob("*_12palaces_labeled.jpg"))
            if cands:
                output_path = cands[0]
        if not output_path or not output_path.exists():
            raise HTTPException(500, "output image not found")

        # 移到 public/
        out_name = f"{uuid.uuid4().hex}.jpg"
        public_path = PUBLIC_DIR / out_name
        shutil.move(str(output_path), str(public_path))
        base = str(request.base_url).rstrip("/")
        result_url = f"{base}/files/{out_name}"

        # 先返回图片
        payload = {"result_url": result_url, "filename": out_name}

        if req.return_data_url:
            b64_out = base64.b64encode(public_path.read_bytes()).decode("ascii")
            payload["data_url"] = "data:image/jpeg;base64," + b64_out

        # with_ai=true：无论成功与否，都返回 ai_job（与旧逻辑一致）
        metrics = rec.get("metrics")
        if req.with_ai:
            if not metrics:
                job = _new_job(status="failed", result_url=result_url, filename=out_name, metrics=None)
                _update_job(job["job_id"], error="metrics not found")
            else:
                job = _new_job(status="running", result_url=result_url, filename=out_name, metrics=metrics)
                _start_llm_background(job["job_id"])
            payload["ai_job"] = {
                "job_id": job["job_id"],
                "status": job["status"],
                "created_at": job["created_at"],
                "expires_at": job["expires_at"],
            }

        logger.info("Success -> %s | with_ai=%s", result_url, req.with_ai)
        return JSONResponse(payload, background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True))

    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True); raise
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        shutil.rmtree(workdir, ignore_errors=True); raise HTTPException(500, f"unexpected error: {e}")

# ---------------- 轮询 ----------------
@app.get("/jobs/{job_id}")
async def get_job(job_id: str, wait: int = 1, timeout: int = 55):
    start = _utcnow()
    poll_interval = 0.8

    def _view(j: dict) -> dict:
        base = {"job_id": j["job_id"], "status": j["status"], "result_url": j["result_url"], "filename": j["filename"]}
        if j["status"] == "completed":
            base["face_analysis"] = j["face_analysis"]
        if j["status"] == "failed":
            base["error"] = j["error"]
        return base

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found or expired")

    if job["status"] in ("completed", "failed") or not wait:
        return JSONResponse(_view(job))

    while (datetime.now(timezone.utc) - start).total_seconds() < timeout:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found or expired")
        if job["status"] in ("completed", "failed"):
            return JSONResponse(_view(job))
        await asyncio.sleep(poll_interval)

    view = _view(job)
    view["hint"] = "still processing, please poll again"
    return JSONResponse(view, status_code=202)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
