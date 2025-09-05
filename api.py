# api.py — 路由与编排（下载/解码 → 调图像分析 或 直接用 metrics → 静态出图 → 可选 LLM 异步任务）
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
from typing import Dict, Any, Optional, Tuple

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

# 新增：占位图
from PIL import Image, ImageDraw, ImageFont

# ---------------- 日志 ----------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("facemesh_api")

# ---------------- 应用与静态 ----------------
app = FastAPI(title="FaceMesh 12 Palaces API", version="3.2.0")
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
        metrics=metrics,               # 仅服务端保存
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
    # 改为可选
    image_url: Optional[str] = None     # http/https 或 data:image/...;base64,
    return_data_url: bool = False
    with_ai: bool = False
    # 可选：预计算几何指标（传了可跳过人脸检测/绘制）
    metrics: Optional[Dict[str, Any]] = None

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

def _make_placeholder(workdir: Path, size: Tuple[int, int]=(1000, 750)) -> Path:
    """在 metrics-only 模式下生成一张占位图，确保依旧有 result_url / filename。"""
    w, h = size
    img = Image.new("RGB", (w, h), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    title = "No image (metrics-only)"
    sub = "Face analysis will use provided geometric metrics."
    try:
        font_main = ImageFont.truetype(str(FONT_PATH), 36) if FONT_PATH.exists() else None
        font_sub = ImageFont.truetype(str(FONT_PATH), 24) if FONT_PATH.exists() else None
    except Exception:
        font_main = font_sub = None
    tw, th = draw.textlength(title, font=font_main), 36
    sw, sh = draw.textlength(sub, font=font_sub), 24
    draw.text(((w - tw) / 2, h/2 - th - 10), title, fill=(60,60,60), font=font_main, anchor=None)
    draw.text(((w - sw) / 2, h/2 + 10), sub, fill=(90,90,90), font=font_sub, anchor=None)
    out = workdir / f"{uuid.uuid4().hex}.jpg"
    img.save(out, format="JPEG", quality=92)
    return out

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

# ---------------- 主流程：下载 → （分析 或 直接用 metrics 或 生成占位图） → 静态出图 →（可选）LLM ----------------
@app.post("/analyze")
def analyze(req: AnalyzeReq, request: Request):
    client_ip = getattr(request.client, "host", "?")
    ct = request.headers.get("content-type")
    url_in = (req.image_url or "").strip()
    url_preview = (url_in[:80] + "...") if len(url_in) > 80 else url_in
    logger.info("Incoming /analyze from %s | CT=%s | image_url(%s) | return_data_url=%s | with_ai=%s | metrics=%s",
                client_ip, ct, ("yes" if url_in else "no"), req.return_data_url, req.with_ai, "yes" if req.metrics else "no")

    # 如果既没有 image，也没有 metrics，才报错
    if not url_in and not req.metrics:
        raise HTTPException(400, "either image_url or metrics must be provided")

    workdir = Path(tempfile.mkdtemp(prefix="facemesh_api_"))
    logger.info("Workdir: %s", workdir)

    try:
        input_path: Optional[Path] = None

        # 1) 如有 image_url：下载/解码图片；否则 metrics-only：跳过下载
        if url_in:
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

        # 2) 走三种路径：
        #   a) metrics-only（无 image_url）：生成占位图；直接使用 req.metrics
        #   b) metrics+image：跳过检测，直接使用 req.metrics，出图用原图
        #   c) 仅 image：做人脸检测并叠加标签
        rec: Dict[str, Any]
        if req.metrics and not input_path:
            # (a) metrics-only
            placeholder = _make_placeholder(workdir)
            rec = {
                "input": None,
                "faces": 1,
                "output": str(placeholder),
                "metrics": req.metrics
            }
        elif req.metrics and input_path:
            # (b) metrics+image：不跑检测，直接把原图作为输出
            rec = {
                "input": str(input_path),
                "faces": 1,
                "output": str(input_path),
                "metrics": req.metrics
            }
        else:
            # (c) 仅 image：原有流程（检测+绘制）
            rec = analyze_image(
                img_path=str(input_path),
                out_dir=str(workdir),
                max_width=1600,
                max_height=1600,
                font_path=(FONT_PATH if FONT_PATH.exists() else None),
            )

        # 3) 找输出图
        output_path = Path(rec.get("output") or "")
        if not output_path.exists():
            cands = sorted(workdir.glob("*_12palaces_labeled.jpg"))
            if cands:
                output_path = cands[0]
        if not output_path or not output_path.exists():
            raise HTTPException(500, "output image not found")

        # 4) 移到 public/
        out_name = f"{uuid.uuid4().hex}.jpg"
        public_path = PUBLIC_DIR / out_name
        shutil.move(str(output_path), str(public_path))
        base = str(request.base_url).rstrip("/")
        result_url = f"{base}/files/{out_name}"

        # 5) 组织响应
        payload = {"result_url": result_url, "filename": out_name}
        if req.return_data_url:
            b64_out = base64.b64encode(public_path.read_bytes()).decode("ascii")
            payload["data_url"] = "data:image/jpeg;base64," + b64_out

        # 6) with_ai：必返 job_id
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

        logger.info("Success -> %s | with_ai=%s | path=%s",
                    result_url, req.with_ai,
                    "metrics-only" if (req.metrics and not url_in) else ("metrics+image" if req.metrics else "image-only"))
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
