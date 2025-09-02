#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖：
  pip install fastapi uvicorn pydantic requests pillow opencv-python mediapipe
环境变量：
  LLM_API_KEY   大模型 API Key（可选，只有 with_ai=true 时需要）
  LLM_BASE_URL  OpenAI 兼容接口基础地址（可选，默认 https://api.openai.com/v1）
  LLM_MODEL     模型名（可选，默认 gpt-4o-mini 或你的网关模型名）
"""

import os
import re
import sys
import uuid
import json
import base64
import binascii
import shutil
import tempfile
import subprocess
import logging
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---------------- 日志设置 ----------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger("facemesh_api")

# ---------------- 基础配置 ----------------
app = FastAPI(title="FaceMesh 12 Palaces API", version="1.4.0")

BASE_DIR = Path(__file__).resolve().parent
SCRIPT = BASE_DIR / "face_img.py"                     # 你的分析脚本（已支持 --save-json）
FONT = BASE_DIR / "NotoSansCJKsc-Regular.otf"         # 可选字体
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024                 # 20MB
SCRIPT_TIMEOUT_SEC = 180                              # 脚本执行超时

# LLM 配置（OpenAI 兼容）
LLM_API_KEY  = os.getenv("LLM_API_KEY", "sk-806253c7049146b29e13a4ed0e52bb75").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
LLM_MODEL    = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT  = int(os.getenv("LLM_TIMEOUT", "60"))

# 静态文件目录（对外可访问）
PUBLIC_DIR = BASE_DIR / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(PUBLIC_DIR)), name="files")

# ---------------- 请求体模型 ----------------
class AnalyzeReq(BaseModel):
    image_url: str               # 支持 http/https 或 data URL
    return_data_url: bool = False
    with_ai: bool = False        # true 时，调用大模型生成 face_analysis

# ---------------- 工具：宽松容错的 Base64 解码 ----------------
def decode_base64_forgiving(b64: str) -> bytes:
    """
    兼容多种客户端发来的 base64：
    - 允许 URL-safe 字符 '-' '_'
    - 允许被 URL 编码（%2B/%2F/%0A 等）
    - 忽略所有空白字符
    - 自动补 '=' 到 4 的倍数
    """
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

# ---------------- 全局异常处理（打印 422 / 其它 HTTP 错误的详情） ----------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        raw = await request.body()
        preview = raw[:200].decode("utf-8", "ignore")
        size = len(raw)
    except Exception:
        preview, size = "", None
    logger.error(
        "422 ValidationError: errors=%s | content-type=%s | body_size=%s | body_preview=%r",
        exc.errors(), request.headers.get("content-type"), size, preview
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body_preview": preview,
            "body_size": size,
            "content_type": request.headers.get("content-type"),
            "hint": "检查字段名/类型与 JSON 语法；image_url 需为 http(s) 或 data:image/...;base64,..."
        },
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error("HTTPException %s on %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code, "path": request.url.path},
    )

# ---------------- 基础路由 ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return JSONResponse({
        "msg": "POST /analyze with JSON: {\"image_url\": \"https://...\" 或 \"data:image/...;base64,...\", "
               "\"return_data_url\": false, \"with_ai\": false}"
    })

# ---------------- 构造给 LLM 的提示词 ----------------
def build_llm_messages(metrics: dict, result_url: str) -> dict:
    """
    metrics：face_img.py --save-json 产生的单张图片的 rec["metrics"]
    result_url：处理后图片的外链（可给模型参考）
    """
    system = (
        "你是面部几何与传统“面相十二宫”对照的分析助手。"
        "用户会提供基于 MediaPipe Face Mesh 的结构化几何特征（区域面积/宽高/Z 均值、三庭比例、五眼比例等），"
        "请仅依据这些**几何数据**做温和、中性的中文解读；不要做医学诊断或绝对化断言；避免迷信化、避免确定性承诺；"
        "面向普通用户，语言简洁友好。"
        "输出**严格 JSON**，结构："
        "{"
        "  \"face_analysis\": {"
        "    \"twelve_palaces\": ["
        "      {\"palace_name\": \"1. 命宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"2. 兄弟宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"3. 夫妻宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"4. 子女宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"5. 财帛宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"6. 疾厄宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"7. 迁移宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"8. 奴仆宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"9. 官禄宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"10. 田宅宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"11. 福德宫\", \"description\": \"...\"},"
        "      {\"palace_name\": \"12. 父母宫\", \"description\": \"...\"}"
        "    ],"
        "    \"three_court_five_eye\": {"
        "      \"three_court\": {\"description\": \"...\"},"
        "      \"five_eye\": {\"description\": \"...\"}"
        "    },"
        "    \"ai_destiny\": \"结合以上几何指标的综合解读（300~600字，避免绝对化）\""
        "  }"
        "}"
        "注意：把左右宫位（如 兄弟宫_左/右、田宅宫_左/右、父母宫_左/右、迁移宫_左/右、仆役宫_左/右、夫妻宫_左/右）合并成一个描述；"
        "不要重复；描述里可引用“饱满/均衡/对称/宽度/高度/面积/Z 值（饱满度）”等术语。"
    )
    user = {
        "tip": "下面是结构化几何指标与可视化图片链接，仅用于参考。",
        "result_image_url": result_url,
        "metrics": metrics
    }
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ]
    }

# ---------------- 调大模型（OpenAI 兼容 Chat Completions） ----------------
def call_llm_for_face_analysis(metrics: dict, result_url: str) -> dict:
    if not LLM_API_KEY:
        raise HTTPException(500, "LLM_API_KEY not set; cannot run with_ai=true")
    url = f"{LLM_BASE_URL}/chat/completions"
    payload = build_llm_messages(metrics, result_url)
    data = {
        "model": LLM_MODEL,
        "messages": payload["messages"],
        # 如果你的网关不支持 response_format，可以删掉这一行
        "response_format": {"type": "json_object"},
        "temperature": 0.6,
        "max_tokens": 1200
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        j = resp.json()
        content = j["choices"][0]["message"]["content"]
        # 解析成 JSON
        try:
            parsed = json.loads(content)
        except Exception:
            # 有些模型可能返回多余文本，尝试提取最外层 JSON
            m = re.search(r"(\{.*\})", content, re.S)
            if not m:
                raise HTTPException(502, f"LLM returned non-JSON content: {content[:200]}...")
            parsed = json.loads(m.group(1))
        if "face_analysis" not in parsed:
            raise HTTPException(502, "LLM JSON missing 'face_analysis'")
        return parsed["face_analysis"]
    except requests.HTTPError as e:
        raise HTTPException(502, f"LLM HTTP error: {e} | body={getattr(e.response, 'text', '')[:200]}")
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")

# ---------------- 核心：兼容 http/https 与 data URL，并（可选）调用大模型 ----------------
@app.post("/analyze")
def analyze(req: AnalyzeReq, request: Request):
    """
    收到图片输入（http/https 或 data URL） -> 保存到临时目录 -> 调 face_img.py 处理并保存 JSON
    -> 将结果图移动到 public/ 生成可访问 URL -> （可选）调用大模型生成 face_analysis -> 返回 JSON
    """
    client_ip = getattr(request.client, "host", "?")
    ct = request.headers.get("content-type")
    url_in = (req.image_url or "").strip()
    url_preview = (url_in[:80] + "...") if len(url_in) > 80 else url_in
    logger.info(
        "Incoming /analyze from %s | Content-Type=%s | image_url(len=%d, preview=%r) | return_data_url=%s | with_ai=%s",
        client_ip, ct, len(url_in), url_preview, req.return_data_url, req.with_ai
    )

    if not url_in:
        raise HTTPException(400, "image_url is required")

    # 1) 临时工作目录
    workdir = Path(tempfile.mkdtemp(prefix="facemesh_api_"))
    logger.info("Workdir: %s", workdir)

    try:
        # 2) 解析输入，得到本地 input_path
        input_path: Path
        if url_in.startswith("data:"):
            if "," not in url_in:
                raise HTTPException(400, "invalid data URL")
            header, b64 = url_in.split(",", 1)
            mime = "image/jpeg"
            if ":" in header:
                h1 = header.split(":", 1)[1]
                mime = (h1.split(";")[0] or "image/jpeg").lower()
            ext_map = {
                "image/jpeg": ".jpg", "image/jpg": ".jpg",
                "image/png": ".png", "image/webp": ".webp", "image/bmp": ".bmp",
            }
            ext = ext_map.get(mime, ".jpg")
            raw = decode_base64_forgiving(b64)
            if len(raw) > MAX_DOWNLOAD_BYTES:
                raise HTTPException(413, "image too large (>20MB)")
            input_path = workdir / f"{uuid.uuid4().hex}{ext}"
            with open(input_path, "wb") as f: f.write(raw)
            logger.info("Saved decoded data URL to %s (%d bytes)", input_path, len(raw))
        else:
            parsed = urlparse(url_in)
            scheme = parsed.scheme.lower()
            if scheme not in ("http", "https"):
                raise HTTPException(400, "image_url must be http(s) or data URL")
            url_path = parsed.path
            ext = (os.path.splitext(url_path)[1] or ".jpg").lower()
            if ext not in ALLOWED_EXT:
                ext = ".jpg"
            input_path = workdir / f"{uuid.uuid4().hex}{ext}"
            try:
                with requests.get(url_in, stream=True, timeout=20) as r:
                    r.raise_for_status()
                    downloaded = 0
                    with open(input_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if not chunk: continue
                            downloaded += len(chunk)
                            if downloaded > MAX_DOWNLOAD_BYTES:
                                raise HTTPException(413, "image too large (>20MB)")
                            f.write(chunk)
                logger.info("Downloaded URL to %s (%d bytes)", input_path, input_path.stat().st_size)
            except Exception as e:
                raise HTTPException(502, f"failed to download image: {e}")

        # 3) 调 face_img.py（附带 --save-json）
        if not SCRIPT.exists():
            raise HTTPException(500, f"script not found: {SCRIPT}")
        summary_json = workdir / "summary.json"
        cmd = [sys.executable, str(SCRIPT),
               "--input", str(input_path),
               "--out", str(workdir),
               "--save-json", str(summary_json)]
        if FONT.exists():
            cmd += ["--font", str(FONT)]
        logger.info("Running command: %r (cwd=%s)", cmd, BASE_DIR)
        try:
            proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, f"face_img.py timeout (> {SCRIPT_TIMEOUT_SEC}s)")

        stdout_tail = (proc.stdout or "")[-400:]
        stderr_tail = (proc.stderr or "")[-400:]
        logger.info("face_img.py return=%s | stdout_tail=%r | stderr_tail=%r",
                    proc.returncode, stdout_tail, stderr_tail)
        if proc.returncode != 0:
            raise HTTPException(500, f"face_img.py failed (code {proc.returncode}). stderr_tail: {stderr_tail}")

        # 4) 找输出图
        expected = workdir / f"{input_path.stem}_12palaces_labeled.jpg"
        output_path = expected if expected.exists() else None
        if not output_path:
            cands = sorted(workdir.glob("*_12palaces_labeled.jpg"))
            if cands: output_path = cands[0]
        if not output_path or not output_path.exists():
            raise HTTPException(500, "output image not found")

        # 5) 移动到 public/ 并生成 URL
        out_name = f"{uuid.uuid4().hex}.jpg"
        public_path = PUBLIC_DIR / out_name
        shutil.move(str(output_path), str(public_path))
        base = str(request.base_url).rstrip("/")
        result_url = f"{base}/files/{out_name}"

        # 6) 读取 metrics（供返回或 LLM 使用）
        metrics = None
        if summary_json.exists():
            try:
                data = json.loads(summary_json.read_text(encoding="utf-8"))
                # face_img.py 返回的是列表；按 input 匹配（或取第一个）
                rec = None
                for r in data:
                    if str(r.get("input")) == str(input_path):
                        rec = r; break
                if not rec and data:
                    rec = data[0]
                metrics = rec.get("metrics") if rec else None
            except Exception as e:
                logger.warning("Failed to read metrics JSON: %s", e)

        # 7) 组织返回（可选：data_url）
        payload = {"result_url": result_url, "filename": out_name}
        if req.return_data_url:
            with open(public_path, "rb") as f:
                b64_out = base64.b64encode(f.read()).decode("ascii")
            payload["data_url"] = "data:image/jpeg;base64," + b64_out

        # 8) 可选：调用 LLM 生成解读
        if req.with_ai:
            if not metrics:
                raise HTTPException(500, "metrics not found; cannot run with_ai=true")
            try:
                face_analysis = call_llm_for_face_analysis(metrics, result_url)
                payload["face_analysis"] = face_analysis
            except HTTPException as e:
                # 不让整个流程失败，返回部分成功 + 错误原因
                payload["face_analysis_error"] = {"code": e.status_code, "detail": e.detail}

        logger.info("Success -> %s | with_ai=%s", result_url, req.with_ai)
        return JSONResponse(
            payload,
            background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True),
        )

    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True); raise
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        shutil.rmtree(workdir, ignore_errors=True); raise HTTPException(500, f"unexpected error: {e}")

# 允许 `python api.py` 直接启动（也可以继续用 `uvicorn api:app`）
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
