# ──────────────────────────────────────────────────────────────────────────────
# settings.py — 全局配置与常量（含 LLM 配置）
# ──────────────────────────────────────────────────────────────────────────────
# settings.py — 全局配置与常量（含 LLM 配置）
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 静态文件目录（对外可访问）
PUBLIC_DIR = BASE_DIR / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# 允许的图片扩展名
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 下载与执行限制
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024   # 20MB
SCRIPT_TIMEOUT_SEC = 180                # 调用分析的超时秒

# —— 关键：从本地 llm_config.py 读取（不再依赖环境变量） —— #
try:
    import llm_config
    LLM_ENABLED  = getattr(llm_config, "LLM_ENABLED", True)
    LLM_API_KEY  = getattr(llm_config, "LLM_API_KEY", "")
    LLM_BASE_URL = getattr(llm_config, "LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    LLM_MODEL    = getattr(llm_config, "LLM_MODEL", "deepseek-chat")

    _timeout_raw = getattr(llm_config, "LLM_TIMEOUT", None)
    if _timeout_raw in (None, "none", "None", 0, "0"):
        LLM_TIMEOUT = None          # 无超时
    else:
        LLM_TIMEOUT = int(_timeout_raw)
except Exception:
    LLM_ENABLED  = False
    LLM_API_KEY  = ""
    LLM_BASE_URL = "https://api.deepseek.com"
    LLM_MODEL    = "deepseek-chat"
    LLM_TIMEOUT  = None  # 兜底为无超时

# 可选中文字体（用于在图上显示中文标签）。不存在则走 ASCII 兜底。
FONT_PATH = BASE_DIR / "NotoSansCJKsc-Regular.otf"
