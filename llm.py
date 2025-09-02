# llm.py — 负责把几何指标交给 LLM，产出“固定结构”的面相 JSON（含兼容转换）
import re
import json
import requests
from fastapi import HTTPException
from typing import Dict, Any, List

from settings import LLM_ENABLED, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT


# --- 工具：把 numpy 类型递归转成原生 Python（防止 json.dumps 报错） ---
def _to_builtin(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


# --- 你前端期望的 12 宫顺序与名称（注意第 8 是“奴仆宫”而不是“仆役宫”） ---
PALACE_ORDER = [
    "命宫",
    "兄弟宫",
    "夫妻宫",
    "子女宫",
    "财帛宫",
    "疾厄宫",
    "迁移宫",
    "奴仆宫",   # ← 前端通常用这个名
    "官禄宫",
    "田宅宫",
    "福德宫",
    "父母宫",
]

# 可能出现在模型/指标里的同义写法，统一到前端预期名
ALIASES = {
    "仆役宫": "奴仆宫",
    "父母宫_左": "父母宫",
    "父母宫_右": "父母宫",
    "田宅宫_左": "田宅宫",
    "田宅宫_右": "田宅宫",
    "兄弟宫_左": "兄弟宫",
    "兄弟宫_右": "兄弟宫",
    "夫妻宫_左": "夫妻宫",
    "夫妻宫_右": "夫妻宫",
    "迁移宫_左": "迁移宫",
    "迁移宫_右": "迁移宫",
    "仆役宫_左": "奴仆宫",
    "仆役宫_右": "奴仆宫",
}


def _merge_left_right(texts: Dict[str, str]) -> Dict[str, str]:
    """把带左右的宫位合并为单一宫位，合并描述（去重拼接）。"""
    merged: Dict[str, List[str]] = {}
    for k, v in texts.items():
        name = ALIASES.get(k, k)
        merged.setdefault(name, [])
        if v and v not in merged[name]:
            merged[name].append(v)
    # 合并成一句或两句
    return {k: " ".join(vs) for k, vs in merged.items()}


def _to_expected_schema_from_flat(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    把“平铺字典”形式：
      {"父母宫":"...", "官禄宫":"...", ...}
    转成前端期望的 schema：
      { "face_analysis": { "twelve_palaces":[...], "three_court_five_eye":{...}, "ai_destiny":"..." } }
    """
    # 只抽取字符串描述的键值
    texts = {ALIASES.get(k, k): str(v) for k, v in flat.items() if isinstance(v, (str, int, float))}
    texts = _merge_left_right(texts)

    twelve = []
    for idx, name in enumerate(PALACE_ORDER, start=1):
        desc = texts.get(name, "数据不足，无法分析当前状态。")
        twelve.append({"palace_name": f"{idx}. {name}", "description": desc})

    # 三庭/五眼若不存在，给个兜底
    three_five = {
        "three_court": {"description": "三庭比例未提供或不足，保持日常作息与姿态更利于面部状态衡量。"},
        "five_eye": {"description": "五眼比例未提供或不足，建议在光线均匀、正面照片下再做测量参考。"}
    }

    return {
        "face_analysis": {
            "twelve_palaces": twelve,
            "three_court_five_eye": three_five,
            "ai_destiny": "综合来看，面部各部位信号呈现一定的个体差异。请以当下生活场景为重心，在沟通、作息、情绪管理与环境布置上做细小而持续的优化，避免过度解读。"
        }
    }


def build_llm_messages(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    恢复为你原先的严格输出规范：
    face_analysis:
      - twelve_palaces: [ {palace_name, description} x12 ]
      - three_court_five_eye: { three_court:{description}, five_eye:{description} }
      - ai_destiny: 300~600字
    """
    system = (
        "你是面部几何与传统“面相十二宫”对照的中文解读助手。"
        "用户会提供基于 Face Mesh 的几何特征。"
        "你的任务是：结合这些几何信号，重点分析每个部位当前的状态代表什么含义，"
        "给出温和、中性的解释和日常化建议。"
        "不要做医学诊断或绝对化断言，也不要使用迷信化的措辞。"
        "语言要简洁友好，更偏向生活场景的解读，而不是比例或数值。"
        "输出严格 JSON："
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
        "注意：把左右宫位合并描述；可使用“饱满/均衡/对称/宽度/高度/面积/Z值（饱满度）”等术语。"
    )

    # 关键：把 metrics 转成原生类型
    safe_metrics = _to_builtin(metrics)

    user = {
        "tip": "下面是结构化几何指标，仅供分析。",
        "metrics": safe_metrics
    }
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ]
    }


def call_llm_for_face_analysis(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not LLM_ENABLED:
        raise HTTPException(500, "LLM disabled by config (llm_config.py)")
    if not LLM_API_KEY:
        raise HTTPException(500, "LLM_API_KEY not set in llm_config.py")

    url = f"{LLM_BASE_URL}/chat/completions"
    payload = build_llm_messages(metrics)
    data = {
        "model": LLM_MODEL,
        "messages": payload["messages"],
        "response_format": {"type": "json_object"},
        "temperature": 0.6,
        "max_tokens": 1400,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

    try:
        kwargs = {"timeout": LLM_TIMEOUT} if LLM_TIMEOUT is not None else {}
        resp = requests.post(url, headers=headers, json=data, **kwargs)
        resp.raise_for_status()
        j = resp.json()
        content = j["choices"][0]["message"]["content"]

        # 先尝试直接解析为 JSON 对象
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r"(\{.*\})", content, re.S)
            if not m:
                raise HTTPException(502, f"LLM returned non-JSON content: {content[:200]}...")
            parsed = json.loads(m.group(1))

        # 期望结构：face_analysis.twelve_palaces([...]) / three_court_five_eye / ai_destiny
        if "face_analysis" not in parsed:
            # 兼容：“平铺字典”→ 转换成期望结构
            converted = _to_expected_schema_from_flat(parsed)
            return converted["face_analysis"]

        fa = parsed["face_analysis"]

        # 如果是标准结构，直接返回；如果缺字段，补全
        if "twelve_palaces" not in fa or not isinstance(fa["twelve_palaces"], list):
            # 可能收到平铺 → 尝试从根级或 face_analysis 里抽字典并转
            base = fa if isinstance(fa, dict) else {}
            converted = _to_expected_schema_from_flat(base)
            return converted["face_analysis"]

        # 补全缺失的两块
        fa.setdefault("three_court_five_eye", {
            "three_court": {"description": "三庭比例未提供或不足。"},
            "five_eye": {"description": "五眼比例未提供或不足。"}
        })
        fa.setdefault("ai_destiny", "综合分析未完整提供。请基于当前生活场景做温和调整。")

        return fa

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:200]
        raise HTTPException(502, f"LLM HTTP error: {e} | body={body}")
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")
