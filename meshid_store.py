# meshid_store.py
# 仅基于 MediaPipe Face Mesh 的 468 点做“形状向量”匹配：
# - 入库（enroll_from_points）
# - 检索（match_from_points）
# 设计要点：
#   * 以左右外眼角做相似变换（平移+旋转+尺度归一化）
#   * 选稳定子集点 + 少量比例特征 → 原始几何向量
#   * 样本充足时训练 StandardScaler+PCA → 嵌入向量（L2）
#   * 样本不足/无模型时自动回退：原始向量 L2 + 余弦相似度
#   * SQLite 做轻量持久化（persons / templates）

import os
import time
import json
import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# ================== 可调参数 ==================
EMB_DIM = 128                 # PCA 目标维度上限
DB_PATH = "meshid.sqlite"     # SQLite 路径
MODEL_DIR = "meshid_models"   # 保存 scaler/pca 的目录
MIN_PCA_SAMPLES = 5           # 少于该模板数时不训练PCA，自动回退
os.makedirs(MODEL_DIR, exist_ok=True)

# 稳定子集索引（根据经验挑选：眼眶/鼻梁/嘴/下巴/脸缘等较稳定部位）
STABLE_IDX = sorted(set([
    33, 133, 362, 263,      # 左右眼外/内角
    1, 2,                   # 鼻梁/鼻底（请确认索引与你的mesh一致）
    61, 291, 13, 14,        # 嘴角/上唇中点/下唇中点
    199, 175, 152, 234, 454 # 脸颊/下巴尖/左右脸缘
]))

# 关键锚点（用于对齐/比例特征）
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_BASE = 2

# ================== DB 基础 ==================
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def db_init():
    """初始化数据库（仅需一次）"""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            template_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            vec BLOB NOT NULL,        -- 原始几何特征（未PCA）
            created_at INTEGER,
            quality REAL,
            meta_json TEXT,
            FOREIGN KEY(person_id) REFERENCES persons(person_id)
        );
    """)
    conn.commit()
    conn.close()

def save_model(scaler: StandardScaler, pca: PCA):
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca.joblib"))

def load_model() -> Tuple[Optional[StandardScaler], Optional[PCA]]:
    sp = os.path.join(MODEL_DIR, "scaler.joblib")
    pp = os.path.join(MODEL_DIR, "pca.joblib")
    if not (os.path.exists(sp) and os.path.exists(pp)):
        return None, None
    return joblib.load(sp), joblib.load(pp)

def fit_or_update_pca():
    """从库中全部模板拟合/更新 Scaler+PCA。
       若样本不足，删除模型文件并回退到原始向量余弦。"""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT vec FROM templates;")
    rows = cur.fetchall()
    conn.close()

    n = len(rows) if rows else 0
    if n < MIN_PCA_SAMPLES:
        # 样本太少：删除已有模型，表示“无模型”，匹配时自动回退
        for fn in ("scaler.joblib", "pca.joblib"):
            p = os.path.join(MODEL_DIR, fn)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
        print(f"[INFO] templates={n} < {MIN_PCA_SAMPLES}, skip PCA (fallback to raw).")
        return

    X = [np.frombuffer(b, dtype=np.float32) for (b,) in rows]
    X = np.stack(X, axis=0)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    # 组件数不能超过样本数/特征维
    ncomp = min(EMB_DIM, Xs.shape[1], Xs.shape[0])
    pca = PCA(n_components=ncomp).fit(Xs)
    save_model(scaler, pca)
    print(f"[INFO] PCA updated: raw_dim={Xs.shape[1]} -> emb_dim={ncomp}, samples={Xs.shape[0]}")

# ================== 形状向量化 ==================
def _similarity_align(pts_xy: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    以左右外眼角做相似变换：
      * 平移到眼中点为原点
      * 旋转使两眼连线水平
      * 缩放使两眼外角距离 = 1
    输入：pts_xy [468,2] 像素坐标
    输出：对齐后的 pts_norm [468,2]、是否成功
    """
    pts = pts_xy.copy().astype(np.float32)
    lx, ly = pts[LEFT_EYE_OUTER]
    rx, ry = pts[RIGHT_EYE_OUTER]
    v = np.array([rx - lx, ry - ly], dtype=np.float32)
    d = np.linalg.norm(v)
    if d < 1e-4:
        return pts, False

    # 平移：眼中点到原点
    mid = np.array([(lx + rx) / 2.0, (ly + ry) / 2.0], dtype=np.float32)
    pts -= mid

    # 旋转：使眼线水平
    theta = -np.arctan2(v[1], v[0])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ R.T

    # 缩放：两眼外角距离 = 1
    pts *= (1.0 / d)
    return pts, True

def _build_geom_feats(pts_norm: np.ndarray) -> np.ndarray:
    """
    构造原始几何特征向量：
      * 稳定子集坐标（展平）
      * 少量比例特征（示例：嘴宽/脸宽、鼻长/脸高）
    """
    sub = pts_norm[STABLE_IDX]          # [S,2]
    vecs = [sub.flatten()]

    face_w = np.max(pts_norm[:, 0]) - np.min(pts_norm[:, 0]) + 1e-6
    face_h = np.max(pts_norm[:, 1]) - np.min(pts_norm[:, 1]) + 1e-6
    mouth_w = np.linalg.norm(pts_norm[291] - pts_norm[61])
    nose_len = np.linalg.norm(pts_norm[NOSE_BASE] - pts_norm[1])

    vecs.append(np.array([mouth_w / face_w, nose_len / face_h], dtype=np.float32))
    return np.concatenate(vecs, axis=0).astype(np.float32)

def vectorize_from_landmarks(points_xyzc: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    输入：你的 face_img.py 中 _to_xyzc 形成的 468×3 列表（像素坐标 x,y,z）
    输出：原始几何特征向量（未PCA）；失败时返回 None
    """
    if points_xyzc is None or len(points_xyzc) < 468:
        return None
    pts_xy = np.array([[p[0], p[1]] for p in points_xyzc], dtype=np.float32)  # 仅用 x,y
    pts_norm, ok = _similarity_align(pts_xy)
    if not ok:
        return None
    return _build_geom_feats(pts_norm)

# ================== 嵌入与相似度 ==================
def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def embed(vec_raw: np.ndarray, scaler: Optional[StandardScaler], pca: Optional[PCA]) -> np.ndarray:
    """
    有模型（scaler+pca）→ 标准化 + PCA + L2
    无模型 → 原始向量 L2（回退）
    """
    if scaler is None or pca is None:
        return _l2(vec_raw.astype(np.float32))
    Xs = scaler.transform(vec_raw[None, :])
    z = pca.transform(Xs)[0]
    return _l2(z.astype(np.float32))

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-12)*(np.linalg.norm(b)+1e-12)))

# ================== 录入 / 匹配 ==================
def enroll_from_points(points_xyzc: List[np.ndarray],
                       display_name: Optional[str] = None,
                       person_id: Optional[int] = None,
                       quality: float = 1.0,
                       meta: Optional[dict] = None) -> Optional[int]:
    """
    将一张 Face Mesh 关键点样本录入数据库。
    - 若 person_id 为空 → 新建身份（可传 display_name）
    - 若提供 person_id → 追加模板
    成功返回 person_id，失败返回 None
    """
    vec_raw = vectorize_from_landmarks(points_xyzc)
    if vec_raw is None:
        return None

    conn = db_connect()
    cur = conn.cursor()

    if person_id is None:
        cur.execute("INSERT INTO persons(display_name) VALUES(?)", (display_name,))
        person_id = cur.lastrowid

    cur.execute(
        "INSERT INTO templates(person_id, vec, created_at, quality, meta_json) VALUES (?,?,?,?,?)",
        (person_id, vec_raw.tobytes(), int(time.time()), float(quality), json.dumps(meta or {}))
    )
    conn.commit()
    conn.close()

    # 每次新增后尝试更新 PCA（模板太少会自动跳过并回退）
    fit_or_update_pca()
    return person_id

def match_from_points(points_xyzc: List[np.ndarray],
                      sim_thresh: float = 0.90,
                      margin: float = 0.04,
                      topk: int = 5):
    """
    与库中所有模板进行匹配，返回 Top 结果与决策：
    - 有模型 → PCA 嵌入；无模型（样本少/未训练）→ 原始向量 L2（回退）
    - 决策规则：Top-1 相似度 >= sim_thresh 且 (Top1-Top2) >= margin → "match"
                否则 → "uncertain"
    """
    vec_raw = vectorize_from_landmarks(points_xyzc)
    if vec_raw is None:
        return {"status": "bad_sample"}

    scaler, pca = load_model()
    q = embed(vec_raw, scaler, pca)  # 有模型用嵌入；无模型自动回退

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT t.template_id, t.person_id, t.vec, t.quality, p.display_name
        FROM templates t LEFT JOIN persons p ON t.person_id = p.person_id
    """)
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return {"status": "empty_db"}

    scores = []
    for tid, pid, bvec, qlt, name in rows:
        v_raw = np.frombuffer(bvec, dtype=np.float32)
        v = embed(v_raw, scaler, pca)
        s = float(np.dot(q, v))  # 两者都已 L2 → 点积即余弦相似度
        scores.append((s, pid, name, tid, float(qlt)))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:max(1, topk)]

    s1 = top[0][0]
    s2 = top[1][0] if len(top) > 1 else -1.0
    decision = "match" if (s1 >= sim_thresh and (s1 - s2) >= margin) else "uncertain"

    return {
        "status": decision,
        "best": {
            "person_id": top[0][1],
            "display_name": top[0][2],
            "score": s1,
            "template_id": top[0][3],
            "quality": top[0][4]
        },
        "topk": [
            {"score": s, "person_id": pid, "display_name": nm, "template_id": tid, "quality": ql}
            for (s, pid, nm, tid, ql) in top
        ]
    }

# （可选）命令行调试入口：仅用于快速自检
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="MeshID store quick test")
    ap.add_argument("--initdb", action="store_true")
    ap.add_argument("--dump_stats", action="store_true")
    args = ap.parse_args()

    if args.initdb:
        db_init()
        print(f"DB initialized at {DB_PATH}")

    if args.dump_stats:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM templates;")
        c, tmin, tmax = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM persons;")
        p, = cur.fetchone()
        conn.close()
        print(json.dumps({
            "persons": int(p or 0),
            "templates": int(c or 0),
            "tmin": int(tmin or 0),
            "tmax": int(tmax or 0)
        }, ensure_ascii=False, indent=2))
