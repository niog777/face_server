# ──────────────────────────────────────────────────────────────────────────────
# face_img.py — 图像几何分析与可视化，仅暴露清晰 API
# ──────────────────────────────────────────────────────────────────────────────
import os
import cv2
import glob
import json
import math
import numpy as np
from typing import List, Tuple, Dict, Any

import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# MediaPipe 句柄
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 公开：供上层调用的配置（必要时也可从上层传入覆盖）
CONFIG = dict(
    forehead_upper_band=0.25,
    forehead_mid_band=0.50,
    under_eye_h=0.30,
    cheek_wedge_w=1.30,
    cheek_wedge_h=0.90,
    nose_width_margin=1.20,
    nose_height_margin=1.10,
    philtrum_width_ratio=0.25,
    jaw_band_h=0.45,
    eyebrow_upper_band_ratio=0.45,
    color_palace=(60, 220, 60),  # BGR
)

# ASCII 安全标签（无中文字体时使用）
PALACE_PINYIN = {
    "父母宫_左": "fumu-zuo",
    "父母宫_右": "fumu-you",
    "官禄宫": "guanlu",
    "福德宫": "fude",
    "田宅宫_左": "tianzhai-zuo",
    "田宅宫_右": "tianzhai-you",
    "命宫": "minggong",
    "兄弟宫_左": "xiongdi-zuo",
    "兄弟宫_右": "xiongdi-you",
    "子女宫": "zinv",
    "夫妻宫_左": "fuqi-zuo",
    "夫妻宫_右": "fuqi-you",
    "财帛宫": "caibo",
    "疾厄宫": "jiee",
    "迁移宫_左": "qianyi-zuo",
    "迁移宫_右": "qianyi-you",
    "仆役宫_左": "puyi-zuo",
    "仆役宫_右": "puyi-you",
}

# -------------------------- 工具函数 --------------------------
def _json_default(o):
    import numpy as _np
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)

def _to_xyzc(lm, w: int, h: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h, lm.z], dtype=np.float32)

def _dist2d(p1, p2) -> float:
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    return float(np.linalg.norm(p1[:2] - p2[:2]))

def _convex_area(points_xy: List[Tuple[float, float]]) -> float:
    if len(points_xy) < 3:
        return 0.0
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    return float(cv2.contourArea(hull))

def _indices_in_box(points: List[np.ndarray], x1: float, y1: float, x2: float, y2: float) -> List[int]:
    x_lo, y_lo = min(x1, x2), min(y1, y2)
    x_hi, y_hi = max(x1, x2), max(y1, y2)
    idxs = []
    for i, p in enumerate(points):
        x, y = float(p[0]), float(p[1])
        if x_lo <= x <= x_hi and y_lo <= y <= y_hi:
            idxs.append(i)
    return idxs

def _region_metrics(points: List[np.ndarray], idxs: List[int]) -> Dict[str, Any]:
    if not idxs:
        return dict(error="insufficient points", count=0, indices=[])
    region = [points[i] for i in idxs]
    xs = [p[0] for p in region]
    ys = [p[1] for p in region]
    zs = [p[2] for p in region]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    area_hull = _convex_area([(p[0], p[1]) for p in region])
    return dict(
        area=float(area_hull),
        width=float(bbox_w),
        height=float(bbox_h),
        fullness_z_mean=float(np.mean(zs)),
        count=len(idxs),
        indices=idxs[:]
    )

# ----------------------- 三庭 / 五眼 -----------------------
def analyze_three_court(points: List[np.ndarray]) -> Dict[str, Any]:
    FOREHEAD_LINE = 10
    BROW_LINE     = 9
    NOSE_BASE     = 2
    CHIN_TIP      = 152
    fy = points[FOREHEAD_LINE][1]
    by = points[BROW_LINE][1]
    ny = points[NOSE_BASE][1]
    cy = points[CHIN_TIP][1]
    upper  = abs(by - fy)
    middle = abs(ny - by)
    lower  = abs(cy - ny)
    total  = upper + middle + lower + 1e-6
    return dict(
        upper_ratio=upper/total, middle_ratio=middle/total, lower_ratio=lower/total,
        values_px=(float(upper), float(middle), float(lower)),
        anchors=dict(forehead=int(10), brow=int(9), nose_base=int(2), chin=int(152))
    )

def analyze_five_eye(points: List[np.ndarray]) -> Dict[str, Any]:
    LEFT_FACE_BORDER  = 234
    LEFT_EYE_OUTER    = 130
    LEFT_EYE_INNER    = 133
    RIGHT_EYE_INNER   = 362
    RIGHT_EYE_OUTER   = 359
    RIGHT_FACE_BORDER = 454
    idx = [LEFT_FACE_BORDER, LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_FACE_BORDER]
    pts = [points[i] for i in idx]
    w1 = _dist2d(pts[0], pts[1]); w2 = _dist2d(pts[1], pts[2]); w3 = _dist2d(pts[2], pts[3]); w4 = _dist2d(pts[3], pts[4]); w5 = _dist2d(pts[4], pts[5])
    total = w1+w2+w3+w4+w5+1e-6
    ratios = [w1/total, w2/total, w3/total, w4/total, w5/total]
    return dict(
        eye_widths_px=(float(w1), float(w2), float(w3), float(w4), float(w5)),
        eye_ratios=[float(r) for r in ratios],
        description="五眼比例: " + ", ".join([f"{r*100:.1f}%" for r in ratios]),
        anchors=dict(L_border=234, L_outer=130, L_inner=133, R_inner=362, R_outer=359, R_border=454)
    )

# ---------------------- 宫位框生成 ----------------------
BROW_LEFT  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_RIGHT = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
EYE_ANCHORS = dict(L_outer=130, L_inner=133, R_inner=362, R_outer=359)
FACE_BORDERS = dict(L=234, R=454)
NOSE_TIP=1; NOSE_BASE=2; NOSE_WINGS=(49, 279)
MOUTH_CORNERS=(61, 291)
UPPER_LIP=13

def _y_mean(points: List[np.ndarray], idxs: List[int]) -> float:
    return float(np.mean([points[i][1] for i in idxs]))

def _palace_boxes(points: List[np.ndarray], w: int, h: int) -> Dict[str, Tuple[float,float,float,float]]:
    cfg = CONFIG
    # 锚点
    p_faceL = points[FACE_BORDERS["L"]]; p_faceR = points[FACE_BORDERS["R"]]
    xL, xR = float(p_faceL[0]), float(p_faceR[0])
    brow_y = (_y_mean(points, BROW_LEFT) + _y_mean(points, BROW_RIGHT)) * 0.5
    forehead_y = float(points[10][1])
    nose_tip = points[NOSE_TIP]; nose_base = points[NOSE_BASE]
    nose_wL, nose_wR = points[NOSE_WINGS[0]], points[NOSE_WINGS[1]]
    mouth_L, mouth_R = points[MOUTH_CORNERS[0]], points[MOUTH_CORNERS[1]]
    upper_lip = points[UPPER_LIP]
    chin = points[152]

    brow2forehead = max(1.0, abs(brow_y - forehead_y))
    nose2chin = max(1.0, abs(chin[1] - nose_base[1]))

    L_outer = points[EYE_ANCHORS["L_outer"]]; L_inner = points[EYE_ANCHORS["L_inner"]]
    R_inner = points[EYE_ANCHORS["R_inner"]]; R_outer = points[EYE_ANCHORS["R_outer"]]
    eye_w = (_dist2d(L_outer, L_inner) + _dist2d(R_inner, R_outer)) * 0.5
    eye_h = (abs(L_outer[1]-L_inner[1]) + abs(R_outer[1]-R_inner[1])) * 0.5
    eye_y_mid = 0.5*(L_inner[1]+R_inner[1])

    # 额头带（父母宫/官禄宫）
    top_band_h = cfg["forehead_upper_band"] * brow2forehead
    top_y1 = min(forehead_y, brow_y)
    forehead_mid_x = 0.5*(xL+xR)
    left_top_box  = (xL, top_y1, forehead_mid_x, top_y1 + top_band_h)
    right_top_box = (forehead_mid_x, top_y1, xR, top_y1 + top_band_h)
    crown_w = 0.30*(xR-xL)
    crown_x1 = forehead_mid_x - 0.5*crown_w
    crown_x2 = forehead_mid_x + 0.5*crown_w
    crown_box = (crown_x1, top_y1, crown_x2, top_y1 + top_band_h)

    # 福德/田宅（眉上带）
    band_h = CONFIG["eyebrow_upper_band_ratio"] * brow2forehead
    fude_w = 0.28*(xR-xL)
    fude_box = (forehead_mid_x - 0.5*fude_w, brow_y - band_h, forehead_mid_x + 0.5*fude_w, brow_y)
    left_tianz_box  = (xL, brow_y - band_h, xL + 0.28*(xR-xL), brow_y)
    right_tianz_box = (xR - 0.28*(xR-xL), brow_y - band_h, xR, brow_y)

    # 命宫（印堂）
    brow_in_L = points[105]; brow_in_R = points[334]
    mid_x = 0.5*(brow_in_L[0]+brow_in_R[0])
    ming_w = abs(brow_in_R[0]-brow_in_L[0]) * 0.6
    ming_h = band_h * 0.9
    ming_box = (mid_x - 0.5*ming_w, brow_y - ming_h, mid_x + 0.5*ming_w, brow_y)

    # 兄弟宫（左右眉框）
    lbx1,lbx2 = min([points[i][0] for i in BROW_LEFT]), max([points[i][0] for i in BROW_LEFT])
    lby = _y_mean(points, BROW_LEFT)
    rbx1,rbx2 = min([points[i][0] for i in BROW_RIGHT]), max([points[i][0] for i in BROW_RIGHT])
    rby = _y_mean(points, BROW_RIGHT)
    brow_h = 1.6*band_h
    bro_left_box  = (lbx1-0.05*(xR-xL), lby - 0.5*brow_h, lbx2+0.05*(xR-xL), lby + 0.5*brow_h)
    bro_right_box = (rbx1-0.05*(xR-xL), rby - 0.5*brow_h, rbx2+0.05*(xR-xL), rby + 0.5*brow_h)

    # 子女宫（眼下带）
    under_h = CONFIG["under_eye_h"] * max(eye_h, 1.0)
    kids_box = (L_inner[0], eye_y_mid, R_inner[0], eye_y_mid + under_h)

    # 夫妻宫（外眼角→颧）
    wedge_w = CONFIG["cheek_wedge_w"] * max(eye_w, 1.0)
    wedge_h = CONFIG["cheek_wedge_h"] * max(eye_h, 1.0)
    couple_left = (L_outer[0]-0.2*wedge_w, L_outer[1]-0.3*wedge_h,
                   L_outer[0]+0.8*wedge_w, L_outer[1]+0.9*wedge_h)
    couple_right = (R_outer[0]-0.8*wedge_w, R_outer[1]-0.3*wedge_h,
                    R_outer[0]+0.2*wedge_w, R_outer[1]+0.9*wedge_h)

    # 财帛宫（鼻翼/鼻头）
    nose_x1 = min(nose_wL[0], nose_wR[0])
    nose_x2 = max(nose_wL[0], nose_wR[0])
    nose_y1 = min(nose_tip[1], nose_base[1])
    nose_y2 = max(nose_tip[1], nose_base[1])
    cx = 0.5*(nose_x1+nose_x2); cy = 0.5*(nose_y1+nose_y2)
    ww = CONFIG["nose_width_margin"]*(nose_x2-nose_x1 + 1e-3)
    hh = CONFIG["nose_height_margin"]*(nose_y2-nose_y1 + 1e-3)
    cai_box = (cx-0.5*ww, cy-0.5*hh, cx+0.5*ww, cy+0.5*hh)

    # 疾厄宫（人中）
    phil_w = CONFIG["philtrum_width_ratio"] * max(abs(mouth_R[0]-mouth_L[0]), 1.0)
    phil_x1 = 0.5*(brow_in_L[0]+brow_in_R[0]) - 0.5*phil_w
    phil_x2 = phil_x1 + phil_w
    phil_y1 = min(nose_base[1], upper_lip[1])
    phil_y2 = max(nose_base[1], upper_lip[1])
    dis_box = (phil_x1, phil_y1, phil_x2, phil_y2)

    # 迁移宫（太阳穴）
    L_brow_tail = points[46]; R_brow_tail = points[276]
    temple_w = 0.18*(xR-xL); temple_h = 0.35*brow2forehead
    temple_left  = (L_brow_tail[0]-temple_w, L_brow_tail[1]-temple_h, L_brow_tail[0], L_brow_tail[1])
    temple_right = (R_brow_tail[0], R_brow_tail[1]-temple_h, R_brow_tail[0]+temple_w, R_brow_tail[1])

    # 仆役宫（下颌两侧）
    jaw_y1 = chin[1] - CONFIG["jaw_band_h"] * nose2chin
    jaw_y2 = chin[1]
    mid_x = 0.5*(xL+xR)
    jaw_left  = (xL, jaw_y1, (xL+mid_x)*0.5, jaw_y2)
    jaw_right = ((mid_x+xR)*0.5, jaw_y1, xR, jaw_y2)

    return dict(
        父母宫_左=left_top_box, 父母宫_右=right_top_box, 官禄宫=crown_box,
        福德宫=fude_box, 田宅宫_左=left_tianz_box, 田宅宫_右=right_tianz_box,
        命宫=ming_box, 兄弟宫_左=bro_left_box, 兄弟宫_右=bro_right_box,
        子女宫=kids_box, 夫妻宫_左=couple_left, 夫妻宫_右=couple_right,
        财帛宫=cai_box, 疾厄宫=dis_box, 迁移宫_左=temple_left, 迁移宫_右=temple_right,
        仆役宫_左=jaw_left, 仆役宫_右=jaw_right,
    )


def analyze_palaces(points: List[np.ndarray], w: int, h: int) -> Dict[str, Any]:
    boxes = _palace_boxes(points, w, h)
    feats = {}
    for name, (x1,y1,x2,y2) in boxes.items():
        idxs = _indices_in_box(points, x1,y1,x2,y2)
        feats[name] = _region_metrics(points, idxs)
        feats[name]["box"] = [float(x1), float(y1), float(x2), float(y2)]
    return feats

# ----------------------- 绘制 -----------------------
def draw_overlay(img, flandmarks):
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=flandmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=flandmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )


def _draw_text(img_bgr, text, x, y, color=(60,220,60), font_path=None, font_size=24):
    # 优先 FreeType（需要 opencv-contrib-python）
    if font_path:
        try:
            if hasattr(cv2, "freetype"):
                ft = cv2.freetype.createFreeType2()
                ft.loadFontData(font_path, 0)
                ft.putText(img_bgr, text, (int(x), int(y)),
                           fontHeight=font_size, color=color, thickness=1, line_type=cv2.LINE_AA, bottomLeftOrigin=False)
                return
        except Exception:
            pass
        # Pillow 回退
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(str(font_path), font_size)
            draw.text((int(x), int(y)), text, fill=(color[2], color[1], color[0]), font=font)
            img_bgr[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
    # ASCII 兜底
    cv2.putText(img_bgr, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_palace_boxes_with_labels(img, pal_feats: Dict[str, Any], font_path=None):
    color = CONFIG["color_palace"]
    for name, feat in pal_feats.items():
        if "box" not in feat:
            continue
        x1,y1,x2,y2 = map(int, feat["box"])
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)
        label = name if font_path else PALACE_PINYIN.get(name, name)
        _draw_text(img, label, x1, max(12, y1 - 4), color=color, font_path=font_path, font_size=12)


def resize_keep_max(img, max_w: int | None, max_h: int | None):
    if (not max_w and not max_h) or (max_w == 0 and max_h == 0):
        return img
    h, w = img.shape[:2]
    if max_w is None: max_w = w
    if max_h is None: max_h = h
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh))


# ----------------------- 对外主函数 -----------------------
def analyze_image(
    img_path: os.PathLike | str,
    out_dir: os.PathLike | str,
    max_width: int = 1600,
    max_height: int = 1600,
    font_path: os.PathLike | None = None,
) -> Dict[str, Any]:
    """对单张图片进行分析与渲染并输出到 out_dir。
    返回 dict: {"output": 输出图片路径, "metrics": {...}}
    若未检测到人脸，也会输出缩放原图，并在返回中带 error。
    """
    img = cv2.imread(str(img_path))
    rec: Dict[str, Any] = {"input": str(img_path)}
    if img is None:
        return {**rec, "error": "cannot read image"}

    h0, w0 = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # FaceMesh（静态图）
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
        result = fm.process(rgb)

    base = os.path.splitext(os.path.basename(str(img_path)))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(str(out_dir), base + "_12palaces_labeled.jpg")

    if not result.multi_face_landmarks:
        img_out = resize_keep_max(img, max_width, max_height)
        cv2.imwrite(out_img, img_out)
        return {**rec, "faces": 0, "output": out_img, "error": "no face detected"}

    fl = result.multi_face_landmarks[0]
    points = [_to_xyzc(lm, w0, h0) for lm in fl.landmark]

    three = analyze_three_court(points)
    five  = analyze_five_eye(points)
    pal   = analyze_palaces(points, w0, h0)

    draw = img.copy()
    draw_overlay(draw, fl)
    draw_palace_boxes_with_labels(draw, pal, font_path=str(font_path) if font_path else None)

    draw = resize_keep_max(draw, max_width, max_height)
    cv2.imwrite(out_img, draw)

    return {
        **rec,
        "faces": 1,
        "output": out_img,
        "metrics": {"three": three, "five": five, "palaces": pal},
    }
