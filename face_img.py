#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_image_analysis_12palaces_labeled.py
------------------------------------
依赖:
  pip install opencv-python mediapipe numpy pillow

功能:
  - 静态图片人脸关键点(468点)分析：MediaPipe Face Mesh
  - 三庭/五眼计算（不在图上写字）
  - 十二宫：动态几何选区（矩形框），并在框旁标注“是什么宫”
  - 标签：默认用拼音(ASCII)；传 --font 使用中文字体文件(.ttf/.otf)即可中文显示
  - 批处理/递归 & 可选 JSON 汇总

用法:
  # 单图
  python face_image_analysis_12palaces_labeled.py --input input.jpg --out out

  # 目录（递归）+ JSON 汇总
  python face_image_analysis_12palaces_labeled.py --input imgs/ --out out --recursive --save-json out/summary.json

  # 指定中文字体
  python face_image_analysis_12palaces_labeled.py --input input.jpg --out out --font "C:/Windows/Fonts/simhei.ttf"
"""

import os
import cv2
import json
import glob
import math
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any

import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont  # 用于中文标签

def _json_default(o):
    import numpy as _np
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# ---------------------------------- 可调配置 ----------------------------------
CONFIG = dict(
    # 额头带（相对“额头代表点10”与“眉骨均值行”的比例划分）
    forehead_upper_band=0.25,
    forehead_mid_band=0.50,

    # 眼下“子女宫”高度（相对两眼高度平均值）
    under_eye_h=0.30,

    # 夫妻宫：外眼角→颧部的楔形区域宽高比例（相对眼宽/眼高）
    cheek_wedge_w=1.30,
    cheek_wedge_h=0.90,

    # 鼻翼/鼻头框扩展（财帛宫）
    nose_width_margin=1.20,
    nose_height_margin=1.10,

    # 人中（疾厄宫）宽度（相对嘴宽）
    philtrum_width_ratio=0.25,

    # 下颌带（仆役宫）高度（相对 鼻底→下巴 的距离）
    jaw_band_h=0.45,

    # 眉上带（福德/田宅）厚度（相对 眉到额头中线 距离）
    eyebrow_upper_band_ratio=0.45,

    # 可视化颜色 (B,G,R)
    color_palace=(60, 220, 60),
)

# 拼音标签（ASCII 安全）。如果使用 --font 指向中文字体，将直接用中文名称。
PALACE_PINYIN = {
    "父母宫_左": "父母宫_左",
    "父母宫_右": "父母宫_右",
    "官禄宫": "官禄宫",
    "福德宫": "福德宫",
    "田宅宫_左": "田宅宫_左",
    "田宅宫_右": "田宅宫_右",
    "命宫": "命宫",
    "兄弟宫_左": "兄弟宫_左",
    "兄弟宫_右": "兄弟宫_右",
    "子女宫": "子女宫",
    "夫妻宫_左": "夫妻宫_左",
    "夫妻宫_右": "夫妻宫_右",
    "财帛宫": "财帛宫",
    "疾厄宫": "疾厄宫",
    "迁移宫_左": "迁移宫_左",
    "迁移宫_右": "迁移宫_右",
    "仆役宫_左": "仆役宫_左",
    "仆役宫_右": "仆役宫_右",
}

# ---------------------------------- 工具函数 ----------------------------------
def _to_xyzc(lm, w: int, h: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h, lm.z], dtype=np.float32)

def _dist2d(p1, p2) -> float:
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    return float(np.linalg.norm(p1[:2] - p2[:2]))

def _poly_area_xy(points_xy: List[Tuple[float, float]]) -> float:
    if len(points_xy) < 3:
        return 0.0
    x = np.array([p[0] for p in points_xy], dtype=np.float32)
    y = np.array([p[1] for p in points_xy], dtype=np.float32)
    s = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return abs(s) * 0.5

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

def _midpoint(p, q):
    return ( (p[0]+q[0]) * 0.5, (p[1]+q[1]) * 0.5 )

# ---------------------------------- 三庭/五眼 ----------------------------------
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

# ---------------------------------- 宫位：动态选区（矩形） ----------------------------------
BROW_LEFT  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_RIGHT = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
EYE_ANCHORS = dict(L_outer=130, L_inner=133, R_inner=362, R_outer=359)
FACE_BORDERS = dict(L=234, R=454)
NOSE_TIP=1; NOSE_BASE=2; NOSE_WINGS=(49, 279)
MOUTH_CORNERS=(61, 291)
UPPER_LIP=13

def _y_mean(points: List[np.ndarray], idxs: List[int]) -> float:
    return float(np.mean([points[i][1] for i in idxs]))

def _x_mean(points: List[np.ndarray], idxs: List[int]) -> float:
    return float(np.mean([points[i][0] for i in idxs]))

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

# ---------------------------------- 绘制 ----------------------------------
def draw_overlay(img, flandmarks):
    # 仅画网格，不写文字，避免乱码
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

def draw_label(img_bgr, text, x, y, color=(60,220,60), font_path=None, font_size=24):
    if font_path:
        # OpenCV FreeType 画中文
        try:
            ft = cv2.freetype.createFreeType2()
            ft.loadFontData(font_path, 0)  # 第二个参数是 TTC 的 face index
            ft.putText(img_bgr, text, (int(x), int(y)),
                       fontHeight=font_size, color=color, thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=False)
            return
        except Exception as e:
            # 回退到 Pillow（如果你也装了 Pillow）
            try:
                from PIL import Image, ImageDraw, ImageFont
                img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype(font_path, font_size)
                draw.text((int(x), int(y)), text, fill=(color[2], color[1], color[0]), font=font)
                img_bgr[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                return
            except:
                pass
    # 最终兜底（ASCII）：仍然会把中文变成 ???，所以仅当没有字体可用时才走到这里
    cv2.putText(img_bgr, text, (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_palace_boxes_with_labels(img, pal_feats: Dict[str, Any], font_path=None):
    color = CONFIG["color_palace"]
    for name, feat in pal_feats.items():
        if "box" not in feat: 
            continue
        x1,y1,x2,y2 = map(int, feat["box"])
        # 边框
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)
        # 标签（左上角上方 4px）
        label = name if font_path else PALACE_PINYIN.get(name, name)
        draw_label(img, label, x1, max(12, y1 - 4), color=color, font_path=font_path, font_size=12)

# ---------------------------------- IO ----------------------------------
def resize_keep_max(img, max_w: int, max_h: int):
    if (max_w is None and max_h is None) or (max_w == 0 and max_h == 0):
        return img
    h, w = img.shape[:2]
    if max_w is None: max_w = w
    if max_h is None: max_h = h
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh))

def list_images(root: str, recursive: bool) -> List[str]:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    files = []
    if os.path.isdir(root):
        if recursive:
            for ext in exts:
                files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        else:
            for ext in exts:
                files.extend(glob.glob(os.path.join(root, ext)))
    else:
        files = [root]
    files = sorted(list(dict.fromkeys(files)))
    return files

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------------------- 单图处理 ----------------------------------
def process_image(path: str, outdir: str, face_mesh, args) -> Dict[str, Any]:
    img = cv2.imread(path)
    rec = {"input": path}
    base = os.path.splitext(os.path.basename(path))[0]
    out_img = os.path.join(outdir, base + "_12palaces_labeled.jpg")

    if img is None:
        rec["error"]="cannot read image"
        return rec

    h0,w0 = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        # 保存原图缩放版
        img_out = resize_keep_max(img, args.max_width, args.max_height)
        cv2.imwrite(out_img, img_out)
        rec.update({"faces":0, "output":out_img, "error":"no face detected"})
        return rec

    fl = result.multi_face_landmarks[0]
    points = [_to_xyzc(lm, w0, h0) for lm in fl.landmark]

    # 计算指标（不在图上写字）
    three = analyze_three_court(points)
    five  = analyze_five_eye(points)
    pal   = analyze_palaces(points, w0, h0)

    # 绘制：网格 + 宫位框 + 标签
    draw = img.copy()
    draw_overlay(draw, fl)
    draw_palace_boxes_with_labels(draw, pal, font_path=(args.font if args.font else None))

    draw = resize_keep_max(draw, args.max_width, args.max_height)
    cv2.imwrite(out_img, draw)

    rec.update({"faces":1, "output":out_img, "metrics":{"three":three,"five":five,"palaces":pal}})
    return rec

# ---------------------------------- 主入口 ----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 图片文件 或 目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--recursive", action="store_true", help="当 input 为目录时递归处理")
    ap.add_argument("--save-json", default="", help="保存汇总 JSON 到此路径（可选）")
    ap.add_argument("--max-width", type=int, default=1600, help="输出图最大宽(仅展示缩放)")
    ap.add_argument("--max-height", type=int, default=1600, help="输出图最大高(仅展示缩放)")
    ap.add_argument("--font", default="", help="中文字体路径(.ttf/.otf)。为空则用拼音标签避免乱码")
    args = ap.parse_args()

    ensure_dir(args.out)
    files = list_images(args.input, args.recursive)
    if not files:
        print("未找到图片。支持: .jpg/.jpeg/.png/.bmp/.webp")
        return

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )

    results = []
    for idx, p in enumerate(files, 1):
        rec = process_image(p, args.out, face_mesh, args)
        results.append(rec)
        status = "OK" if "error" not in rec else f"ERR:{rec['error']}"
        print(f"[{idx}/{len(files)}] {os.path.basename(p)} -> {status} -> {rec.get('output','-')}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"JSON汇总已保存: {args.save_json}")

if __name__ == "__main__":
    main()
