# id_ops.py
import argparse
import json
from face_img import extract_landmarks_xyzc
from meshid_store import db_init, enroll_from_points, match_from_points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initdb", action="store_true")
    ap.add_argument("--enroll", type=str, help="要录入的图片路径")
    ap.add_argument("--display_name", type=str, default=None)
    ap.add_argument("--person_id", type=int, default=None)
    ap.add_argument("--match", type=str, help="要查询匹配的图片路径")
    ap.add_argument("--sim_thresh", type=float, default=0.90)
    ap.add_argument("--margin", type=float, default=0.04)
    args = ap.parse_args()

    if args.initdb:
        db_init()
        print("DB OK")
        return

    if args.enroll:
        rec = extract_landmarks_xyzc(args.enroll)
        if "points_xyzc" not in rec:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
            return
        pid = enroll_from_points(rec["points_xyzc"], display_name=args.display_name, person_id=args.person_id)
        print({"status": "ok" if pid else "fail", "person_id": pid})
        return

    if args.match:
        rec = extract_landmarks_xyzc(args.match)
        if "points_xyzc" not in rec:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
            return
        out = match_from_points(rec["points_xyzc"], sim_thresh=args.sim_thresh, margin=args.margin)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    ap.print_help()

if __name__ == "__main__":
    main()
