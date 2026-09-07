#!/usr/bin/env python3
"""benchmark_eval.py - 3D Sonar Diver Detection 공통 벤치마크 evaluator (팀 공유).

프레임워크 무관 standalone. 예측 JSON + GT JSON을 받아 전 지표 + SDS를 낸다.
의존성: numpy, shapely 뿐 (torch/spconv/ultralytics 불필요).
geometry/AP 함수는 VoxelNet/model/{eval_voxelnet,eval_extra_metrics,anchors}.py에서
verbatim 복제 - ours 파이프라인과 동일 수치(Monte-Carlo IoU seed=0 고정).

사용:
  python benchmark_eval.py --pred preds.json --gt gt.json --out metrics.json
  python benchmark_eval.py --selftest      # 합성 데이터로 동작 검증

입력 포맷 (benchmark_config.yaml의 prediction_format):
  GT:   {"frames": {"<fid>": [ {"center":[x,y,z], "dims":[l,w,h], "rotation":[[3x3]]}, ...] }}
  PRED: {"frames": {"<fid>": [ {"center":[x,y,z], "dims":[l,w,h],
                                "rotation":[[3x3]] (또는 "yaw":rad), "score":float}, ...] }}
  - dims = [length(local x), width(local y), height(local z)] (m)
  - rotation R: 열이 local 축(col0=x/length, col1=y/width, col2=z/height), local->world.
    yaw-only baseline은 "yaw"(rad)만 줘도 됨 -> R_z(yaw)로 변환(tilt=0).
  - GT는 항상 full 3D rotation. baseline이 yaw만 내도 tilt/AOE3D가 GT 대비 계산됨.
"""
import argparse, json, math
import numpy as np
from shapely.geometry import Polygon

# ============================ geometry core (verbatim) =======================
_LOCAL_CORNERS_UNIT = np.array([
    [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
    [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1],
], dtype=np.float64)


def obb_corners(center, dims, R):
    return _LOCAL_CORNERS_UNIT * (np.asarray(dims) / 2) @ np.asarray(R).T + np.asarray(center)


def point_in_obb(points, center, dims, R):
    local = (points - center) @ R
    return np.all(np.abs(local) <= np.asarray(dims) / 2, axis=1)


def iou_3d_obb(center_a, dims_a, R_a, center_b, dims_b, R_b, n_samples=8000, rng=None):
    """진짜 3D OBB-OBB IoU, Monte-Carlo(부피는 해석값, 교집합만 샘플링)."""
    center_a, dims_a = np.asarray(center_a, float), np.asarray(dims_a, float)
    center_b, dims_b = np.asarray(center_b, float), np.asarray(dims_b, float)
    if not (np.all(np.isfinite(center_a)) and np.all(np.isfinite(dims_a))
            and np.all(np.isfinite(center_b)) and np.all(np.isfinite(dims_b))):
        return 0.0
    rng = rng or np.random.default_rng()
    vol_a, vol_b = float(np.prod(dims_a)), float(np.prod(dims_b))
    if vol_a <= 0 or vol_b <= 0:
        return 0.0
    ca, cb = obb_corners(center_a, dims_a, R_a), obb_corners(center_b, dims_b, R_b)
    lo = np.minimum(ca.min(axis=0), cb.min(axis=0))
    hi = np.maximum(ca.max(axis=0), cb.max(axis=0))
    if np.any(hi <= lo) or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return 0.0
    sample_vol = float(np.prod(hi - lo))
    pts = rng.uniform(lo, hi, size=(n_samples, 3))
    inter_frac = float(np.mean(point_in_obb(pts, center_a, dims_a, R_a)
                               & point_in_obb(pts, center_b, dims_b, R_b)))
    inter_vol = inter_frac * sample_vol
    union_vol = vol_a + vol_b - inter_vol
    return float(inter_vol / union_vol) if union_vol > 0 else 0.0


def rotated_rect_corners(x, y, l, w, theta):
    hl, hw = l / 2, w / 2
    local = np.array([[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]])
    c, s = np.cos(theta), np.sin(theta)
    return local @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def bev_iou(ca, da, Ra, cb, db, Rb):
    pa = Polygon(rotated_rect_corners(ca[0], ca[1], da[0], da[1], yaw_of(Ra)))
    pb = Polygon(rotated_rect_corners(cb[0], cb[1], db[0], db[1], yaw_of(Rb)))
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return inter / union if union > 0 else 0.0


def aligned_iou_3d(da, db):
    da, db = np.asarray(da), np.asarray(db)
    inter = np.prod(np.minimum(da, db))
    union = np.prod(da) + np.prod(db) - inter
    return inter / union if union > 0 else 0.0


def yaw_err_folded(ya, yb):
    d = abs((ya - yb + math.pi) % (2 * math.pi) - math.pi)
    d = d % math.pi
    return min(d, math.pi - d)


_RZ_PI = np.diag([-1.0, -1.0, 1.0])


def geo_err_deg_folded(Ra, Rb):
    best = math.pi
    for S in (np.eye(3), _RZ_PI):
        Rd = Ra.T @ (Rb @ S)
        tr = float(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))
        best = min(best, math.acos(tr))
    return math.degrees(best)


def tilt_err_deg(Ra, Rb):
    za, zb = Ra[:, 2], Rb[:, 2]
    c = float(np.clip(np.dot(za, zb) / (np.linalg.norm(za) * np.linalg.norm(zb) + 1e-9), -1.0, 1.0))
    a = math.degrees(math.acos(c))
    return min(a, 180.0 - a)


def compute_ap(detections, n_gt):
    if n_gt == 0 or not detections:
        return 0.0
    confs = np.array([d[0] for d in detections])
    is_tp = np.array([d[1] for d in detections])
    order = np.argsort(-confs)
    is_tp = is_tp[order]
    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(~is_tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ============================ box parsing ====================================
def _R_of(b):
    if b.get("rotation") is not None:
        return np.asarray(b["rotation"], dtype=float)
    yaw = float(b.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _parse(frame_boxes, with_score):
    out = []
    for b in frame_boxes:
        rec = (np.asarray(b["center"], float), np.asarray(b["dims"], float), _R_of(b))
        out.append((float(b["score"]),) + rec if with_score else rec)
    return out


# ============================ evaluation =====================================
AP_IOUS = [0.25, 0.3, 0.35, 0.4, 0.5]
BEV_IOUS = [0.25, 0.35, 0.5]
SDS_AP_THRS = [0.3, 0.35, 0.4]
W_AP, ATE_NORM, AOE3D_NORM = 3.0, 1.0, math.pi / 2


def evaluate(pred, gt, score_thresh=0.3, tp_iou=0.3, n_samples=8000):
    ap_dets = {t: [] for t in AP_IOUS}
    bev_dets = {t: [] for t in BEV_IOUS}
    tp = {k: [] for k in ("iou3d", "ate", "ase", "aoe", "aoe3d", "tilt")}
    n_gt_total = 0

    for fid, gt_boxes in gt["frames"].items():
        gts = _parse(gt_boxes, with_score=False)
        preds = [p for p in _parse(pred["frames"].get(fid, []), with_score=True) if p[0] >= score_thresh]
        preds.sort(key=lambda p: -p[0])
        n_gt_total += len(gts)
        n_p, n_g = len(preds), len(gts)
        if n_g == 0:
            for p in preds:
                for t in AP_IOUS: ap_dets[t].append((p[0], False))
                for t in BEV_IOUS: bev_dets[t].append((p[0], False))
            continue
        rng = np.random.default_rng(0)
        i3 = np.zeros((n_p, n_g)); ib = np.zeros((n_p, n_g))
        for pi, (_, pc, pd, pR) in enumerate(preds):
            for gi, (gc, gd, gR) in enumerate(gts):
                i3[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, n_samples=n_samples, rng=rng)
                ib[pi, gi] = bev_iou(pc, pd, pR, gc, gd, gR)

        for t, mat, dets in [(a, i3, ap_dets) for a in AP_IOUS] + [(b, ib, bev_dets) for b in BEV_IOUS]:
            matched = set()
            for pi in range(n_p):
                cand = [gi for gi in range(n_g) if gi not in matched]
                is_tp = False
                if cand:
                    bg = cand[int(np.argmax(mat[pi, cand]))]
                    if mat[pi, bg] >= t:
                        matched.add(bg); is_tp = True
                dets[t].append((preds[pi][0], is_tp))

        matched = set()
        for pi in range(n_p):
            cand = [gi for gi in range(n_g) if gi not in matched]
            if not cand:
                break
            bg = cand[int(np.argmax(i3[pi, cand]))]
            if i3[pi, bg] >= tp_iou:
                matched.add(bg)
                _, pc, pd, pR = preds[pi]; gc, gd, gR = gts[bg]
                tp["iou3d"].append(i3[pi, bg])
                tp["ate"].append(float(np.linalg.norm(pc - gc)))
                tp["ase"].append(1.0 - aligned_iou_3d(pd, gd))
                tp["aoe"].append(math.degrees(yaw_err_folded(yaw_of(pR), yaw_of(gR))))
                tp["aoe3d"].append(geo_err_deg_folded(pR, gR))
                tp["tilt"].append(tilt_err_deg(pR, gR))

    def _m(x): return float(np.mean(x)) if len(x) else 0.0
    ap3d = {t: compute_ap(ap_dets[t], n_gt_total) for t in AP_IOUS}
    bev_ap = {t: compute_ap(bev_dets[t], n_gt_total) for t in BEV_IOUS}

    mAP = float(np.mean([ap3d[t] for t in SDS_AP_THRS]))
    ate_m, ase_m, aoe3d_m = _m(tp["ate"]), _m(tp["ase"]), _m(tp["aoe3d"])
    sds = (W_AP * mAP + (1 - min(1.0, ate_m / ATE_NORM)) + (1 - min(1.0, ase_m))
           + (1 - min(1.0, math.radians(aoe3d_m) / AOE3D_NORM))) / (W_AP + 3.0)

    return {
        "n_gt": n_gt_total, "n_tp@%.2f" % tp_iou: len(tp["iou3d"]),
        "AP3D": {str(t): ap3d[t] for t in AP_IOUS},
        "BEV_AP": {str(t): bev_ap[t] for t in BEV_IOUS},
        "TP_IoU3D": _m(tp["iou3d"]),
        "ATE_m": ate_m, "ASE": ase_m,
        "AOE_yaw_deg": _m(tp["aoe"]), "AOE3D_deg": aoe3d_m, "tilt_deg": _m(tp["tilt"]),
        "mAP": mAP, "SDS": sds,
    }


# ============================ selftest =======================================
def _selftest():
    def Rz(a): c, s = math.cos(a), math.sin(a); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    def Rx(a): c, s = math.cos(a), math.sin(a); return [[1, 0, 0], [0, c, -s], [0, s, c]]
    gt = {"frames": {"f0": [{"center": [6, 0, 0.1], "dims": [1.5, 1.0, 1.1],
                             "rotation": (np.array(Rz(0.3)) @ np.array(Rx(0.5))).tolist()}]}}
    # perfect full-3D pred
    good = {"frames": {"f0": [{"center": [6, 0, 0.1], "dims": [1.5, 1.0, 1.1],
                              "rotation": gt["frames"]["f0"][0]["rotation"], "score": 0.9}]}}
    # yaw-only pred (tilt=0) - 같은 위치/크기/yaw지만 tilt 못 맞춤
    yawonly = {"frames": {"f0": [{"center": [6, 0, 0.1], "dims": [1.5, 1.0, 1.1],
                                 "yaw": 0.3, "score": 0.9}]}}
    rg = evaluate(good, gt); ry = evaluate(yawonly, gt)
    print("[selftest] full-3D pred : AP3D@0.5=%.3f tilt=%.1f AOE3D=%.1f SDS=%.3f"
          % (rg["AP3D"]["0.5"], rg["tilt_deg"], rg["AOE3D_deg"], rg["SDS"]))
    print("[selftest] yaw-only pred: AP3D@0.5=%.3f tilt=%.1f AOE3D=%.1f SDS=%.3f"
          % (ry["AP3D"]["0.5"], ry["tilt_deg"], ry["AOE3D_deg"], ry["SDS"]))
    assert rg["tilt_deg"] < 2 and ry["tilt_deg"] > 20, "tilt metric sanity"
    assert rg["SDS"] > ry["SDS"], "full-3D should beat yaw-only on SDS"
    print("[selftest] OK - full-3D가 yaw-only를 tilt/SDS에서 이김(기대대로)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--gt"); ap.add_argument("--out")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--tp-iou", type=float, default=0.3)
    ap.add_argument("--n-samples", type=int, default=8000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        pred = json.load(open(a.pred)); gt = json.load(open(a.gt))
        res = evaluate(pred, gt, a.score_thresh, a.tp_iou, a.n_samples)
        print(json.dumps(res, indent=2))
        if a.out:
            json.dump(res, open(a.out, "w"), indent=2)
            print(f"[saved] {a.out}")
