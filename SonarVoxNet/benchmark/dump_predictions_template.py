"""dump_predictions_template.py - 팀원용: 자기 모델 추론 결과를 공통 예측 JSON으로 저장.

benchmark_eval.py가 먹는 포맷으로 dump하는 최소 템플릿. 자기 추론 루프에 아래
build_record()/save_predictions() 두 함수만 가져다 쓰면 된다. 의존성 없음(표준 라이브러리).

핵심 규약 (꼭 지킬 것):
  - center = [x, y, z]  (m, ROI [0,-5,-2.5,12,5,2.5] 안, x=forward)
  - dims   = [length, width, height] = [local-x, local-y, local-z extent] (m)
  - 회전: 둘 중 하나로 주면 됨
      (A) yaw-only baseline (VoxelNet/SECOND/VoxelNeXt/PointPillars/CenterPoint/TriBand):
          yaw = heading(rad, world-z 축 회전) 하나만. evaluator가 R_z(yaw)로 만듦(tilt=0).
      (B) full-3D (ours 등): rotation = 3x3 리스트. 열이 local 축(col0=length축, col1=width축,
          col2=height축), local->world (world = R @ local + center).
  - score = confidence (float)

★ 대부분의 3D 검출 프레임워크(OpenPCDet/mmdet3d 등)는 박스를 [x,y,z,dx,dy,dz,heading]로
  낸다 → center=[x,y,z], dims=[dx,dy,dz], yaw=heading 으로 그대로 매핑(아래 예시).
"""
import json


def build_record(center, dims, score, yaw=None, rotation=None):
    """박스 하나 -> dict. yaw(rad) 또는 rotation(3x3) 중 하나 제공."""
    rec = {"center": [float(c) for c in center],
           "dims": [float(d) for d in dims],
           "score": float(score)}
    if rotation is not None:
        rec["rotation"] = [[float(v) for v in row] for row in rotation]
    else:
        rec["yaw"] = float(yaw)  # yaw-only
    return rec


def save_predictions(frames, path):
    """frames: {frame_id: [record, ...]} -> JSON 저장."""
    json.dump({"frames": frames}, open(path, "w"), indent=2)
    print(f"[saved] {path}  ({sum(len(v) for v in frames.values())} boxes, {len(frames)} frames)")


# ============================ 사용 예시 ======================================
if __name__ == "__main__":
    frames = {}

    # ---- 예시 A: yaw-only baseline (프레임워크 출력 [x,y,z,l,w,h,heading] + score) ----
    # 실제로는 자기 추론 루프에서 프레임별 예측 박스 배열을 순회하면 된다.
    # pred_boxes: (N,7) [x,y,z,dx,dy,dz,heading],  scores: (N,)
    def dump_one_frame_yaw(frame_id, pred_boxes, scores):
        recs = []
        for (x, y, z, dx, dy, dz, heading), sc in zip(pred_boxes, scores):
            recs.append(build_record([x, y, z], [dx, dy, dz], sc, yaw=heading))
        frames[frame_id] = recs

    # (더미) 한 프레임 예시
    dump_one_frame_yaw("scene_0001_000011",
                       pred_boxes=[[4.1, -0.95, 0.0, 1.55, 1.0, 1.1, -0.48]],
                       scores=[0.86])

    # ---- 예시 B: full-3D (ours) - 회전행렬 R(3x3) 있는 경우 ----
    #   frames["scene_0001_000010"] = [build_record([6.05,0.02,0.1],[1.5,1.0,1.1],0.91, rotation=R)]

    save_predictions(frames, "my_predictions.json")
    print("이제: python benchmark_eval.py --pred my_predictions.json --gt <공통 GT>.json")
