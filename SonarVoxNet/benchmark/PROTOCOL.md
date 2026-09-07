# 3D Sonar Diver Detection — 공통 벤치마크 프로토콜

팀 벤치마크 표를 **비교 가능**하게 만들기 위한 최소 합의. 모델 코드는 각자 원 논문에 충실하게(공유 불필요) 구현하되, **아래 5개는 반드시 공통**이어야 한다.

## 공통 대상 모델
- Voxel 류: **VoxelNet**, **CenterPoint**, **SECOND**, **VoxelNeXt**
- 2D 방법: **PointPillars**, **TriBand BEV**

## 반드시 공통인 5가지

1. **ROI / 좌표계** — `point_cloud_range = [0,-5,-2.5, 12,5,2.5]` (전 방법 동일). voxel 계열은 voxel_size `[0.1,0.1,0.5]`도 공통. PointPillars(pillar)·TriBand(image)는 동일 ROI 위에 자기 인코딩.
2. **Split** — `splits.json` **동일본**, scene 단위, seed {0,1}. eval split = `test_full`.
3. **Checkpoint 선정** — val **AP3D@IoU0.35** 기준 best epoch (전 방법 동일). 원 데이터셋 HP(예: VoxelNet 160ep, LiDAR 360° aug)를 리터럴로 쓰지 말 것 — 데이터셋에 맞춘 budget + val-best.
4. **Augmentation 정책** — 고정 forward-looking FOV(±45°) 때문에 **global rotation·전후 flip은 전 방법 제외**(안 그러면 물리적 무효 데이터). 좌우 flip / scale / translation / GT-sampling은 허용. → **전 방법 동일 적용**해야 공정.
5. **평가 harness + 지표** — 아래 참고.

## 역할 분담 (누가 뭘 하나)

| | 하는 일 |
|---|---|
| **팀원** | 규약 준수 학습 → best-ckpt로 추론 → **예측 dump JSON** 생성 → 채점자에게 전달. **GT 불필요, eval 안 돌려도 됨.** |
| **채점자(우리)** | `common_gt.json` 보유 → 모두의 예측 dump를 **하나의 `benchmark_eval.py`**로 일괄 채점 → 벤치 표. |

- **GT JSON은 채점자 전용**이다. 팀원 배포물 아님. (팀원이 보내기 전 자가확인하고 싶으면 `examples/`의 미니 GT로 도구 동작만 보면 됨.)
- **frame_id 규약(필수)**: 예측 JSON의 frame_id = **공유된 `splits.json`의 test 항목과 동일 키**. 같은 split을 쓰므로 자동으로 우리 GT 키와 정합된다. (키가 어긋나면 매칭이 통째로 깨짐.)

**팀원이 알아야 할 것은 딱 2개**: ① 규약(이 문서 + `benchmark_config.yaml`) ② 예측 dump 만드는 법(`dump_predictions_template.py` + `examples/`).

## ★ eval는 "체크포인트 모으기"가 아니라 "예측 dump 모으기"

프레임워크가 달라(spconv / ultralytics / pillar) 한 컴퓨터에서 모든 체크포인트를 직접 eval할 수 없다. 대신:

1. 각자 자기 best-checkpoint로 **추론**을 돌려 **공통 예측 포맷 JSON**으로 dump.
2. 그 예측 dump들만 모아 **하나의 공통 evaluator**에 투입.

**공통 evaluator = `benchmark_protocol/benchmark_eval.py`** (standalone, numpy+shapely만, torch 불필요).
```bash
python benchmark_eval.py --pred <my_predictions>.json --gt <공통 GT>.json --out metrics.json
```

**예측 dump가 뭔지 헷갈리면 이 3개를 보면 된다(전부 실행 가능):**
- `examples/example_gt.json` — GT 포맷 예시(2 프레임)
- `examples/example_pred.json` — 예측 포맷 예시. **full-3D는 `rotation`(3x3), yaw-only baseline은 `yaw`(rad) 하나만** — 두 방식 다 들어있음.
- `dump_predictions_template.py` — 자기 추론 루프에 붙여 쓰는 dump 헬퍼. 대부분 프레임워크 출력 `[x,y,z,l,w,h,heading]+score` → `center/dims/yaw/score`로 그대로 매핑하는 예시 포함.

검증: `python benchmark_eval.py --pred examples/example_pred.json --gt examples/example_gt.json` 이 돌면 환경 OK.

- **왜 이 evaluator를 꼭 써야 하나**: 자기 프레임워크 eval이 yaw-AP만 내면 우리 논문 핵심(**full-3D tilt 기여**)이 표에 안 나온다. 이 도구는 방향을 **AOE3D(full-3D geodesic)·tilt°**까지 산출한다(정의: `reports_v2/orientation_metrics_and_sparse_decision.html`).
- **baseline은 yaw만 내도 됨**: `yaw` 하나 넣으면 evaluator가 R_z(yaw)(tilt=0)로 만들어, 진짜 3D GT 대비 "yaw는 맞는데 tilt는 못 잡음"이 tilt°/AOE3D에 그대로 드러난다.
- geometry/AP 함수는 ours 파이프라인(`eval_extra_metrics.py`)에서 verbatim 복제 → 수치 일치(MC IoU seed 고정).

## 보고 지표 (표 열)
`AP3D@0.3 · AP3D@0.35 · AP3D@0.4 · AP3D@0.5 · BEV_AP@0.35 · ATE · ASE · AOE(yaw) · AOE3D · tilt · SDS`

- **SDS** (종합): `SDS = (1/6)[3·mAP + (1−min(1,ATE/1.0)) + (1−ASE) + (1−min(1, AOE3D_rad/(π/2)))]`, mAP = mean(AP3D@0.3/0.35/0.4). (자세한 정의·근거: `benchmark_config.yaml`의 `sds`, 및 orientation_metrics 문서.)
- 스토리: baseline은 **BEV AP·yaw**에선 경쟁력, **AP3D@0.5·tilt·AOE3D**에서 붕괴 → ours가 3D/tilt에서 우위.

## 체크리스트 (배포 전 확인)
- [ ] 모두 같은 `splits.json` 쓰는가
- [ ] 모두 ROI `[0,-5,-2.5,12,5,2.5]` 안에서 학습/추론하는가
- [ ] 모두 val AP@0.35로 best 선정하는가
- [ ] 모두 global rotation / 전후 flip 제외했는가 (ours 포함)
- [ ] baseline은 yaw-only(native), ours만 full-3D인가
- [ ] 예측을 공통 포맷으로 dump → 공통 eval(eval_extra+compute_sds) 통과하는가

## 파일 (이 폴더 통째로 팀 공유)
- `benchmark_config.yaml` — 공통 수치(ROI/voxel/anchor/split/eval/SDS/예측포맷)
- `benchmark_eval.py` — **공통 evaluator**(standalone, numpy+shapely). `--selftest` 내장.
- `dump_predictions_template.py` — 예측 dump 헬퍼 템플릿(자기 추론 코드에 복사)
- `examples/example_gt.json`, `examples/example_pred.json` — 포맷 예시(실행 가능)
- 지표 정의 문서: `../reports_v2/orientation_metrics_and_sparse_decision.html`
