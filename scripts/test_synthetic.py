"""splat_utils 동작 확인용 합성 장면 테스트 (실제 복원 결과 아님).

바닥면 + 바퀴 모양 고리 몇 개 + 원호 위 카메라 24대를 만들고, 노트북과 같은 순서로
이상치 제외 → 측정 → PLY 저장 → scene_meta.json 생성까지 수행한다.
출력: tests_out/ (git에 올리지 않음). 뷰어 확인: index.html?src=tests_out/synthetic.ply&meta=tests_out/scene_meta.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splat_utils import (FLIP, bbox_stats, camera_metrics, read_ply_vertex_count, trim_outliers,  # noqa: E402
                         viewer_cameras, voxel_downsample, write_inria_ply)

rng = np.random.default_rng(0)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests_out")
os.makedirs(out, exist_ok=True)

# OpenCV world: +Y 아래. 바닥은 y = +1.0 평면
g = rng.uniform([-3, 0.98, 0], [3, 1.02, 6], (200_000, 3))
gc = np.where(((np.floor(g[:, 0] * 5) + np.floor(g[:, 2] * 5)) % 2)[:, None] == 0, [0.62, 0.38, 0.3], [0.5, 0.3, 0.25])
wheels, wc = [], []
for k in range(5):
    t = rng.uniform(0, 2 * np.pi, 20_000)
    r = 0.33 + rng.normal(0, 0.01, t.size)
    wheels.append(np.c_[-2 + k * 1.0 + np.zeros_like(t) + rng.normal(0, 0.01, t.size), 0.65 - r * np.sin(t), 3 + r * np.cos(t)])
    wc.append(np.tile([0.2, 0.7, 0.3], (t.size, 1)))
outl = rng.uniform(-50, 50, (2_000, 3))  # 멀리 떨어진 이상치
P = np.r_[g, *wheels, outl]
C = np.r_[gc, *wc, np.ones((2_000, 3))]
S = np.full(len(P), 0.01)

poses = []
for a in np.linspace(-0.6, 0.6, 24):
    p = np.array([3.2 * np.sin(a), 0.0, 3 - 3.2 * np.cos(a)])
    fwd = np.array([0, 0.6, 3]) - p
    fwd /= np.linalg.norm(fwd)
    right = np.cross([0, 1, 0], fwd) * -1
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.c_[right, down, fwd]
    T[:3, 3] = p
    poses.append(T)
poses = np.stack(poses)

n0 = len(P)
keep, lo, hi = trim_outliers(P, 0.5)
P, C, S = P[keep], C[keep], S[keep]
idx = voxel_downsample(P, 0.01)
P, C, S = P[idx], C[idx], S[idx]
nbytes = write_inria_ply(os.path.join(out, "synthetic.ply"), P, C, S)
cam = camera_metrics(poses, P)
diag = bbox_stats(P)["diagonal"]

import json  # noqa: E402
with open(os.path.join(out, "scene_meta.json"), "w") as f:
    json.dump({"target": (FLIP @ np.array(cam["scene_bbox"]["center"])).tolist(),
               "cameras": viewer_cameras(poses, diag * 0.25),
               "frames": [f"synthetic_{i + 1:03d}" for i in range(24)],
               "points": len(P), "scene_diagonal": diag, "spread_ratio": cam["spread_ratio"]}, f, indent=1)

assert read_ply_vertex_count(os.path.join(out, "synthetic.ply")) == len(P)
assert os.path.getsize(os.path.join(out, "synthetic.ply")) == nbytes
assert P.min() > -10 and P.max() < 10, "outliers should be trimmed"
print(f"points {n0} -> trimmed {keep.sum()} -> voxel {len(P)}; ply {nbytes} B ({nbytes / len(P):.1f} B/pt)")
print("scene extent", np.round(cam["scene_bbox"]["extent"], 2), "spread_ratio", round(cam["spread_ratio"], 3))
print("OK")
