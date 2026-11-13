"""MapAnything 출력 → 측정값 + INRIA Gaussian Splat PLY 변환 유틸리티.

numpy만 사용하므로 Colab과 로컬 양쪽에서 동작합니다.
좌표계: MapAnything의 world 좌표(OpenCV: +X 오른쪽, +Y 아래, +Z 앞)를 그대로 PLY에 저장합니다.
뷰어(three.js, +Y 위)에서는 메시를 X축 기준 180° 회전(F = diag(1,-1,-1))해서 보여줍니다.
"""
from __future__ import annotations

import json
import math

import numpy as np

SH_C0 = 0.28209479177387814  # 0차 구면조화 상수: f_dc = (rgb - 0.5) / SH_C0
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV world → three.js 표시 좌표


def trim_outliers(pts: np.ndarray, pct: float = 0.5):
    """좌표축(x, y, z)별 상·하위 pct% 를 벗어나는 점을 제외하는 마스크를 반환."""
    lo = np.percentile(pts, pct, axis=0)
    hi = np.percentile(pts, 100.0 - pct, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    return keep, lo, hi


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """voxel 크기 격자마다 첫 번째 점만 남기는 인덱스(정렬됨)를 반환. voxel<=0 이면 전체."""
    if voxel <= 0:
        return np.arange(len(pts))
    key = np.floor((pts - pts.min(axis=0)) / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    return idx


def bbox_stats(pts: np.ndarray) -> dict:
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    ext = mx - mn
    return {
        "min": mn.tolist(),
        "max": mx.tolist(),
        "extent": ext.tolist(),
        "diagonal": float(np.linalg.norm(ext)),
        "center": ((mn + mx) / 2).tolist(),
    }


def camera_metrics(cam2world: np.ndarray, scene_pts: np.ndarray) -> dict:
    """카메라 위치와 퍼짐 비(spread ratio) 계산.

    spread_ratio = (카메라 위치 bbox 대각선) / (장면 bbox 대각선)
    path_length  = 프레임 순서대로 이은 카메라 이동 거리 합
    """
    pos = cam2world[:, :3, 3]
    cam_box = bbox_stats(pos)
    scene_box = bbox_stats(scene_pts)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    centroid = pos.mean(axis=0)
    return {
        "camera_positions": pos.tolist(),
        "camera_bbox": cam_box,
        "camera_path_length": float(steps.sum()),
        "camera_mean_dist_to_centroid": float(np.linalg.norm(pos - centroid, axis=1).mean()),
        "scene_bbox": scene_box,
        "spread_ratio": float(cam_box["diagonal"] / scene_box["diagonal"]) if scene_box["diagonal"] > 0 else None,
    }


def viewer_cameras(cam2world: np.ndarray, look_dist: float) -> list[dict]:
    """뷰어용 촬영 시점 목록 (three.js 표시 좌표로 변환)."""
    out = []
    for T in cam2world:
        R, p = T[:3, :3], T[:3, 3]
        fwd = R[:, 2]
        up = -R[:, 1]
        out.append({
            "position": (FLIP @ p).tolist(),
            "target": (FLIP @ (p + fwd * look_dist)).tolist(),
            "up": (FLIP @ up).tolist(),
        })
    return out


def write_inria_ply(path: str, xyz: np.ndarray, rgb01: np.ndarray, scale: np.ndarray,
                    opacity: float = 0.9) -> int:
    """INRIA 3DGS 형식(binary_little_endian) PLY 저장. 반환값: 파일 바이트 수.

    필드: x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3 (float32 17개 = 68 B/점)
    - f_dc: (rgb - 0.5) / SH_C0
    - opacity: logit(opacity)  (렌더러가 sigmoid 적용)
    - scale: log(반지름)        (렌더러가 exp 적용), 등방성
    - rot: 단위 쿼터니언 (w, x, y, z) = (1, 0, 0, 0)
    """
    n = len(xyz)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros((n, len(names)), dtype=np.float32)
    data[:, 0:3] = xyz
    data[:, 6:9] = (np.clip(rgb01, 0, 1) - 0.5) / SH_C0
    data[:, 9] = math.log(opacity / (1 - opacity))
    log_s = np.log(np.clip(scale, 1e-6, None)).astype(np.float32)
    data[:, 10:13] = log_s[:, None]
    data[:, 13] = 1.0
    header = "ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype("<f4").tobytes())
    return len(header) + data.nbytes


def read_ply_vertex_count(path: str) -> int:
    with open(path, "rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"vertex count not found in {path}")


def dump_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
