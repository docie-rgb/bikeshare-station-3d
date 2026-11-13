"""colab/bikeshare_mapanything.ipynb 생성기.

scripts/splat_utils.py, scripts/convert_formats.py, privacy/redact_boxes.json 을 노트북 안에
%%writefile / JSON 리터럴로 넣어서, Colab에는 노트북과 입력 ZIP만 올리면 되도록 만든다.
사용법: python scripts/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


boxes_path = os.path.join(ROOT, "privacy", "redact_boxes.json")
boxes = read("privacy/redact_boxes.json") if os.path.exists(boxes_path) else "{}"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# 서울 공공자전거 대여소 3D 복원 (MapAnything → Gaussian Splat)

**런타임 → 런타임 유형 변경 → T4 GPU**로 바꾼 뒤 위에서부터 순서대로 실행하세요.

| 단계 | 내용 | 산출물 |
|---|---|---|
| 1 | GPU 확인, MapAnything 설치 | |
| 2 | `bikeshare_frames_24.zip` 업로드 · 압축 구조/크기 확인 | |
| 3 | (선택) 번호판·얼굴 등 식별정보 영역 블러 | `frames_redacted/`, `redaction_sheet.jpg` |
| 4 | `facebook/map-anything-apache` 추론 (긴 변 축소 + fp16) | |
| 5 | 축별 상·하위 0.5% 이상치 제외, 측정, INRIA PLY 저장 | `results/scene.ply`, `metrics.json`, `scene_meta.json` |
| 6 | splat-transform 3.10.3으로 형식 변환·비교 | `scene.sog`, `scene.spz`, `scene.compressed.ply`, `formats.md` |
| 7 | 결과 ZIP 다운로드 | `bikeshare_results.zip` |

이 노트북의 모든 숫자는 실행할 때 실제로 측정됩니다. 결과는 `results/`에 JSON으로 저장됩니다.
""")

md("## 1. GPU 확인 및 MapAnything 설치")
code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
""")
code("""
%cd /content
!test -d map-anything || git clone --depth 1 https://github.com/facebookresearch/map-anything.git
%cd /content/map-anything
MAPANYTHING_COMMIT = !git rev-parse HEAD
MAPANYTHING_COMMIT = MAPANYTHING_COMMIT[0]
print("map-anything commit:", MAPANYTHING_COMMIT)
!pip install -q -e .
%cd /content
# editable 설치는 커널을 재시작해야 import 경로에 반영되므로, 재시작 없이 바로 쓰도록 경로를 직접 추가
import sys
if "/content/map-anything" not in sys.path:
    sys.path.insert(0, "/content/map-anything")
""")

md("""
## 2. 입력 ZIP 업로드 및 구조 확인
`/content/bikeshare_frames_24.zip`이 없으면 업로드 창이 열립니다. 왼쪽 파일 패널에 직접 끌어다 놓아도 됩니다.
""")
code("""
import os, zipfile, glob
from PIL import Image
ZIP = "/content/bikeshare_frames_24.zip"
if not os.path.exists(ZIP):
    from google.colab import files
    up = files.upload()
    os.rename(list(up.keys())[0], ZIP)
with zipfile.ZipFile(ZIP) as z:
    names = [n for n in z.namelist() if not n.endswith("/")]
    print(len(names), "files; folders inside:", sorted({os.path.dirname(n) or "(root)" for n in names}))
    z.extractall("/content/frames_raw")
FRAMES = sorted(glob.glob("/content/frames_raw/**/*.jpg", recursive=True))
sizes = {}
for f in FRAMES:
    with Image.open(f) as im:
        sizes[os.path.basename(f)] = im.size
print("image sizes (W,H):", sorted(set(sizes.values())), "| count:", len(FRAMES))
""")

md("""
## 3. (선택) 식별정보 블러
`REDACT_BOXES`는 프레임마다 블러할 영역 `[x0, y0, x1, y1]`입니다. 원본 720×1280 픽셀 기준이며, 사람이 직접 눈으로 확인해 정했습니다.
블러는 복원 **전에** 적용합니다. 그래야 splat 색상에 번호판 같은 식별정보가 남지 않습니다.
실행 후 `redaction_sheet.jpg`를 열어 **빠진 곳이 없는지 반드시 직접 확인**하세요.
""")
code(f"""
import json, shutil
from PIL import ImageFilter, ImageDraw
REDACT = True
REDACT_BOXES = json.loads(r'''{boxes}''')
SRC_DIR = "/content/frames_raw"
IMG_DIR = "/content/frames_redacted" if REDACT else SRC_DIR
if REDACT:
    shutil.rmtree(IMG_DIR, ignore_errors=True); os.makedirs(IMG_DIR)
    thumbs = []
    for f in FRAMES:
        name = os.path.basename(f)
        im = Image.open(f).convert("RGB")
        for item in REDACT_BOXES.get(name, []):
            x0, y0, x1, y1 = [int(v) for v in item["box"]]
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(im.width, x1), min(im.height, y1)
            region = im.crop((x0, y0, x1, y1))
            small = region.resize((max(1, region.width // 16), max(1, region.height // 16)))
            im.paste(small.resize(region.size, Image.NEAREST).filter(ImageFilter.GaussianBlur(6)), (x0, y0))
        im.save(os.path.join(IMG_DIR, name), quality=95)
        t = im.copy(); t.thumbnail((180, 320)); ImageDraw.Draw(t).text((4, 4), name, fill=(255, 0, 0)); thumbs.append(t)
    sheet = Image.new("RGB", (180 * 6, 320 * 4), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % 6) * 180, (i // 6) * 320))
    sheet.save("/content/redaction_sheet.jpg"); display(sheet)
print("reconstruction input:", IMG_DIR, "| redacted regions:", sum(len(v) for v in REDACT_BOXES.values()) if REDACT else 0)
""")

md("## 4. 보조 스크립트 작성 (저장소 `scripts/` 와 동일한 코드)")
code("%%writefile /content/splat_utils.py\n" + read("scripts/splat_utils.py"))
code("%%writefile /content/convert_formats.py\n" + read("scripts/convert_formats.py"))

md("""
## 5. MapAnything 추론
- 긴 변 축소: `load_images(resize_mode="longest_side", size=LONG_SIDE)`. 기본값은 518이고, ViT patch 14의 배수입니다.
- fp16: T4는 bf16 가속을 지원하지 않아서 `amp_dtype="fp16"`을 명시합니다.
- 메모리가 부족하면 `LONG_SIDE`를 434나 392로 줄이세요.
""")
code("""
import os, time, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np, torch
sys.path.insert(0, "/content")
sys.path.insert(0, "/content/map-anything")  # editable 설치 직후(커널 재시작 전)에도 import 되도록
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

LONG_SIDE = 518
AMP_DTYPE = "fp16"
TIMES = {}
t0 = time.perf_counter()
model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
TIMES["model_load_s"] = time.perf_counter() - t0

t0 = time.perf_counter()
views = load_images(IMG_DIR, resize_mode="longest_side", size=LONG_SIDE)
TIMES["image_load_s"] = time.perf_counter() - t0
INPUT_SHAPE = [int(x) for x in views[0]["true_shape"][0]]  # [H, W]
print(len(views), "views | model input H,W =", INPUT_SHAPE, "|", TIMES)
""")
code("""
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    preds = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype=AMP_DTYPE,
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
    )
torch.cuda.synchronize()
TIMES["inference_s"] = time.perf_counter() - t0
PEAK_GB = torch.cuda.max_memory_allocated() / 2**30
print(f"inference {TIMES['inference_s']:.1f}s | peak GPU mem {PEAK_GB:.2f} GiB | keys: {sorted(preds[0].keys())}")
""")

md("""
## 6. 점 데이터 정리 · 측정 · INRIA PLY 저장
- 마스크(`mask`)를 통과한 픽셀만 사용합니다.
- **좌표축별 상·하위 0.5% 이상치 제외**를 적용합니다 (`trim_outliers`).
- 점이 `MAX_POINTS`보다 많으면 voxel 격자로 줄입니다. 각 단계의 점 개수는 모두 기록합니다.
- Gaussian 반지름은 픽셀 하나의 크기를 3D로 옮긴 값(`depth / fx × SCALE_K`)입니다. 회전은 단위 쿼터니언, 불투명도는 0.9로 둡니다.
""")
code("""
from splat_utils import trim_outliers, voxel_downsample, camera_metrics, viewer_cameras, write_inria_ply, bbox_stats, dump_json, FLIP
SCALE_K = 1.0
MAX_POINTS = 1_000_000
os.makedirs("/content/results", exist_ok=True)

P, C, S, POSES, per_view = [], [], [], [], []
for i, p in enumerate(preds):
    pts = p["pts3d"][0].float().cpu().numpy()
    img = p["img_no_norm"][0].float().cpu().numpy()
    if img.max() > 1.5:
        img = img / 255.0
    depth = p["depth_z"][0].squeeze(-1).float().cpu().numpy()
    fx = float(p["intrinsics"][0][0, 0])
    m = p["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & np.isfinite(pts).all(-1) & (depth > 0)
    P.append(pts[m]); C.append(img[m]); S.append(depth[m] / fx * SCALE_K)
    POSES.append(p["camera_poses"][0].float().cpu().numpy())
    per_view.append({"frame": os.path.basename(FRAMES[i]), "pixels": int(m.size), "valid": int(m.sum()), "fx": fx})
P, C, S, POSES = np.concatenate(P), np.concatenate(C), np.concatenate(S), np.stack(POSES)
n_masked = len(P)

keep, lo, hi = trim_outliers(P, 0.5)
P, C, S = P[keep], C[keep], S[keep]
n_trimmed = len(P)

diag = bbox_stats(P)["diagonal"]
voxel = 0.0
if len(P) > MAX_POINTS:
    voxel = diag / 4000
    while True:
        idx = voxel_downsample(P, voxel)
        if len(idx) <= MAX_POINTS: break
        voxel *= 1.25
    P, C, S = P[idx], C[idx], np.maximum(S[idx], voxel * 0.5)
n_final = len(P)

t0 = time.perf_counter()
ply_bytes = write_inria_ply("/content/results/scene.ply", P, C, S)
TIMES["ply_write_s"] = time.perf_counter() - t0

cam = camera_metrics(POSES, P)
metrics = {
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "model": "facebook/map-anything-apache", "mapanything_commit": MAPANYTHING_COMMIT,
    "frames": len(views), "original_size_wh": sorted(set(sizes.values())), "model_input_hw": INPUT_SHAPE,
    "long_side": LONG_SIDE, "amp_dtype": AMP_DTYPE, "redacted": REDACT,
    "timings_s": {k: round(v, 3) for k, v in TIMES.items()}, "peak_gpu_mem_gib": round(PEAK_GB, 3),
    "points": {"masked_valid": n_masked, "after_trim_0.5pct": n_trimmed, "final": n_final,
               "trim_removed": n_masked - n_trimmed, "voxel_size": voxel},
    "trim_bounds": {"low": lo.tolist(), "high": hi.tolist()},
    "per_view": per_view, "ply_bytes": ply_bytes,
    "units_note": "MapAnything 출력 단위(메트릭 스케일 추정, 미터 근사). 실측 기준과 비교하지 않았음.",
    **cam,
}
dump_json("/content/results/metrics.json", metrics)

center = np.array(cam["scene_bbox"]["center"])
dump_json("/content/results/scene_meta.json", {
    "target": (FLIP @ center).tolist(),
    "cameras": viewer_cameras(POSES, look_dist=diag * 0.25),
    "frames": [v["frame"] for v in per_view],
    "points": n_final, "scene_diagonal": diag, "spread_ratio": cam["spread_ratio"],
})
print(json.dumps({k: metrics[k] for k in ["points", "timings_s", "peak_gpu_mem_gib", "spread_ratio"]}, indent=1, ensure_ascii=False))
print("scene bbox extent:", np.round(cam["scene_bbox"]["extent"], 3), "| camera bbox extent:", np.round(cam["camera_bbox"]["extent"], 3))
""")
code("""
# 위에서 내려다본 점 분포 + 카메라 궤적 (README용 미리보기)
import matplotlib.pyplot as plt
sel = np.random.default_rng(0).choice(len(P), min(150_000, len(P)), replace=False)
pos = np.array(cam["camera_positions"])
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(P[sel, 0], P[sel, 2], c=C[sel], s=0.3, linewidths=0)
ax.plot(pos[:, 0], pos[:, 2], "r.-", lw=1); ax.text(pos[0, 0], pos[0, 2], " 1", color="r")
ax.set_xlabel("x"); ax.set_ylabel("z (forward)"); ax.set_aspect("equal"); ax.set_title("top view: points + cameras")
plt.savefig("/content/results/preview_topdown.png", dpi=120, bbox_inches="tight"); plt.show()
""")

md("""
## 7. splat-transform 3.10.3 형식 변환 · 비교
splat-transform 3.10.3은 Node.js 22 이상이 필요합니다. 그래서 Colab에 Node 22를 설치합니다.
SOG 압축은 WebGPU를 사용하고, 실패하면 자동으로 `-g cpu`로 다시 시도합니다. 이때 비고 칸에 그 사실이 기록됩니다.
""")
code("""
!node --version || true
!curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1 && apt-get install -y -qq nodejs > /dev/null
!node --version
!cd /content && python convert_formats.py results/scene.ply results/
""")

md("## 8. 결과 다운로드\n받은 ZIP을 풀어서 저장소의 `results/` 폴더에 넣으세요. `scene.sog`는 저장소 루트에 둡니다.")
code("""
import shutil
if os.path.exists("/content/redaction_sheet.jpg"):
    shutil.copy("/content/redaction_sheet.jpg", "/content/results/")
shutil.make_archive("/content/bikeshare_results", "zip", "/content/results")
print({f: os.path.getsize(os.path.join("/content/results", f)) for f in sorted(os.listdir("/content/results"))})
from google.colab import files
files.download("/content/bikeshare_results.zip")
""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": cells,
}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
out = os.path.join(ROOT, "colab", "bikeshare_mapanything.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, len(cells), "cells")
