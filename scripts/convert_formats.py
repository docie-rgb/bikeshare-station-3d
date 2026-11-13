"""scene.ply → .compressed.ply / .sog / .spz 변환과 크기·시간·점 개수 측정.

사용법:  python scripts/convert_formats.py results/scene.ply results/
요구사항: Node.js >= 22 (splat-transform 3.10.3의 engines 조건), npx

측정 방식
- 변환 시간: 각 변환 명령의 wall-clock (time.perf_counter). npx 패키지 다운로드 시간을 빼기 위해
  측정 전에 `--help`를 한 번 실행해 캐시를 채운다.
- 점 개수: 각 출력 파일을 다시 splat-transform으로 .ply로 되돌려(round-trip) 헤더의 vertex 수를 읽는다.
- 점당 바이트: 출력 파일 크기 / 출력 점 개수.
SOG는 WebGPU를 사용한다. GPU를 못 쓰면 `-g cpu`로 다시 시도하고, 그 사실을 기록한다.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splat_utils import read_ply_vertex_count  # noqa: E402

PKG = "@playcanvas/splat-transform@3.10.3"
FORMATS = [("compressed.ply", "scene.compressed.ply"), ("sog", "scene.sog"), ("spz", "scene.spz")]


def st_cmd(*args: str) -> list[str]:
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        sys.exit("npx를 찾을 수 없습니다. Node.js 22 이상을 설치하세요.")
    return [npx, "-y", "-p", PKG, "splat-transform", *args]


def run(cmd: list[str]) -> tuple[float, subprocess.CompletedProcess]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return time.perf_counter() - t0, proc


def main(src: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    node_ver = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    print("node", node_ver)
    run(st_cmd("--help"))  # npx 캐시 준비 (측정 제외)

    src_n = read_ply_vertex_count(src)
    src_size = os.path.getsize(src)
    rows = [{"format": "ply (INRIA, 원본)", "file": os.path.basename(src), "bytes": src_size,
             "seconds": None, "points": src_n, "bytes_per_point": src_size / src_n, "note": "변환 입력"}]

    rt_dir = os.path.join(outdir, "_roundtrip")
    os.makedirs(rt_dir, exist_ok=True)
    for fmt, name in FORMATS:
        out = os.path.join(outdir, name)
        secs, proc = run(st_cmd("-w", src, out))
        note = ""
        if proc.returncode != 0 and fmt == "sog":
            print("SOG GPU 변환 실패, CPU로 재시도:\n", proc.stderr[-800:])
            secs, proc = run(st_cmd("-w", "-g", "cpu", src, out))
            note = "WebGPU 사용 불가로 -g cpu 사용"
        if proc.returncode != 0:
            print(f"[{fmt}] 실패\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
            rows.append({"format": fmt, "file": name, "bytes": None, "seconds": secs, "points": None,
                         "bytes_per_point": None, "note": "변환 실패: " + proc.stderr.strip()[-200:]})
            continue
        size = os.path.getsize(out)
        rt = os.path.join(rt_dir, name.replace(".", "_") + ".ply")
        _, rtp = run(st_cmd("-w", out, rt))
        pts = read_ply_vertex_count(rt) if rtp.returncode == 0 else None
        rows.append({"format": fmt, "file": name, "bytes": size, "seconds": round(secs, 3), "points": pts,
                     "bytes_per_point": (size / pts) if pts else None, "note": note})
        print(f"[{fmt}] {size:,} B, {secs:.2f} s, {pts} pts")
    shutil.rmtree(rt_dir, ignore_errors=True)

    result = {"tool": PKG, "node": node_ver, "input": os.path.basename(src), "rows": rows}
    with open(os.path.join(outdir, "formats.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    def fmt_num(v, spec):
        return "-" if v is None else format(v, spec)

    lines = ["| 형식 | 파일 | 크기 (MB) | 변환 시간 (s) | 점 개수 | 점당 바이트 | 비고 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        mb = None if r["bytes"] is None else r["bytes"] / 1e6
        lines.append(f"| {r['format']} | `{r['file']}` | {fmt_num(mb, '.2f')} | {fmt_num(r['seconds'], '.2f')} | "
                     f"{fmt_num(r['points'], ',')} | {fmt_num(r['bytes_per_point'], '.2f')} | {r['note']} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "formats.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
