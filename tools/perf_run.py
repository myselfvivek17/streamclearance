"""Run one clip through a running StreamClearance and print where the time went.

    .venv/bin/python -m tools.perf_run eval/clips/speech_26s.mp4 MY IN AU SG \
        [--url http://127.0.0.1:8431] [--code-file .secrets/access_code.txt]

Uploads the clip, starts an analysis, polls until it ends, then prints wall
time, the job's `timings` (per stage) and `calls` (Gemini calls and tokens per
stage), findings per country and the drift checks. The access code is read
from the file and sent as a header; it is never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("countries", nargs="+")
    ap.add_argument("--url", default="http://127.0.0.1:8431")
    ap.add_argument("--code-file")
    ap.add_argument("--out", help="write the final job JSON here")
    args = ap.parse_args()

    headers = {}
    if args.code_file:
        headers["X-Access-Code"] = Path(args.code_file).read_text().strip()
    clip = Path(args.clip)
    with httpx.Client(base_url=args.url, headers=headers, timeout=300) as c:
        t0 = time.monotonic()
        with clip.open("rb") as fh:
            r = c.post("/api/upload", files={"file": (clip.name, fh, "video/mp4")})
        r.raise_for_status()
        t_upload = time.monotonic() - t0
        r = c.post("/api/analyze", json={"video_id": r.json()["video_id"],
                                         "countries": args.countries})
        r.raise_for_status()
        job_id = r.json()["job_id"]
        t1 = time.monotonic()
        while True:
            job = c.get(f"/api/jobs/{job_id}").json()
            if job["status"] not in ("queued", "running"):
                break
            time.sleep(1)
        wall = time.monotonic() - t1

    if args.out:
        Path(args.out).write_text(json.dumps(job, ensure_ascii=False, indent=1))
    print(f"job {job_id} {job['status']}  upload {t_upload:.1f}s  analysis wall {wall:.1f}s")
    if job.get("error"):
        print("error:", job["error"])
    for k, v in (job.get("timings") or {}).items():
        print(f"  t {k:28} {v:7.1f}s")
    calls = job.get("calls") or {}
    for k, v in calls.items():
        print(f"  c {k:28} {v['calls']:3d} calls  {v['prompt']:7d} in  {v['out']:6d} out  "
              f"{v['thoughts']:6d} think")
    rep = job.get("report") or {}
    for cr in rep.get("countries", []):
        fs = [f"{f.get('leaf') or f['category']}:{f['risk']}@{f['start_s']:.0f}s" for f in cr["findings"]]
        drift = [f"{d['language']}:{d['drift']}@{d['start_s']:.0f}s" for d in cr["translation"]]
        print(f"  {cr['country']} {cr.get('badge', cr['overall_risk'])}  findings={fs}")
        print(f"     drift={drift}  note={cr.get('translation_note', '')!r}")
    for e in rep.get("errors", []):
        print("  report error:", e)
    return 0 if job["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
