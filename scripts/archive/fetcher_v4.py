#!/usr/bin/env python3
import json, sys, time, os, uuid
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"data/fetcher/config.sample.json"
RUNS_DIR = ROOT/"data/fetcher/runs"
LOGS_DIR = ROOT/"outputs/logs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def write_log(line):
    with open(LOGS_DIR/"fetcher.log", "a") as f:
        f.write(line.rstrip() + "\n")

def main():
    cfg = load_config(CONFIG_PATH)
    started = time.time()
    run_id = cfg.get("run_id", f"fetch_{uuid.uuid4().hex[:8]}").replace("${ISO_NOW}", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))

    # DRY_RUN: fabricate 3–5 posts
    posts = []
    for i in range(3, 6):
        posts.append({
            "id": f"mock_{i}",
            "platform": cfg["targets"][0]["platform"],
            "creator": cfg["targets"][0]["creator_name"],
            "handle": cfg["targets"][0]["handle_or_channel_id"],
            "published_at": now_iso(),
            "caption": f"Mock post #{i} about calm parenting.",
            "engagement": {"likes": i*10, "comments": i, "shares": i//2},
            "url": f"https://example.com/post/{i}"
        })

    out = {
        "status": "ok",
        "run_id": run_id,
        "mode": cfg.get("mode", "DRY_RUN"),
        "since_window_days": cfg.get("since_window_days", 7),
        "targets": cfg.get("targets", []),
        "posts_returned": len(posts),
        "posts": posts,
        "generated_at": now_iso(),
        "duration_ms": int((time.time() - started) * 1000)
    }

    # Write file
    out_path = RUNS_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Log
    write_log(f"{now_iso()} | run_id={run_id} | status=ok | posts={len(posts)} | file={out_path}")

    print("ok", out_path)

if __name__ == "__main__":
    main()