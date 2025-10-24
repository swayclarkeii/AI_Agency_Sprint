#!/usr/bin/env python3
# Fetcher_v5.py — contract-compliant writer for Fetcher → Analyzer
# Policy: empty-string/empty-array (no nulls)

import json, sys, os, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/fetcher/config.sample.json"
RUNS_DIR = ROOT / "data/fetcher/runs"
LOGS_DIR = ROOT / "outputs/logs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Constants / Contract ----
PLATFORM_ENUM = {"instagram", "tiktok", "youtube"}
CONTRACT_VERSION = "fetcher_v4"   # keep as v4 to match canonical JSON contract
MAX_POSTS_PER_RUN = 100
TEXT_CAP = 600
MAX_TAGS = 10

# ---- Time helpers ----
def now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def parse_iso_z(s: str) -> str:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# ---- IO helpers ----
def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_log_line(msg: str) -> None:
    line = f"{now_iso_z()} | {msg}\n"
    with open(LOGS_DIR / "fetcher.log", "a", encoding="utf-8") as f:
        f.write(line)

def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ---- Simulated fetch (replace with real clients as needed) ----
def simulate_fetch_posts(targets: list, window_days: int) -> list:
    posts = []
    now = datetime.now(timezone.utc)
    for t in targets:
        creator = (t.get("creator") or "unknown_creator").strip()
        platform = (t.get("platform") or "").lower()
        platform = platform if platform in PLATFORM_ENUM else "instagram"
        for i in range(2):  # simulate a couple per target
            ts = (now - timedelta(days=min(i, max(0, window_days - 1)), hours=i)).replace(microsecond=0)
            posts.append({
                "creator": creator,
                "platform": platform,
                "post_id": f"{platform[:2]}_{creator}_{int(ts.timestamp())}_{i}",
                "post_url": f"https://{platform}.com/p/{creator}_{i}",
                "published_at": ts.isoformat().replace("+00:00", "Z"),
                "text": "Example caption about calm parenting and a micro-shift that reduces meltdowns.",
                "metrics": {"views": 12345 + i, "likes": 678 + i, "comments": 12 + i},
                "tags": ["parenting", "calm_parenting", "micro_habits"],
                "language": "en",
                "sentiment_hint": "",
                "thumbnail_url": ""
            })
    return posts

# ---- Normalization / Validation ----
def normalize_post(p: dict, log: list) -> dict | None:
    # Required presence
    req_fields = ["creator", "platform", "post_id", "post_url", "published_at", "text", "metrics"]
    for k in req_fields:
        if k not in p or p[k] in (None, ""):
            log.append({"level": "warn", "msg": f"skip_post: missing {k}"})
            return None

    # Platform enum
    platform = (p["platform"] or "").lower()
    if platform not in PLATFORM_ENUM:
        log.append({"level": "warn", "msg": f"skip_post: bad platform '{platform}'"})
        return None

    # Datetime format
    try:
        published_at = parse_iso_z(p["published_at"])
    except Exception:
        log.append({"level": "warn", "msg": "skip_post: bad published_at"})
        return None

    # Metrics ints
    m = p.get("metrics") or {}
    try:
        views = int(m.get("views", 0))
        likes = int(m.get("likes", 0))
        comments = int(m.get("comments", 0))
    except Exception:
        log.append({"level": "warn", "msg": "skip_post: bad metrics types"})
        return None

    # Text cap
    text = p.get("text") or ""
    if len(text) > TEXT_CAP:
        text = text[:TEXT_CAP]
        log.append({"level": "info", "msg": "truncate:text>600"})

    # Tags cap
    tags = list(p.get("tags") or [])
    if len(tags) > MAX_TAGS:
        tags = tags[:MAX_TAGS]
        log.append({"level": "info", "msg": "truncate:tags>10"})

    # Optional defaults (empty-string/array policy)
    language = p.get("language") or ""
    sentiment_hint = p.get("sentiment_hint") or ""
    thumbnail_url = p.get("thumbnail_url") or ""

    # Assemble in exact contract order
    return {
        "creator": p["creator"],
        "platform": platform,
        "post_id": p["post_id"],
        "post_url": p["post_url"],
        "published_at": published_at,
        "text": text,
        "metrics": {"views": views, "likes": likes, "comments": comments},
        "tags": tags,
        "language": language,
        "sentiment_hint": sentiment_hint,
        "thumbnail_url": thumbnail_url
    }

# ---- Main build ----
def build_output(cfg: dict) -> dict:
    started = time.time()
    # Config mapping: since_window_days (config) → window_days (contract)
    window_days = int(cfg.get("window_days", cfg.get("since_window_days", 7)))
    targets = list(cfg.get("targets", []))

    # Fetch & normalize
    log = []
    raw_posts = simulate_fetch_posts(targets, window_days)
    posts = []
    for p in raw_posts:
        np = normalize_post(p, log)
        if np:
            posts.append(np)
        if len(posts) >= MAX_POSTS_PER_RUN:
            log.append({"level": "info", "msg": "cap:posts=100"})
            break

    status = "ok" if posts else "error"
    reason = "" if posts else "no_valid_posts"
    notes = ""

    run_id = f"fetch_{now_iso_z()}"

    # Assemble top-level in exact contract order
    out = {
        "status": status,
        "run_id": run_id,
        "version": CONTRACT_VERSION,
        "window_days": window_days,
        "targets_total": len(targets),
        "posts": posts,
        "log": log,
        "reason": reason,
        "notes": notes
    }

    # Append deterministic timing into notes
    duration_ms = int((time.time() - started) * 1000)
    out["notes"] = f"duration_ms={duration_ms}"

    return out

def main():
    try:
        cfg = load_config(CONFIG_PATH)
        out = build_output(cfg)
        out_path = RUNS_DIR / f"{out['run_id']}.json"
        atomic_write_json(out_path, out)
        write_log_line(f"run_id={out['run_id']} status={out['status']} posts={len(out['posts'])} file={out_path}")
        # Single-line console echo for Make routers
        print(f"{out['status']} {out_path}")
    except Exception as e:
        # Best-effort error object (still contract-shaped)
        run_id = f"fetch_{now_iso_z()}"
        fail = {
            "status": "error",
            "run_id": run_id,
            "version": CONTRACT_VERSION,
            "window_days": 7,
            "targets_total": 0,
            "posts": [],
            "log": [{"level": "error", "msg": str(e)}],
            "reason": "exception",
            "notes": ""
        }
        out_path = RUNS_DIR / f"{run_id}.json"
        try:
            atomic_write_json(out_path, fail)
        except Exception:
            pass
        write_log_line(f"run_id={run_id} status=error err={e}")
        print("error", str(e))

if __name__ == "__main__":
    main()