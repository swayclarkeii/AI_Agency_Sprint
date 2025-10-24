#!/usr/bin/env python3
# analyzer_v5.0.0.py — Contract-compliant Analyzer
# Policy: empty-string / empty-array for unknowns (no nulls)
# JSON-only output downstream (script writes file + prints single status line)

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ---------- Paths ----------
# Repo root heuristic: parent of this script if it contains /data; else cwd.
ROOT_PARENT = Path(__file__).resolve().parents[1]
ROOT = ROOT_PARENT if (ROOT_PARENT / "data").exists() else Path.cwd()

ANALYZER_RUNS_DIR = ROOT / "data" / "analyzer" / "runs"
FETCHER_RUNS_DIR = ROOT / "data" / "fetcher" / "runs"
ANALYZER_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Constants ----------
TOP_N_PER_CREATOR = 3
ANGLE_ENUM = {"emotional", "practical", "logical"}
VERSION = "analyzer_v4"  # locked to contract

# ---------- Time helpers ----------
def now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# ---------- IO ----------
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def latest_fetch_path() -> Path:
    """Return the most recently modified fetch_*.json in FETCHER_RUNS_DIR, or None if none."""
    if not FETCHER_RUNS_DIR.exists():
        return None
    files = sorted(
        FETCHER_RUNS_DIR.glob("fetch_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None

# ---------- Normalization ----------
def to_snake(s: str) -> str:
    return (s or "").strip().replace("-", "_").replace(" ", "_").lower()

def normalize_fetcher_posts(fetcher: dict) -> list:
    posts = []
    for p in fetcher.get("posts", []):
        m = p.get("metrics") or {}
        posts.append({
            "creator": (p.get("creator") or "").strip(),
            "platform": to_snake(p.get("platform") or ""),
            "post_url": p.get("post_url") or "",
            "published_at": p.get("published_at") or "",
            "text": p.get("text") or "",
            "views": int(m.get("views", 0)),
            "likes": int(m.get("likes", 0)),
            "comments": int(m.get("comments", 0)),
            "tags": list(p.get("tags") or [])
        })
    return posts

# ---------- Scoring ----------
def engagement_priority_score(views: int, likes: int, comments: int) -> float:
    # Locked formula
    return views * 0.6 + likes * 0.3 + comments * 0.1

# ---------- Core analysis ----------
def rank_and_select(posts: list) -> list:
    """Basic validity + per-creator ranking by engagement score; return selected posts."""
    by_creator = defaultdict(list)
    for p in posts:
        if not p["creator"] or not p["text"] or not p["published_at"] or not p["post_url"]:
            continue
        p["_eps"] = engagement_priority_score(p["views"], p["likes"], p["comments"])
        by_creator[p["creator"]].append(p)

    selected = []
    for creator, arr in by_creator.items():
        arr.sort(key=lambda x: x["_eps"], reverse=True)
        selected.extend(arr[:TOP_N_PER_CREATOR])
    return selected

def derive_themes(selected: list) -> list:
    """Light heuristic: extract up to 4 lowercase snake_case themes."""
    themes = set()
    for p in selected:
        text = (p["text"] or "").lower()
        if "calm" in text: themes.add("calm_parenting")
        if "tantrum" in text or "meltdown" in text: themes.add("behavior")
        if "micro" in text or "one " in text: themes.add("micro_habits")
        if "confidence" in text: themes.add("confidence")
    if not themes:
        themes.update(["parenting", "tips"])
    return list(themes)[:4]

def derive_hooks(selected: list) -> list:
    """Build 1–3 hooks from selected posts; archetypes are lowercase snake_case."""
    hooks = []
    for p in selected[:3]:
        line = (p["text"] or "").split(".")[0].strip()
        if len(line) < 15:
            line = "This one shift reduces meltdowns fast"
        archetype = "curiosity_gap" if "one " in (p["text"] or "").lower() else "relief_promise"
        hooks.append({"line": line[:100], "archetype": archetype})
    if not hooks:
        hooks = [{"line": "A simple way to stay calm under pressure", "archetype": "relief_promise"}]
    return hooks

def coerce_angle(s: str) -> str:
    s = to_snake(s)
    return s if s in ANGLE_ENUM else "emotional"

def build_idea_seeds(selected: list) -> list:
    """Return exactly 3 contract-compliant idea seeds."""
    base = [
        ("Parenting Under Pressure", "Your calm teaches more than your words.", "emotional",
         ["Short, relief-focused tips perform best.", "Empathy + authority drives saves.", "Model before you manage."]),
        ("90-Second Calm Reset", "One breath that diffuses chaos.", "practical",
         ["Tactical micro-steps beat long lists.", "Use second-person phrasing.", "Lead with a vivid micro-outcome."]),
        ("From Reacting to Coaching", "What to try before you raise your voice.", "logical",
         ["Modeling language beats correction.", "Hooks promising control reduce drop-off.", "Concrete examples win."])
    ]
    ideas = []
    for title, hook, angle, takeaways in base:
        ideas.append({
            "idea_title": title[:60],
            "hook": hook[:100],
            "angle": coerce_angle(angle),
            "takeaways": [t[:100] for t in takeaways][:5]
        })
    return ideas[:3]

# ---------- Validation ----------
def validate_contract(obj: dict) -> (bool, str):
    try:
        if obj.get("status") not in ("ok", "error"): return False, "bad status"
        if not isinstance(obj.get("run_id",""), str): return False, "missing run_id"
        if not isinstance(obj.get("source_run_id",""), str): return False, "missing source_run_id"
        if obj.get("version") != VERSION: return False, "bad version"
        stats = obj.get("stats") or {}
        for k in ("posts_input","posts_analyzed","creators"):
            if not isinstance(stats.get(k, 0), int): return False, f"bad stats.{k}"
        if obj["status"] == "ok":
            if not isinstance(obj.get("top_themes"), list): return False, "top_themes not list"
            if not isinstance(obj.get("top_hooks"), list): return False, "top_hooks not list"
            seeds = obj.get("idea_seeds")
            if not isinstance(seeds, list) or len(seeds) != 3: return False, "idea_seeds length != 3"
            for i in seeds:
                if i.get("angle") not in ANGLE_ENUM: return False, "bad angle enum"
            if not isinstance(obj.get("summary",""), str): return False, "bad summary"
            if "reason" not in obj or "notes" not in obj: return False, "missing reason/notes"
        else:
            if obj.get("summary","") != "": return False, "summary not empty on error"
        return True, ""
    except Exception as e:
        return False, str(e)

# ---------- Main ----------
def main():
    # Accept explicit path, or auto-pick latest fetch file if no arg/--latest
    fetch_path = None
    if len(sys.argv) < 2 or sys.argv[1] in {"--latest", "-L"}:
        fetch_path = latest_fetch_path()
        if not fetch_path:
            print("error no fetch_*.json files found in data/fetcher/runs")
            sys.exit(2)
    else:
        arg = Path(os.path.expanduser(sys.argv[1]))
        fetch_path = arg if arg.is_file() else (ROOT / arg)
        if not fetch_path.is_file():
            print(f"error expected fetcher json at:\n - {arg}\n - {ROOT / arg}")
            sys.exit(2)

    fetcher = load_json(fetch_path)
    source_run_id = fetcher.get("run_id","")
    run_id = f"analyzer_{now_iso_z()}"

    posts_norm = normalize_fetcher_posts(fetcher)
    posts_input = len(posts_norm)
    selected = rank_and_select(posts_norm)
    creators = len({p["creator"] for p in selected})
    posts_analyzed = len(selected)

    if posts_analyzed == 0:
        out = {
            "status": "error",
            "run_id": run_id,
            "source_run_id": source_run_id,
            "version": VERSION,
            "stats": { "posts_input": posts_input, "posts_analyzed": 0, "creators": 0 },
            "top_themes": [],
            "top_hooks": [],
            "idea_seeds": [],
            "summary": "",
            "reason": "no_valid_posts",
            "notes": ""
        }
    else:
        top_themes = derive_themes(selected)
        top_hooks = derive_hooks(selected)
        idea_seeds = build_idea_seeds(selected)
        out = {
            "status": "ok",
            "run_id": run_id,
            "source_run_id": source_run_id,
            "version": VERSION,
            "stats": {
                "posts_input": posts_input,
                "posts_analyzed": posts_analyzed,
                "creators": creators
            },
            "top_themes": top_themes,
            "top_hooks": top_hooks,
            "idea_seeds": idea_seeds,
            "summary": "Concise, relief-promising hooks win; empathy + authority is the top tone.",
            "reason": "",
            "notes": ""
        }

    ok, err = validate_contract(out)
    if not ok:
        out = {
            "status": "error",
            "run_id": run_id,
            "source_run_id": source_run_id,
            "version": VERSION,
            "stats": { "posts_input": posts_input, "posts_analyzed": 0, "creators": 0 },
            "top_themes": [],
            "top_hooks": [],
            "idea_seeds": [],
            "summary": "",
            "reason": f"validation_failed:{err}",
            "notes": ""
        }

    out_path = ANALYZER_RUNS_DIR / f"{run_id}.json"
    atomic_write_json(out_path, out)
    # Keep stdout machine-friendly for Make routers:
    print(f"{out['status']} {out_path}")

if __name__ == "__main__":
    main()