#!/usr/bin/env python3
# analyzer_v6.0.0.py — LLM-powered Analyzer (contract-compliant)
# - Uses OpenAI to analyze top posts and produce JSON-only output
# - Keeps v5 features: auto-pick latest fetch file, strict schema, validation, Drive paths
# - Fallback to deterministic local heuristics if the API fails
#
# ENV:
#   OPENAI_API_KEY=sk-...
#   OPENAI_BASE (optional, for proxies/azure)
#   ANALYZER_MODEL (optional, default: gpt-4o-mini or gpt-4-0613 fallback)
#
# USAGE:
#   python scripts/analyzer_v6.0.0.py            # auto-picks latest fetch_*.json
#   python scripts/analyzer_v6.0.0.py --latest   # same as above
#   python scripts/analyzer_v6.0.0.py /path/to/fetch_*.json

import json, sys, os, time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ---------- Absolute Google Drive root (locked as requested) ----------
DRIVE_ROOT = Path("/Users/swayclarke/Library/CloudStorage/GoogleDrive-swayclarkeii@gmail.com/My Drive/AI_Agency_Sprint")
ANALYZER_RUNS_DIR = DRIVE_ROOT / "data" / "analyzer" / "runs"
FETCHER_RUNS_DIR  = DRIVE_ROOT / "data" / "fetcher" / "runs"
ANALYZER_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Constants ----------
TOP_N_PER_CREATOR = 3
ANGLE_ENUM = {"emotional", "practical", "logical"}
VERSION = "analyzer_v4"  # contract-locked
MAX_TAKEAWAYS = 5
MODEL_DEFAULTS = ["gpt-4o-mini", "gpt-4-0613", "gpt-3.5-turbo-1106"]  # try in this order

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

def latest_fetch_path() -> Path | None:
    if not FETCHER_RUNS_DIR.exists():
        return None
    files = sorted(FETCHER_RUNS_DIR.glob("fetch_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
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

# ---------- Scoring / selection ----------
def engagement_priority_score(views: int, likes: int, comments: int) -> float:
    return views * 0.6 + likes * 0.3 + comments * 0.1

def rank_and_select(posts: list) -> list:
    by_creator = defaultdict(list)
    for p in posts:
        if not p["creator"] or not p["text"] or not p["published_at"] or not p["post_url"]:
            continue
        p["_eps"] = engagement_priority_score(p["views"], p["likes"], p["comments"])
        by_creator[p["creator"]].append(p)
    selected = []
    for _, arr in by_creator.items():
        arr.sort(key=lambda x: x["_eps"], reverse=True)
        selected.extend(arr[:TOP_N_PER_CREATOR])
    return selected

# ---------- Heuristic fallbacks (if LLM fails) ----------
def derive_themes_local(selected: list) -> list:
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

def derive_hooks_local(selected: list) -> list:
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

def idea_seeds_local() -> list:
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
            "angle": angle if angle in ANGLE_ENUM else "emotional",
            "takeaways": [t[:100] for t in takeaways][:MAX_TAKEAWAYS]
        })
    return ideas[:3]

# ---------- Contract validation ----------
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

# ---------- LLM call ----------
def openai_client():
    """
    Create a client compatible with both new and legacy SDKs.
    Tries v1 `openai` import first; falls back to legacy style.
    """
    try:
        # New-style SDK (openai>=1.0)
        from openai import OpenAI
        base = os.getenv("OPENAI_BASE")
        kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        if base: kwargs["base_url"] = base
        return ("new", OpenAI(**kwargs))
    except Exception:
        # Legacy SDK (openai<=0.28)
        import openai
        if os.getenv("OPENAI_BASE"):
            openai.api_base = os.getenv("OPENAI_BASE")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return ("legacy", openai)

def choose_model() -> str:
    env = os.getenv("ANALYZER_MODEL", "").strip()
    if env:
        return env
    # try defaults in order; first is our preferred
    return MODEL_DEFAULTS[0]

def build_prompt_block(selected_posts: list, stats: dict) -> str:
    """
    Compact, token-frugal context for the LLM.
    We pass ONLY the selected (ranked) posts (max TOP_N_PER_CREATOR per creator).
    """
    lines = []
    lines.append("ANALYZE THESE SELECTED POSTS (ranked by engagement):")
    for i, p in enumerate(selected_posts, 1):
        lines.append(f"- #{i} | creator:{p['creator']} | platform:{p['platform']} | views:{p['views']} likes:{p['likes']} comments:{p['comments']}")
        txt = (p['text'] or "").replace("\n"," ").strip()
        lines.append(f"  text: {txt[:600]}")
        if p.get("tags"):
            lines.append(f"  tags: {', '.join([t for t in p['tags'][:10]])}")
    lines.append(f"INPUT_STATS: posts_input={stats['posts_input']}, posts_analyzed={stats['posts_analyzed']}, creators={stats['creators']}")
    return "\n".join(lines)

def system_message_contract():
    return (
        "You are an Analyzer that turns selected social posts into structured seeds for Ideation.\n"
        "Return ONLY a single valid JSON object that EXACTLY matches the Output Contract. No markdown, no prose.\n"
        "Use lower_snake_case for all tokens. Angles must be one of: emotional, practical, logical."
    )

def user_message_contract(source_run_id: str):
    return f"""OUTPUT CONTRACT (STRICT)
Return exactly this object. All keys must be present on success; use "" or [] if unknown:

{{
  "status": "ok",
  "run_id": "analyzer_<ISO8601_UTC_Z>",
  "source_run_id": "{source_run_id}",
  "version": "analyzer_v4",
  "stats": {{ "posts_input": <int>, "posts_analyzed": <int>, "creators": <int> }},
  "top_themes": ["<string>", "..."],
  "top_hooks": [ {{ "line": "<string>", "archetype": "<string>" }} ],
  "idea_seeds": [
    {{
      "idea_title": "<string>",
      "hook": "<string>",
      "angle": "emotional|practical|logical",
      "takeaways": ["<string>", "..."]
    }},
    {{ /* exactly 3 items total */ }},
    {{ /* exactly 3 items total */ }}
  ],
  "summary": "<string>",
  "reason": "",
  "notes": ""
}}

ERROR MODE (if no valid posts)
{{
  "status": "error",
  "run_id": "analyzer_<ISO8601_UTC_Z>",
  "source_run_id": "{source_run_id}",
  "version": "analyzer_v4",
  "stats": {{ "posts_input": <int>, "posts_analyzed": 0, "creators": 0 }},
  "top_themes": [],
  "top_hooks": [],
  "idea_seeds": [],
  "summary": "",
  "reason": "no_valid_posts",
  "notes": ""
}}
"""

def call_llm(selected_posts: list, stats: dict, source_run_id: str) -> dict | None:
    mode, client = openai_client()
    model = choose_model()
    run_id = f"analyzer_{now_iso_z()}"

    # Compose messages
    system_msg = system_message_contract()
    context_block = build_prompt_block(selected_posts, stats)
    contract_block = user_message_contract(source_run_id)

    # Prefer JSON-mode if available
    try:
        if mode == "new":
            # New SDK `chat.completions.create`
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},  # JSON-only if model supports it
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": context_block + "\n\n" + contract_block}
                ],
            )
            content = resp.choices[0].message.content
        else:
            # Legacy SDK
            resp = client.ChatCompletion.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": context_block + "\n\n" + contract_block}
                ],
            )
            content = resp.choices[0].message["content"]
        obj = json.loads(content)
        # Inject run_id if model returned a placeholder
        obj["run_id"] = run_id
        obj["source_run_id"] = source_run_id
        obj["version"] = VERSION
        return obj
    except Exception as e:
        # Could log e to a file if needed
        return None

# ---------- Main ----------
def main():
    # Resolve fetch path (explicit or auto-latest)
    if len(sys.argv) < 2 or sys.argv[1] in {"--latest", "-L"}:
        fetch_path = latest_fetch_path()
        if not fetch_path:
            print(f"error no fetch_*.json files found in {FETCHER_RUNS_DIR}")
            sys.exit(2)
    else:
        arg = Path(os.path.expanduser(sys.argv[1]))
        fetch_path = arg if arg.is_file() else (DRIVE_ROOT / arg)
        if not fetch_path.is_file():
            print(f"error expected fetcher json at:\n - {arg}\n - {DRIVE_ROOT / arg}")
            sys.exit(2)

    fetcher = load_json(fetch_path)
    source_run_id = fetcher.get("run_id", "")
    run_id = f"analyzer_{now_iso_z()}"

    posts_norm = normalize_fetcher_posts(fetcher)
    posts_input = len(posts_norm)
    selected = rank_and_select(posts_norm)
    creators = len({p["creator"] for p in selected})
    posts_analyzed = len(selected)

    # No valid posts → error object w/out LLM
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
        stats_obj = {"posts_input": posts_input, "posts_analyzed": posts_analyzed, "creators": creators}

        # Try LLM up to 2 attempts, then local fallback
        out = None
        for attempt in range(2):
            obj = call_llm(selected, stats_obj, source_run_id)
            if obj:
                ok, err = validate_contract(obj)
                if ok:
                    out = obj
                    break
                else:
                    # ask model to self-correct by appending a brief validator hint on retry
                    # (light touch: we simply retry; the system prompt already enforces strict shape)
                    pass
            time.sleep(0.5)

        if out is None:
            # Fallback: local heuristics (keeps pipeline running)
            out = {
                "status": "ok",
                "run_id": run_id,
                "source_run_id": source_run_id,
                "version": VERSION,
                "stats": stats_obj,
                "top_themes": derive_themes_local(selected),
                "top_hooks": derive_hooks_local(selected),
                "idea_seeds": idea_seeds_local(),
                "summary": "Concise, relief-promising hooks win; empathy + authority is the top tone.",
                "reason": "",
                "notes": "llm_fallback"
            }

    # Final validation guardrail (never write malformed JSON)
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

    out_path = ANALYZER_RUNS_DIR / f"{out['run_id']}.json"
    atomic_write_json(out_path, out)
    print(f"{out['status']} {out_path}")

if __name__ == "__main__":
    main()