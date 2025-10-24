# --- standard library ---
import json
import os
import glob
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add project root
from paths import resolve_project_drive

# --- project helper ---
from paths import resolve_project_drive  # uses your .zshrc env or auto-detect

# --- third-party ---
from dotenv import load_dotenv
import openai


# === Setup ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Resolve Google Drive Project Root ===
GDRIVE = resolve_project_drive()

# Define key directories
fetcher_dir = GDRIVE / "data" / "fetcher" / "runs"
prompt_dir = GDRIVE / "outputs" / "day2_prompts"
output_dir = GDRIVE / "data" / "analyzer" / "runs"

# === Load Fetcher data ===
# (You can switch this to the latest file automatically later if you want.)
src = fetcher_dir / "fetch_20251017T211544Z.json"

if not src.is_file():
    raise FileNotFoundError(f"Fetcher file not found: {src}")

with src.open("r") as f:
    fetch_data = json.load(f)

posts = fetch_data.get("posts", [])
normalized = []
for p in posts:
    metrics = p.get("engagement", {})
    views = metrics.get("views", 0)
    likes = metrics.get("likes", 0)
    comments = metrics.get("comments", 0)
    score = views * 0.6 + likes * 0.3 + comments * 0.1
    normalized.append({
        "creator": p.get("creator"),
        "platform": p.get("platform"),
        "post_url": p.get("url"),
        "published_at": p.get("published_at"),
        "text": p.get("caption"),
        "metrics": {"views": views, "likes": likes, "comments": comments},
        "tags": p.get("tags", []),
        "engagement_priority_score": score
    })

# === Rank top 3 posts per creator ===
by_creator = {}
for item in normalized:
    c = item["creator"]
    by_creator.setdefault(c, []).append(item)
for c, arr in by_creator.items():
    arr.sort(key=lambda x: x["engagement_priority_score"], reverse=True)
    by_creator[c] = arr[:3]

# === Find analyzer prompt file ===
matches = sorted(prompt_dir.glob("analyzer_v*.txt"))
if not matches:
    raise FileNotFoundError(f"No analyzer_v*.txt files found in {prompt_dir}")

latest_prompt = matches[-1]
print(f"Using analyzer prompt: {latest_prompt}")

with latest_prompt.open("r") as f:
    prompt = f.read()

# === Analyze with LLM ===
summary = []
for creator, arr in by_creator.items():
    text_block = "\n\n".join([p["text"] or "" for p in arr])
    user_prompt = f"{prompt}\n\nAnalyze posts for creator {creator}:\n{text_block}"
    try:
        res = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=600
        )
        analysis = res.choices[0].message.content
    except Exception as e:
        analysis = f"Error: {e}"
    summary.append({"creator": creator, "analysis": analysis})

# === Save output ===
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

if not output_dir.is_dir():
    raise FileNotFoundError(f"Missing output directory: {output_dir}")

out_path = output_dir / f"analyzer_{ts}.json"

with out_path.open("w") as f:
    json.dump({"status": "ok", "summary": summary}, f, indent=2)

print(f"Saved → {out_path}")