# --- standard library ---
import json
import os
from datetime import datetime, UTC  # modern, timezone-aware

# --- third-party ---
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Load Fetcher data ===
src = "/Users/computer/Library/CloudStorage/GoogleDrive-swayclarkeii@gmail.com/My Drive/AI_Agency_Sprint/data/fetcher/runs/fetch_20251017T211544Z.json"
with open(src, "r") as f:
    fetch_data = json.load(f)

posts = fetch_data.get("posts", [])
normalized = []
for p in posts:
    metrics = p.get("engagement", {})
    views = metrics.get("views", 0)
    likes = metrics.get("likes", 0)
    comments = metrics.get("comments", 0)
    score = views*0.6 + likes*0.3 + comments*0.1
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

# === Analyze with LLM ===
prompt = open("/Users/computer/Library/CloudStorage/GoogleDrive-swayclarkeii@gmail.com/My Drive/AI_Agency_Sprint/outputs/day2_prompts/analyzer_v4.0.0.txt").read()
summary = []
for creator, arr in by_creator.items():
    text_block = "\n\n".join([p["text"] or "" for p in arr])
    user_prompt = f"{prompt}\n\nAnalyze posts for creator {creator}:\n{text_block}"
    try:
        res = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":user_prompt}],
            max_tokens=600
        )
        analysis = res.choices[0].message.content
    except Exception as e:
        analysis = f"Error: {e}"
    summary.append({"creator": creator, "analysis": analysis})

# === Save output ===
ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")  # ✅ no deprecation

# Directory ONLY (Make pickup directory)
output_dir = "/Users/computer/Library/CloudStorage/GoogleDrive-swayclarkeii@gmail.com/My Drive/AI_Agency_Sprint/data/analyzer/runs"

# Fail loudly if the directory doesn't exist (no auto-creation)
if not os.path.isdir(output_dir):
    raise FileNotFoundError(f"Missing output directory: {output_dir}")

# Final file path
out_path = os.path.join(output_dir, f"analyzer_{ts}.json")

with open(out_path, "w") as f:
    json.dump({"status": "ok", "summary": summary}, f, indent=2)
print(f"Saved → {out_path}")