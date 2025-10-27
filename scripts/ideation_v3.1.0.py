#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ideation_v3.1.0.py
Local 3-call Ideation runner:
  Call 1: Hooks (spoken_hook_line per idea)
  Call 2: Scripts + Text Hooks + B-roll Ideas
  Kling prompts: local template-fill (no LLM)
  Call 3: Newsletter (subject_lines, preview_lines, body_md)
  Merge -> ideation_v2_3_0 final JSON + Markdown export

Policy: empty-string / empty-array (no nulls).
"""

import os
import re
import json
import glob
import time
import math
import textwrap
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

# --- Path resolution (portable) ---
import os

HERE = os.path.abspath(os.path.dirname(__file__))                         # .../AI_Agency_Sprint/scripts
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))                     # .../AI_Agency_Sprint

# Optional: allow overriding the whole base via env var
DRIVE_ROOT = os.getenv("DRIVE_ROOT", REPO_ROOT)

# ---------- Inputs ----------
ANALYZER_DIR = os.path.join(DRIVE_ROOT, "data", "analyzer", "runs")

# ---------- Outputs ----------
IDEATION_RUNS_DIR    = os.path.join(DRIVE_ROOT, "data", "ideation", "runs")
IDEATION_EXPORTS_DIR = os.path.join(DRIVE_ROOT, "data", "ideation", "exports")
IDEATION_LOGS_DIR    = os.path.join(DRIVE_ROOT, "logs", "ideation")

# ---------- Frameworks ----------
SF_FRAMEWORK           = os.path.join(DRIVE_ROOT, "frameworks", "shortform_script_framework.yaml")
NEWS_FRAMEWORK         = os.path.join(DRIVE_ROOT, "frameworks", "newsletter_framework.yaml")
BROLL_PROMPT_FRAMEWORK = os.path.join(DRIVE_ROOT, "frameworks", "broll_prompt_framework.yaml")  # renamed earlier

# ---------- Databases (env-overridable) ----------
HOOKS_DB = os.getenv(
    "HOOKS_DB",
    os.path.join(DRIVE_ROOT, "data", "kb", "Kallaway Hook Database_all_v2_with metadata_with_embeddings_v3_23.10.2025.csv"),
)
VIDEO_DB = os.getenv(
    "VIDEO_DB",
    os.path.join(DRIVE_ROOT, "data", "kb", "transcripts", "transcripts_video_embeddings_v1.0.0 23.10.2025_with_metadata.csv"),
)
NEWS_DB = os.getenv(
    "NEWS_DB",
    os.path.join(DRIVE_ROOT, "data", "kb", "transcripts", "transcripts_newsletter_embeddings_v1.0.0 23.10.2025_with_metadata.csv"),
)

# ---------- Prompts ----------
PROMPTS_DIR  = os.path.join(DRIVE_ROOT, "prompts", "ideation", "current")
CALL1_PROMPT = os.path.join(PROMPTS_DIR, "Call #1_hooks_prompt_v1_0_0.txt")
CALL2_PROMPT = os.path.join(PROMPTS_DIR, "Call #2_script+text+broll_prompt_v1_0_0 copy.txt")
CALL3_PROMPT = os.path.join(PROMPTS_DIR, "Call #3_newsletter_prompt_v1_0_0 copy 2.txt")

# ---------- Model ----------
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # overridable; safe default
TEMPERATURE = 0.1
MAX_TOKENS = 4000

TOPK_HOOKS = 6
TOPK_VIDEO = 8
TOPK_NEWS  = 8

# ---------- Utilities ----------

def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def latest_json(path: str) -> str:
    files = sorted(glob.glob(os.path.join(path, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {path}")
    return files[-1]

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Dict[str, Any], path: str) -> None:
    # Enforce no nulls: coerce None to "" or [] where sensible.
    def cleanse(v):
        if v is None:
            return ""
        if isinstance(v, dict):
            return {k: cleanse(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [cleanse(x) for x in v]
        return v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleanse(obj), f, ensure_ascii=False, indent=2)

def read_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("Please install pyyaml: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def log_event(message: str, level: str="INFO") -> None:
    ts = utc_iso()
    line = f"{ts} [{level}] {message}\n"
    os.makedirs(IDEATION_LOGS_DIR, exist_ok=True)
    write_text(os.path.join(IDEATION_LOGS_DIR, f"ideation_{ts[:10]}.log"), line if not os.path.exists(os.path.join(IDEATION_LOGS_DIR, f"ideation_{ts[:10]}.log")) else read_text(os.path.join(IDEATION_LOGS_DIR, f"ideation_{ts[:10]}.log")) + line)

# ---------- Simple Top-K Retrieval (keyword filter baseline) ----------

def load_csv_lines(path: str, limit: int = 5000) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i == 0 and "," in line.lower() and "header" in line.lower():
                continue
            lines.append(line.strip())
            if len(lines) >= limit:
                break
    return lines

def topk_by_keywords(lines: List[str], query_terms: List[str], k: int) -> List[str]:
    scored = []
    qset = {t.lower() for t in query_terms if t}
    for ln in lines:
        score = sum(1 for t in qset if t in ln.lower())
        if score > 0:
            scored.append((score, ln))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    out = [ln for _, ln in scored[:k]]
    return out

def seeds_from_analyzer(analyzer: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Analyzer input assumed compliant (idea_seeds exactly 3)
    return analyzer.get("idea_seeds", [])

def hooks_rules_block() -> Dict[str, Any]:
    # Minimal, explicit rubric labels used by Call #1 prompt.
    return {
        "criteria": ["attention_grab", "clarity", "curiosity_gap", "emotional_resonance", "relevance"],
        "notes": "Rate 1-10 per criterion; produce final spoken_hook_line."
    }

def summarize_shortform_framework(yaml_dict: Dict[str, Any]) -> str:
    # Short string summary to keep tokens low.
    prod = yaml_dict.get("production", {})
    beats = yaml_dict.get("production", {}).get("beat_timing", [])
    deriv = yaml_dict.get("derivations", {})
    return json.dumps({
        "length": prod.get("length", ""),
        "hook_window_sec": prod.get("hook_window_sec", ""),
        "beats": beats,
        "text_hooks_from": deriv.get("text_hooks", {}).get("source", "")
    }, ensure_ascii=False)

def summarize_newsletter_framework(yaml_dict: Dict[str, Any]) -> str:
    policy = yaml_dict.get("policy", {})
    return json.dumps({
        "structure": policy.get("structure", {}),
        "format": policy.get("format", {}),
        "keys": ["subject_line", "preview_text", "body", "cta"]
    }, ensure_ascii=False)

def summarize_broll_prompt_framework(yaml_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "policy": yaml_dict.get("policy", {}),
        "cling_prompt_template": yaml_dict.get("cling_prompt_template", ""),
        "defaults": yaml_dict.get("defaults", {})
    }

def build_hooks_exemplars_block(analyzer: Dict[str, Any]) -> Dict[str, Any]:
    seeds = seeds_from_analyzer(analyzer)
    terms = []
    for s in seeds:
        terms += [s.get("idea_title",""), s.get("hook","")] + s.get("takeaways", [])
    lines = load_csv_lines(HOOKS_DB)
    examples = topk_by_keywords(lines, terms, TOPK_HOOKS)
    return {"hook_examples": examples}

def build_video_exemplars_block(analyzer: Dict[str, Any]) -> Dict[str, Any]:
    seeds = seeds_from_analyzer(analyzer)
    terms = []
    for s in seeds:
        terms += [s.get("idea_title",""), s.get("hook","")] + s.get("takeaways", [])
    lines = load_csv_lines(VIDEO_DB)
    examples = topk_by_keywords(lines, terms, TOPK_VIDEO)
    return {"video_examples": examples}

def build_news_exemplars_block(scripts_output: Dict[str, Any]) -> Dict[str, Any]:
    # Use scripts' key phrases as queries
    items = scripts_output.get("items", [])
    phrases = []
    for it in items:
        phrases.append(it.get("hook_line",""))
        # grab a couple lines from script if present
        scr = it.get("short_form_video", {}).get("script","")
        if scr:
            first_line = scr.splitlines()[0] if scr.splitlines() else ""
            phrases.append(first_line)
    lines = load_csv_lines(NEWS_DB)
    examples = topk_by_keywords(lines, phrases, TOPK_NEWS)
    return {"newsletter_examples": examples}

# ---------- Prompt Assembly ----------

def fill_template(template_text: str, mapping: Dict[str, Any]) -> str:
    # Replace {placeholders} with JSON strings when dicts/lists
    filled = template_text
    for k, v in mapping.items():
        if isinstance(v, (dict, list)):
            vv = json.dumps(v, ensure_ascii=False, indent=2)
        else:
            vv = str(v)
        filled = filled.replace("{"+k+"}", vv)
    return filled

# ---------- OpenAI JSON Call ----------

def call_openai_json(prompt_text: str, model: str = MODEL_NAME, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> dict:
    """
    Chat Completions with JSON mode when available.
    Falls back across a small model list if model_not_found occurs.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Please install openai: pip install openai>=1.0.0")

    client = OpenAI()

    # small, sensible fallbacks
    candidates = [model, "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
    last_err = None

    for m in candidates:
        try:
            # try JSON mode first
            resp = client.chat.completions.create(
                model=m,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a JSON-only generator. Return exactly one valid JSON object that matches the contract."},
                    {"role": "user", "content": prompt_text},
                ],
            )
            txt = resp.choices[0].message.content
            return json.loads(txt)
        except TypeError:
            # older SDK without response_format support
            try:
                resp = client.chat.completions.create(
                    model=m,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": "Return ONLY a single valid JSON object that matches the contract. No prose."},
                        {"role": "user", "content": prompt_text},
                    ],
                )
                txt = resp.choices[0].message.content
                try:
                    return json.loads(txt)
                except Exception:
                    mobj = re.search(r'\{[\s\S]*\}\s*$', txt)
                    if mobj:
                        return json.loads(mobj.group(0))
                    raise ValueError(f"Model {m} did not return valid JSON.")
            except Exception as e:
                last_err = e
                continue
        except Exception as e:
            # includes NotFoundError for unavailable model
            last_err = e
            continue

    # if all candidates failed
    raise SystemExit(f"All model attempts failed. Last error: {last_err}")

# ---------- Validators ----------

def words_count(s: str) -> int:
    toks = re.findall(r"[^\s]+", s.strip())
    return len(toks)

def is_4_to_6_words(s: str) -> bool:
    wc = words_count(s)
    return 4 <= wc <= 6

def validate_call1(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    if obj.get("status","") != "ok":
        errs.append("status != ok")
    hooks = obj.get("hooks", [])
    if len(hooks) != 3:
        errs.append("hooks length != 3")
    for i, h in enumerate(hooks):
        if not h.get("idea_title",""):
            errs.append(f"idea {i}: missing idea_title")
        if h.get("angle","") not in ["emotional","practical","logical"]:
            errs.append(f"idea {i}: invalid angle")
        line = h.get("spoken_hook_line","")
        if not line:
            errs.append(f"idea {i}: missing spoken_hook_line")
        rating = h.get("rating", {})
        for k in ["attention_grab","clarity","curiosity_gap","emotional_resonance","relevance"]:
            v = rating.get(k, 0)
            if not isinstance(v, int) or not (1 <= v <= 10):
                errs.append(f"idea {i}: rating.{k} must be int 1-10")
        if "average_score" not in rating:
            errs.append(f"idea {i}: rating.average_score missing")
    return (len(errs) == 0, errs)

def validate_call2(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    if obj.get("status","") != "ok":
        errs.append("status != ok")
    items = obj.get("items", [])
    if len(items) != 3:
        errs.append("items length != 3")
    for i, it in enumerate(items):
        if it.get("angle","") not in ["emotional","practical","logical"]:
            errs.append(f"idea {i}: invalid angle")
        hook_line = it.get("hook_line","")
        if not hook_line:
            errs.append(f"idea {i}: missing hook_line")
        sf = it.get("short_form_video", {})
        dur = sf.get("duration_sec", 0)
        if not isinstance(dur, int) or not (60 <= dur <= 90):
            errs.append(f"idea {i}: duration_sec out of range")
        script = sf.get("script","")
        if not script or hook_line not in script.splitlines()[0]:
            # Require first line contains hook_line (HOOK 0–6s)
            # (Relaxation: we just check first line contains it)
            errs.append(f"idea {i}: script does not begin with hook_line")
        th = sf.get("text_hooks", [])
        if len(th) != 3:
            errs.append(f"idea {i}: text_hooks length != 3")
        for j, t in enumerate(th):
            if not is_4_to_6_words(t):
                errs.append(f"idea {i}: text_hook {j} not 4-6 words")
        br = sf.get("b_roll_ideas", [])
        if len(br) != 3:
            errs.append(f"idea {i}: b_roll_ideas length != 3")
        # Timing: first fixed, subsequent ≈+20–30s. We accept ranges with ≈ markers.
        expected_first = "00:00–00:06"
        if br and br[0].get("timecode","") != expected_first:
            errs.append(f"idea {i}: b-roll[0] must be {expected_first}")
        # We won't enforce exact numerics for ≈ windows; presence check only.
        for j in [1,2]:
            if j < len(br):
                if "≈00:" not in br[j].get("timecode",""):
                    errs.append(f"idea {i}: b-roll[{j}] should be approx 20–30s windows")
            if not br[j].get("line_ref","") or not br[j].get("idea",""):
                errs.append(f"idea {i}: b-roll[{j}] missing line_ref/idea")
    return (len(errs) == 0, errs)

def validate_call3(obj: Dict[str, Any], scripts_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    if obj.get("status","") != "ok":
        errs.append("status != ok")
    newsletters = obj.get("newsletters", [])
    if len(newsletters) != 3:
        errs.append("newsletters length != 3")
    # Build script phrase map for containment check
    scripts_map = {it.get("idea_title",""): it.get("short_form_video", {}).get("script","") for it in scripts_obj.get("items", [])}
    for i, n in enumerate(newsletters):
        subs = n.get("subject_lines", [])
        prevs = n.get("preview_lines", [])
        body = n.get("body_md","")
        if len(subs) != 3:
            errs.append(f"newsletter {i}: subject_lines length != 3")
        if len(prevs) != 3:
            errs.append(f"newsletter {i}: preview_lines length != 3")
        title = n.get("idea_title","")
        script_text = scripts_map.get(title, "")
        if script_text:
            # Must reference at least one token (simple heuristic)
            tokens = [w for w in re.findall(r"[A-Za-z']+", script_text) if len(w) > 4]
            if tokens:
                if not any(t in body for t in tokens[:20]):
                    errs.append(f"newsletter {i}: body_md does not reference script content")
        else:
            errs.append(f"newsletter {i}: missing matching script for containment check")
    return (len(errs) == 0, errs)

# ---------- Kling Prompt Template-fill ----------

def build_kling_prompt(broll_idea: Dict[str, str], tpl: str, defaults: Dict[str, Any]) -> str:
    ctx = {
        "subject": broll_idea.get("idea",""),
        "micro_action": broll_idea.get("idea",""),
        "camera_ops": defaults.get("camera_ops",""),
        "visual_style": defaults.get("visual_style",""),
        "format_ratio": defaults.get("format_ratio","9x16") if "format_ratio" in defaults else "9x16",
        "lighting": defaults.get("lighting",""),
        "safety": defaults.get("safety","PG; no logos; avoid direct minor faces unless consented"),
        "duration_sec": defaults.get("duration_sec", 6),
        "notes": defaults.get("notes","")
    }
    # Fill template variables {var}
    out = tpl
    for k, v in ctx.items():
        out = out.replace("{"+k+"}", str(v))
    return out

def attach_kling_prompts(scripts_output: Dict[str, Any], broll_tpl: str, broll_defaults: Dict[str, Any]) -> Dict[str, Any]:
    obj = json.loads(json.dumps(scripts_output))  # deep copy
    for it in obj.get("items", []):
        bri = it.get("short_form_video", {}).get("b_roll_ideas", [])
        new_list = []
        for b in bri:
            new_b = dict(b)
            new_b["kling_prompt"] = build_kling_prompt(new_b, broll_tpl, broll_defaults)
            new_list.append(new_b)
        it["short_form_video"]["b_roll_ideas"] = new_list
    return obj

# ---------- Merge to Final (Ideation v2_3_0) ----------

def merge_final(hooks: Dict[str, Any], scripts: Dict[str, Any], news: Dict[str, Any], db_versions: Dict[str, str]) -> Dict[str, Any]:
    run_id = f"ideas_{utc_iso()}"
    hooks_map = {h.get("idea_title",""): h for h in hooks.get("hooks", [])}
    scripts_map = {s.get("idea_title",""): s for s in scripts.get("items", [])}
    final_ideas = []
    for n in news.get("newsletters", []):
        title = n.get("idea_title","")
        angle = n.get("angle","")
        h = hooks_map.get(title, {})
        s = scripts_map.get(title, {})
        sf = s.get("short_form_video", {})
        idea = {
            "idea_title": title,
            "hook": h.get("spoken_hook_line",""),
            "angle": angle,
            "short_form_video": {
                "duration_sec": sf.get("duration_sec", 60),
                "script": sf.get("script",""),
                "text_hooks": sf.get("text_hooks", []),
                "b_roll": sf.get("b_roll_ideas", [])  # now includes kling_prompt
            },
            "newsletter_email": {
                "subject_lines": n.get("subject_lines", []),
                "preview_lines": n.get("preview_lines", []),
                "body_md": n.get("body_md","")
            },
            "tags": []
        }
        final_ideas.append(idea)
    return {
        "status": "ok",
        "run_id": run_id,
        "source_run_id": scripts.get("run_id",""),
        "version": "ideation_v2_3_0",
        "summary": "Generated 3 short-form scripts and 3 newsletters.",
        "db_versions": db_versions,
        "ideas": final_ideas,
        "reason": "",
        "notes": ""
    }

def regenerate_call2_for_items_len(call2_template: str,
                                   hooks_out: dict,
                                   sf_block: str,
                                   video_block: dict,
                                   last_output: dict,
                                   model: str = MODEL_NAME) -> dict:
    """Retry Call 2 once if items length != 3 with a corrective instruction."""
    corrective_header = (
        "Your previous response had the wrong length for `items`.\n"
        "You must return exactly 3 items—one for each input hook—in the same order.\n"
        "Regenerate now with items.length == 3. No prose.\n"
    )
    corrective_prompt = fill_template(call2_template, {
        "hooks_output": hooks_out,
        "shortform_script_framework_block": json.loads(sf_block),
        "video_exemplars_block": video_block
    })
    corrective_prompt = corrective_header + "\n\n---\n\n" + corrective_prompt + "\n\n---\n\nPrevious output:\n" + json.dumps(last_output, ensure_ascii=False, indent=2)
    return call_openai_json(corrective_prompt, model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS + 1000)


# ---------- Main Orchestration ----------

def main():
    ensure_dirs(IDEATION_RUNS_DIR, IDEATION_EXPORTS_DIR, IDEATION_LOGS_DIR)

    analyzer_path = latest_json(ANALYZER_DIR)
    analyzer_json = read_json(analyzer_path)
    log_event(f"Loaded analyzer JSON: {analyzer_path}")

    # Build blocks
    sf_yaml   = read_yaml(SF_FRAMEWORK)
    news_yaml = read_yaml(NEWS_FRAMEWORK)
    bpf_yaml  = read_yaml(BROLL_PROMPT_FRAMEWORK)

    sf_block   = summarize_shortform_framework(sf_yaml)
    news_block = summarize_newsletter_framework(news_yaml)
    bpf_block  = summarize_broll_prompt_framework(bpf_yaml)

    hooks_block = build_hooks_exemplars_block(analyzer_json)
    video_block = build_video_exemplars_block(analyzer_json)

    # ---- Call 1: HOOKS ----
    call1_template = read_text(CALL1_PROMPT)
    call1_prompt = fill_template(call1_template, {
        "analyzer_json": analyzer_json,
        "hooks_exemplars_block": hooks_block,
        "hooks_rules_block": hooks_rules_block()
    })
    hooks_out = call_openai_json(call1_prompt)
    ok, errs = validate_call1(hooks_out)
    hooks_path = os.path.join(IDEATION_RUNS_DIR, f"hooks_{utc_iso()}.json")
    write_json(hooks_out, hooks_path)
    if not ok:
        log_event(f"Call 1 validation failed: {errs}", "ERROR")
        raise SystemExit(f"Call 1 failed: {errs}")
    log_event(f"Call 1 OK. Saved: {hooks_path}")

    # ---- Call 2: SCRIPTS ----
    call2_template = read_text(CALL2_PROMPT)
    call2_prompt = fill_template(call2_template, {
        "hooks_output": hooks_out,
        "shortform_script_framework_block": json.loads(sf_block),
        "video_exemplars_block": video_block
    })
    scripts_out = call_openai_json(call2_prompt)

    ok, errs = validate_call2(scripts_out)
    if not ok and any("items length != 3" in e for e in errs):
        log_event(f"Call 2 validation failed (items length). Retrying once...", "WARN")
        # one corrective retry
        scripts_out = regenerate_call2_for_items_len(
            call2_template, hooks_out, sf_block, video_block, scripts_out, model=MODEL_NAME
        )
        ok, errs = validate_call2(scripts_out)

    scripts_path = os.path.join(IDEATION_RUNS_DIR, f"scripts_{utc_iso()}.json")
    write_json(scripts_out, scripts_path)
    if not ok:
        log_event(f"Call 2 validation failed after retry: {errs}", "ERROR")
        raise SystemExit(f"Call 2 failed: {errs}")
    log_event(f"Call 2 OK. Saved: {scripts_path}")

    # ---- Kling prompts (local template-fill) ----
    scripts_with_kling = attach_kling_prompts(
        scripts_out,
        bpf_block.get("cling_prompt_template",""),
        bpf_block.get("defaults", {})
    )
    scripts_kling_path = os.path.join(IDEATION_RUNS_DIR, f"scripts_kling_{utc_iso()}.json")
    write_json(scripts_with_kling, scripts_kling_path)
    log_event(f"Kling prompts attached. Saved: {scripts_kling_path}")

    # ---- Call 3: NEWSLETTER ----
    news_ex_block = build_news_exemplars_block(scripts_with_kling)
    call3_template = read_text(CALL3_PROMPT)
    call3_prompt = fill_template(call3_template, {
        "scripts_output": scripts_with_kling,
        "newsletter_framework_block": json.loads(news_block),
        "newsletter_exemplars_block": news_ex_block
    })
    news_out = call_openai_json(call3_prompt)
    ok, errs = validate_call3(news_out, scripts_with_kling)
    news_path = os.path.join(IDEATION_RUNS_DIR, f"news_{utc_iso()}.json")
    write_json(news_out, news_path)
    if not ok:
        log_event(f"Call 3 validation failed: {errs}", "ERROR")
        raise SystemExit(f"Call 3 failed: {errs}")
    log_event(f"Call 3 OK. Saved: {news_path}")

    # ---- Merge final ----
    db_versions = {
        "hooks_database": os.path.basename(HOOKS_DB),
        "video_database": os.path.basename(VIDEO_DB),
        "newsletter_database": os.path.basename(NEWS_DB),
        "broll_prompt_framework": os.path.basename(BROLL_PROMPT_FRAMEWORK),
        "newsletter_framework": os.path.basename(NEWS_FRAMEWORK),
        "video_framework": os.path.basename(SF_FRAMEWORK)
    }
    final_obj = merge_final(hooks_out, scripts_with_kling, news_out, db_versions)
    final_path = os.path.join(IDEATION_RUNS_DIR, f"ideas_{utc_iso()}.json")
    write_json(final_obj, final_path)
    log_event(f"Final merged JSON saved: {final_path}")

    # ---- Markdown export ----
    md_lines = [f"# Ideation Output — {final_obj.get('run_id','')}", ""]
    for idea in final_obj.get("ideas", []):
        md_lines.append(f"## {idea.get('idea_title','')}")
        md_lines.append("")
        md_lines.append("**Hook**")
        md_lines.append("")
        md_lines.append(f"> {idea.get('hook','')}")
        md_lines.append("")
        md_lines.append("**Script**")
        md_lines.append("```")
        md_lines.append(idea.get("short_form_video",{}).get("script",""))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("**Text Hooks**")
        for t in idea.get("short_form_video",{}).get("text_hooks", []):
            md_lines.append(f"- {t}")
        md_lines.append("")
        md_lines.append("**B-roll + Kling prompts**")
        for br in idea.get("short_form_video",{}).get("b_roll", []):
            tc = br.get("timecode","")
            lr = br.get("line_ref","")
            ic = br.get("idea","")
            kp = br.get("kling_prompt","")
            md_lines.append(f"1. {tc} — *{lr}* → {ic}")
            md_lines.append("")
            md_lines.append("   ```")
            md_lines.append(kp)
            md_lines.append("   ```")
        md_lines.append("")
        md_lines.append("**Newsletter**")
        subs = idea.get("newsletter_email",{}).get("subject_lines", [])
        prevs= idea.get("newsletter_email",{}).get("preview_lines", [])
        body = idea.get("newsletter_email",{}).get("body_md","")
        md_lines.append("")
        for i in range(min(3, len(subs), len(prevs))):
            md_lines.append(f"- **Subject:** {subs[i]}")
            md_lines.append(f"  **Preview:** {prevs[i]}")
        md_lines.append("")
        md_lines.append(body)
        md_lines.append("\n---\n")
    md_path = os.path.join(IDEATION_EXPORTS_DIR, f"ideas_{utc_iso()}.md")
    write_text(md_path, "\n".join(md_lines))
    log_event(f"Markdown export saved: {md_path}")
    print(f"OK\nFinal JSON: {final_path}\nMarkdown: {md_path}")

# ---------- Optional JSON Schema Validation ----------

def get_json_schemas() -> Dict[str, Dict[str, Any]]:
    """Return minimal JSON Schemas for Calls 1–3 + final merge."""
    return {
        "call1": {
            "type": "object",
            "required": ["status", "hooks"],
            "properties": {
                "status": {"enum": ["ok", "error"]},
                "hooks": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "required": ["idea_title", "spoken_hook_line", "angle", "rating"],
                        "properties": {
                            "idea_title": {"type": "string"},
                            "spoken_hook_line": {"type": "string"},
                            "angle": {"enum": ["emotional","practical","logical"]},
                            "rating": {
                                "type": "object",
                                "required": [
                                    "attention_grab","clarity","curiosity_gap",
                                    "emotional_resonance","relevance","average_score"
                                ],
                                "properties": {k: {"type": "integer"} for k in [
                                    "attention_grab","clarity","curiosity_gap",
                                    "emotional_resonance","relevance","average_score"
                                ]}
                            }
                        }
                    }
                }
            }
        },
        "call2": {
            "type": "object",
            "required": ["status","items"],
            "properties": {
                "status": {"enum":["ok","error"]},
                "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "required":["idea_title","hook_line","angle","short_form_video"],
                        "properties": {
                            "idea_title":{"type":"string"},
                            "hook_line":{"type":"string"},
                            "angle":{"enum":["emotional","practical","logical"]},
                            "short_form_video":{
                                "type":"object",
                                "required":["duration_sec","script","text_hooks","b_roll_ideas"],
                                "properties":{
                                    "duration_sec":{"type":"integer"},
                                    "script":{"type":"string"},
                                    "text_hooks":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"string"}},
                                    "b_roll_ideas":{"type":"array","minItems":3,"maxItems":3,"items":{
                                        "type":"object",
                                        "required":["timecode","line_ref","idea"],
                                        "properties":{
                                            "timecode":{"type":"string"},
                                            "line_ref":{"type":"string"},
                                            "idea":{"type":"string"}
                                        }
                                    }}
                                }
                            }
                        }
                    }
                }
            }
        },
        "call3": {
            "type": "object",
            "required":["status","newsletters"],
            "properties":{
                "status":{"enum":["ok","error"]},
                "newsletters":{
                    "type":"array",
                    "minItems":3,
                    "maxItems":3,
                    "items":{
                        "type":"object",
                        "required":["idea_title","angle","subject_lines","preview_lines","body_md"],
                        "properties":{
                            "idea_title":{"type":"string"},
                            "angle":{"enum":["emotional","practical","logical"]},
                            "subject_lines":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"string"}},
                            "preview_lines":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"string"}},
                            "body_md":{"type":"string"}
                        }
                    }
                }
            }
        },
        "final": {
            "type":"object",
            "required":["status","ideas"],
            "properties":{
                "status":{"enum":["ok","error"]},
                "ideas":{"type":"array","minItems":3,"maxItems":3}
            }
        }
    }


def schema_validate(obj: Dict[str, Any], stage: str) -> Tuple[bool, str]:
    """Validate a dict against the stage schema using jsonschema (if installed)."""
    try:
        import jsonschema
    except ImportError:
        return True, "jsonschema not installed (skipped)"
    schemas = get_json_schemas()
    sch = schemas.get(stage)
    if not sch:
        return False, f"Unknown schema stage: {stage}"
    try:
        jsonschema.validate(instance=obj, schema=sch)
        return True, "ok"
    except jsonschema.ValidationError as e:
        return False, f"Schema validation failed at {list(e.path)}: {e.message}"

# ---------- Quick Unit Tests (pytest/CLI) ----------

def _dummy_json(stage: str) -> Dict[str, Any]:
    """Generate minimal dummy objects for smoke tests."""
    if stage == "call1":
        return {
            "status":"ok",
            "hooks":[
                {"idea_title":"A","spoken_hook_line":"Hook line","angle":"emotional",
                 "rating":{"attention_grab":9,"clarity":8,"curiosity_gap":8,
                           "emotional_resonance":9,"relevance":8,"average_score":8}},
                {"idea_title":"B","spoken_hook_line":"Hook line","angle":"practical",
                 "rating":{"attention_grab":8,"clarity":8,"curiosity_gap":7,
                           "emotional_resonance":8,"relevance":7,"average_score":8}},
                {"idea_title":"C","spoken_hook_line":"Hook line","angle":"logical",
                 "rating":{"attention_grab":7,"clarity":8,"curiosity_gap":8,
                           "emotional_resonance":7,"relevance":8,"average_score":8}}
            ]
        }
    if stage == "call2":
        return {
            "status":"ok",
            "items":[
                {"idea_title":"A","hook_line":"Hook","angle":"emotional",
                 "short_form_video":{"duration_sec":70,"script":"Hook...\nScene",
                                     "text_hooks":["A B C D","E F G H","I J K L"],
                                     "b_roll_ideas":[
                                         {"timecode":"00:00–00:06","line_ref":"line","idea":"idea"},
                                         {"timecode":"≈00:20","line_ref":"line","idea":"idea"},
                                         {"timecode":"≈00:50","line_ref":"line","idea":"idea"}]}},
                {"idea_title":"B","hook_line":"Hook","angle":"practical",
                 "short_form_video":{"duration_sec":65,"script":"Hook...\nScene",
                                     "text_hooks":["A B C D","E F G H","I J K L"],
                                     "b_roll_ideas":[
                                         {"timecode":"00:00–00:06","line_ref":"line","idea":"idea"},
                                         {"timecode":"≈00:20","line_ref":"line","idea":"idea"},
                                         {"timecode":"≈00:50","line_ref":"line","idea":"idea"}]}},
                {"idea_title":"C","hook_line":"Hook","angle":"logical",
                 "short_form_video":{"duration_sec":80,"script":"Hook...\nScene",
                                     "text_hooks":["A B C D","E F G H","I J K L"],
                                     "b_roll_ideas":[
                                         {"timecode":"00:00–00:06","line_ref":"line","idea":"idea"},
                                         {"timecode":"≈00:20","line_ref":"line","idea":"idea"},
                                         {"timecode":"≈00:50","line_ref":"line","idea":"idea"}]}}
            ]
        }
    if stage == "call3":
        return {
            "status":"ok",
            "newsletters":[
                {"idea_title":"A","angle":"emotional",
                 "subject_lines":["s1","s2","s3"],
                 "preview_lines":["p1","p2","p3"],
                 "body_md":"story text A"},
                {"idea_title":"B","angle":"practical",
                 "subject_lines":["s1","s2","s3"],
                 "preview_lines":["p1","p2","p3"],
                 "body_md":"story text B"},
                {"idea_title":"C","angle":"logical",
                 "subject_lines":["s1","s2","s3"],
                 "preview_lines":["p1","p2","p3"],
                 "body_md":"story text C"}
            ]
        }
    return {}


def run_unit_tests():
    stages = ["call1","call2","call3"]
    for s in stages:
        obj = _dummy_json(s)
        ok, msg = schema_validate(obj, s)
        print(f"{s}: schema -> {ok}, {msg}")
        if s == "call1":
            print("rules ->", validate_call1(obj))
        elif s == "call2":
            print("rules ->", validate_call2(obj))
        else:
            print("rules ->", validate_call3(obj, _dummy_json('call2')))


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_unit_tests()
    else:
        main()    
