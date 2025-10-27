# clean_transcript.py
import re
import argparse
from pathlib import Path

TIMESTAMP_PATTERNS = [
    r'\b(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\b',      # 1:23 or 01:02:03 or 00:05.123
    r'[\(\[\{]\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\s*[\)\]\}]'  # [00:05], (1:02:03)
]

FILLERS = r'\b(?:uh|um|er|erm|hmm)\b'  # optional

def remove_timestamps(text: str) -> str:
    for pat in TIMESTAMP_PATTERNS:
        text = re.sub(pat, '', text)
    # collapse leftover double spaces from timestamp removal
    text = re.sub(r'[ \t]+', ' ', text)
    return text

def unwrap_lines(text: str) -> str:
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Split into paragraphs on blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    cleaned = []
    for p in paragraphs:
        # Strip leading/trailing spaces per line, drop empty lines
        lines = [ln.strip() for ln in p.split('\n') if ln.strip()]
        # Join with a single space so mid-sentence wraps are fixed
        if not lines:
            continue
        joined = ' '.join(lines)
        # Normalize spaces around dashes/quotes
        joined = re.sub(r'\s+', ' ', joined).strip()
        cleaned.append(joined)
    return '\n\n'.join(cleaned).strip()

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    # Keep a single blank line between paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def remove_fillers(text: str) -> str:
    # Remove isolated filler words; preserve punctuation nearby
    return re.sub(FILLERS, '', text, flags=re.IGNORECASE)

def clean_text(raw: str, drop_fillers: bool = False) -> str:
    t = remove_timestamps(raw)
    t = unwrap_lines(t)
    t = normalize_whitespace(t)
    if drop_fillers:
        t = remove_fillers(t)
        t = normalize_whitespace(t)
    return t

def process_file(src: Path, dst: Path, drop_fillers: bool):
    raw = src.read_text(encoding='utf-8', errors='ignore')
    cleaned = clean_text(raw, drop_fillers=drop_fillers)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding='utf-8')
    print(f"✔ cleaned: {src.name} -> {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean transcripts: remove timestamps, unwrap lines, tidy spaces.")
    ap.add_argument("--in", dest="inp", required=True, help="Input .txt file OR folder with .txt files")
    ap.add_argument("--out", dest="out", required=True, help="Output file OR folder for cleaned .txt")
    ap.add_argument("--drop-fillers", action="store_true", help="Remove simple fillers like 'um', 'uh' (optional)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    if inp.is_dir():
        # Process all .txt files in the folder
        for p in sorted(inp.glob("*.txt")):
            dst = out / p.name if out.is_dir() or not out.suffix else out
            if not out.suffix:  # if output is a folder path
                dst = out / p.name
            process_file(p, dst, args.drop_fillers)
    else:
        # Single file
        if out.is_dir() or not out.suffix:
            out = out / inp.name  # write with same name inside folder
        process_file(inp, out, args.drop_fillers)