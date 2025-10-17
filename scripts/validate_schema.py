#!/usr/bin/env python3
"""
Validate a JSON file against the Fetcher config schema.

Usage:
  python scripts/validate_schema.py data/fetcher/config.sample.json
"""

import json, sys, pathlib
from jsonschema import validate, Draft202012Validator
from jsonschema.exceptions import ValidationError

# ---- JSON Schema for Fetcher config (Day 3) ----
FETCHER_CONFIG_SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Fetcher Config v1",
  "type": "object",
  "required": ["mode", "run_id", "since_window_days", "targets", "output_contract"],
  "additionalProperties": False,
  "properties": {
    "mode": {
      "type": "string",
      "enum": ["DRY_RUN", "LIVE"],
      "description": "Run mode"
    },
    "run_id": {
      "type": "string",
      "minLength": 1,
      "description": "Unique run identifier, e.g., fetch_2025-10-17T08:00:00Z"
    },
    "since_window_days": {
      "type": "integer",
      "minimum": 1,
      "maximum": 30,
      "description": "Lookback window in days"
    },
    "targets": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["creator_name", "platform", "handle_or_channel_id"],
        "additionalProperties": False,
        "properties": {
          "creator_name": {"type": "string", "minLength": 1},
          "platform": {"type": "string", "enum": ["instagram","tiktok","youtube"]},
          "handle_or_channel_id": {"type": "string", "minLength": 1},
          "notes": {"type": "string"}
        }
      }
    },
    "output_contract": {
      "type": "object",
      "required": ["write_destination","dataset_name"],
      "additionalProperties": False,
      "properties": {
        "write_destination": {"type": "string", "enum": ["local_json","drive_json"]},
        "dataset_name": {"type": "string", "minLength": 1}
      }
    }
  }
}

def main():
  if len(sys.argv) != 2:
    print("Usage: python scripts/validate_schema.py <path-to-json>")
    sys.exit(2)

  path = pathlib.Path(sys.argv[1])
  if not path.exists():
    print(f"ERROR: file not found: {path}")
    sys.exit(2)

  try:
    data = json.loads(path.read_text())
  except json.JSONDecodeError as e:
    print(f"ERROR: invalid JSON: {e}")
    sys.exit(1)

  try:
    Draft202012Validator(FETCHER_CONFIG_SCHEMA).validate(data)
  except ValidationError as e:
    print("INVALID")
    print(f"reason: {e.message}")
    # show where it failed (json-pointer-ish)
    loc = "/".join(str(p) for p in e.path)
    if loc:
      print(f"path: {loc}")
    sys.exit(1)

  print("OK")

if __name__ == "__main__":
  main()