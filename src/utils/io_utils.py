"""File I/O helpers."""
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Any, Dict, List


def save_json(data: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def save_csv(rows: List[Dict], path: str):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Collect all fieldnames across all rows
    fieldnames = sorted(set(k for row in rows for k in row.keys()))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_csv(path: str) -> List[Dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)
