from __future__ import annotations
import json
from pathlib import Path


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(path: str, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

