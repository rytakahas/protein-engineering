
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


@dataclass
class PromptBundle:
    text: str
    path: str


def load_prompt(prompt_path: str | Path) -> PromptBundle:
    p = Path(prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return PromptBundle(text=p.read_text(encoding="utf-8"), path=str(p))


def render_prompt(
    prompt_path: str | Path,
    *,
    variables: Optional[Dict[str, Any]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    variables = variables or {}
    bundle = load_prompt(prompt_path)

    snapshot_json = ""
    if snapshot is not None:
        snapshot_json = json.dumps(snapshot, indent=2, ensure_ascii=False)

    ctx: Dict[str, Any] = dict(variables)
    ctx["snapshot_json"] = snapshot_json

    def _repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in ctx:
            raise KeyError(
                f"Missing placeholder variable '{key}' when rendering {bundle.path}. "
                f"Provided keys: {sorted(ctx.keys())}"
            )
        return str(ctx[key])

    return _PLACEHOLDER_RE.sub(_repl, bundle.text)
