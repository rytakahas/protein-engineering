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


def _compact_json(obj: Any) -> str:
    # Much smaller than indent=2; helps avoid token blowups.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def render_prompt(
    prompt_path: str | Path,
    *,
    variables: Optional[Dict[str, Any]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Renders a prompt file with {{placeholders}}.

    Special placeholders supported:
      - {{snapshot_json}} : compact JSON representation of snapshot (or a prebuilt compact string)

    Safety:
      - If snapshot is huge and NOT pre-compacted, we do a last-resort truncation.
        (Best practice is to compact/prune snapshot before calling render_prompt.)
    """
    variables = variables or {}
    bundle = load_prompt(prompt_path)

    snapshot_json = ""
    if snapshot is not None:
        # If caller already compacted it, use that directly
        if isinstance(snapshot, dict) and "_snapshot_json_compact" in snapshot:
            snapshot_json = str(snapshot["_snapshot_json_compact"])
        else:
            snapshot_json = _compact_json(snapshot)

            # Last-resort safety cap by character length
            # (prevents accidental mega prompts when user calls render_prompt directly)
            if len(snapshot_json) > 200_000 and isinstance(snapshot, dict) and "nodes" in snapshot and "edges" in snapshot:
                slim = dict(snapshot)
                slim["nodes"] = (slim.get("nodes") or [])[:220]
                slim["edges"] = (slim.get("edges") or [])[:800]
                snapshot_json = _compact_json(slim)

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

