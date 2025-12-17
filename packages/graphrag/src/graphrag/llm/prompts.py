# packages/graphrag/src/graphrag/llm/prompts.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json


@dataclass
class PromptBundle:
    """Loaded prompt + metadata."""
    text: str
    path: str


def load_prompt(prompt_path: str | Path) -> PromptBundle:
    """
    Load a prompt template from a .md/.txt file.
    """
    p = Path(prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    text = p.read_text(encoding="utf-8")
    return PromptBundle(text=text, path=str(p))


def render_prompt(
    prompt_path: str | Path,
    *,
    variables: Optional[Dict[str, Any]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Render a prompt template.

    Supported placeholders (simple):
      - {snapshot_json}  -> JSON dump of retrieval snapshot
      - any {var} from `variables`

    This is intentionally lightweight (no Jinja dependency).
    """
    variables = variables or {}
    bundle = load_prompt(prompt_path)

    # Provide a stable snapshot serialization for the LLM
    snapshot_json = ""
    if snapshot is not None:
        snapshot_json = json.dumps(snapshot, indent=2, ensure_ascii=False)

    # Merge variables
    ctx: Dict[str, Any] = dict(variables)
    ctx["snapshot_json"] = snapshot_json

    try:
        return bundle.text.format(**ctx)
    except KeyError as e:
        missing = str(e).strip("'")
        raise KeyError(
            f"Missing placeholder variable '{missing}' when rendering {bundle.path}. "
            f"Provided keys: {sorted(ctx.keys())}"
        ) from e

