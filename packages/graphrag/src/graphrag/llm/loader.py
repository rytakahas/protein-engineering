from __future__ import annotations
from .base import LLM
from .hf_local import HFLocalLLM
from ..config import LLMConfig


class NoopLLM(LLM):
    def generate(self, prompt: str) -> str:
        # Deterministic fallback: just return prompt header + note
        return (
            "NOOP_LLM_RESPONSE\n"
            "This is a placeholder response because LLM_PROVIDER=noop.\n\n"
            "PROMPT:\n"
            + prompt[:4000]
        )


def load_llm(cfg: LLMConfig) -> LLM:
    if cfg.provider == "hf_local":
        return HFLocalLLM(
            model_id=cfg.model,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
    return NoopLLM()

